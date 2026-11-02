import json
import itertools
import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher

from RoleRAG.config.types import Document, RetrieverResult
from RoleRAG.rolerag.pipelines.prompts import PromptManager


def _str2json(text: str) -> Any:
    if not isinstance(text, str):
        return text

    candidates = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates.extend(fenced)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            return data.get("Response", data)
        except Exception:
            continue

    return text.strip()


def have_not_support_word(answer: str) -> bool:
    not_support_words = [
        "unknown",
        "support",
        "describe",
        "unclear",
        "insufficient",
        "unspecified",
        "mention",
        "evidence",
        "undefined",
        "none",
        "available",
        "information",
        "found",
    ]
    if any(ns_word in answer.lower() for ns_word in ["[]", "[ ]", "not "]):
        return True

    answer_words = re.split(r"[\s,.?!]+", answer)
    return any(ans_word.lower() in not_support_words for ans_word in answer_words if ans_word)


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def doc_to_retrieval(doc: Document) -> Dict[str, Any]:
    lines = doc.content.splitlines()
    metadata = doc.metadata or {}
    title = str(metadata.get("title", "")).strip().strip('"')
    if title:
        text = doc.content
        if lines:
            first_line = lines[0].strip().strip('"')
            if first_line == title:
                text = " ".join(line.strip() for line in lines[1:] if line.strip())
    else:
        title = lines[0].strip().strip('"') if lines else ""
        text = " ".join(line.strip() for line in lines[1:] if line.strip())
    if not text:
        text = doc.content
    return {
        "id": doc.id,
        "title": title,
        "contents": text,
        "document": doc,
        "score": doc.score,
    }


class PERQARetrievalStrategy:

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _generate_json(self, reasoner: Any, system_prompt: str, user_prompt: str, **kwargs) -> Any:
        answer = reasoner.generate(
            user_prompt.format(**kwargs),
            system_prompt=system_prompt,
        )
        return _str2json(answer)

    def _get_prompt_template(self, reasoner: Any, name: str) -> Dict[str, str]:
        prompt_templates = getattr(reasoner, "prompt_templates", {}) or {}
        if name not in prompt_templates:
            raise KeyError(f"PERQA reasoner missing prompt template: {name}")
        return prompt_templates[name]

    def get_type(self, query: str, planner_reasoner: Any) -> str:
        prompt_template = self._get_prompt_template(planner_reasoner, "type_system")
        question_type = self._generate_json(
            planner_reasoner,
            prompt_template.system_prompt,
            prompt_template.user_prompt,
            question=query,
        )
        if isinstance(question_type, str):
            return question_type
        return str(question_type)

    def get_bridge_plan(self, query: str, planner_reasoner: Any) -> Dict[str, List[str]]:
        prompt_template = self._get_prompt_template(planner_reasoner, "bridge_system")
        plan = self._generate_json(
            planner_reasoner,
            prompt_template.system_prompt,
            prompt_template.user_prompt,
            question=query,
        )
        return plan if isinstance(plan, dict) else {}

    def get_comparison_plan(self, query: str, planner_reasoner: Any) -> Dict[str, List[str]]:
        prompt_template = self._get_prompt_template(planner_reasoner, "comparison_system")
        plan = self._generate_json(
            planner_reasoner,
            prompt_template.system_prompt,
            prompt_template.user_prompt,
            question=query,
        )
        return plan if isinstance(plan, dict) else {}

    def run(self, query: str, planner_reasoner: Any) -> Tuple[str, Dict[str, List[str]]]:
        question_type = self.get_type(query, planner_reasoner)
        if question_type.lower() == "comparison":
            return question_type, self.get_comparison_plan(query, planner_reasoner)
        return question_type, self.get_bridge_plan(query, planner_reasoner)

    def get_answers_with_or_without_retrieval(
        self,
        query: str,
        retriever: Any,
        reasoner: Any,
        retrieve_query: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        perqa_cfg = (self.config.get("inference", {}) or {}).get("perqa", {}) or {}
        if retrieve_query is None:
            retrieve_query = query

        retrieval_result = retriever.search(retrieve_query)
        retrievals = [doc_to_retrieval(doc) for doc in retrieval_result.documents]

        if not retrievals:
            return [], ["Not support"]

        retrievals2str = [f"{r['title']}: {r['contents']}" for r in retrievals]
        paragraphs = "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(retrievals2str))
        prompt_name = getattr(reasoner, "prompt_name", None)
        if not prompt_name:
            raise ValueError("reasoner.prompt_name is required for PERQA retrieval")
        prompt_template = PromptManager.get_template(prompt_name)
        answers = self._generate_json(
            reasoner,
            prompt_template.system_prompt,
            prompt_template.user_prompt,
            question=query,
            paragraphs=paragraphs,
        )
        return retrievals, [str(answer) for answer in ensure_list(answers)]

    @staticmethod
    def get_support_answers(answers: List[str]) -> List[str]:
        support_answers = []
        for answer in answers:
            if any(ns_word in answer.lower() for ns_word in ["[]", "[ ]", "not ", "there is no"]):
                continue
            if have_not_support_word(answer):
                continue
            if "no " in answer.lower() and " no " not in answer.lower():
                continue
            support_answers.append(answer)
        return support_answers

    @staticmethod
    def get_mid2nid(intermediate_logs: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        mid2nid: Dict[str, List[str]] = {}
        for key in intermediate_logs:
            key_prefix, full_key = key.split("_")[0], key
            mid2nid.setdefault(key_prefix, []).append(full_key)
        return mid2nid

    @staticmethod
    def get_sample_nodes(
        path: List[Tuple[str, str]],
        graph: nx.DiGraph,
        sample_nodes: List[List[str]],
    ) -> List[List[str]]:
        for node in graph.nodes():
            graph.nodes[node]["name"] = node

        sample_nodes_ = []
        for sample in sample_nodes:
            subgraph = nx.DiGraph()
            combinations = list(itertools.combinations(sample, 2))
            for combination in combinations:
                path_ = tuple(node.split("_")[0] for node in combination)
                if path_ in path:
                    subgraph.add_edges_from([combination])
            for node in subgraph.nodes():
                subgraph.nodes[node]["name"] = node
            matcher = DiGraphMatcher(graph, subgraph, node_match=lambda x, y: x["name"] == y["name"])
            if matcher.subgraph_is_isomorphic():
                sample_nodes_.append(sample)

        return sample_nodes_
    def execute(
        self,
        predicted_plan: Dict[str, List[str]],
        retriever: Any,
        reasoner: Any,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        perqa_cfg = (self.config.get("inference", {}) or {}).get("perqa", {}) or {}
        graph = nx.DiGraph()
        path: List[Tuple[str, str]] = []
        intermediate_logs: Dict[str, Dict[str, Any]] = {}

        for mid, meta in predicted_plan.items():
            if not isinstance(meta, list) or len(meta) < 2:
                continue

            template, tag = meta[0], meta[1]
            placeholders = re.findall(r"<A(\d+)>", template)
            suffix = 1

            current_tag_match = re.findall(r"<A(\d+)>", tag)
            current_mid = f"Q{current_tag_match[0]}" if current_tag_match else mid
            for placeholder in placeholders:
                path.append((f"Q{placeholder}", current_mid))

            if not placeholders:
                combinations = [[]]
            else:
                previous_info = []
                for placeholder in placeholders:
                    current = [
                        [node_id, value["tag"], value["answer"]]
                        for node_id, value in intermediate_logs.items()
                        if node_id.split("_")[0] == f"Q{placeholder}" and value["answer"] != "[Not support]"
                    ]
                    if not current:
                        previous_info = []
                        break
                    previous_info.append(current)
                if len(previous_info) != len(placeholders):
                    break
                combinations = [list(combo) for combo in itertools.product(*previous_info)]

            for combination in combinations:
                current_query = template
                for _, placeholder_tag, padding in combination:
                    current_query = current_query.replace(placeholder_tag, padding)

                supports: List[Dict[str, Any]] = []
                try:
                    supports, answers = self.get_answers_with_or_without_retrieval(
                        current_query,
                        retriever,
                        reasoner,
                    )
                    answers = self.get_support_answers(answers)[: perqa_cfg.get("max_answer_num", 1)]
                except Exception:
                    answers = []

                if answers:
                    for answer in answers:
                        node_id = f"{mid}_{suffix}"
                        intermediate_logs[node_id] = {
                            "template": template,
                            "previous": [f"Q{p}" for p in placeholders],
                            "query": current_query,
                            "rewrite_query": None,
                            "supports": supports,
                            "tag": tag,
                            "answer": answer,
                        }
                        graph.add_node(node_id)
                        for previous_node_id, _, _ in combination:
                            graph.add_edge(previous_node_id, node_id)
                        suffix += 1
                else:
                    node_id = f"{mid}_{suffix}"
                    intermediate_logs[node_id] = {
                        "template": template,
                        "previous": [f"Q{p}" for p in placeholders],
                        "query": current_query,
                        "rewrite_query": None,
                        "supports": supports,
                        "tag": tag,
                        "answer": "[Not support]",
                    }
                    suffix += 1

        intermediate_logs_without_not_support = {
            key: value for key, value in intermediate_logs.items() if value["answer"] != "[Not support]"
        }
        mid_lst = list(predicted_plan.keys())
        nid_lst = list({key.split("_")[0] for key in intermediate_logs_without_not_support})
        mid2nid = self.get_mid2nid(intermediate_logs_without_not_support)

        candidate_graphs: List[Dict[str, Any]] = []
        incomplete_graphs: List[Dict[str, Any]] = []
        sample_nodes = list(mid2nid.values())
        if sample_nodes:
            sample_nodes = [list(sample) for sample in itertools.product(*sample_nodes)]
        else:
            sample_nodes = []

        sample_nodes = self.get_sample_nodes(path, graph, sample_nodes)

        if not (len(mid_lst) != len(nid_lst) or len(mid_lst) == 0):
            for sample in sample_nodes:
                candidate_graphs.append({node_id.split("_")[0]: intermediate_logs[node_id] for node_id in sample})

        if not candidate_graphs and intermediate_logs_without_not_support:
            for sample in sample_nodes:
                incomplete_graphs.append({node_id.split("_")[0]: intermediate_logs[node_id] for node_id in sample})

        metadata = {
            "have_candidate": "[NONE]",
            "intermediate_logs": intermediate_logs,
        }
        if len(candidate_graphs) == 1:
            metadata["have_candidate"] = "[SINGLE]"
        elif len(candidate_graphs) > 1:
            metadata["have_candidate"] = f"[MULTIPLE]__NUM_{len(candidate_graphs)}"
        elif incomplete_graphs:
            metadata["have_candidate"] = "[INCOMPLETE]"

        return incomplete_graphs, candidate_graphs, metadata

    def retrieve(
        self,
        query: str,
        retriever: Any,
        planner_reasoner: Any,
        executor_reasoner: Any,
        **kwargs,
    ) -> RetrieverResult:
        question_type, predicted_plan = self.run(query, planner_reasoner)
        incomplete_graphs, candidate_graphs, metadata = self.execute(
            predicted_plan,
            retriever,
            executor_reasoner,
        )

        metadata.update(
            {
                "question_type": question_type,
                "predicted_plan": predicted_plan,
                "candidate_graphs": candidate_graphs,
                "incomplete_graphs": incomplete_graphs,
            }
        )

        return RetrieverResult(
            documents=[],
            scores=[],
            metadata=metadata,
        )
