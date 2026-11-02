import json
import re
from typing import Any, Dict, List, Tuple

from RoleRAG.config.types import RetrieverResult
from RoleRAG.rolerag.pipelines.prompts.prompt_manager import PromptTemplate


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


def _build_templates_from_reasoner(reasoner: Any) -> Tuple[PromptTemplate, PromptTemplate]:
    prompt_templates = getattr(reasoner, "prompt_templates", {}) or {}
    if "bridge_aggregate" not in prompt_templates:
        raise KeyError("PERQA reasoner missing prompt template: bridge_aggregate")
    if "comparison_aggregate" not in prompt_templates:
        raise KeyError("PERQA reasoner missing prompt template: comparison_aggregate")
    return (
        prompt_templates["bridge_aggregate"],
        prompt_templates["comparison_aggregate"],
    )


class PERQAGenerationStrategy:
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def aggregate(
        self,
        question_type: str,
        question: str,
        candidate_graphs: List[Dict[str, Any]],
        reasoner: Any,
    ) -> List[List[Any]]:
        bridge_template, comparison_template = _build_templates_from_reasoner(reasoner)
        aggregate_template = comparison_template if question_type.lower() == "comparison" else bridge_template

        final_graphs = []
        for candidate_graph in candidate_graphs:
            evidence = [f"{value['query']} >> {value['answer']}" for _, value in candidate_graph.items()]
            paragraphs = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(evidence))
            answer = reasoner.generate(
                aggregate_template.user_prompt.format(question=question, paragraphs=paragraphs),
                system_prompt=aggregate_template.system_prompt,
            )
            final_answer = _str2json(answer)
            final_graphs.append([candidate_graph, str(final_answer).strip()])

        if not final_graphs:
            return []

        answer_counts: Dict[str, int] = {}
        for _, answer in final_graphs:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        max_count = max(answer_counts.values())
        return [graph for graph in final_graphs if answer_counts[graph[1]] == max_count]

    def generate(
        self,
        question: str,
        retrieval_result: RetrieverResult,
        reasoner: Any,
    ) -> Tuple[str, List[List[Any]]]:
        metadata = retrieval_result.metadata or {}
        question_type = str(metadata.get("question_type"))
        candidate_graphs = metadata.get("candidate_graphs")

        if not candidate_graphs:
            return "", []

        final_graphs = self.aggregate(
            question_type=question_type,
            question=question,
            candidate_graphs=candidate_graphs,
            reasoner=reasoner,
        )
        if not final_graphs:
            return "", []
        return final_graphs[0][1], final_graphs


def aggregate(
    question_type: str,
    question: str,
    candidate_graphs: List[Dict[str, Any]],
    reasoner: Any,
) -> List[List[Any]]:
    return PERQAGenerationStrategy().aggregate(
        question_type=question_type,
        question=question,
        candidate_graphs=candidate_graphs,
        reasoner=reasoner,
    )


def generate_perqa_answer(
    question: str,
    retrieval_result: RetrieverResult,
    reasoner: Any,
) -> Tuple[str, List[List[Any]]]:
    return PERQAGenerationStrategy().generate(
        question=question,
        retrieval_result=retrieval_result,
        reasoner=reasoner,
    )
