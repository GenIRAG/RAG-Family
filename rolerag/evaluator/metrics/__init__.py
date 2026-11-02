import re
import warnings
from abc import ABC
from typing import List, Dict, Any, Set
from enum import Enum
from collections import Counter

import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import DiGraphMatcher

from RoleRAG.config.types import EvalItem, EvalRecord, RAGOutput


class MetricCategory(Enum):

    GENERATION = "generation"
    RETRIEVAL = "retrieval"


class BaseMetric(ABC):

    metric_name = "base"

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def compute(self, response: RAGOutput, sample: EvalItem) -> float:
        raise NotImplementedError

    def calculate_metric(self, data: List[EvalRecord]) -> tuple:

        raise NotImplementedError

    def get_dataset_answer(self, data: List[EvalRecord]) -> List[List[str]]:

        if not data:
            return []

        samples = [record.sample for record in data]
        choices_list = [sample.choices for sample in samples]
        if any(choices == [] for choices in choices_list):
            return [sample.golden_answers for sample in samples]

        golden_choice_idx_list = [sample.golden_answers for sample in samples]
        return [[choices[idx] for idx in idx_list] for choices, idx_list in zip(choices_list, golden_choice_idx_list)]


def normalize_answer(s: str) -> str:

    import re
    import string

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def get_retrieved_texts(response: RAGOutput) -> List[str]:
    retrieved_contexts = response.get_retrieved_contexts()
    texts = []
    for doc in retrieved_contexts:
        if hasattr(doc, 'content'):
            texts.append(doc.content.lower())
        elif isinstance(doc, str):
            texts.append(doc.lower())
    return texts


def get_gt_tokens(sample: EvalItem) -> Set[str]:
    tokens = set()
    for gt in sample.golden_answers:
        normalized = normalize_answer(gt)
        tokens.update(normalized.split())
    return tokens


class ExactMatch(BaseMetric):

    metric_name = "em"
    category = MetricCategory.GENERATION

    def compute(self, response: RAGOutput, sample: EvalItem) -> float:
        return self.calculate_em(response.answer, sample.golden_answers)

    def calculate_em(self, prediction: str, golden_answers: List[str]) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) == normalized_prediction:
                score = 1.0
                break
        return score

    def calculate_metric(self, data: List[EvalRecord]) -> tuple:
        pred_list = [record.prediction.answer for record in data]
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.calculate_em(pred, golden_answers) 
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        em_score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        return {"em": em_score}, metric_score_list


class SubExactMatch(BaseMetric):

    metric_name = "acc"
    category = MetricCategory.GENERATION

    def compute(self, response: RAGOutput, sample: EvalItem) -> float:
        return self.calculate_sub_em(response.answer, sample.golden_answers)

    def calculate_sub_em(self, prediction: str, golden_answers: List[str]) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalized_prediction:
                score = 1.0
                break
        return score

    def calculate_metric(self, data: List[EvalRecord]) -> tuple:
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = [record.prediction.answer for record in data]

        metric_score_list = [
            self.calculate_sub_em(pred, golden_answers) 
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        sub_em_score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        return {"acc": sub_em_score}, metric_score_list


class F1Score(BaseMetric):

    metric_name = "f1"
    category = MetricCategory.GENERATION

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    def compute(self, response: RAGOutput, sample: EvalItem) -> float:
        return self.token_level_scores(response.answer, sample.golden_answers)["f1"]

    def token_level_scores(self, prediction: str, ground_truths: List[str]) -> Dict[str, float]:
        final_metric = {"f1": 0, "precision": 0, "recall": 0}
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        for ground_truth in ground_truths:
            normalized_prediction = normalize_answer(prediction)
            normalized_ground_truth = normalize_answer(ground_truth)

            if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                continue
            
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            for k in ["f1", "precision", "recall"]:
                if final_metric[k] == 0 or f1 > final_metric[k]:
                    final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

    def calculate_metric(self, data: List[EvalRecord]) -> tuple:
        pred_list = [record.prediction.answer for record in data]
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["f1"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        f1 = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        return {"f1": f1}, metric_score_list


class RecallScore(F1Score):

    metric_name = "recall"
    category = MetricCategory.GENERATION

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    def compute(self, response: RAGOutput, sample: EvalItem) -> float:
        return self.token_level_scores(response.answer, sample.golden_answers)["recall"]

    def calculate_metric(self, data: List[EvalRecord]) -> tuple:
        pred_list = [record.prediction.answer for record in data]
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["recall"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        recall = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        return {"recall": recall}, metric_score_list


class PrecisionScore(F1Score):

    metric_name = "precision"
    category = MetricCategory.GENERATION

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    def compute(self, response: RAGOutput, sample: EvalItem) -> float:
        return self.token_level_scores(response.answer, sample.golden_answers)["precision"]

    def calculate_metric(self, data: List[EvalRecord]) -> tuple:
        pred_list = [record.prediction.answer for record in data]
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["precision"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        precision = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        return {"precision": precision}, metric_score_list


_SIM_MODEL_CACHE: Dict[str, Any] = {}

warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set.*",
    category=FutureWarning,
)


def _get_sentence_transformer(model_name: str):
    if model_name in _SIM_MODEL_CACHE:
        return _SIM_MODEL_CACHE[model_name]
    from sentence_transformers import SentenceTransformer

    _SIM_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _SIM_MODEL_CACHE[model_name]


class _PERQAMetricMixin:

    def _get_sim_model(self):
        inference_cfg = self.config.get("inference", {}) or {}
        perqa_cfg = inference_cfg.get("perqa", {}) or {}
        model_name = perqa_cfg.get("plan_similarity_model", "sentence-transformers/all-MiniLM-L6-v2")
        return _get_sentence_transformer(model_name)

    def _dict2graph(self, graph_dict: Dict[str, Any]) -> nx.DiGraph:
        graph = nx.DiGraph()
        pattern = r"<A(\d+)>"

        for nid, node in graph_dict.items():
            template = node[0] if isinstance(node, list) else node.get("template", "")
            graph.add_node(nid, template=template)
            placeholders = re.findall(pattern, template)
            for placeholder in placeholders:
                graph.add_edge(f"Q{placeholder}", nid)

        return graph

    def _template2embedding(self, graph: nx.DiGraph):
        pattern = r"<A(\d+)>"
        nids = [nid for nid in graph.nodes()]
        templates = [attr["template"] for _, attr in graph.nodes(data=True)]
        templates_ = []
        for template in templates:
            placeholders = re.findall(pattern, template)
            for placeholder in placeholders:
                template = template.replace(f"<A{placeholder}>", "entity")
            templates_.append(template)
        embeddings = np.asarray(self._get_sim_model().encode(templates_))
        return nids, embeddings

    def _graph_edit_distance(self, mapping, golden_graph, predicted_graph, beta=0.1):
        golden_nodes_relabel = {}
        predicted_nodes_relabel = {}
        common_nid_suffix = 1
        for key, value in mapping.items():
            golden_nodes_relabel[key] = f"C{common_nid_suffix}"
            predicted_nodes_relabel[value] = f"C{common_nid_suffix}"
            common_nid_suffix += 1

        golden_nodes_relabel.update({node: f"Qg{idx + 1}" for idx, node in enumerate(golden_graph.nodes()) if node not in golden_nodes_relabel})
        predicted_nodes_relabel.update({node: f"Qp{idx + 1}" for idx, node in enumerate(predicted_graph.nodes()) if node not in predicted_nodes_relabel})
        golden_relabel = nx.relabel_nodes(golden_graph, golden_nodes_relabel)
        predicted_relabel = nx.relabel_nodes(predicted_graph, predicted_nodes_relabel)

        a = set(predicted_relabel.nodes()).union(set(golden_relabel.nodes()))
        b = set(predicted_relabel.nodes()).intersection(set(golden_relabel.nodes()))
        c = set(predicted_relabel.edges()).union(set(golden_relabel.edges()))
        d = set(predicted_relabel.edges()).intersection(set(golden_relabel.edges()))
        ged_score = len(a - b) + len(c - d)
        normalized_ged_score = np.exp(-beta * ged_score)
        return ged_score, round(float(normalized_ged_score), 6)

    def _plan_score(self, average_graph_similarity, ged_score, alpha=0.9):
        return alpha * average_graph_similarity + (1 - alpha) * ged_score

    def calculate_plan_score(self, predicted_plan, golden_plan, threshold=0.7, alpha=0.9, beta=1.0):
        if not predicted_plan or not golden_plan:
            return {}, {
                "graph_similarity": [],
                "average_graph_similarity": 0.0,
                "ged_score": 0,
                "normalized_ged_score": 0.0,
                "plan_score": 0.0,
            }

        from sentence_transformers import util

        golden_graph = self._dict2graph(golden_plan)
        predicted_graph = self._dict2graph(predicted_plan)
        golden_zero_in_degree_nodes = [n for n, d in golden_graph.in_degree() if d == 0]
        predicted_zero_in_degree_nodes = [n for n, d in predicted_graph.in_degree() if d == 0]
        golden_nid, golden_embedding = self._template2embedding(golden_graph)
        predicted_nid, predicted_embedding = self._template2embedding(predicted_graph)
        similarity = util.cos_sim(
            np.asarray(golden_embedding),
            np.asarray(predicted_embedding),
        ).tolist()

        mapping = {}
        graph_similarity = []
        init_mapping = {}
        for node in golden_zero_in_degree_nodes:
            node_index = golden_nid.index(node)
            if not similarity[node_index] or max(similarity[node_index]) < threshold:
                continue
            max_sim_index = similarity[node_index].index(max(similarity[node_index]))
            if predicted_nid[max_sim_index] in predicted_zero_in_degree_nodes and predicted_nid[max_sim_index] not in init_mapping.values():
                init_mapping[node] = predicted_nid[max_sim_index]
                graph_similarity.append(similarity[node_index][max_sim_index])

        extra_mapping = {}

        def map_subsequent_nodes(golden_node, predicted_node):
            for successor in golden_graph.successors(golden_node):
                successor_index = golden_nid.index(successor)
                predicted_successors = list(predicted_graph.successors(predicted_node))
                if not predicted_successors:
                    continue
                predicted_successor_embeddings = np.asarray(
                    [predicted_embedding[predicted_nid.index(node)] for node in predicted_successors]
                )
                similarities = util.cos_sim(
                    np.asarray(golden_embedding[successor_index]),
                    predicted_successor_embeddings,
                ).tolist()[0]
                if max(similarities) < threshold:
                    continue
                max_sim_index = similarities.index(max(similarities))
                candidate_successor = predicted_successors[max_sim_index]
                if len(list(golden_graph.predecessors(successor))) != len(list(predicted_graph.predecessors(candidate_successor))):
                    continue
                if candidate_successor not in extra_mapping.values():
                    extra_mapping[successor] = candidate_successor
                    graph_similarity.append(similarities[max_sim_index])
                    map_subsequent_nodes(successor, candidate_successor)

        for golden_node, predicted_node in init_mapping.items():
            map_subsequent_nodes(golden_node, predicted_node)

        mapping.update(init_mapping)
        mapping.update(extra_mapping)

        del_mapping_keys = []
        del_sim_indexes = []
        for golden_node, predicted_node in mapping.items():
            predecessors = list(predicted_graph.predecessors(predicted_node))
            if len(predecessors) > 1:
                for predecessor in predecessors:
                    if predecessor not in mapping.values():
                        del_mapping_keys.append(golden_node)
                        del_sim_indexes.append(list(mapping.keys()).index(golden_node))

        for key in del_mapping_keys:
            mapping.pop(key, None)

        graph_similarity = [round(float(score), 6) for idx, score in enumerate(graph_similarity) if idx not in del_sim_indexes]
        average_graph_similarity = sum(graph_similarity) / len(golden_graph.nodes()) if golden_graph.nodes() else 0.0
        ged_score, normalized_ged_score = self._graph_edit_distance(mapping, golden_graph, predicted_graph, beta=beta)
        plan_score = round(float(self._plan_score(average_graph_similarity, normalized_ged_score, alpha=alpha)), 6)
        return mapping, {
            "graph_similarity": graph_similarity,
            "average_graph_similarity": round(float(average_graph_similarity), 6),
            "ged_score": ged_score,
            "normalized_ged_score": normalized_ged_score,
            "plan_score": plan_score,
        }

    def calculate_step_score(self, final_graphs, incomplete_graphs, golden_graphs, mapping):
        if final_graphs:
            evaluate_graphs = [final_graph for final_graph, _ in final_graphs]
        elif incomplete_graphs:
            evaluate_graphs = incomplete_graphs
            mapping = {golden_node: predicted_node for golden_node, predicted_node in mapping.items() if predicted_node in list(incomplete_graphs[0].keys())}
        else:
            return {"step_score": 0.0}

        if not golden_graphs or not mapping:
            return {"step_score": 0.0}

        pred_eval_order = {}
        gold_eval_order = {}
        for golden_node, predicted_node in mapping.items():
            gold_eval_order[golden_node] = list(set(graph[golden_node]["answer"] for graph in golden_graphs if golden_node in graph))
            pred_eval_order[predicted_node] = list(set(graph[predicted_node]["answer"] for graph in evaluate_graphs if predicted_node in graph))

        pred_step_scores = []
        f1_evaluator = F1Score(self.config)
        for golden_node, predicted_node in mapping.items():
            gold_step_answers = gold_eval_order.get(golden_node, [])
            pred_step_answers = pred_eval_order.get(predicted_node, [])
            if not gold_step_answers or not pred_step_answers:
                pred_step_scores.append(0.0)
                continue
            pred_step_score = [f1_evaluator.token_level_scores(pred_step_answer, gold_step_answers)["f1"] for pred_step_answer in pred_step_answers]
            pred_step_scores.append(sum(pred_step_score) / len(pred_step_score))

        final_step_score = round(sum(pred_step_scores) / len(golden_graphs[0]), 6) if golden_graphs and golden_graphs[0] else 0.0
        return {"step_score": final_step_score}

    def calculate_final_score(self, plan_score, step_score):
        return (2 * plan_score * step_score) / (plan_score + step_score + 1e-6)

    def calculate_pse_bundle(self, item: EvalRecord) -> Dict[str, Any]:
        sample_metadata = item.sample.metadata or {}
        output_metadata = item.prediction.metadata or {}
        golden_plan = sample_metadata.get("plan", {})
        golden_graphs = sample_metadata.get("graph", [])
        predicted_plan = output_metadata.get("predicted_plan", {})
        final_graphs = output_metadata.get("final_graphs", [])
        incomplete_graphs = output_metadata.get("incomplete_graphs", [])
        mapping, plan_scores = self.calculate_plan_score(predicted_plan, golden_plan)
        step_score = self.calculate_step_score(final_graphs, incomplete_graphs, golden_graphs, mapping)
        pse_g_score = self.calculate_final_score(plan_scores["plan_score"], step_score["step_score"])
        result = {
            "mapping": mapping,
            "pse_g": round(float(pse_g_score), 6),
        }
        result.update(plan_scores)
        result.update(step_score)
        return result


class PlanScore(_PERQAMetricMixin, BaseMetric):

    metric_name = "plan_score"
    category = MetricCategory.GENERATION

    def calculate_metric(self, data: List[EvalRecord]) -> tuple:
        metric_score_list = []
        for item in data:
            metric_score_list.append(self.calculate_pse_bundle(item)["plan_score"])
        score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        return {"plan_score": score}, metric_score_list


class StepScore(_PERQAMetricMixin, BaseMetric):

    metric_name = "step_score"
    category = MetricCategory.GENERATION

    def calculate_metric(self, data: List[EvalRecord]) -> tuple:
        metric_score_list = []
        for item in data:
            metric_score_list.append(self.calculate_pse_bundle(item)["step_score"])
        score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        return {"step_score": score}, metric_score_list


class PSEGScore(_PERQAMetricMixin, BaseMetric):

    metric_name = "pse_g"
    category = MetricCategory.GENERATION

    def calculate_metric(self, data: List[EvalRecord]) -> tuple:
        metric_score_list = []
        for item in data:
            metric_score_list.append(self.calculate_pse_bundle(item)["pse_g"])
        score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        return {"pse_g": score}, metric_score_list


class GraphSimilarityScore(_PERQAMetricMixin, BaseMetric):

    metric_name = "graph_similarity"
    category = MetricCategory.GENERATION

    def calculate_metric(self, data: List[EvalRecord]) -> tuple:
        metric_score_list = []
        for item in data:
            graph_similarity = self.calculate_pse_bundle(item)["graph_similarity"]
            metric_score_list.append(sum(graph_similarity) / len(graph_similarity) if graph_similarity else 0.0)
        score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        return {"graph_similarity": score}, metric_score_list


class AverageGraphSimilarityScore(_PERQAMetricMixin, BaseMetric):

    metric_name = "average_graph_similarity"
    category = MetricCategory.GENERATION

    def calculate_metric(self, data: List[EvalRecord]) -> tuple:
        metric_score_list = []
        for item in data:
            metric_score_list.append(self.calculate_pse_bundle(item)["average_graph_similarity"])
        score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        return {"average_graph_similarity": score}, metric_score_list


class GEDScore(_PERQAMetricMixin, BaseMetric):

    metric_name = "ged_score"
    category = MetricCategory.GENERATION

    def calculate_metric(self, data: List[EvalRecord]) -> tuple:
        metric_score_list = []
        for item in data:
            metric_score_list.append(self.calculate_pse_bundle(item)["ged_score"])
        score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        return {"ged_score": score}, metric_score_list


class NormalizedGEDScore(_PERQAMetricMixin, BaseMetric):

    metric_name = "normalized_ged_score"
    category = MetricCategory.GENERATION

    def calculate_metric(self, data: List[EvalRecord]) -> tuple:
        metric_score_list = []
        for item in data:
            metric_score_list.append(self.calculate_pse_bundle(item)["normalized_ged_score"])
        score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        return {"normalized_ged_score": score}, metric_score_list



def is_relevant_doc(doc_text: str, golden_answers: List[str]) -> bool:
    doc_normalized = normalize_answer(doc_text)
    for gt in golden_answers:
        gt_normalized = normalize_answer(gt)
        if gt_normalized and gt_normalized in doc_normalized:
            return True
    return False


def get_retrieval_docs(retrieval_result) -> List[str]:
    if retrieval_result is None:
        return []
    if hasattr(retrieval_result, 'documents'):
        docs = retrieval_result.documents
    else:
        docs = retrieval_result if isinstance(retrieval_result, list) else []
    texts = []
    for doc in docs:
        if hasattr(doc, 'content'):
            texts.append(doc.content)
        elif isinstance(doc, dict) and 'contents' in doc:
            texts.append(doc['contents'])
        elif isinstance(doc, str):
            texts.append(doc)
    return texts


class RetrievalMetric(BaseMetric):

    category = MetricCategory.RETRIEVAL

    def __init__(self, k: int = 10, config: Dict[str, Any] = None):
        super().__init__(config)
        self.k = k

    def compute(self, response: RAGOutput, sample: EvalItem) -> float:
        if not sample.golden_answers:
            return 0.0
        retrieved_texts = get_retrieved_texts(response)
        return self._compute_metric(retrieved_texts, sample.golden_answers)

    def _compute_metric(self, retrieved_texts: List[str], golden_answers: List[str]) -> float:
        raise NotImplementedError

    def calculate_metric(self, data: List[EvalRecord]) -> tuple:
        retrieval_results = [record.prediction.retrieval_result for record in data]
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = []
        for retrieval_result, golden_answers in zip(retrieval_results, golden_answers_list):
            retrieved_texts = get_retrieval_docs(retrieval_result)
            score = self._compute_metric(retrieved_texts, golden_answers)
            metric_score_list.append(score)

        avg_score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        return {self.metric_name: avg_score}, metric_score_list


class Acc4R(RetrievalMetric):

    metric_name = "acc4r"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(k=1, config=config)

    def _compute_metric(self, retrieved_texts: List[str], golden_answers: List[str]) -> float:
        if not retrieved_texts:
            return 0.0
        return 1.0 if is_relevant_doc(retrieved_texts[0], golden_answers) else 0.0


class RecallAtK(RetrievalMetric):

    metric_name = "recall_at_k"

    def _compute_metric(self, retrieved_texts: List[str], golden_answers: List[str]) -> float:
        k_docs = retrieved_texts[:self.k]
        if not k_docs:
            return 0.0
        return 1.0 if any(is_relevant_doc(doc, golden_answers) for doc in k_docs) else 0.0

    def __init__(self, k: int = 10, config: Dict[str, Any] = None):
        super().__init__(k, config)
        self.metric_name = f"recall@{k}"


class PrecisionAtK(RetrievalMetric):

    def __init__(self, k: int = 10, config: Dict[str, Any] = None):
        super().__init__(k, config)
        self.metric_name = f"precision@{k}"

    def _compute_metric(self, retrieved_texts: List[str], golden_answers: List[str]) -> float:
        k_docs = retrieved_texts[:self.k]
        if not k_docs:
            return 0.0
        relevant_in_k = sum(1 for doc in k_docs if is_relevant_doc(doc, golden_answers))
        return relevant_in_k / len(k_docs)


class F1AtK(RetrievalMetric):

    def __init__(self, k: int = 10, config: Dict[str, Any] = None):
        super().__init__(k, config)
        self.metric_name = f"f1@{k}"

    def _compute_metric(self, retrieved_texts: List[str], golden_answers: List[str]) -> float:
        k_docs = retrieved_texts[:self.k]
        if not k_docs:
            return 0.0
        relevant_in_k = sum(1 for doc in k_docs if is_relevant_doc(doc, golden_answers))
        precision = relevant_in_k / len(k_docs)
        recall = 1.0 if relevant_in_k > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class MAPAtK(RetrievalMetric):

    def __init__(self, k: int = 10, config: Dict[str, Any] = None):
        super().__init__(k, config)
        self.metric_name = f"map@{k}"

    def _compute_metric(self, retrieved_texts: List[str], golden_answers: List[str]) -> float:
        k_docs = retrieved_texts[:self.k]
        if not k_docs:
            return 0.0
        hit_precisions = []
        relevant_count = 0
        for i, doc in enumerate(k_docs):
            if is_relevant_doc(doc, golden_answers):
                relevant_count += 1
                hit_precisions.append(relevant_count / (i + 1))
        if not hit_precisions:
            return 0.0
        return sum(hit_precisions) / len(hit_precisions)


class MRRAtK(RetrievalMetric):

    def __init__(self, k: int = 10, config: Dict[str, Any] = None):
        super().__init__(k, config)
        self.metric_name = f"mrr@{k}"

    def _compute_metric(self, retrieved_texts: List[str], golden_answers: List[str]) -> float:
        k_docs = retrieved_texts[:self.k]
        for i, doc in enumerate(k_docs):
            if is_relevant_doc(doc, golden_answers):
                return 1.0 / (i + 1)
        return 0.0


class NDCGAtK(RetrievalMetric):

    def __init__(self, k: int = 10, config: Dict[str, Any] = None):
        super().__init__(k, config)
        self.metric_name = f"ndcg@{k}"

    def _compute_dcg(self, rel_list: List[float]) -> float:
        dcg = 0.0
        for i, rel in enumerate(rel_list[:self.k]):
            dcg += rel / np.log2(i + 2)
        return dcg

    def _compute_idcg(self, rel_list: List[float]) -> float:
        sorted_rel = sorted(rel_list, reverse=True)
        return self._compute_dcg(sorted_rel)

    def _compute_metric(self, retrieved_texts: List[str], golden_answers: List[str]) -> float:
        k_docs = retrieved_texts[:self.k]
        relevances = [1.0 if is_relevant_doc(doc, golden_answers) else 0.0 for doc in k_docs]
        if not relevances:
            return 0.0
        dcg = self._compute_dcg(relevances)
        idcg = self._compute_idcg(relevances)
        if idcg == 0:
            return 0.0
        return dcg / idcg


GENERATION_METRIC_CLASSES = {
    "em": ExactMatch,
    "f1": F1Score,
    "recall": RecallScore,
    "precision": PrecisionScore,
    "acc": SubExactMatch,
}

PERQA_METRIC_CLASSES = {
    "plan_score": PlanScore,
    "step_score": StepScore,
    "pse_g": PSEGScore,
    "graph_similarity": GraphSimilarityScore,
    "average_graph_similarity": AverageGraphSimilarityScore,
    "ged_score": GEDScore,
    "normalized_ged_score": NormalizedGEDScore,
}

STATIC_RETRIEVAL_METRIC_FACTORIES = {
    "acc4r": lambda config=None: Acc4R(config=config),
    "recall@5": lambda config=None: RecallAtK(k=5, config=config),
    "recall@10": lambda config=None: RecallAtK(k=10, config=config),
    "recall@20": lambda config=None: RecallAtK(k=20, config=config),
    "precision@5": lambda config=None: PrecisionAtK(k=5, config=config),
    "precision@10": lambda config=None: PrecisionAtK(k=10, config=config),
    "precision@20": lambda config=None: PrecisionAtK(k=20, config=config),
    "f1@5": lambda config=None: F1AtK(k=5, config=config),
    "f1@10": lambda config=None: F1AtK(k=10, config=config),
    "f1@20": lambda config=None: F1AtK(k=20, config=config),
    "map@5": lambda config=None: MAPAtK(k=5, config=config),
    "map@10": lambda config=None: MAPAtK(k=10, config=config),
    "map@20": lambda config=None: MAPAtK(k=20, config=config),
    "mrr@5": lambda config=None: MRRAtK(k=5, config=config),
    "mrr@10": lambda config=None: MRRAtK(k=10, config=config),
    "mrr@20": lambda config=None: MRRAtK(k=20, config=config),
    "ndcg@5": lambda config=None: NDCGAtK(k=5, config=config),
    "ndcg@10": lambda config=None: NDCGAtK(k=10, config=config),
    "ndcg@20": lambda config=None: NDCGAtK(k=20, config=config),
}

DYNAMIC_RETRIEVAL_METRIC_PREFIXES = {
    "recall@": RecallAtK,
    "precision@": PrecisionAtK,
    "f1@": F1AtK,
    "map@": MAPAtK,
    "mrr@": MRRAtK,
    "ndcg@": NDCGAtK,
}

GENERATION_METRICS = {
    **GENERATION_METRIC_CLASSES,
    **PERQA_METRIC_CLASSES,
}
RETRIEVAL_METRICS = dict(STATIC_RETRIEVAL_METRIC_FACTORIES)
METRICS = {
    **GENERATION_METRICS,
    **RETRIEVAL_METRICS,
}


def _build_dynamic_topk_metric(name: str, config: Dict[str, Any] = None) -> BaseMetric:
    for prefix, metric_cls in DYNAMIC_RETRIEVAL_METRIC_PREFIXES.items():
        if not name.startswith(prefix):
            continue
        try:
            k = int(name.split("@", 1)[1])
        except (IndexError, ValueError):
            k = 10
        return metric_cls(k=k, config=config)
    return None


def get_metric(name: str, config: Dict[str, Any] = None) -> BaseMetric:
    metric_name = str(name).lower()

    if metric_name in GENERATION_METRIC_CLASSES:
        return GENERATION_METRIC_CLASSES[metric_name](config)

    if metric_name in PERQA_METRIC_CLASSES:
        return PERQA_METRIC_CLASSES[metric_name](config)

    if metric_name in STATIC_RETRIEVAL_METRIC_FACTORIES:
        return STATIC_RETRIEVAL_METRIC_FACTORIES[metric_name](config)

    dynamic_metric = _build_dynamic_topk_metric(metric_name, config)
    if dynamic_metric is not None:
        return dynamic_metric

    available_metrics = sorted(METRICS.keys())
    dynamic_prefixes = sorted(DYNAMIC_RETRIEVAL_METRIC_PREFIXES.keys())
    raise ValueError(
        f"Unknown metric '{name}'. "
        f"Available metrics: {available_metrics}. "
        f"Dynamic metric prefixes: {dynamic_prefixes}"
    )
