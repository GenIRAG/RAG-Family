from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Type

from RoleRAG.config.types import EvalItem, RAGOutput


_PIPELINE_REGISTRY: Dict[str, Type["BasePipeline"]] = {}


def register_pipeline(name: str, pipeline_class: Type["BasePipeline"]) -> None:
    _PIPELINE_REGISTRY[name] = pipeline_class


def create_pipeline(config: Dict[str, Any]) -> "BasePipeline":
    global_cfg = config["global"]
    pipeline_type = global_cfg["pipeline_type"]

    if pipeline_type not in _PIPELINE_REGISTRY:
        raise ValueError(
            f"Unknown pipeline type: {pipeline_type}. "
            f"Available types: {list(_PIPELINE_REGISTRY.keys())}"
        )

    return _PIPELINE_REGISTRY[pipeline_type](config)


class BasePipeline(ABC):

    config: Dict[str, Any]
    retriever: Any = None
    reasoner: Any = None

    def __init__(self, config: Dict[str, Any], **kwargs):
        self.config = config
        self._log_buffer: Optional[Dict[str, Any]] = None

    def log_begin(self, sample: Optional[EvalItem] = None) -> None:
        self._log_buffer = {}

    def logset(self, key: str, value: Any) -> None:
        if self._log_buffer is not None:
            self._log_buffer[key] = value

    def logappend(self, key: str, value: Any) -> None:
        if self._log_buffer is not None:
            self._log_buffer.setdefault(key, []).append(value)

    def log_end(self) -> Dict[str, Any]:
        log_data = self._log_buffer or {}
        self._log_buffer = None
        return log_data

    def _get_eval_option(self, key: str, evaluation: Optional[Dict[str, Any]]) -> Any:
        if isinstance(evaluation, dict) and key in evaluation and evaluation[key] is not None:
            return evaluation[key]
        global_cfg = self.config["global"]
        evaluation_cfg = global_cfg["evaluation"]
        if key not in evaluation_cfg or evaluation_cfg[key] is None:
            raise KeyError(f"Evaluation config key not found: global.evaluation.{key}")
        return evaluation_cfg[key]

    def _get_method_inference_config(self) -> Dict[str, Any]:
        pipeline_type = self.config["global"]["pipeline_type"]
        inference_cfg = self.config.get("inference", {}) or {}
        method_cfg = inference_cfg.get(pipeline_type)
        if method_cfg is None and pipeline_type.startswith("graphrag_"):
            method_cfg = inference_cfg.get("graphrag")
        return method_cfg or {}

    def _is_retrieval_only(self) -> bool:
        return bool(self._get_method_inference_config().get("retrieval_only", False))

    def _retrieval_only_output(self, query: str, retrieval_result: Any, metadata: Optional[Dict[str, Any]] = None) -> RAGOutput:
        output_metadata = dict(metadata or {})
        output_metadata["retrieval_only"] = True
        return RAGOutput(
            query=query,
            answer="",
            retrieval_result=retrieval_result,
            metadata=output_metadata,
        )

    def run(self, query: Union[str, List[str]], **kwargs) -> Union[RAGOutput, List[RAGOutput]]:
        if isinstance(query, str):
            return self._run_item(query, **kwargs)
        return [self._run_item(q, **kwargs) for q in query]

    def predict(self, sample: EvalItem, **kwargs) -> RAGOutput:
        return self._run_item(sample.query, eval_item=sample, **kwargs)

    @abstractmethod
    def _run_item(self, query: str, **kwargs) -> RAGOutput:

        pass


    @abstractmethod
    def create(self) -> None:
        raise NotImplementedError("Subclasses must implement create()")

    def evaluate(self, dataset, evaluation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from RoleRAG.rolerag.evaluator.evaluators import RAGEvaluator

        metrics = self._get_eval_option("metrics", evaluation)
        show_progress = self._get_eval_option("show_progress", evaluation)
        save_dir = self._get_eval_option("save_dir", evaluation)
        save_metric = self._get_eval_option("save_metric", evaluation)
        verbose = self._get_eval_option("verbose", evaluation)
        print_results = self._get_eval_option("print_results", evaluation)

        evaluator = RAGEvaluator(
            pipeline=self,
            metrics=metrics,
            show_progress=show_progress,
            config=self.config,
            save_dir=save_dir,
            save_metric=save_metric,
            verbose=verbose,
        )
        results = evaluator.run(dataset)
        if print_results:
            evaluator.print_results(results)
        return results

    def evalute(self, *args, **kwargs) -> Dict[str, Any]:
        return self.evaluate(*args, **kwargs)
