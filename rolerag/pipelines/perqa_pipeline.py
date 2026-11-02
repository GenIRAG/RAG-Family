import logging
from typing import Any, Dict

from RoleRAG.roles.utils import assist_role, get_method_role_config
from RoleRAG.config.types import RAGOutput
from RoleRAG.rolerag.pipelines.pipelines_base import BasePipeline, register_pipeline
from RoleRAG.models import get_model
from RoleRAG.rolerag.pipelines.generation_strategy.perqa_generation_strategy import PERQAGenerationStrategy
from RoleRAG.rolerag.pipelines.retrieval_strategy import PERQARetrievalStrategy

class PERQAPipeline(BasePipeline):
    def __init__(
        self,
        config: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.index_cfg = config["index"]
        self.paths_cfg = self.index_cfg["paths"]
        self.bm25_index_path = self.paths_cfg["bm25_index_path"]

        self.retriever_cfg = get_method_role_config(config, "retriever", "BM25Model")
        self.retriever_cfg["index_path"] = self.bm25_index_path
        self.reasoner_cfg = get_method_role_config(config, "reasoner", "OpenAIGenerator", "answer_reasoner")
        self.planner_reasoner_cfg = get_method_role_config(config, "reasoner", "OpenAIGenerator", "planner_reasoner")
        self.executor_reasoner_cfg = get_method_role_config(config, "reasoner", "OpenAIGenerator", "executor_reasoner")

        self.retriever = None
        self.reasoner = None
        self.planner_reasoner = None
        self.executor_reasoner = None
        self.retrieval_strategy = PERQARetrievalStrategy(config)
        self.generation_strategy = PERQAGenerationStrategy(config)
        self.create()

    def create(self) -> None:
        bm25_model = get_model("BM25Model")
        self.retriever = assist_role(bm25_model, "retriever", self.retriever_cfg)

        reasoner_model = get_model("OpenAIGenerator")

        self.reasoner = assist_role(
            reasoner_model,
            "reasoner",
            self.reasoner_cfg,
        )
        self.planner_reasoner = assist_role(
            reasoner_model,
            "reasoner",
            self.planner_reasoner_cfg,
        )
        self.executor_reasoner = assist_role(
            reasoner_model,
            "reasoner",
            self.executor_reasoner_cfg,
        )
    def _run_item(self, query: str, **kwargs) -> RAGOutput:
        
        retrieval_result = self.retrieval_strategy.retrieve(
            query=query,
            retriever=self.retriever,
            planner_reasoner=self.planner_reasoner,
            executor_reasoner=self.executor_reasoner,
        )

        if self._is_retrieval_only():
            metadata = retrieval_result.metadata or {}
            return self._retrieval_only_output(query, retrieval_result, metadata)
        
        answer, final_graphs = self.generation_strategy.generate(
            question=query,
            retrieval_result=retrieval_result,
            reasoner=self.reasoner,
        )

        metadata = retrieval_result.metadata or {}
        
        self.logset("type", metadata.get("question_type"))
        self.logset("predicted_plan", metadata.get("predicted_plan"))
        self.logset("have_candidate", metadata.get("have_candidate"))
        self.logset("intermediate_logs", self._compact_supports_for_log(metadata.get("intermediate_logs")))
        self.logset("candidate_graphs", self._compact_supports_for_log(metadata.get("candidate_graphs")))
        self.logset("incomplete_graphs", self._compact_supports_for_log(metadata.get("incomplete_graphs")))
        self.logset("final_graphs", self._compact_supports_for_log(final_graphs))
        metadata["final_graphs"] = final_graphs

        return RAGOutput(
            query=query,
            answer=answer,
            retrieval_result=retrieval_result,
            metadata=metadata,
        )

    def _compact_supports_for_log(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._compact_support_items(item) if key == "supports" else self._compact_supports_for_log(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._compact_supports_for_log(item) for item in value]
        return value

    def _compact_support_items(self, supports: Any) -> Any:
        if not isinstance(supports, list):
            return self._compact_supports_for_log(supports)

        compact_supports = []
        for support in supports:
            if isinstance(support, dict):
                compact_supports.append(
                    {
                        key: self._compact_supports_for_log(support[key])
                        for key in ("title", "contents")
                        if key in support
                    }
                )
            else:
                compact_supports.append(self._compact_supports_for_log(support))
        return compact_supports
    
register_pipeline("perqa", PERQAPipeline)
