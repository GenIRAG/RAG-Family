from typing import Any, Dict

from RoleRAG.config.types import RAGOutput
from RoleRAG.models import get_model
from RoleRAG.roles.utils import assist_role, get_method_role_config
from RoleRAG.rolerag.pipelines.generation_strategy.refine_generation_strategy import (
    RefineGenerationStrategy,
)
from RoleRAG.rolerag.pipelines.pipelines_base import BasePipeline, register_pipeline
from RoleRAG.rolerag.pipelines.retrieval_strategy import BasicRetrievalStrategy


class LLMLinguaPipeline(BasePipeline):

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.bm25_index_path = config["index"]["paths"]["bm25_index_path"]

        self.retriever_cfg = get_method_role_config(config, "retriever", "BM25Model")
        self.retriever_cfg["index_path"] = self.bm25_index_path
        self.reasoner_cfg = get_method_role_config(config, "reasoner", "OpenAIGenerator")
        self.refiner_cfg = get_method_role_config(config, "refiner", "LLMLinguaModel")

        self.retriever = None
        self.reasoner = None
        self.refiner = None
        self.retrieval_strategy = BasicRetrievalStrategy(config)
        self.generation_strategy = RefineGenerationStrategy(config)
        self.create()

    def create(self) -> None:
        self.retriever = assist_role(
            get_model("BM25Model"), "retriever", self.retriever_cfg
        )
        self.reasoner = assist_role(
            get_model("OpenAIGenerator"), "reasoner", self.reasoner_cfg
        )
        self.refiner = assist_role(
            get_model("LLMLinguaModel"), "refiner", self.refiner_cfg
        )

    def _run_item(self, query: str, **kwargs) -> RAGOutput:
        retrieval_result = self.retrieval_strategy.retrieve(
            query=query,
            retriever=self.retriever,
        )
        self.logset("retrieval_result", retrieval_result)
        if self._is_retrieval_only():
            return self._retrieval_only_output(query, retrieval_result)

        answer = self.generation_strategy.generate(
            question=query,
            retrieval_result=retrieval_result,
            reasoner=self.reasoner,
            refiner=self.refiner,
        )
        return RAGOutput(
            query=query,
            answer=answer,
            retrieval_result=retrieval_result,
        )


register_pipeline("llmlingua", LLMLinguaPipeline)
