from typing import Any, Dict

from RoleRAG.roles.utils import assist_role, get_method_role_config
from RoleRAG.config.types import RAGOutput
from RoleRAG.rolerag.pipelines.pipelines_base import BasePipeline, register_pipeline
from RoleRAG.models import get_model
from RoleRAG.rolerag.pipelines.generation_strategy.naive_generation_strategy import NaiveGenerationStrategy
from RoleRAG.rolerag.pipelines.retrieval_strategy import BasicRetrievalStrategy


class NaiveRAGPipeline(BasePipeline):

    def __init__(
        self,
        config: Dict[str, Any],
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.index_cfg = config["index"]
        self.paths_cfg = self.index_cfg["paths"]
        self.bm25_index_path = self.paths_cfg["bm25_index_path"]

        self.retriever_cfg = get_method_role_config(config, "retriever", "BM25Model")
        self.retriever_cfg["index_path"] = self.bm25_index_path
        self.rewriter_cfg = get_method_role_config(config, "rewriter", "OpenAIGenerator")
        self.reasoner_cfg = get_method_role_config(config, "reasoner", "OpenAIGenerator")

        self.retriever = None
        self.rewriter = None
        self.reasoner = None
        self.retrieval_strategy = BasicRetrievalStrategy(config)
        self.generation_strategy = NaiveGenerationStrategy(config)
        self.create()

    def create(self) -> None:
        bm25_model = get_model("BM25Model")
        self.retriever = assist_role(bm25_model, "retriever", self.retriever_cfg)
        rewriter_model = get_model("OpenAIGenerator")
        self.rewriter = assist_role(
            rewriter_model,
            "rewriter",
            self.rewriter_cfg,
        )
        
        reasoner_model = get_model("OpenAIGenerator")
        self.reasoner = assist_role(
            reasoner_model,
            "reasoner",
            self.reasoner_cfg,
        )


    def _run_item(self, query: str, **kwargs) -> RAGOutput:
        retrieval_result = self.retrieval_strategy.retrieve(
            query=query,
            retriever=self.retriever,
            rewriter=self.rewriter,
        )
        self.logset("retrieval_result", retrieval_result)
        if self._is_retrieval_only():
            return self._retrieval_only_output(query, retrieval_result)

        answer = self.generation_strategy.generate(
            question=query,
            retrieval_result=retrieval_result,
            reasoner=self.reasoner,
        )
        return RAGOutput(
            query=query,
            answer=answer,
            retrieval_result=retrieval_result,
        )


register_pipeline("naive", NaiveRAGPipeline)
