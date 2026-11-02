from typing import Any, Dict

from RoleRAG.roles.utils import assist_role, get_method_role_config
from RoleRAG.config.types import RAGOutput
from RoleRAG.rolerag.pipelines.pipelines_base import BasePipeline, register_pipeline
from RoleRAG.models import get_model
from RoleRAG.rolerag.pipelines.generation_strategy.graphrag_generation_strategy import GraphRAGLocalGenerationStrategy
from RoleRAG.rolerag.pipelines.retrieval_strategy import GraphRAGRetrievalStrategy


class GraphRAGLocalPipeline(BasePipeline):
    def __init__(
        self,
        config: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.index_cfg = config["index"]
        self.paths_cfg = self.index_cfg["paths"]
        self.graph_index_path = self.paths_cfg["graph_index_path"]

        self.retriever_cfg = get_method_role_config(config, "retriever", "HFEncoder")
        self.retriever_cfg["graph_index_path"] = self.graph_index_path
        self.reasoner_cfg = get_method_role_config(config, "reasoner", "OpenAIGenerator")

        self.retriever = None
        self.reasoner = None
        self.retrieval_strategy = GraphRAGRetrievalStrategy(config, mode="local")
        self.generation_strategy = GraphRAGLocalGenerationStrategy(config)
        self.create()

    def create(self) -> None:
        encode_model = get_model("HFEncoder")
        self.retriever = assist_role(encode_model, "retriever", self.retriever_cfg)

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
        )
        self.logset("retrieval_result", retrieval_result)
        if self._is_retrieval_only():
            return self._retrieval_only_output(query, retrieval_result)

        answer = self.generation_strategy.generate(
            question=query,
            retrieval_result=retrieval_result,
            reasoner=self.reasoner,
        )

        return RAGOutput(query=query, answer=answer, retrieval_result=retrieval_result)


register_pipeline("graphrag_local", GraphRAGLocalPipeline)
