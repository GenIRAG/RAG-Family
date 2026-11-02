from typing import Any, Dict

from RoleRAG.roles.utils import assist_role, get_method_role_config
from RoleRAG.config.types import RAGOutput
from RoleRAG.rolerag.pipelines.pipelines_base import BasePipeline, register_pipeline
from RoleRAG.models import get_model
from RoleRAG.rolerag.pipelines.retrieval_strategy import UCGRRetrievalStrategy


class UCGRPipeline(BasePipeline):

    def __init__(
        self,
        config: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.index_cfg = config["index"]
        self.paths_cfg = self.index_cfg["paths"]
        self.fm_index_path = self.paths_cfg["FM_index_path"]

        self.reasoner_cfg = get_method_role_config(config, "reasoner", "HFDecoder")
        self.reasoner_cfg["fm_index_path"] = self.fm_index_path

        self.reasoner = None
        self.retrieval_strategy = UCGRRetrievalStrategy(config)
        self.create()

    def create(self) -> None:
        decoder_model = get_model("HFDecoder")
        self.reasoner = assist_role(decoder_model, "reasoner", self.reasoner_cfg)

    def _run_item(self, query: str, **kwargs) -> RAGOutput:
        context = kwargs.get("context")
        eval_item = kwargs.get("eval_item")
        if context is None and eval_item is not None:
            context = (eval_item.metadata or {}).get("context", "")
        effective_query = self._build_retrieval_query(query, context)
        
        retrieval_result = self.retrieval_strategy.retrieve(
            query=effective_query,
            reasoner=self.reasoner,
        )

        retrieval_result.metadata["original_query"] = query
        retrieval_result.metadata["effective_query"] = effective_query
        self.logset("retrieval_result", retrieval_result)

        answer = retrieval_result.documents[0].content if retrieval_result.documents else ""
        return RAGOutput(query=query, answer=answer, retrieval_result=retrieval_result)

    @staticmethod
    def _format_context(context: Any) -> str:
        if isinstance(context, list):
            context = "\n".join(str(item).strip() for item in context if str(item).strip())
        elif context is None:
            context = ""
        else:
            context = str(context).strip()
        return context or "None"

    @staticmethod
    def _build_retrieval_query(query: str, context: Any) -> str:
        context_text = UCGRPipeline._format_context(context)
        if context_text == "None":
            return query
        return f"{context_text}\nQ: {query}"


register_pipeline("ucgr", UCGRPipeline)
