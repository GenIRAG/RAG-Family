from pathlib import Path
from typing import Optional

from RoleRAG.config.types import RetrieverResult
from RoleRAG.errors import error_context
from RoleRAG.roles.registry import register_role
from RoleRAG.roles.role_base import BaseRetrieverContract
from RoleRAG.roles.validation import model_capabilities, validate_model_capability


@register_role("retriever")
class RetrieverRoleAdapter(BaseRetrieverContract):

    def __init__(
        self,
        model,
        top_k: int = 5,
        index_path: Optional[str] = None,
        graph_index_path: Optional[str] = None,
    ):
        if graph_index_path:
            validate_model_capability(model, "retriever (graph)", "encode", "encode")
        self._model = model
        self.top_k = top_k
        self.index_path = index_path
        self.graph_index_path = graph_index_path
        self.model_name = self._get_model_name()

        self._backend: Optional[str] = None
        self._indexer = None
        self._search_fn = None

        if not self.index_path and not self.graph_index_path:
            raise ValueError("RetrieverRoleAdapter requires index_path or graph_index_path")

        if self.index_path:
            self._backend = self._resolve_backend()
            if self._backend == "bm25":
                self._setup_fn = self._setup_bm25_retriever
                self._search_fn = self._search_bm25
            else:
                self._setup_fn = self._setup_dense_retriever
                self._search_fn = self._search_dense

            with error_context(
                f"role=retriever, model={self.model_name}, loading {self._backend} index: {self.index_path}",
                "Check index.paths in the method YAML and build the matching index before running queries.",
            ):
                if not Path(self.index_path).is_dir():
                    raise FileNotFoundError(f"Index directory not found: {self.index_path}")
                self._setup_fn()

    def _get_model_name(self):
        return getattr(self._model, "model_name", type(self._model).__name__)

    def _resolve_backend(self) -> str:
        capabilities = model_capabilities(self._model, "retriever")

        if "sparse_search" in capabilities:
            return "bm25"
        if "encode" in capabilities:
            validate_model_capability(self._model, "retriever (dense)", "encode", "encode")
            return "dense"
        raise TypeError(
            f"Model {type(self._model).__name__} cannot serve as retriever. "
            f"Capabilities: {sorted(capabilities)}; "
            f"required one of: ['encode', 'sparse_search']. Select a compatible retrieval model in the pipeline."
        )

    def _setup_bm25_retriever(self):
        from RoleRAG.indexes.indexing.chunk_indexing import BM25Indexing

        self._indexer = BM25Indexing()
        self._indexer.load(self.index_path)

    def _setup_dense_retriever(self):
        from RoleRAG.indexes.indexing.chunk_indexing import DenseIndexing

        self._indexer = DenseIndexing(encoder=self._model)
        self._indexer.load(self.index_path)

    def encode(self, texts, is_query: bool = False, **kwargs):
        validate_model_capability(self._model, "retriever", "encode", "encode")
        with error_context(f"role=retriever, model={self.model_name}, operation=encode"):
            return self._model.encode(texts, is_query=is_query, **kwargs)

    def search(self, query: str, **kwargs) -> RetrieverResult:
        if self._search_fn is None:
            raise ValueError("RetrieverRoleAdapter.search() requires index_path")
        top_k = kwargs.get("top_k", self.top_k)
        with error_context(f"role=retriever, model={self.model_name}, operation=search, index={self.index_path}"):
            return self._search_fn(query, top_k)

    def _search_bm25(self, query: str, top_k: int) -> RetrieverResult:
        documents, scores = self._indexer.search(query, top_k=top_k)

        return RetrieverResult(
            documents=documents,
            scores=scores,
            metadata={"model_type": "bm25", "query": query}
        )

    def _search_dense(self, query: str, top_k: int) -> RetrieverResult:
        query_embedding = self._model.encode(query, is_query=True)
        documents, scores = self._indexer.search(query_embedding, top_k=top_k)

        return RetrieverResult(
            documents=documents,
            scores=scores,
            metadata={"model_type": "dense", "query": query}
        )
