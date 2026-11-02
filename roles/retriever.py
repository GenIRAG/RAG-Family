from typing import Optional

from RoleRAG.config.types import RetrieverResult
from RoleRAG.roles.role_base import BaseRetrieverContract


class RetrieverRoleAdapter(BaseRetrieverContract):


    def __init__(
        self,
        model,
        top_k: int = 5,
        index_path: Optional[str] = None,
        graph_index_path: Optional[str] = None,
        **kwargs
    ):
        self._model = model
        self.top_k = top_k
        self.index_path = index_path
        self.graph_index_path = graph_index_path
        self.model_name = self._get_model_name()
        self._backend = self._detect_backend()
        if not self.index_path and not self.graph_index_path:
            raise ValueError("RetrieverRoleAdapter requires index_path or graph_index_path")

        self._indexer = None
        self._search_fn = None
        if self.index_path:
            if self._backend == "bm25":
                self._setup_fn = self._setup_bm25_retriever
                self._search_fn = self._search_bm25
            else:
                self._setup_fn = self._setup_dense_retriever
                self._search_fn = self._search_dense

            self._setup_fn()

    def _get_model_name(self):
        return getattr(self._model, "model_name", type(self._model).__name__)

    def _detect_backend(self) -> str:
        if hasattr(self._model, "get_top_k"):
            return "bm25"
        if hasattr(self._model, "encode"):
            return "dense"
        raise TypeError(f"Model {type(self._model).__name__} does not support search")

    def _setup_bm25_retriever(self):
        from RoleRAG.indexes.indexing.chunk_indexing import BM25Indexing

        self._indexer = BM25Indexing()
        self._indexer.load(self.index_path)

    def _setup_dense_retriever(self):
        from RoleRAG.indexes.indexing.chunk_indexing import DenseIndexing

        self._indexer = DenseIndexing(encoder=self._model)
        self._indexer.load(self.index_path)

    def encode(self, texts, is_query: bool = False, **kwargs):
        if not hasattr(self._model, "encode"):
            raise TypeError(
                f"Retriever backend {type(self._model).__name__} does not support encode"
            )
        return self._model.encode(texts, is_query=is_query, **kwargs)

    def search(self, query: str, **kwargs) -> RetrieverResult:
        if self._search_fn is None:
            raise ValueError("RetrieverRoleAdapter.search() requires index_path")
        top_k = kwargs.get("top_k", self.top_k)
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
    
