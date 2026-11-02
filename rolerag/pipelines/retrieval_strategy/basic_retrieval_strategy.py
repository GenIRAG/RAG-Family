from typing import Any, List, Optional, Tuple

from RoleRAG.roles.role_base import BaseRetrieverContract
from RoleRAG.config.types import Document, RetrieverResult


class BasicRetrievalStrategy:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        inference_cfg = (self.config.get("inference", {}) or {})
        self.method_name = self.config.get("global", {}).get("pipeline_type")
        method_cfg = inference_cfg.get(self.method_name, {}) if self.method_name else {}
        self.default_return_top_k = self._to_int_or_none(method_cfg.get("return_top_k"))
        self.default_rerank_top_k = self._to_int_or_none(method_cfg.get("rerank_top_k"))

    @staticmethod
    def _to_int_or_none(value: Any) -> Optional[int]:
        if value is None:
            return None
        return int(value)

    def _resolve_return_top_k(self, return_top_k: Optional[int], fallback: int = 20) -> int:
        if return_top_k is not None:
            return int(return_top_k)
        if self.default_return_top_k is not None:
            return self.default_return_top_k
        return fallback

    def _resolve_rerank_top_k(self, rerank_top_k: Optional[int], fallback: int) -> int:
        if rerank_top_k is not None:
            return int(rerank_top_k)
        if self.default_rerank_top_k is not None:
            return self.default_rerank_top_k
        return fallback

    def retrieve(
        self,
        query: str,
        retriever: BaseRetrieverContract,
        reranker: Any = None,
        rewriter: Any = None,
        return_top_k: Optional[int] = None,
        **kwargs,
    ) -> RetrieverResult:
        if retriever is None:
            raise ValueError("BasicRetrievalStrategy.retrieve() requires a retriever")

        effective_query = query
        if rewriter is not None:
            effective_query = rewriter.rewrite(query, **kwargs)

        if reranker is None:
            result = self.simple(
                query=effective_query,
                retriever=retriever,
                return_top_k=return_top_k,
                **kwargs,
            )
        else:
            result = self.retrieve_then_rerank(
                query=effective_query,
                retriever=retriever,
                reranker=reranker,
                return_top_k=return_top_k,
                **kwargs,
            )

        if result.metadata is None:
            result.metadata = {}
        if rewriter is not None:
            result.metadata["original_query"] = query
            result.metadata["rewritten_query"] = effective_query
        return result

    def simple(
        self,
        query: str,
        retriever: BaseRetrieverContract,
        return_top_k: int = None,
        **kwargs,
    ) -> RetrieverResult:
        return_top_k = self._resolve_return_top_k(return_top_k)

        result = retriever.search(query, top_k=return_top_k, **kwargs)

        return RetrieverResult(
            documents=result.documents[:return_top_k],
            scores=result.scores[:return_top_k],
            metadata={
                "strategy": "simple",
                "query": query,
            },
        )

    def retrieve_then_rerank(
        self,
        query: str,
        retriever: BaseRetrieverContract,
        reranker: Any,
        return_top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        **kwargs,
    ) -> RetrieverResult:
        return_top_k = self._resolve_return_top_k(return_top_k)
        rerank_top_k = self._resolve_rerank_top_k(rerank_top_k, fallback=return_top_k)

        result = self.simple(
            query=query,
            retriever=retriever,
            return_top_k=return_top_k,
            **kwargs,
        )

        reranked_result = self._rerank_retrieval_result(
            retrieval_result=result,
            reranker=reranker,
            query=query,
            top_k=rerank_top_k,
        )
        reranked_result.metadata["strategy"] = "retrieve_then_rerank"
        return reranked_result

    @staticmethod
    def _rerank_documents(
        documents: List[Document],
        reranker: Any,
        query: str,
        top_k: int = 10,
    ) -> Tuple[List[Document], List[float]]:
        if not documents:
            return [], []

        doc_texts = [doc.content for doc in documents]
        reranked = reranker.rerank(query, doc_texts, top_k=top_k)

        if not reranked:
            return documents, [doc.score for doc in documents if doc.score is not None]

        text_to_doc = {doc.content: doc for doc in documents}
        reranked_docs = []
        reranked_scores = []

        for doc_text, score in reranked:
            doc = text_to_doc.get(doc_text)
            if doc:
                doc.score = score
                reranked_docs.append(doc)
                reranked_scores.append(score)

        return reranked_docs, reranked_scores

    def _rerank_retrieval_result(
        self,
        retrieval_result: RetrieverResult,
        reranker: Any,
        query: str,
        top_k: int = 5,
    ) -> RetrieverResult:
        reranked_docs, reranked_scores = self._rerank_documents(
            documents=retrieval_result.documents,
            reranker=reranker,
            query=query,
            top_k=top_k,
        )

        return RetrieverResult(
            documents=reranked_docs,
            scores=reranked_scores,
            metadata=retrieval_result.metadata.copy() if retrieval_result.metadata else {},
        )
