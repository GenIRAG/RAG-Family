from typing import Any, List, Optional, Tuple

from RoleRAG.roles.role_base import BaseRetrieverContract
from RoleRAG.config.types import Document, RetrieverResult


class HybridRetrievalStrategy:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        hybrid_cfg = (self.config.get("inference", {}) or {}).get("hybrid", {}) or {}
        self.rrf_k = int(hybrid_cfg.get("rrf_k", 60))
        self.retrieve_top_k = self._to_int_or_none(hybrid_cfg.get("retrieve_top_k")) or 5
        self.return_top_k = self._to_int_or_none(hybrid_cfg.get("return_top_k")) or 5

    @staticmethod
    def _to_int_or_none(value: Any) -> Optional[int]:
        if value is None:
            return None
        return int(value)

    def retrieve(
        self,
        query: str,
        retrievers: List[BaseRetrieverContract],
        retrieve_top_k: Optional[int] = None,
        return_top_k: Optional[int] = None,
        **kwargs,
    ) -> RetrieverResult:
        if not retrievers:
            return RetrieverResult(
                documents=[],
                scores=[],
                metadata={"strategy": "hybrid", "query": query},
            )

        effective_retrieve_top_k = int(retrieve_top_k if retrieve_top_k is not None else self.retrieve_top_k)
        effective_return_top_k = int(return_top_k if return_top_k is not None else self.return_top_k)
        results = [retriever.search(query, top_k=effective_retrieve_top_k, **kwargs) for retriever in retrievers]
        fused_docs, fused_scores = self._reciprocal_rank_fusion_multi(
            results,
            self.rrf_k,
            effective_return_top_k,
        )

        return RetrieverResult(
            documents=fused_docs,
            scores=fused_scores,
            metadata={
                "strategy": "hybrid",
                "query": query,
            },
        )

    @staticmethod
    def _reciprocal_rank_fusion_multi(
        results: List[RetrieverResult],
        rrf_k: int,
        top_k: int,
    ) -> Tuple[List[Document], List[float]]:
        doc_scores = {}

        for result in results:
            if result.documents:
                for rank, (doc, score) in enumerate(zip(result.documents, result.scores)):
                    doc_id = doc.id
                    rrf_score = 1 / (rrf_k + rank + 1)
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        result_docs = []
        result_scores = []
        doc_id_to_obj = {}
        for result in results:
            if result.documents:
                doc_id_to_obj.update({doc.id: doc for doc in result.documents})

        for doc_id, score in sorted_docs:
            doc = doc_id_to_obj.get(doc_id)
            if doc:
                doc.score = score
                result_docs.append(doc)
                result_scores.append(score)

        return result_docs, result_scores
