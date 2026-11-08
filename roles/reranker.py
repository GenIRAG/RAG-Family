from typing import List, Optional, Tuple, Union

from RoleRAG.config.types import Document
from RoleRAG.errors import error_context
from RoleRAG.roles.registry import register_role
from RoleRAG.roles.role_base import BaseRerankerContract
from RoleRAG.roles.validation import validate_model_capability


@register_role("reranker")
class RerankerRoleAdapter(BaseRerankerContract):

    def __init__(self, model, top_k: int = 10):
        validate_model_capability(model, "reranker", "rerank", "rerank")
        self._model = model
        self.top_k = top_k
        self.model_name = self._get_model_name()

    def _get_model_name(self):
        return getattr(self._model, "model_name", type(self._model).__name__)

    def rerank(
        self,
        query: str,
        documents: Union[List[Document], List[str]],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Tuple[Union[Document, str], float]]:
        top_k = self.top_k if top_k is None else top_k

        if not documents:
            return []

        if isinstance(documents[0], Document):
            doc_strings = [doc.content for doc in documents]
            remaining_docs = list(documents)
        else:
            doc_strings = documents
            remaining_docs = None

        with error_context(f"role=reranker, model={self.model_name}, operation=rerank, top_k={top_k}"):
            results = self._model.rerank(query, doc_strings, top_k=top_k, **kwargs)

        if remaining_docs is not None:
            reranked_results = []
            for text, score in results:
                for idx, doc in enumerate(remaining_docs):
                    if doc.content == text:
                        reranked_results.append((doc, score))
                        remaining_docs.pop(idx)
                        break
            return reranked_results
        return results
