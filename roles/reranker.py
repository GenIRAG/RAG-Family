from typing import List, Tuple, Union

from RoleRAG.config.types import Document
from RoleRAG.roles.role_base import BaseRerankerContract


class RerankerRoleAdapter(BaseRerankerContract):

    def __init__(self, model, top_k: int = 10, **kwargs):
        self._model = model
        self.top_k = top_k
        self.model_name = self._get_model_name()

    def _get_model_name(self):
        return getattr(self._model, "model_name", type(self._model).__name__)

    def rerank(
        self,
        query: str,
        documents: Union[List[Document], List[str]],
        top_k: int = 10,
        **kwargs
    ) -> List[Tuple[Union[Document, str], float]]:
        top_k = kwargs.pop("top_k", self.top_k)

        if not documents:
            return []

        if isinstance(documents[0], Document):
            doc_strings = [doc.content for doc in documents]
            remaining_docs = list(documents)
        else:
            doc_strings = documents
            remaining_docs = None

        results = self._model.rerank(query, doc_strings, top_k=top_k)

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
