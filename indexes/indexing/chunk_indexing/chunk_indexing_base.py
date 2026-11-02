from typing import List

from RoleRAG.config.types import Document


class BaseIndexing:

    def build_index(self, documents: List[Document], **kwargs):
        raise NotImplementedError

    def search(self, query_embedding, top_k: int = 5) -> tuple:
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError
