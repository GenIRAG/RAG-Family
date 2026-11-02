from typing import List

from RoleRAG.config.types import Document


class BaseChunking:
    def chunk(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError
