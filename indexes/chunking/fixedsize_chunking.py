from typing import List

from RoleRAG.config.types import Document
from RoleRAG.indexes.chunking.chunking_base import BaseChunking


class FixedSizeChunking(BaseChunking):
    def __init__(
        self,
        chunk_size: int = 256,
        overlap: int = 32,
        separator: str = "\n"
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator

    def chunk(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []
        for doc in documents:
            chunks = self._split_text(doc.content)
            for idx, chunk in enumerate(chunks):
                chunked_docs.append(Document(
                    id=f"{doc.id}_chunk_{idx}",
                    content=chunk,
                    metadata={**doc.metadata, "parent_id": doc.id, "chunk_idx": idx}
                ))
        return chunked_docs

    def _split_text(self, text: str) -> List[str]:
        if not text.strip():
            return []

        if self.separator:
            paragraphs = text.split(self.separator)
        else:
            paragraphs = [text]

        chunks = []
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += (self.separator + para) if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                while len(para) > self.chunk_size:
                    chunks.append(para[:self.chunk_size])
                    para = para[self.chunk_size - self.overlap:]
                if self.overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.overlap:]
                    current_chunk = overlap_text + self.separator + para
                else:
                    current_chunk = para
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
