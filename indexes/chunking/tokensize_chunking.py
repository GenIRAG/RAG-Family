from typing import Callable, List, Optional

from RoleRAG.config.types import Document
from RoleRAG.indexes.chunking.chunking_base import BaseChunking


class TokenSizeChunking(BaseChunking):

    def __init__(
        self,
        chunk_size: int = 512,
        overlap_token_size: int = 64,
        tokenizer: Optional[Callable[[str], List[int]]] = None,
        separator: str = "\n",
    ):
        self.chunk_size = chunk_size
        self.overlap_token_size = overlap_token_size
        self.separator = separator

        if tokenizer is not None:
            self._encode = tokenizer
        else:
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                self._encode = enc.encode
            except ImportError:
                self._encode = None

    def chunk(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []
        for doc in documents:
            chunks = self._split_by_tokens(doc.content)
            for idx, chunk_text in enumerate(chunks):
                chunked_docs.append(Document(
                    id=f"{doc.id}_chunk_{idx}",
                    content=chunk_text,
                    metadata={**doc.metadata, "parent_id": doc.id, "chunk_idx": idx},
                ))
        return chunked_docs

    def _count_tokens(self, text: str) -> int:
        if self._encode is not None:
            return len(self._encode(text))
        return max(1, len(text) // 4)

    def _split_by_tokens(self, text: str) -> List[str]:
        paragraphs = text.split(self.separator) if self.separator else [text]

        chunks = []
        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            if para_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                chunks.extend(self._force_split_paragraph(para))
                continue

            if current_tokens + para_tokens <= self.chunk_size:
                current_chunk += (self.separator + para) if current_chunk else para
                current_tokens += para_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if self.overlap_token_size > 0 and current_chunk:
                    overlap_text = self._get_last_n_tokens_text(current_chunk, self.overlap_token_size)
                    current_chunk = overlap_text + self.separator + para
                else:
                    current_chunk = para
                current_tokens = self._count_tokens(current_chunk)

        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _force_split_paragraph(self, para: str) -> List[str]:
        if self._encode is not None:
            tokens = self._encode(para)
            chunks = []
            step = self.chunk_size - self.overlap_token_size
            for start in range(0, len(tokens), step):
                chunk_tokens = tokens[start:start + self.chunk_size]
                try:
                    import tiktoken
                    enc = tiktoken.get_encoding("cl100k_base")
                    chunk_text = enc.decode(chunk_tokens)
                except ImportError:
                    chunk_text = para[start * 4:(start + self.chunk_size) * 4]
                chunks.append(chunk_text)
            return chunks

        chars_per_token = 4
        step = (self.chunk_size - self.overlap_token_size) * chars_per_token
        chunk_chars = self.chunk_size * chars_per_token
        chunks = []
        for start in range(0, len(para), step):
            chunks.append(para[start:start + chunk_chars])
        return chunks

    def _get_last_n_tokens_text(self, text: str, n_tokens: int) -> str:
        if self._encode is not None:
            tokens = self._encode(text)
            overlap_tokens = tokens[-n_tokens:] if len(tokens) > n_tokens else tokens
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                return enc.decode(overlap_tokens)
            except ImportError:
                pass
        return text[-n_tokens * 4:]
