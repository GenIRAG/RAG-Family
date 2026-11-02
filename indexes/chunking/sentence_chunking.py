from typing import List

from RoleRAG.config.types import Document
from RoleRAG.indexes.chunking.chunking_base import BaseChunking


class SentenceChunking(BaseChunking):

    def __init__(self, max_chunk_size: int = 256):
        self.max_chunk_size = max_chunk_size

    def chunk(self, documents: List[Document]) -> List[Document]:
        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            import nltk
            nltk.download('punkt', quiet=True)

        import nltk
        chunked_docs = []
        for doc in documents:
            if not doc.content.strip():
                continue
            sentences = nltk.sent_tokenize(doc.content)
            chunks = []
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) <= self.max_chunk_size:
                    current_chunk += (" " + sent) if current_chunk else sent
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    if len(sent) > self.max_chunk_size:
                        for i in range(0, len(sent), self.max_chunk_size):
                            chunks.append(sent[i:i + self.max_chunk_size])
                        current_chunk = ""
                    else:
                        current_chunk = sent
            if current_chunk:
                chunks.append(current_chunk)

            for idx, chunk in enumerate(chunks):
                chunked_docs.append(Document(
                    id=f"{doc.id}_chunk_{idx}",
                    content=chunk,
                    metadata={**doc.metadata, "parent_id": doc.id, "chunk_idx": idx},
                ))
        return chunked_docs
