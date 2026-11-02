import os
import json
import bm25s
import Stemmer
from typing import List, Optional

from RoleRAG.config.types import Document
from RoleRAG.indexes.indexing.chunk_indexing.chunk_indexing_base import BaseIndexing


class BM25Indexing(BaseIndexing):

    def __init__(
        self,
        index_path: Optional[str] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.index_path = index_path
        self.k1 = k1
        self.b = b
        self.index = None
        self.documents = []
        self.corpus_tokens = None
        self._tokenizer = None
        self._stemmer = Stemmer.Stemmer("english")

    def build_index(self, documents: List[Document], **kwargs):
        self.documents = documents

        corpus_text = [doc.content for doc in documents]
        self.corpus_tokens = bm25s.tokenize(corpus_text, stemmer=self._stemmer)

        self.index = bm25s.BM25(k1=self.k1, b=self.b)
        self.index.index(self.corpus_tokens)

    def search(self, query: str, top_k: int = 5) -> tuple:
        if self.index is None:
            return [], []

        query_tokens = bm25s.tokenize([query], stemmer=self._stemmer)
        results, scores = self.index.retrieve(query_tokens, k=top_k)

        results = results[0]
        scores = scores[0]

        retrieved_docs = []
        retrieved_scores = []
        for idx, score in zip(results, scores):
            if int(idx) < len(self.documents):
                doc = self.documents[int(idx)]
                doc.score = float(score)
                retrieved_docs.append(doc)
                retrieved_scores.append(float(score))

        return retrieved_docs, retrieved_scores

    def save(self, path: str):
        if self.index is None:
            return

        save_dir = path if path.endswith("/") else path + "/"
        os.makedirs(save_dir, exist_ok=True)

        self.index.save(save_dir)

        corpus_data = [doc.to_dict() for doc in self.documents]
        with open(os.path.join(save_dir, "corpus.jsonl"), "w", encoding="utf-8") as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        load_dir = path if path.endswith("/") else path + "/"

        self.index = bm25s.BM25.load(load_dir)
        
        with open(os.path.join(load_dir, "params.index.json"), "r", encoding="utf-8") as f:
            params = json.load(f)
            self.k1 = params.get("k1", self.k1)
            self.b = params.get("b", self.b)

        with open(os.path.join(load_dir, "corpus.jsonl"), "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
            self.documents = [Document.from_dict(d) for d in corpus_data]
