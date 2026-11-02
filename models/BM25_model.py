from typing import Any, List


class BM25Model:

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self._documents = []
        self._bm25 = None
        self._corpus_tokens = None
        self._stemmer = None

    def add_documents(self, documents: List[Any]):
        """Add documents to the index"""
        self._documents.extend(documents)
        self._initialize()

    def _initialize(self):
        try:
            import bm25s
        except ImportError:
            raise ImportError("Please install bm25s: pip install bm25s")

        try:
            import Stemmer
            self._stemmer = Stemmer.Stemmer("english")
        except ImportError:
            self._stemmer = None

        if not self._documents:
            return

        corpus_text = [doc.content for doc in self._documents]
        self._corpus_tokens = bm25s.tokenize(corpus_text, stemmer=self._stemmer)
        self._bm25 = bm25s.BM25(k1=self.k1, b=self.b)
        self._bm25.index(self._corpus_tokens)

    def get_scores(self, query_tokens: List[str]) -> List[float]:

        if self._bm25 is None:
            return [0.0] * len(self._documents)
        return self._bm25.get_scores(query_tokens).tolist()

    def get_top_k(self, query_tokens: List[str], top_k: int = 10) -> List[tuple]:

        if self._bm25 is None:
            return []

        import bm25s
        query_text = ' '.join(query_tokens)
        query_tokens_input = bm25s.tokenize([query_text], stemmer=self._stemmer)
        results, scores = self._bm25.retrieve(query_tokens_input, k=top_k)

        results = results[0]
        scores = scores[0]
        return [(int(idx), float(score)) for idx, score in zip(results, scores)]

    @property
    def documents(self):
        return self._documents

    def clone(self) -> "BM25Model":
        cloned = type(self)(
            k1=self.k1,
            b=self.b,
        )
        if self._documents:
            cloned.add_documents(list(self._documents))
        return cloned
