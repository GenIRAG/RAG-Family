from RoleRAG.indexes.loading import load_corpus, load_corpus_records, BaseLoading, JsonLoading, TextLoading
from RoleRAG.indexes.chunking import BaseChunking, FixedSizeChunking, SentenceChunking, TokenSizeChunking
from RoleRAG.indexes.indexing.chunk_indexing import BaseIndexing
from RoleRAG.indexes.utils import make_index_name

__all__ = [
    "load_corpus",
    "load_corpus_records",
    "BaseLoading",
    "JsonLoading",
    "TextLoading",
    "BaseChunking",
    "FixedSizeChunking",
    "SentenceChunking",
    "TokenSizeChunking",
    "BaseIndexing",
    "BM25Indexing",
    "DenseIndexing",
    "FMIndexing",
    "make_index_name",
]


def __getattr__(name):
    if name == "BM25Indexing":
        from RoleRAG.indexes.indexing.chunk_indexing import BM25Indexing
        return BM25Indexing
    if name == "DenseIndexing":
        from RoleRAG.indexes.indexing.chunk_indexing import DenseIndexing
        return DenseIndexing
    if name == "FMIndexing":
        from RoleRAG.indexes.indexing.chunk_indexing import FMIndexing
        return FMIndexing
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
