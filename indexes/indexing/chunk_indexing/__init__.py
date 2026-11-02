from RoleRAG.indexes.indexing.chunk_indexing.chunk_indexing_base import BaseIndexing

__all__ = [
    "BaseIndexing",
    "BM25Indexing",
    "DenseIndexing",
    "FMIndexing",
]


def __getattr__(name):
    if name == "BM25Indexing":
        from RoleRAG.indexes.indexing.chunk_indexing.bm25_indexing import BM25Indexing
        return BM25Indexing
    if name == "DenseIndexing":
        from RoleRAG.indexes.indexing.chunk_indexing.dense_indexing import DenseIndexing
        return DenseIndexing
    if name == "FMIndexing":
        from RoleRAG.indexes.indexing.chunk_indexing.fm_indexing import FMIndexing
        return FMIndexing
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
