import os

from RoleRAG.indexes.indexing.graph_indexing.graph_indexer import GraphIndexer


def load_graph_index(path: str) -> GraphIndexer:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Graph index not found at {path}. "
            "Run `python main.py --build_index hipporag` or `python main.py --build_index graphrag` first."
        )
    return GraphIndexer.load(path)

def get_graph_index_path(method: str, dataset_name: str, storage_dir: str = "storage") -> str:
    return os.path.join(storage_dir, f"{method}_{dataset_name}")
