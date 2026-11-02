import pickle
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


class UniversalGraph:
    NODE_DEFAULTS: Dict[str, Any] = {
        "entity_name": None,
        "entity_type": None,
        "description": None,
        "source_id": None,
        "clusters": None,
    }
    EDGE_DEFAULTS: Dict[str, Any] = {
        "weight": 1.0,
        "source_id": None,
        "relation_name": None,
        "keywords": None,
        "description": None,
        "src_id": None,
        "tgt_id": None,
    }

    def __init__(self, directed: bool = False):
        self._graph = nx.DiGraph() if directed else nx.Graph()
        self._node_list: Optional[List[str]] = None
        self._edge_list: Optional[List[Tuple[str, str]]] = None

    @property
    def raw_graph(self) -> nx.Graph:
        return self._graph

    @property
    def is_directed(self) -> bool:
        return self._graph.is_directed()

    def add_node(self, node_id: str, **attrs):
        data = {**self.NODE_DEFAULTS, **attrs}
        data["entity_name"] = data.get("entity_name") or node_id
        self._graph.add_node(node_id, **data)
        self._invalidate_cache()

    def update_node(self, node_id: str, **attrs):
        if not self.has_node(node_id):
            self.add_node(node_id, **attrs)
            return
        self._graph.nodes[node_id].update(attrs)

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def get_node(self, node_id: str) -> Optional[dict]:
        data = self._graph.nodes.get(node_id)
        return data if data is None else dict(data)

    def node_ids(self) -> List[str]:
        return list(self._graph.nodes())

    @property
    def node_num(self) -> int:
        return self._graph.number_of_nodes()

    def add_edge(self, src_id: str, tgt_id: str, **attrs):
        data = {**self.EDGE_DEFAULTS, **attrs}
        data["src_id"] = src_id
        data["tgt_id"] = tgt_id
        self._graph.add_edge(src_id, tgt_id, **data)
        self._invalidate_cache()

    def update_edge(self, src_id: str, tgt_id: str, **attrs):
        if not self.has_edge(src_id, tgt_id):
            self.add_edge(src_id, tgt_id, **attrs)
            return
        self._graph.edges[src_id, tgt_id].update(attrs)

    def has_edge(self, src_id: str, tgt_id: str) -> bool:
        return self._graph.has_edge(src_id, tgt_id)

    def get_edge(self, src_id: str, tgt_id: str) -> Optional[dict]:
        data = self._graph.edges.get((src_id, tgt_id))
        return data if data is None else dict(data)

    def edge_ids(self) -> List[Tuple[str, str]]:
        return list(self._graph.edges())

    @property
    def edge_num(self) -> int:
        return self._graph.number_of_edges()

    def degree(self, node_id: str) -> int:
        return int(self._graph.degree(node_id)) if self.has_node(node_id) else 0

    def get_node_edges(self, node_id: str) -> List[Tuple[str, str]]:
        if not self.has_node(node_id):
            return []
        return list(self._graph.edges(node_id))

    def get_edge_weight(self, src_id: str, tgt_id: str) -> float:
        edge = self._graph.edges.get((src_id, tgt_id))
        if edge is None:
            return 0.0
        try:
            return float(edge.get("weight", 1.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def node_index(self, node_id: str) -> int:
        if self._node_list is None:
            self._node_list = list(self._graph.nodes())
        try:
            return self._node_list.index(node_id)
        except ValueError:
            return -1

    def to_pickle(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self._graph, f)

    @staticmethod
    def from_pickle(path: str) -> "UniversalGraph":
        graph = UniversalGraph()
        with open(path, "rb") as f:
            graph._graph = pickle.load(f)
        return graph

    def to_graphml(self, path: str):
        export_graph = nx.DiGraph() if self._graph.is_directed() else nx.Graph()

        for node_id, attrs in self._graph.nodes(data=True):
            export_graph.add_node(
                node_id,
                **{
                    key: safe_value
                    for key, value in attrs.items()
                    if (safe_value := self._to_graphml_value(value)) is not None
                },
            )

        for src, tgt, attrs in self._graph.edges(data=True):
            export_graph.add_edge(
                src,
                tgt,
                **{
                    key: safe_value
                    for key, value in attrs.items()
                    if (safe_value := self._to_graphml_value(value)) is not None
                },
            )

        nx.write_graphml(export_graph, path)

    @staticmethod
    def _to_graphml_value(value):
        if value is None:
            return "null"
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _invalidate_cache(self):
        self._node_list = None
        self._edge_list = None
