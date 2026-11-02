from collections import Counter, defaultdict
from typing import Any, Iterable, List, Optional

from RoleRAG.indexes.indexing.graph_indexing.extraction_schema import Entity, ExtractionResult, Relationship
from RoleRAG.indexes.indexing.graph_indexing.universal_graph import UniversalGraph

GRAPH_FIELD_SEP = "<SEP>"


class GraphBuilder:

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def build(self, result: ExtractionResult) -> UniversalGraph:
        graph = UniversalGraph()

        for entity_name, entities in result.nodes.items():
            if not entity_name:
                continue
            graph.add_node(entity_name, **self._merge_entities(entity_name, entities))

        grouped_edges = defaultdict(list)
        
        for (src_id, tgt_id), relationships in result.edges.items():
            if not src_id or not tgt_id:
                continue
            
            key = (src_id, tgt_id) if graph.is_directed else tuple(sorted((src_id, tgt_id)))
            grouped_edges[key].extend(relationships)
        for (src_id, tgt_id), relationships in grouped_edges.items():
            
            merged_relationship = self._merge_relationships(src_id, tgt_id, relationships)
            for node_id in (src_id, tgt_id):
                if not graph.has_node(node_id):
                    
                    graph.add_node(
                        node_id,
                        entity_name=node_id,
                        entity_type="UNKNOWN",
                        description=merged_relationship.get("description"),
                        source_id=merged_relationship.get("source_id"),
                    )
            graph.add_edge(src_id, tgt_id, **merged_relationship)

        return graph

    def _merge_entities(self, entity_name: str, entities: List[Entity]) -> dict:
        source_ids = _unique(e.source_id for e in entities if e.source_id)
        descriptions = _unique(e.description for e in entities if e.description)
        entity_types = [e.entity_type for e in entities if e.entity_type]

        return {
            "entity_name": entity_name,
            "entity_type": Counter(entity_types).most_common(1)[0][0] if entity_types else None,
            "description": GRAPH_FIELD_SEP.join(descriptions) if descriptions else None,
            "source_id": GRAPH_FIELD_SEP.join(source_ids) if source_ids else None,
        }

    def _merge_relationships(self, src_id: str, tgt_id: str, relationships: List[Relationship]) -> dict:
        source_ids = _unique(rel.source_id for rel in relationships if rel.source_id)
        relation_names = _unique(rel.relation_name for rel in relationships if rel.relation_name)
        descriptions = _unique(rel.description for rel in relationships if rel.description)
        keywords = _unique(rel.keywords for rel in relationships if rel.keywords)
        total_weight = sum(float(rel.weight or 0.0) for rel in relationships) or 1.0

        return {
            "weight": total_weight,
            "source_id": GRAPH_FIELD_SEP.join(source_ids) if source_ids else None,
            "relation_name": GRAPH_FIELD_SEP.join(relation_names) if relation_names else None,
            "keywords": GRAPH_FIELD_SEP.join(keywords) if keywords else None,
            "description": GRAPH_FIELD_SEP.join(descriptions) if descriptions else None,
        }


def _unique(values: Iterable[Any]) -> List[Any]:
    result = []
    seen = set()
    for value in values:
        if value is None:
            continue
        marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result