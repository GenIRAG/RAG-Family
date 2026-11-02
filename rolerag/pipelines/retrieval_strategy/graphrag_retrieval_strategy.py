import json
from typing import Any, Dict, List, Optional

from RoleRAG.config.types import Document, RetrieverResult
from RoleRAG.indexes.indexing.graph_indexing.community import (
    list_to_quoted_csv_string,
    parse_json_object,
    truncate_list_by_token_size,
)

GRAPH_FIELD_SEP = "<SEP>"


class GraphRAGRetrievalStrategy:

    _graph_index_cache = {}

    def __init__(self, config: dict | None = None, mode: str = "local"):
        self.config = config or {}
        self.mode = mode
        graphrag_cfg = (self.config.get("inference", {}) or {}).get("graphrag", {}) or {}
        self.level = int(graphrag_cfg.get("level", 2))
        self.response_type = graphrag_cfg.get("response_type", "Multiple Paragraphs")
        self.local_top_k = int(graphrag_cfg.get("local_top_k", 10))
        self.local_max_token_for_text_unit = int(graphrag_cfg.get("local_max_token_for_text_unit", 4000))
        self.local_max_token_for_community_report = int(graphrag_cfg.get("local_max_token_for_community_report", 3200))
        self.local_community_single_one = bool(graphrag_cfg.get("local_community_single_one", False))
        self.max_token_for_local_context = int(graphrag_cfg.get("max_token_for_local_context", 4800))
        self.global_min_community_rating = float(graphrag_cfg.get("global_min_community_rating", 0))
        self.global_max_consider_community = int(graphrag_cfg.get("global_max_consider_community", 512))
        self.global_max_token_for_community_report = int(graphrag_cfg.get("global_max_token_for_community_report", 16384))

    def retrieve(self, query: str, retriever=None, reasoner=None) -> RetrieverResult:
        if self.mode == "global":
            graph_indexer = self._load_graph_index(reasoner)
            return self.global_(
                query=query,
                graph_indexer=graph_indexer,
                reasoner=reasoner,
                level=self.level,
                response_type=self.response_type,
                global_min_community_rating=self.global_min_community_rating,
                global_max_consider_community=self.global_max_consider_community,
                global_max_token_for_community_report=self.global_max_token_for_community_report,
            )
        graph_indexer = self._load_graph_index(retriever)
        return self.local(
            query=query,
            graph_indexer=graph_indexer,
            retriever=retriever,
            top_k=self.local_top_k,
            level=self.level,
            local_max_token_for_text_unit=self.local_max_token_for_text_unit,
            local_max_token_for_community_report=self.local_max_token_for_community_report,
            local_community_single_one=self.local_community_single_one,
            max_token_for_local_context=self.max_token_for_local_context,
        )

    def local(
        self,
        query: str,
        graph_indexer,
        retriever,
        top_k: Optional[int] = None,
        level: int = 2,
        local_max_token_for_text_unit: int = 4000,
        local_max_token_for_community_report: int = 3200,
        local_community_single_one: bool = False,
        max_token_for_local_context: int = 4800,
    ) -> RetrieverResult:
        if graph_indexer is None or graph_indexer.graph is None:
            return RetrieverResult(
                documents=[],
                scores=[],
                metadata={"strategy": "graphrag_local", "query": query, "error": "missing graph index"},
            )
        if graph_indexer.entity_vdb is None:
            return RetrieverResult(
                documents=[],
                scores=[],
                metadata={"strategy": "graphrag_local", "query": query, "error": "missing entity_vdb"},
            )

        node_datas = self._retrieve_entity_nodes(query, graph_indexer, retriever, top_k=top_k)
        if not node_datas:
            return RetrieverResult(
                documents=[],
                scores=[],
                metadata={"strategy": "graphrag_local", "query": query, "error": "no entities"},
            )

        use_communities = self._find_most_related_communities(
            node_datas=node_datas,
            community_reports=graph_indexer.community_reports,
            level=level,
            max_token_size=local_max_token_for_community_report,
            single_one=local_community_single_one,
        )
        use_text_units = self._find_most_related_text_units(
            node_datas=node_datas,
            graph_indexer=graph_indexer,
            max_token_size=local_max_token_for_text_unit,
        )
        use_relations = self._find_most_related_edges(
            node_datas=node_datas,
            graph_indexer=graph_indexer,
            max_token_size=max_token_for_local_context,
        )

        context = self._format_local_context(
            node_datas=node_datas,
            relations=use_relations,
            communities=use_communities,
            text_units=use_text_units,
        )
        doc = Document(
            id="graphrag-local-context",
            content=context,
            metadata={
                "strategy": "graphrag_local",
                "entities": node_datas,
                "relationships": use_relations,
                "communities": use_communities,
                "sources": [item.get("id") for item in use_text_units],
            },
            score=1.0,
        )
        return RetrieverResult(
            documents=[doc],
            scores=[1.0],
            metadata={
                "query": query,
            },
        )

    @classmethod
    def _load_graph_index(cls, role_holder):
        from RoleRAG.indexes.loading.graph_loading import load_graph_index

        graph_index_path = getattr(role_holder, "graph_index_path", None)
        if not graph_index_path:
            return None
        if graph_index_path not in cls._graph_index_cache:
            cls._graph_index_cache[graph_index_path] = load_graph_index(graph_index_path)
        return cls._graph_index_cache[graph_index_path]

    def global_(
        self,
        query: str,
        graph_indexer,
        reasoner,
        level: int = 2,
        response_type: str = "Multiple Paragraphs",
        global_min_community_rating: float = 0,
        global_max_consider_community: int = 512,
        global_max_token_for_community_report: int = 16384,
    ) -> RetrieverResult:
        community_datas = self._select_global_communities(
            graph_indexer=graph_indexer,
            level=level,
            global_min_community_rating=global_min_community_rating,
            global_max_consider_community=global_max_consider_community,
        )
        if not community_datas:
            return RetrieverResult(
                documents=[],
                scores=[],
                metadata={
                    "strategy": "graphrag_global",
                    "query": query,
                    "error": "no communities",
                    "selected_community_count": 0,
                    "map_results": [],
                    "support_points": [],
                },
            )

        map_results = self._map_global_communities(
            query=query,
            communities_data=community_datas,
            reasoner=reasoner,
            max_token_size=global_max_token_for_community_report,
        )
        support_points = []
        for analyst_idx, map_result in enumerate(map_results):
            points = map_result.get("parsed_points", [])
            for point in points:
                if "description" not in point:
                    continue
                support_points.append(
                    {
                        "analyst": analyst_idx,
                        "answer": point["description"],
                        "score": point.get("score", 1),
                    }
                )
        support_points = [point for point in support_points if self._safe_float(point.get("score", 0)) > 0]
        if not support_points:
            return RetrieverResult(
                documents=[],
                scores=[],
                metadata={
                    "strategy": "graphrag_global",
                    "query": query,
                    "error": "no support points",
                    "selected_community_count": len(community_datas),
                    "map_results": map_results,
                    "support_points": support_points,
                },
            )

        support_points.sort(key=lambda item: self._safe_float(item.get("score", 0)), reverse=True)
        support_points = truncate_list_by_token_size(
            support_points,
            key=lambda item: item["answer"],
            max_token_size=global_max_token_for_community_report,
        )
        points_context = "\n".join(
            f"""----Analyst {point['analyst']}----
Importance Score: {point['score']}
{point['answer']}
"""
            for point in support_points
        )

        doc = Document(
            id="graphrag-global-context",
            content=points_context,
            metadata={
                "strategy": "graphrag_global",
                "communities": [item.get("title") for item in community_datas],
                "support_points": support_points,
            },
            score=1.0,
        )
        return RetrieverResult(
            documents=[doc],
            scores=[1.0],
            metadata={
                "strategy": "graphrag_global",
                "query": query,
                "response_type": response_type,
                "selected_community_count": len(community_datas),
                "map_results": map_results,
                "support_points": support_points,
            },
        )

    @staticmethod
    def _retrieve_entity_nodes(query: str, graph_indexer, retriever, top_k: Optional[int] = None) -> List[dict]:
        import faiss

        top_k = int(top_k if top_k is not None else getattr(retriever, "top_k", 5))
        query_embedding = retriever.encode([query], is_query=True).reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_embedding)
        scores, indices = graph_indexer.entity_vdb.search(query_embedding, min(top_k, len(graph_indexer.node_ids)))
        node_datas = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            node_id = graph_indexer.idx_to_entity_id.get(int(idx))
            if not node_id:
                continue
            node = graph_indexer.graph.get_node(node_id) or {}
            node_datas.append(
                {
                    **node,
                    "entity_name": node.get("entity_name") or node_id,
                    "rank": graph_indexer.graph.degree(node_id),
                    "score": float(score),
                }
            )
        return node_datas

    def _find_most_related_communities(
        self,
        node_datas: List[dict],
        community_reports: Dict[str, dict],
        level: int,
        max_token_size: int,
        single_one: bool,
    ) -> List[dict]:
        related_keys = []
        for node in node_datas:
            clusters = self._load_clusters(node.get("clusters"))
            related_keys.extend(
                str(cluster.get("cluster"))
                for cluster in clusters
                if int(cluster.get("level", 0)) <= level
            )
        counts = {}
        for key in related_keys:
            counts[key] = counts.get(key, 0) + 1
        valid_keys = [key for key in counts if key in community_reports]
        valid_keys.sort(
            key=lambda key: (
                counts[key],
                self._safe_float(community_reports[key].get("report_json", {}).get("rating", -1)),
            ),
            reverse=True,
        )
        reports = [community_reports[key] for key in valid_keys]
        reports = truncate_list_by_token_size(
            reports,
            key=lambda item: item.get("report_string", ""),
            max_token_size=max_token_size,
        )
        return reports[:1] if single_one else reports

    def _find_most_related_text_units(self, node_datas: List[dict], graph_indexer, max_token_size: int) -> List[dict]:
        chunk_lookup = {chunk.id: chunk for chunk in graph_indexer.chunks}
        text_units = [self._split_field(node.get("source_id")) for node in node_datas]
        edges_by_node = [
            graph_indexer.graph.get_node_edges(node["entity_name"])
            for node in node_datas
        ]
        all_one_hop_nodes = set()
        for node, edges in zip(node_datas, edges_by_node):
            entity_name = node["entity_name"]
            for src, tgt in edges:
                all_one_hop_nodes.add(tgt if src == entity_name else src)

        one_hop_text_lookup = {}
        for node_id in all_one_hop_nodes:
            node = graph_indexer.graph.get_node(node_id) or {}
            one_hop_text_lookup[node_id] = set(self._split_field(node.get("source_id")))

        all_text_units = {}
        for order, (node, chunk_ids, edges) in enumerate(zip(node_datas, text_units, edges_by_node)):
            entity_name = node["entity_name"]
            for chunk_id in chunk_ids:
                if chunk_id in all_text_units:
                    continue
                relation_counts = 0
                for src, tgt in edges:
                    neighbor = tgt if src == entity_name else src
                    if chunk_id in one_hop_text_lookup.get(neighbor, set()):
                        relation_counts += 1
                chunk = chunk_lookup.get(chunk_id)
                if chunk is None:
                    continue
                all_text_units[chunk_id] = {
                    "id": chunk_id,
                    "content": chunk.content,
                    "order": order,
                    "relation_counts": relation_counts,
                }
        units = sorted(all_text_units.values(), key=lambda item: (item["order"], -item["relation_counts"]))
        return truncate_list_by_token_size(units, key=lambda item: item["content"], max_token_size=max_token_size)

    def _find_most_related_edges(self, node_datas: List[dict], graph_indexer, max_token_size: int) -> List[dict]:
        all_edges = set()
        for node in node_datas:
            for edge in graph_indexer.graph.get_node_edges(node["entity_name"]):
                all_edges.add(tuple(sorted(edge)))
        result = []
        for src, tgt in all_edges:
            edge = graph_indexer.graph.get_edge(src, tgt)
            if not edge:
                continue
            rank = graph_indexer.graph.degree(src) + graph_indexer.graph.degree(tgt)
            result.append({"src_tgt": [src, tgt], "rank": rank, **edge})
        result.sort(key=lambda item: (item.get("rank", 0), self._safe_float(item.get("weight", 0))), reverse=True)
        return truncate_list_by_token_size(
            result,
            key=lambda item: item.get("description", ""),
            max_token_size=max_token_size,
        )

    @staticmethod
    def _format_local_context(
        node_datas: List[dict],
        relations: List[dict],
        communities: List[dict],
        text_units: List[dict],
    ) -> str:
        entity_rows = [["id", "entity", "type", "description", "rank"]]
        for idx, node in enumerate(node_datas):
            entity_rows.append(
                [
                    idx,
                    node.get("entity_name", ""),
                    node.get("entity_type", "UNKNOWN"),
                    node.get("description", "UNKNOWN"),
                    node.get("rank", 0),
                ]
            )

        relation_rows = [["id", "source", "target", "description", "weight", "rank"]]
        for idx, edge in enumerate(relations):
            src_tgt = edge.get("src_tgt") or [edge.get("src_id", ""), edge.get("tgt_id", "")]
            relation_rows.append(
                [
                    idx,
                    src_tgt[0],
                    src_tgt[1],
                    edge.get("description", ""),
                    edge.get("weight", 0),
                    edge.get("rank", 0),
                ]
            )

        community_rows = [["id", "content"]]
        for idx, community in enumerate(communities):
            community_rows.append([idx, community.get("report_string", "")])

        source_rows = [["id", "content"]]
        for idx, unit in enumerate(text_units):
            source_rows.append([idx, unit.get("content", "")])

        return f"""
-----Communities-----
```csv
{list_to_quoted_csv_string(community_rows)}
```
-----Entities-----
```csv
{list_to_quoted_csv_string(entity_rows)}
```
-----Relationships-----
```csv
{list_to_quoted_csv_string(relation_rows)}
```
-----Sources-----
```csv
{list_to_quoted_csv_string(source_rows)}
```
"""
    
    def _select_global_communities(
        self,
        graph_indexer,
        level: int,
        global_min_community_rating: float,
        global_max_consider_community: int,
    ) -> List[dict]:
        schema_items = [
            (key, value)
            for key, value in graph_indexer.community_schema.items()
            if int(value.get("level", 0)) <= level
        ]
        schema_items.sort(key=lambda item: item[1].get("occurrence", 0.0), reverse=True)
        schema_items = schema_items[: int(global_max_consider_community)]
        reports = [
            graph_indexer.community_reports[key]
            for key, _ in schema_items
            if key in graph_indexer.community_reports
        ]
        reports = [
            report
            for report in reports
            if self._safe_float(report.get("report_json", {}).get("rating", 0)) >= global_min_community_rating
        ]
        reports.sort(
            key=lambda item: (
                item.get("occurrence", 0.0),
                self._safe_float(item.get("report_json", {}).get("rating", 0)),
            ),
            reverse=True,
        )
        return reports

    @staticmethod
    def _map_global_communities(
        query: str,
        communities_data: List[dict],
        reasoner,
        max_token_size: int,
    ) -> List[dict]:
        from RoleRAG.rolerag.pipelines.prompts import PromptManager

        groups = []
        remaining = list(communities_data)
        while remaining:
            group = truncate_list_by_token_size(
                remaining,
                key=lambda item: item.get("report_string", ""),
                max_token_size=max_token_size,
            )
            if not group:
                group = [remaining[0]]
            groups.append(group)
            remaining = remaining[len(group):]

        map_prompt = PromptManager.get_template("graphrag_global_map")
        map_results = []
        for group_idx, group in enumerate(groups):
            rows = [["id", "content", "rating", "importance"]]
            for idx, community in enumerate(group):
                rows.append(
                    [
                        idx,
                        community.get("report_string", ""),
                        community.get("report_json", {}).get("rating", 0),
                        community.get("occurrence", 0.0),
                    ]
                )
            context = list_to_quoted_csv_string(rows)
            prompt_variables = {
                "question": query,
                "context_data": context,
            }
            system_prompt = map_prompt.system_prompt.format(**prompt_variables)
            user_prompt = map_prompt.user_prompt.format(**prompt_variables)
            response = reasoner.generate(
                user_prompt,
                system_prompt=system_prompt,
                max_tokens=getattr(reasoner, "max_tokens", 1024),
                response_format={"type": "json_object"},
            )
            data = parse_json_object(response)
            parsed_points = data.get("points", []) if isinstance(data, dict) else []
            map_results.append(
                {
                    "group_index": group_idx,
                    "community_titles": [community.get("title") for community in group],
                    "raw_response": response,
                    "parsed_points": parsed_points,
                }
            )
        return map_results

    @staticmethod
    def _split_field(value: Any) -> List[str]:
        if value in (None, "", "N/A"):
            return []
        return [item for item in str(value).split(GRAPH_FIELD_SEP) if item and item != "N/A"]

    @staticmethod
    def _load_clusters(value: Any) -> List[dict]:
        if value in (None, ""):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
            return parsed if isinstance(parsed, list) else []
        return []

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
