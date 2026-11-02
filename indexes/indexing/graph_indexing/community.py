import csv
import html
import io
import json
import re
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from RoleRAG.indexes.indexing.graph_indexing.universal_graph import UniversalGraph
from RoleRAG.rolerag.pipelines.prompts import PromptManager

GRAPH_FIELD_SEP = "<SEP>"


def clean_graph_str(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = html.unescape(value.strip())
    value = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", value)
    return re.sub(r"[^A-Za-z0-9 ]", " ", value.lower()).strip()


def list_to_quoted_csv_string(rows: List[List[Any]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerows(rows)
    return output.getvalue().strip()


def community_report_from_json(data: dict) -> str:
    title = data.get("title", "Report")
    summary = data.get("summary", "")
    findings = data.get("findings", [])

    report_sections = []
    for finding in findings:
        if isinstance(finding, str):
            report_sections.append(f"## {finding}\n")
            continue
        if isinstance(finding, dict):
            report_sections.append(
                f"## {finding.get('summary', '')}\n\n{finding.get('explanation', '')}"
            )
    return f"# {title}\n\n{summary}\n\n" + "\n\n".join(report_sections)


def parse_json_object(text: Any) -> dict:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str) or not text.strip():
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start:end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def truncate_list_by_token_size(
    items: List[Any],
    key: Callable[[Any], str],
    max_token_size: int,
    tokenizer: Optional[Any] = None,
) -> List[Any]:
    if max_token_size <= 0:
        return []
    total = 0
    result = []
    for item in items:
        text = str(key(item) or "")
        size = _token_len(text, tokenizer)
        if result and total + size > max_token_size:
            break
        if not result and size > max_token_size:
            result.append(item)
            break
        result.append(item)
        total += size
    return result


class GraphRAGCommunityBuilder:
    def __init__(
        self,
        config: Optional[dict] = None,
        llm_call: Optional[Callable[[str], str]] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config or {}
        self.llm_call = llm_call
        self.tokenizer = tokenizer

    def build(self, graph: UniversalGraph) -> Tuple[dict, dict, dict]:
        community_node_map = self.cluster(graph)
        community_schema = self.build_schema(graph)
        community_reports = self.generate_reports(graph, community_schema)
        return community_node_map, community_schema, community_reports

    def cluster(self, graph: UniversalGraph) -> Dict[str, List[dict]]:
        try:
            from graspologic.partition import hierarchical_leiden
        except ImportError as exc:
            raise ImportError(
                "GraphRAG community clustering requires graspologic. "
                "Install it before building a GraphRAG index."
            ) from exc

        largest_cc = stable_largest_connected_component(graph.raw_graph)
        max_cluster_size = int(self.config.get("max_graph_cluster_size", 10))
        random_seed = int(self.config.get("graph_cluster_seed", 0xDEADBEEF))
        community_mapping = hierarchical_leiden(
            largest_cc,
            max_cluster_size=max_cluster_size,
            random_seed=random_seed,
        )

        community_node_map: Dict[str, List[dict]] = defaultdict(list)
        for partition in community_mapping:
            node_id = clean_graph_str(partition.node)
            community_node_map[node_id].append(
                {"level": int(partition.level), "cluster": str(partition.cluster)}
            )

        community_node_map = dict(community_node_map)
        for node_id, clusters in community_node_map.items():
            if graph.has_node(node_id):
                graph.update_node(node_id, clusters=json.dumps(clusters, ensure_ascii=False))
        return community_node_map

    def build_schema(self, graph: UniversalGraph) -> Dict[str, dict]:
        schemas = defaultdict(
            lambda: {
                "level": None,
                "title": None,
                "edges": set(),
                "nodes": set(),
                "chunk_ids": set(),
                "occurrence": 0.0,
                "sub_communities": [],
            }
        )
        levels = defaultdict(set)
        max_num_ids = 0

        for node_id, node_data in graph.raw_graph.nodes(data=True):
            clusters = _load_clusters(node_data.get("clusters"))
            if not clusters:
                continue
            node_edges = graph.get_node_edges(node_id)
            for cluster in clusters:
                level = int(cluster["level"])
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                schemas[cluster_key]["level"] = level
                schemas[cluster_key]["title"] = f"Cluster {cluster_key}"
                schemas[cluster_key]["nodes"].add(node_id)
                schemas[cluster_key]["edges"].update(tuple(sorted(edge)) for edge in node_edges)
                for chunk_id in _split_field(node_data.get("source_id")):
                    schemas[cluster_key]["chunk_ids"].add(chunk_id)
                max_num_ids = max(max_num_ids, len(schemas[cluster_key]["chunk_ids"]))

        ordered_levels = sorted(levels.keys())
        for idx, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[idx + 1]
            for comm in levels[curr_level]:
                schemas[comm]["sub_communities"] = [
                    child
                    for child in levels[next_level]
                    if schemas[child]["nodes"].issubset(schemas[comm]["nodes"])
                ]

        result = {}
        for key, value in schemas.items():
            nodes = sorted(value["nodes"])
            edges = [list(edge) for edge in sorted(value["edges"])]
            chunk_ids = sorted(value["chunk_ids"])
            result[key] = {
                "level": value["level"],
                "title": value["title"],
                "edges": edges,
                "nodes": nodes,
                "chunk_ids": chunk_ids,
                "occurrence": len(chunk_ids) / max_num_ids if max_num_ids else 0.0,
                "sub_communities": sorted(value["sub_communities"]),
            }
        return result

    def generate_reports(self, graph: UniversalGraph, community_schema: Dict[str, dict]) -> Dict[str, dict]:
        if self.llm_call is None:
            raise ValueError("GraphRAG community report generation requires an llm_call.")
        reports = {}
        levels = sorted({data["level"] for data in community_schema.values()}, reverse=True)
        for level in levels:
            level_items = [
                (key, value)
                for key, value in community_schema.items()
                if value["level"] == level
            ]
            for community_id, community in level_items:
                describe = self.pack_community_describe(graph, community, reports)
                prompt = PromptManager.get_template("graphrag_community_report").user_prompt
                response = self.llm_call(prompt.format(input_text=describe))
                data = parse_json_object(response)
                if not data:
                    data = {
                        "title": community.get("title") or f"Cluster {community_id}",
                        "summary": str(response or ""),
                        "rating": 0,
                        "rating_explanation": "",
                        "findings": [],
                    }
                reports[community_id] = {
                    "report_string": community_report_from_json(data),
                    "report_json": data,
                    **community,
                }
        return reports

    def pack_community_describe(
        self,
        graph: UniversalGraph,
        community: dict,
        already_reports: Dict[str, dict],
    ) -> str:
        max_token_size = int(self.config.get("community_describe_max_tokens", 12000))
        nodes_in_order = sorted(community.get("nodes", []))
        edges_in_order = sorted(community.get("edges", []), key=lambda x: x[0] + x[1])

        report_describe = ""
        contain_nodes = set()
        contain_edges = set()
        need_sub_reports = (
            bool(community.get("sub_communities"))
            and bool(already_reports)
            and (
                bool(self.config.get("enforce_sub_communities", False))
                or len(nodes_in_order) > 100
                or len(edges_in_order) > 100
            )
        )
        if need_sub_reports:
            report_describe, _, contain_nodes, contain_edges = self._pack_sub_communities(
                community,
                max_token_size=max_token_size // 3,
                already_reports=already_reports,
            )

        node_rows = []
        for idx, node_id in enumerate(nodes_in_order):
            if node_id in contain_nodes:
                continue
            node = graph.get_node(node_id) or {}
            node_rows.append(
                [
                    idx,
                    node_id,
                    node.get("entity_type") or "UNKNOWN",
                    node.get("description") or "UNKNOWN",
                    graph.degree(node_id),
                ]
            )
        node_rows.sort(key=lambda row: row[-1], reverse=True)

        edge_rows = []
        for idx, edge in enumerate(edges_in_order):
            src, tgt = edge
            if (src, tgt) in contain_edges or (tgt, src) in contain_edges:
                continue
            data = graph.get_edge(src, tgt) or {}
            edge_rows.append(
                [
                    idx,
                    src,
                    tgt,
                    data.get("description") or "UNKNOWN",
                    graph.degree(src) + graph.degree(tgt),
                ]
            )
        edge_rows.sort(key=lambda row: row[-1], reverse=True)

        node_fields = ["id", "entity", "type", "description", "degree"]
        edge_fields = ["id", "source", "target", "description", "rank"]
        node_budget = max_token_size // 2
        edge_budget = max_token_size // 2
        node_rows = truncate_list_by_token_size(node_rows, key=lambda row: str(row[3]), max_token_size=node_budget, tokenizer=self.tokenizer)
        edge_rows = truncate_list_by_token_size(edge_rows, key=lambda row: str(row[3]), max_token_size=edge_budget, tokenizer=self.tokenizer)

        return """-----Reports-----
```csv
{reports}
```
-----Entities-----
```csv
{entities}
```
-----Relationships-----
```csv
{relationships}
```""".format(
            reports=report_describe,
            entities=list_to_quoted_csv_string([node_fields] + node_rows),
            relationships=list_to_quoted_csv_string([edge_fields] + edge_rows),
        )

    def _pack_sub_communities(
        self,
        community: dict,
        max_token_size: int,
        already_reports: Dict[str, dict],
    ) -> Tuple[str, int, set, set]:
        sub_reports = [
            already_reports[key]
            for key in community.get("sub_communities", [])
            if key in already_reports
        ]
        sub_reports.sort(key=lambda item: item.get("occurrence", 0.0), reverse=True)
        sub_reports = truncate_list_by_token_size(
            sub_reports,
            key=lambda item: item.get("report_string", ""),
            max_token_size=max_token_size,
            tokenizer=self.tokenizer,
        )
        rows = [["id", "report", "rating", "importance"]]
        contain_nodes = set()
        contain_edges = set()
        for idx, report in enumerate(sub_reports):
            rows.append(
                [
                    idx,
                    report.get("report_string", ""),
                    report.get("report_json", {}).get("rating", -1),
                    report.get("occurrence", 0.0),
                ]
            )
            contain_nodes.update(report.get("nodes", []))
            contain_edges.update(tuple(edge) for edge in report.get("edges", []))
        content = list_to_quoted_csv_string(rows)
        return content, _token_len(content, self.tokenizer), contain_nodes, contain_edges


def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph.copy()
    try:
        from graspologic.utils import largest_connected_component

        graph = largest_connected_component(graph.copy())
    except Exception:
        if graph.is_directed():
            components = nx.weakly_connected_components(graph)
        else:
            components = nx.connected_components(graph)
        largest = max(components, key=len)
        graph = graph.subgraph(largest).copy()

    mapping = {node: clean_graph_str(node) for node in graph.nodes()}
    graph = nx.relabel_nodes(graph, mapping)
    return _stabilize_graph(graph)


def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
    fixed = nx.DiGraph() if graph.is_directed() else nx.Graph()
    fixed.add_nodes_from(sorted(graph.nodes(data=True), key=lambda item: item[0]))
    edges = list(graph.edges(data=True))
    if not graph.is_directed():
        normalized_edges = []
        for src, tgt, data in edges:
            if src > tgt:
                src, tgt = tgt, src
            normalized_edges.append((src, tgt, data))
        edges = normalized_edges
    edges = sorted(edges, key=lambda item: f"{item[0]} -> {item[1]}")
    fixed.add_edges_from(edges)
    return fixed


def _split_field(value: Any) -> List[str]:
    if value in (None, "", "N/A"):
        return []
    return [item for item in str(value).split(GRAPH_FIELD_SEP) if item and item != "N/A"]


def _load_clusters(value: Any) -> List[dict]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []
    return []


def _token_len(text: str, tokenizer: Optional[Any] = None) -> int:
    if tokenizer is not None:
        if hasattr(tokenizer, "encode"):
            try:
                return len(tokenizer.encode(text))
            except Exception:
                pass
    return max(1, len(str(text)) // 4)
