import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from RoleRAG.config.types import Document
from RoleRAG.indexes.indexing.graph_indexing.universal_graph import UniversalGraph

GRAPH_FIELD_SEP = "<SEP>"


class GraphIndexer:

    def __init__(self, config: Optional[dict] = None, encoder=None):
        self.config = config or {}
        self.encoder = encoder

        self.graph: Optional[UniversalGraph] = None
        self.chunks: List[Document] = []

        self.entity_vdb = None
        self.relation_vdb = None

        self.node_ids: List[str] = []
        self.edge_ids: List[Tuple[str, str]] = []
        self.entity_id_to_idx: Dict[str, int] = {}
        self.idx_to_entity_id: Dict[int, str] = {}
        self.entity_texts: List[str] = []
        self.relation_id_to_idx: Dict[str, int] = {}
        self.idx_to_relation_id: Dict[int, Tuple[str, str]] = {}
        self.relation_texts: List[str] = []

        self.e2r_matrix: Optional[csr_matrix] = None
        self.r2c_matrix: Optional[csr_matrix] = None

        self.community_node_map: Dict[str, List[dict]] = {}
        self.community_schema: Dict[str, dict] = {}
        self.community_reports: Dict[str, dict] = {}

    def build_indices(self, graph: UniversalGraph, chunks: List[Document], llm_call=None, tokenizer=None):
        self.graph = graph
        self.chunks = chunks
        self._refresh_ids()

        if llm_call is not None and self.config.get("use_community", False):
            self._summarize_long_descriptions(llm_call=llm_call, tokenizer=tokenizer)

        if self.config.get("use_entities_vdb", False):
            self._build_entity_vdb()

        if self.config.get("enable_graph_augmentation", False):
            self._augment_graph()
            self._refresh_ids()

        if self.config.get("use_relations_vdb", False):
            self._build_relation_vdb()

        if self.config.get("use_e2r_r2c", False):
            self._build_matrices()

        if self.config.get("use_community", False):
            self._build_communities(llm_call=llm_call, tokenizer=tokenizer)

    def _refresh_ids(self):
        if self.graph is None:
            self.node_ids, self.edge_ids = [], []
            return
        self.node_ids = self.graph.node_ids()
        self.edge_ids = self.graph.edge_ids()
        self.entity_id_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
        self.idx_to_entity_id = {idx: node_id for node_id, idx in self.entity_id_to_idx.items()}
        self.relation_id_to_idx = {self._edge_key(*edge): idx for idx, edge in enumerate(self.edge_ids)}
        self.idx_to_relation_id = {idx: edge for idx, edge in enumerate(self.edge_ids)}

    def _build_entity_vdb(self):
        import faiss

        self.entity_texts = []
        text_fields = self.config.get("entity_vdb_text_fields") or ["entity_name"]
        for node_id in self.node_ids:
            node = self.graph.get_node(node_id) or {}
            parts = []
            for field in text_fields:
                value = node.get(field)
                if value not in (None, ""):
                    parts.append(str(value))
            self.entity_texts.append(" ".join(parts) if parts else str(node.get("entity_name") or node_id))

        if not self.entity_texts:
            return
        embeddings = self._encode(self.entity_texts, is_query=False)
        faiss.normalize_L2(embeddings)
        self.entity_vdb = faiss.index_factory(embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        self.entity_vdb.add(embeddings)

    def _build_relation_vdb(self):
        import faiss

        self.relation_texts = []
        for src, tgt in self.edge_ids:
            edge = self.graph.get_edge(src, tgt) or {}
            text = edge.get("relation_name") or edge.get("description") or f"{src} -> {tgt}"
            self.relation_texts.append(str(text))

        if not self.relation_texts:
            return
        embeddings = self._encode(self.relation_texts, is_query=False)
        faiss.normalize_L2(embeddings)
        self.relation_vdb = faiss.index_factory(embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        self.relation_vdb.add(embeddings)

    def _augment_graph(self):
        import faiss
        from tqdm import tqdm

        if self.graph is None or self.entity_vdb is None or len(self.node_ids) < 2:
            return

        top_k = int(self.config.get("augment_top_k", 10))
        threshold = float(self.config.get("augment_threshold", 0.8))
        sim_max = float(self.config.get("similarity_max", 1.0))

        added = 0
        for node_id in tqdm(self.node_ids, desc="Augmenting graph"):
            query = self._encode([node_id], is_query=True)
            faiss.normalize_L2(query)
            scores, indices = self.entity_vdb.search(query, min(top_k, len(self.node_ids)))

            for idx, score in zip(indices[0], scores[0]):
                if idx < 0:
                    continue
                neighbor = self.idx_to_entity_id.get(int(idx))
                if not neighbor or neighbor == node_id:
                    continue

                raw_score = float(score)
                if raw_score < threshold:
                    break
                weight = sim_max * raw_score
                if self.graph.has_edge(node_id, neighbor):
                    self._merge_similarity_edge(node_id, neighbor, weight)
                else:
                    self.graph.add_edge(
                        node_id,
                        neighbor,
                        relation_name="similarity",
                        weight=weight,
                        source_id="N/A",
                    )
                added += 1
        print(f"Graph augmentation: added {added} similarity edges")

    def _merge_similarity_edge(self, src_id: str, tgt_id: str, weight: float):
        edge = self.graph.get_edge(src_id, tgt_id) or {}
        merged_source_id = self._merge_field_values(edge.get("source_id"), "N/A")
        merged_relation_name = self._merge_field_values(edge.get("relation_name"), "similarity")
        try:
            merged_weight = float(edge.get("weight", 0.0) or 0.0) + float(weight)
        except (TypeError, ValueError):
            merged_weight = float(weight)
        self.graph.update_edge(
            src_id,
            tgt_id,
            weight=merged_weight,
            source_id=merged_source_id,
            relation_name=merged_relation_name,
        )

    def _build_matrices(self):
        chunk_id_to_idx = {chunk.id: idx for idx, chunk in enumerate(self.chunks)}
        e2r_rows, e2r_cols = [], []
        r2c_rows, r2c_cols = [], []

        for edge_idx, (src, tgt) in enumerate(self.edge_ids):
            for node_id in (src, tgt):
                node_idx = self.entity_id_to_idx.get(node_id)
                if node_idx is not None:
                    e2r_rows.append(node_idx)
                    e2r_cols.append(edge_idx)

            edge = self.graph.get_edge(src, tgt) or {}
            for chunk_id in self._edge_chunk_ids(edge):
                chunk_idx = chunk_id_to_idx.get(chunk_id)
                if chunk_idx is not None:
                    r2c_rows.append(edge_idx)
                    r2c_cols.append(chunk_idx)

        self.e2r_matrix = csr_matrix(
            (np.ones(len(e2r_rows)), (e2r_rows, e2r_cols)),
            shape=(len(self.node_ids), len(self.edge_ids)),
        )
        self.r2c_matrix = csr_matrix(
            (np.ones(len(r2c_rows)), (r2c_rows, r2c_cols)),
            shape=(len(self.edge_ids), len(self.chunks)),
        )

    def _build_communities(self, llm_call=None, tokenizer=None):
        if self.graph is None:
            return
        from RoleRAG.indexes.indexing.graph_indexing.community import GraphRAGCommunityBuilder

        builder = GraphRAGCommunityBuilder(
            config=self.config,
            llm_call=llm_call,
            tokenizer=tokenizer,
        )
        self.community_node_map, self.community_schema, self.community_reports = builder.build(self.graph)

    def _summarize_long_descriptions(self, llm_call, tokenizer=None):
        if self.graph is None:
            return
        from RoleRAG.indexes.indexing.graph_indexing.community import _token_len
        from RoleRAG.rolerag.pipelines.prompts import PromptManager

        max_tokens = int(self.config.get("summary_max_tokens", 500))
        prompt = PromptManager.get_template("graphrag_summarize_descriptions").user_prompt

        for node_id in self.node_ids:
            node = self.graph.get_node(node_id) or {}
            description = node.get("description")
            if not description or _token_len(str(description), tokenizer) < max_tokens:
                continue
            response = llm_call(
                prompt.format(
                    entity_name=node_id,
                    description_list=str(description).split(GRAPH_FIELD_SEP),
                )
            )
            summary = _extract_summary_text(response)
            if summary:
                self.graph.update_node(node_id, description=summary)

        for src, tgt in self.edge_ids:
            edge = self.graph.get_edge(src, tgt) or {}
            description = edge.get("description")
            if not description or _token_len(str(description), tokenizer) < max_tokens:
                continue
            response = llm_call(
                prompt.format(
                    entity_name=f"{src}, {tgt}",
                    description_list=str(description).split(GRAPH_FIELD_SEP),
                )
            )
            summary = _extract_summary_text(response)
            if summary:
                self.graph.update_edge(src, tgt, description=summary)

    def _encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if self.encoder is None:
            raise ValueError("GraphIndexer needs an encoder to build vector indices.")
        batch_size = int(getattr(self.encoder, "batch_size", 32) or 32)
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            embeddings.append(self.encoder.encode(batch, is_query=is_query))
        return np.concatenate(embeddings, axis=0).astype("float32")

    @staticmethod
    def _edge_chunk_ids(edge: dict) -> List[str]:
        source_id = edge.get("source_id")
        if source_id in (None, "", "N/A"):
            return []
        return [item for item in str(source_id).split(GRAPH_FIELD_SEP) if item and item != "N/A"]

    @staticmethod
    def _edge_key(src_id: str, tgt_id: str) -> str:
        return f"{src_id}{GRAPH_FIELD_SEP}{tgt_id}"

    @staticmethod
    def _merge_field_values(existing: Any, new_value: str) -> str:
        values = []
        if existing not in (None, ""):
            values.extend(str(existing).split(GRAPH_FIELD_SEP))
        if new_value not in (None, ""):
            values.append(str(new_value))
        return GRAPH_FIELD_SEP.join(sorted({value for value in values if value}))

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self._save_pickle_bundle(os.path.join(path, "index.pkl"))

        if self.graph is not None:
            self.graph.to_pickle(os.path.join(path, "graph.pkl"))
            try:
                self.graph.to_graphml(os.path.join(path, "graph.graphml"))
            except Exception:
                pass

        with open(os.path.join(path, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        if self.entity_vdb is not None:
            import faiss
            faiss.write_index(self.entity_vdb, os.path.join(path, "entity_vdb.index"))

        if self.relation_vdb is not None:
            import faiss
            faiss.write_index(self.relation_vdb, os.path.join(path, "relation_vdb.index"))

        if self.e2r_matrix is not None:
            save_npz(os.path.join(path, "e2r.npz"), self.e2r_matrix)
        if self.r2c_matrix is not None:
            save_npz(os.path.join(path, "r2c.npz"), self.r2c_matrix)

        self._save_json(os.path.join(path, "community_node_map.json"), self.community_node_map)
        self._save_json(os.path.join(path, "community_schema.json"), self.community_schema)
        self._save_json(os.path.join(path, "community_reports.json"), self.community_reports)

        with open(os.path.join(path, "index_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self._metadata(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "GraphIndexer":
        bundle_path = os.path.join(path, "index.pkl")
        return GraphIndexer._load_pickle_bundle(bundle_path)

    def _save_pickle_bundle(self, path: str):
        bundle = self._metadata()
        bundle.update(
            {
                "graph": self.graph,
                "chunks": [chunk.to_dict() for chunk in self.chunks],
                "e2r_matrix": self.e2r_matrix,
                "r2c_matrix": self.r2c_matrix,
                "entity_vdb": self._serialize_faiss_index(self.entity_vdb),
                "relation_vdb": self._serialize_faiss_index(self.relation_vdb),
                "community_node_map": self.community_node_map,
                "community_schema": self.community_schema,
                "community_reports": self.community_reports,
            }
        )
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

    @staticmethod
    def _load_pickle_bundle(path: str) -> "GraphIndexer":
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        indexer = GraphIndexer(config=bundle.get("config", {}))
        indexer.graph = bundle.get("graph")
        indexer.chunks = [Document.from_dict(item) for item in bundle.get("chunks", [])]
        indexer.node_ids = bundle.get("node_ids", [])
        indexer.edge_ids = [tuple(item) for item in bundle.get("edge_ids", [])]
        indexer.entity_id_to_idx = bundle.get("entity_id_to_idx", {})
        indexer.idx_to_entity_id = {int(k): v for k, v in bundle.get("idx_to_entity_id", {}).items()}
        indexer.entity_texts = bundle.get("entity_texts", [])
        indexer.relation_id_to_idx = bundle.get("relation_id_to_idx", {})
        indexer.idx_to_relation_id = {int(k): tuple(v) for k, v in bundle.get("idx_to_relation_id", {}).items()}
        indexer.relation_texts = bundle.get("relation_texts", [])
        indexer.e2r_matrix = bundle.get("e2r_matrix")
        indexer.r2c_matrix = bundle.get("r2c_matrix")
        indexer.community_node_map = bundle.get("community_node_map", {})
        indexer.community_schema = bundle.get("community_schema", {})
        indexer.community_reports = bundle.get("community_reports", {})
        indexer.entity_vdb = GraphIndexer._deserialize_faiss_index(bundle.get("entity_vdb"))
        indexer.relation_vdb = GraphIndexer._deserialize_faiss_index(bundle.get("relation_vdb"))
        if indexer.graph is not None and not indexer.node_ids:
            indexer._refresh_ids()
        return indexer

    def _metadata(self) -> dict:
        return {
            "config": self.config,
            "node_ids": self.node_ids,
            "edge_ids": [list(edge) for edge in self.edge_ids],
            "entity_id_to_idx": self.entity_id_to_idx,
            "idx_to_entity_id": self.idx_to_entity_id,
            "entity_texts": self.entity_texts,
            "relation_id_to_idx": self.relation_id_to_idx,
            "idx_to_relation_id": {idx: list(edge) for idx, edge in self.idx_to_relation_id.items()},
            "relation_texts": self.relation_texts,
            "stats": {
                "nodes": len(self.node_ids),
                "edges": len(self.edge_ids),
                "chunks": len(self.chunks),
                "e2r_nnz": int(self.e2r_matrix.nnz) if self.e2r_matrix is not None else 0,
                "r2c_nnz": int(self.r2c_matrix.nnz) if self.r2c_matrix is not None else 0,
                "communities": len(self.community_schema),
                "community_reports": len(self.community_reports),
            },
        }

    @staticmethod
    def _save_json(path: str, payload: Any):
        if payload in (None, {}, []):
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _load_json(path: str):
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _serialize_faiss_index(index):
        if index is None:
            return None
        import faiss
        return faiss.serialize_index(index)

    @staticmethod
    def _deserialize_faiss_index(payload):
        if payload is None:
            return None
        import faiss
        return faiss.deserialize_index(payload)


def _extract_summary_text(response: Any) -> str:
    if response is None:
        return ""
    text = str(response).strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(data, dict):
        return str(data.get("summary") or data.get("description") or text).strip()
    return text
