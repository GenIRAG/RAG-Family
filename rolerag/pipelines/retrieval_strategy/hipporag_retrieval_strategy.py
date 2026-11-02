from typing import Any, Dict, List, Tuple

import numpy as np

from RoleRAG.config.types import RetrieverResult


class HippoRAGRetrievalStrategy:

    _entity_chunk_count_cache = {}
    _graph_index_cache = {}

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        ppr_cfg = (self.config.get("inference", {}) or {}).get("ppr", {}) or {}
        self.ppr_top_k = int(ppr_cfg.get("ppr_top_k", 5))
        self.ppr_damping = float(ppr_cfg.get("ppr_damping", 0.1))
        self.use_node_specificity = bool(ppr_cfg.get("use_node_specificity", True))

    def retrieve(
        self,
        query: str,
        retriever,
        reasoner,
    ) -> RetrieverResult:
        graph_indexer = self._load_graph_index(retriever)
        if graph_indexer is None or graph_indexer.graph is None:
            return RetrieverResult(
                documents=[],
                scores=[],
                metadata={"strategy": "hipporag", "query": query, "error": "missing graph index"},
            )
        if graph_indexer.e2r_matrix is None or graph_indexer.r2c_matrix is None:
            return RetrieverResult(
                documents=[],
                scores=[],
                metadata={"strategy": "hipporag", "query": query, "error": "missing graph matrices"},
            )

        query_entities = self._extract_query_entities(reasoner, query)
        linked = self._link_query_entities(query_entities, graph_indexer, retriever)

        if not linked:
            fallback_result = retriever.search(query, top_k=self.ppr_top_k)
            return RetrieverResult(
                documents=fallback_result.documents,
                scores=fallback_result.scores,
                metadata={
                    "strategy": "hipporag",
                    "query": query,
                    "entities": query_entities,
                    "linked": [],
                    "linked_scores": [],
                    "fallback": True,
                    "fallback_type": "dense_chunks",
                    "r2c_nnz": int(graph_indexer.r2c_matrix.nnz),
                },
            )

        reset_prob = self._build_reset_vector(
            graph_indexer=graph_indexer,
            linked_entities=linked,
            use_node_specificity=self.use_node_specificity,
        )
        ppr_scores = self._run_personalized_pagerank(graph_indexer.graph, reset_prob, self.ppr_damping)

        docs, chunk_scores = self._propagate_ppr_to_chunks(
            ppr_scores=ppr_scores,
            e2r_matrix=graph_indexer.e2r_matrix,
            r2c_matrix=graph_indexer.r2c_matrix,
            chunks=graph_indexer.chunks,
            top_k=self.ppr_top_k,
        )

        return RetrieverResult(
            documents=docs,
            scores=chunk_scores,
            metadata={
                "strategy": "hipporag",
                "query": query,
                "entities": query_entities,
                "linked": [entity["entity_name"] for entity in linked],
                "linked_scores": [entity.get("score") for entity in linked],
                "fallback": not bool(linked),
                "r2c_nnz": int(graph_indexer.r2c_matrix.nnz),
            },
        )

    @classmethod
    def _load_graph_index(cls, retriever):
        from RoleRAG.indexes.loading.graph_loading import load_graph_index

        graph_index_path = getattr(retriever, "graph_index_path", None)
        if not graph_index_path:
            return None
        if graph_index_path not in cls._graph_index_cache:
            cls._graph_index_cache[graph_index_path] = load_graph_index(graph_index_path)
        return cls._graph_index_cache[graph_index_path]

    def _build_reset_vector(self, graph_indexer, linked_entities: List[Dict[str, Any]], use_node_specificity: bool) -> np.ndarray:
        n_nodes = graph_indexer.graph.node_num
        reset_prob = np.zeros(n_nodes, dtype="float64")
        if not linked_entities:
            if n_nodes:
                reset_prob[:] = 1.0 / n_nodes
            return reset_prob

        for entity in linked_entities:
            idx = graph_indexer.entity_id_to_idx.get(entity["entity_name"])
            if idx is None:
                idx = graph_indexer.graph.node_index(entity["entity_name"])
            if idx is None or idx < 0:
                continue
            if use_node_specificity:
                count = self._get_entity_chunk_count(graph_indexer, idx)
                reset_prob[idx] = 1.0 / float(count) if count > 0 else 1.0
            else:
                reset_prob[idx] = 1.0
        if reset_prob.sum() == 0 and n_nodes:
            reset_prob[:] = 1.0 / n_nodes
        return reset_prob

    def _get_entity_chunk_count(self, graph_indexer, idx: int) -> float:
        cache_key = id(graph_indexer)
        if cache_key not in self._entity_chunk_count_cache:
            if graph_indexer.e2r_matrix is None or graph_indexer.r2c_matrix is None:
                return 0
            e2r = graph_indexer.e2r_matrix
            r2c = graph_indexer.r2c_matrix
            c2e = e2r.dot(r2c).T
            c2e[c2e.nonzero()] = 1
            self._entity_chunk_count_cache[cache_key] = np.array(c2e.sum(axis=0)).flatten()
        counts = self._entity_chunk_count_cache[cache_key]
        return counts[idx] if idx < len(counts) else 0

    @staticmethod
    def _extract_query_entities(reasoner, query: str) -> List[str]:
        from RoleRAG.indexes.indexing.graph_indexing.extraction.hipporag_extraction import clean_entity_name, ner_extract

        try:
            generate = getattr(reasoner, "generate", reasoner)
            raw_entities = ner_extract(generate, query)
        except Exception:
            raw_entities = []
        return [clean_entity_name(entity) for entity in raw_entities if isinstance(entity, str) and entity.strip()]

    @staticmethod
    def _link_query_entities(query_entities: List[str], graph_indexer, retriever) -> List[Dict[str, Any]]:
        import faiss

        candidates = []
        if graph_indexer.entity_vdb is None:
            return candidates

        for name in query_entities:
            try:
                q_emb = retriever.encode([name], is_query=True)
                q_emb = q_emb.reshape(1, -1).astype("float32")
                faiss.normalize_L2(q_emb)
                scores, indices = graph_indexer.entity_vdb.search(q_emb, 1)
                if indices[0][0] >= 0:
                    idx = int(indices[0][0])
                    entity_id = graph_indexer.idx_to_entity_id.get(idx)
                    score = float(scores[0][0])
                    if entity_id:
                        candidates.append(
                            {
                                "entity_name": entity_id,
                                "score": score,
                            }
                        )
            except Exception:
                pass
        return candidates

    @staticmethod
    def _run_personalized_pagerank(graph, reset_prob: np.ndarray, damping: float = 0.1) -> np.ndarray:
        n_nodes = graph.node_num
        if n_nodes == 0:
            return np.zeros(0)
        if reset_prob is None or len(reset_prob) != n_nodes or reset_prob.sum() <= 0:
            reset_prob = np.ones(n_nodes, dtype="float64") / n_nodes
        else:
            reset_prob = reset_prob.astype("float64")
            reset_prob = reset_prob / reset_prob.sum()

        import igraph as ig

        ig_graph = ig.Graph.from_networkx(graph.raw_graph)
        weights = []
        for src, tgt in graph.edge_ids():
            weight = graph.get_edge_weight(src, tgt)
            weights.append(weight if weight > 0 else 1.0)
        if weights:
            ig_graph.es["weight"] = weights
        scores = ig_graph.personalized_pagerank(
            vertices=range(n_nodes),
            damping=damping,
            directed=False,
            weights="weight" if weights else None,
            reset=reset_prob,
            implementation="prpack",
        )
        return np.asarray(scores, dtype="float64")

    @staticmethod
    def _propagate_ppr_to_chunks(
        ppr_scores: np.ndarray,
        e2r_matrix,
        r2c_matrix,
        chunks: list,
        top_k: int = 5,
        normalize: bool = True,
    ) -> Tuple[list, List[float]]:
        if e2r_matrix is None or r2c_matrix is None or len(chunks) == 0:
            return [], []
        if e2r_matrix.shape[1] != r2c_matrix.shape[0]:
            raise ValueError(f"e2r/r2c shape mismatch: {e2r_matrix.shape} vs {r2c_matrix.shape}")

        edge_scores = e2r_matrix.T.dot(ppr_scores)
        chunk_scores = r2c_matrix.T.dot(edge_scores)
        chunk_scores = np.asarray(chunk_scores).reshape(-1)

        if normalize and chunk_scores.size:
            cmin = float(chunk_scores.min())
            cmax = float(chunk_scores.max())
            if cmax > cmin:
                chunk_scores = (chunk_scores - cmin) / (cmax - cmin)

        top_k = min(int(top_k), len(chunks), chunk_scores.size)
        if top_k <= 0:
            return [], []

        sorted_indices = np.argsort(chunk_scores, kind="mergesort")[::-1][:top_k]
        docs, scores = [], []
        for idx in sorted_indices:
            doc = chunks[int(idx)]
            score = float(chunk_scores[int(idx)])
            doc.score = score
            docs.append(doc)
            scores.append(score)
        return docs, scores
