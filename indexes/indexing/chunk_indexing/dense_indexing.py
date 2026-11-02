import json
import os

import faiss
import numpy as np
from typing import List, Optional

from RoleRAG.config.types import Document
from RoleRAG.indexes.indexing.chunk_indexing.chunk_indexing_base import BaseIndexing
from RoleRAG.models.hf_encoder import HFEncoder


class DenseIndexing(BaseIndexing):

    def __init__(
        self,
        encoder: Optional[HFEncoder] = None,
        index_path: Optional[str] = None,
        batch_size: int = 32,
        faiss_type: str = "Flat",
        faiss_nlist: int = 1024,
        faiss_nprobe: int = 10,
        faiss_hnsw_m: int = 32,
        faiss_ef_search: int = 64,
    ):
        self.encoder = encoder
        self.index_path = index_path
        self.batch_size = batch_size
        self.faiss_type = faiss_type
        self.faiss_nlist = faiss_nlist
        self.faiss_nprobe = faiss_nprobe
        self.faiss_hnsw_m = faiss_hnsw_m
        self.faiss_ef_search = faiss_ef_search
        self.index = None
        self.documents = []
        self.id_to_doc = {}

    def _build_faiss_index(self, dimension: int, embeddings: np.ndarray):
        
        faiss_type = self.faiss_type.upper()
        if faiss_type == "FLAT":
            index = faiss.IndexFlatIP(dimension)
        elif faiss_type == "IVF":
            nlist = min(self.faiss_nlist, len(embeddings))
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.nprobe = min(self.faiss_nprobe, nlist)
        elif faiss_type == "HNSW":
            index = faiss.IndexHNSWFlat(dimension, self.faiss_hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = self.faiss_ef_search
        else:
            raise ValueError("Unsupported faiss_type. Expected one of: Flat, IVF, HNSW")
        return index

    def _apply_search_params(self):
        if self.index is None:
            return
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = min(self.faiss_nprobe, self.index.nlist)
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = self.faiss_ef_search

    def _index_meta(self) -> dict:
        return {
            "faiss_type": self.faiss_type,
            "faiss_nlist": self.faiss_nlist,
            "faiss_nprobe": self.faiss_nprobe,
            "faiss_hnsw_m": self.faiss_hnsw_m,
            "faiss_ef_search": self.faiss_ef_search,
        }

    def build_index(self, documents: List[Document], **kwargs):
        self.documents = documents
        self.id_to_doc = {doc.id: doc for doc in documents}

        batch_size = kwargs.get("batch_size", self.batch_size)
        corpus_text = [doc.content for doc in documents]
        all_embeddings = []
        for start_idx in range(0, len(corpus_text), batch_size):
            batch_text = corpus_text[start_idx : start_idx + batch_size]
            batch_embeddings = self.encoder.encode(batch_text)
            all_embeddings.append(batch_embeddings)

        embeddings = np.concatenate(all_embeddings, axis=0).astype('float32')
        dimension = embeddings.shape[1]
        
        faiss.normalize_L2(embeddings)
        self.index = self._build_faiss_index(dimension, embeddings)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> tuple:
        
        if self.index is None:
            return [], []

        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        scores = scores[0].tolist()
        indices = indices[0].tolist()

        retrieved_docs = []
        retrieved_scores = []
        for idx, score in zip(indices, scores):
            if idx >= 0 and idx < len(self.documents):
                
                doc = self.documents[idx]
                doc.score = score
                retrieved_docs.append(doc)
                retrieved_scores.append(score)

        return retrieved_docs, retrieved_scores

    def save(self, path: str):
        
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.index"))
        with open(os.path.join(path, "index.json"), "w", encoding="utf-8") as f:
            json.dump([doc.to_dict() for doc in self.documents], f, ensure_ascii=False, indent=2)
        with open(os.path.join(path, "index_meta.json"), "w", encoding="utf-8") as f:
            json.dump(self._index_meta(), f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        
        files = os.listdir(path)
        index_file = next((f for f in files if f.endswith(".index")), None)
        json_file = next((f for f in files if f == "index.json"), None)
        meta_file = next((f for f in files if f == "index_meta.json"), None)
        if meta_file:
            with open(os.path.join(path, meta_file), "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.faiss_type = meta.get("faiss_type", self.faiss_type)
                self.faiss_nlist = meta.get("faiss_nlist", self.faiss_nlist)
                self.faiss_nprobe = meta.get("faiss_nprobe", self.faiss_nprobe)
                self.faiss_hnsw_m = meta.get("faiss_hnsw_m", self.faiss_hnsw_m)
                self.faiss_ef_search = meta.get("faiss_ef_search", self.faiss_ef_search)
        if index_file:
            self.index = faiss.read_index(os.path.join(path, index_file))
            self._apply_search_params()
        if json_file:
            with open(os.path.join(path, json_file), "r", encoding="utf-8") as f:
                docs_data = json.load(f)
                self.documents = [Document.from_dict(d) for d in docs_data]
                self.id_to_doc = {doc.id: doc for doc in self.documents}
