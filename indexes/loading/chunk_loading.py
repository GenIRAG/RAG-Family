import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from RoleRAG.config.types import Document

__all__ = [
    "BaseLoading",
    "JsonLoading",
    "TextLoading",
    "load_corpus",
    "load_corpus_records",
]


def _stable_content_id(content: str) -> str:
    return hashlib.sha256(str(content).encode("utf-8")).hexdigest()


class BaseLoading:

    def load(self, path: str) -> List[Document]:
        raise NotImplementedError


class JsonLoading(BaseLoading):

    def load(self, path: str) -> List[Document]:
        documents = []
        extracted_docs = {}
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                for line in f:
                    data = json.loads(line)
                    docs = self._extract_dataset_documents(data)
                    if docs:
                        for doc in docs:
                            extracted_docs.setdefault(doc.id, doc)
                    else:
                        docs = self._parse_document(data)
                        documents.extend(docs)
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        docs = self._extract_dataset_documents(item)
                        if docs:
                            for doc in docs:
                                extracted_docs.setdefault(doc.id, doc)
                        else:
                            docs = self._parse_document(item)
                            documents.extend(docs)
                else:
                    docs = self._extract_dataset_documents(data)
                    if docs:
                        for doc in docs:
                            extracted_docs.setdefault(doc.id, doc)
                    else:
                        docs = self._parse_document(data)
                        documents.extend(docs)
        if extracted_docs:
            return list(extracted_docs.values())
        return documents

    def _extract_dataset_documents(self, data: Dict[str, Any]) -> List[Document]:
        if not isinstance(data, dict):
            return []

        graph_items = data.get("graph")
        if graph_items is None:
            graph_items = data.get("metadata", {}).get("graph")
        if not isinstance(graph_items, list):
            return []

        extracted_docs = {}
        for graph_item in graph_items:
            if not isinstance(graph_item, dict):
                continue
            for node in graph_item.values():
                if not isinstance(node, dict):
                    continue
                for support in node.get("supports", []):
                    doc = self._support_to_document(support)
                    if doc is not None:
                        extracted_docs.setdefault(doc.id, doc)

        return list(extracted_docs.values())

    def _support_to_document(self, support: Dict[str, Any]) -> Optional[Document]:
        if not isinstance(support, dict):
            return None

        title = support.get("title", "")
        contents = support.get("contents", "")

        if isinstance(contents, list):
            text = "\n".join(str(item) for item in contents if item)
        elif isinstance(contents, str):
            text = contents
        else:
            text = str(contents) if contents else ""

        if not title and not text:
            return None

        content = f"{title}\n{text}" if title else text
        doc_id = support.get("id") or _stable_content_id(content)
        metadata = {k: v for k, v in support.items() if k not in ["id", "title", "contents"]}
        if title:
            metadata["title"] = title

        return Document(id=doc_id, content=content, metadata=metadata)

    def extract_dataset_records(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [doc.to_dict() for doc in self._extract_dataset_documents(data)]

    def _parse_document(self, data: Dict[str, Any], parent_key: str = None, doc_id_prefix: str = "") -> List[Document]:
        if isinstance(data, dict):
            if "title" in data and ("text" in data or "context" in data):
                title = data["title"]
                body = data.get("text") or data.get("context", "")
                if isinstance(body, list):
                    body = "\n".join(str(item) for item in body if item)
                elif not isinstance(body, str):
                    body = str(body)
                content = f"{title}\n{body}"
                doc_id = data.get("id") or data.get("idx") or doc_id_prefix or _stable_content_id(content)
                if not isinstance(doc_id, str):
                    doc_id = str(doc_id)
                metadata = {k: v for k, v in data.items() if k not in ["title", "text", "context", "id", "idx"]}
                return [Document(id=doc_id, content=content, metadata=metadata)]

            if "content" in data or "contents" in data or "text" in data:
                content = data.get("content", data.get("contents", data.get("text", "")))
                if isinstance(content, list):
                    content = "\n".join(str(item) for item in content if item)
                elif not isinstance(content, str):
                    content = str(content)
                doc_id = data.get("id", doc_id_prefix or _stable_content_id(content))
                metadata = {k: v for k, v in data.items() if k not in ["content", "contents", "text", "id"]}
                if parent_key:
                    metadata["section"] = parent_key
                return [Document(id=doc_id, content=content, metadata=metadata)]
            else:
                docs = []
                for key, value in data.items():
                    current_prefix = f"{doc_id_prefix}_{key}" if doc_id_prefix else key
                    docs.extend(self._parse_document(value, parent_key=key, doc_id_prefix=current_prefix))
                return docs
        elif isinstance(data, list):
            docs = []
            for idx, item in enumerate(data):
                item_id_prefix = f"{doc_id_prefix}_{idx}" if doc_id_prefix else str(idx)
                docs.extend(self._parse_document(item, parent_key=parent_key, doc_id_prefix=item_id_prefix))
            return docs
        elif isinstance(data, str):
            return [Document(id=doc_id_prefix or _stable_content_id(data), content=data, metadata={"section": parent_key} if parent_key else {})]
        return []


class TextLoading(BaseLoading):

    def load(self, path: str) -> List[Document]:
        documents = []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        paragraphs = []
        current_para = []
        for line in lines:
            line = line.strip()
            if line:
                current_para.append(line)
            elif current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
        if current_para:
            paragraphs.append(" ".join(current_para))
        
        for idx, para in enumerate(paragraphs):
            if para:
                documents.append(Document(
                    id=f"{Path(path).stem}_{idx}",
                    content=para,
                    metadata={"source": str(path)}
                ))
        return documents


def load_corpus(corpus_path: str) -> List[Document]:
    if corpus_path is None:
        raise ValueError("corpus_path is required.")
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    if path.suffix == ".json" or path.suffix == ".jsonl":
        loader = JsonLoading()
    elif path.suffix == ".txt":
        loader = TextLoading()
    else:
        loader = JsonLoading()

    return loader.load(corpus_path)


def load_corpus_records(corpus_path: str) -> List[Dict[str, Any]]:
    if corpus_path is None:
        raise ValueError("corpus_path is required.")

    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    if path.suffix not in {".json", ".jsonl"}:
        return [doc.to_dict() for doc in load_corpus(corpus_path)]

    loader = JsonLoading()
    records = []
    record_map = {}

    with open(corpus_path, "r", encoding="utf-8") as f:
        raw = f.read()
        if path.suffix == ".jsonl":
            items = (json.loads(line) for line in raw.splitlines() if line.strip())
        else:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                items = (json.loads(line) for line in raw.splitlines() if line.strip())
            else:
                if isinstance(data, list):
                    items = iter(data)
                else:
                    items = iter([data])
        for item in items:
            dataset_records = loader.extract_dataset_records(item)
            if dataset_records:
                for record in dataset_records:
                    record_map.setdefault(record["id"], record)
            else:
                docs = loader._parse_document(item)
                for doc in docs:
                    record_map.setdefault(doc.id, doc.to_dict())

    records.extend(record_map.values())
    return records
