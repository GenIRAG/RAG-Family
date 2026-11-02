from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from RoleRAG.config.types import Document


@dataclass
class Entity:
    entity_name: str
    source_id: str
    entity_type: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Relationship:
    src_id: str
    tgt_id: str
    source_id: str
    relation_name: Optional[str] = None
    weight: float = 1.0
    description: Optional[str] = None
    keywords: Optional[str] = None


@dataclass
class ChunkExtraction:
    chunk_id: str
    chunk_content: str
    named_entities: List[str] = field(default_factory=list)
    triples: List[List[str]] = field(default_factory=list)
    ner_raw_response: Optional[str] = None
    openie_raw_response: Optional[str] = None
    ner_error: Optional[str] = None
    openie_error: Optional[str] = None


@dataclass
class ExtractionResult:
    nodes: Dict[str, List[Entity]] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], List[Relationship]] = field(default_factory=dict)
    chunks: List[Document] = field(default_factory=list)
    chunk_results: List[ChunkExtraction] = field(default_factory=list)
    method: str = ""
