import asyncio
import html
import inspect
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from RoleRAG.config.types import Document
from RoleRAG.indexes.indexing.graph_indexing.extraction_schema import (
    ChunkExtraction,
    Entity,
    ExtractionResult,
    Relationship,
)
from RoleRAG.rolerag.pipelines.prompts import PromptManager

logger = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    pass


@dataclass
class ExtractionProgress:
    ner: Any
    openie: Any


def clean_entity_name(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    text = html.unescape(text.strip())
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return re.sub(r"[^A-Za-z0-9 ]", " ", text.lower()).strip()


def parse_json_object(text: Any) -> dict:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str) or not text.strip():
        return {}

    cleaned = _strip_json_fence(text.strip())
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


def normalize_entities(data: dict) -> List[str]:
    entities = data.get("named_entities", [])
    if isinstance(entities, str):
        return [entities] if entities.strip() else []
    if isinstance(entities, list):
        return [item for item in entities if isinstance(item, str) and item.strip()]
    return []


def normalize_triples(data: dict) -> List[List[str]]:
    triples = []
    for triple in data.get("triples", []):
        normalized = _normalize_triple_shape(triple)
        if normalized is not None:
            triples.append(normalized)
    return triples


def ner_extract(llm_call: Callable[[str], str], passage: str, prompt_template: Optional[str] = None) -> List[str]:
    template = prompt_template or PromptManager.get_template("hipporag_ner").user_prompt
    response = llm_call(template.replace("__USER_INPUT__", passage))
    entities = normalize_entities(parse_json_object(response))
    if not entities:
        logger.warning("NER returned no entities. raw_response[:200]=%s", str(response)[:200])
    return entities


class HippoRAGExtraction:
    def __init__(
        self,
        ask_fn: Callable[[str], str],
        prompt_config: Optional[Dict[str, dict]] = None,
        max_concurrency: int = 1,
        max_retries: int = 2,
        fail_on_empty: bool = True,
    ):
        self.ask = ask_fn
        self.prompts = prompt_config or {}
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_retries = max(0, int(max_retries))
        self.fail_on_empty = fail_on_empty

    async def extract(self, chunks: List[Document]) -> ExtractionResult:
        chunk_results = await self._extract_chunks(chunks)
        nodes, edges = self._build_graph_items(chunk_results)

        if self.fail_on_empty:
            self._validate_extraction(chunk_results, nodes, edges)

        return ExtractionResult(
            nodes=nodes,
            edges=edges,
            chunks=chunks,
            chunk_results=chunk_results,
            method="hipporag",
        )

    async def _extract_chunks(self, chunks: List[Document]) -> List[ChunkExtraction]:
        from tqdm import tqdm

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_one(chunk: Document, progress: ExtractionProgress) -> ChunkExtraction:
            async with semaphore:
                return await self._extract_chunk(chunk, progress)

        progress = ExtractionProgress(
            ner=tqdm(total=len(chunks), desc="NER", unit="chunk", position=0),
            openie=tqdm(total=len(chunks), desc="OpenIE", unit="chunk", position=1),
        )
        try:
            tasks = [asyncio.create_task(run_one(chunk, progress)) for chunk in chunks]
            results = []
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting", unit="chunk", position=2):
                results.append(await task)
        finally:
            progress.ner.close()
            progress.openie.close()
        return results

    async def _extract_chunk(self, chunk: Document, progress: ExtractionProgress) -> ChunkExtraction:
        ner_response = ""
        openie_response = ""

        try:
            ner_response, ner_data = await self._call_json_with_retry(
                self._ner_prompt(chunk.content),
                stage="NER",
                chunk_id=chunk.id,
            )
            entities = normalize_entities(ner_data)
        except Exception as exc:
            progress.ner.update(1)
            progress.openie.update(1)
            return ChunkExtraction(
                chunk_id=chunk.id,
                chunk_content=chunk.content,
                ner_raw_response=ner_response,
                ner_error=str(exc),
            )
        progress.ner.update(1)

        try:
            openie_response, openie_data = await self._call_json_with_retry(
                self._openie_prompt(chunk.content, entities),
                stage="OpenIE",
                chunk_id=chunk.id,
            )
            triples = normalize_triples(openie_data)
        except Exception as exc:
            progress.openie.update(1)
            return ChunkExtraction(
                chunk_id=chunk.id,
                chunk_content=chunk.content,
                named_entities=entities,
                ner_raw_response=ner_response,
                openie_raw_response=openie_response,
                openie_error=str(exc),
            )
        progress.openie.update(1)

        return ChunkExtraction(
            chunk_id=chunk.id,
            chunk_content=chunk.content,
            named_entities=entities,
            triples=triples,
            ner_raw_response=ner_response,
            openie_raw_response=openie_response,
        )

    def _build_graph_items(
        self,
        chunk_results: List[ChunkExtraction],
    ) -> Tuple[Dict[str, List[Entity]], Dict[tuple, List[Relationship]]]:
        nodes: Dict[str, List[Entity]] = {}
        edges: Dict[tuple, List[Relationship]] = {}

        for result in chunk_results:
            self._add_named_entities(nodes, result.named_entities, result.chunk_id)
            self._add_triples(nodes, edges, result.triples, result.chunk_id)

        return nodes, edges

    def _add_named_entities(self, nodes: Dict[str, List[Entity]], raw_entities: List[str], chunk_id: str) -> None:
        for raw_entity in raw_entities:
            entity_name = clean_entity_name(raw_entity)
            if entity_name:
                self._append_entity(nodes, entity_name, chunk_id)

    def _add_triples(
        self,
        nodes: Dict[str, List[Entity]],
        edges: Dict[tuple, List[Relationship]],
        raw_triples: List[List[str]],
        chunk_id: str,
    ) -> None:
        for raw_src, raw_rel, raw_tgt in raw_triples:
            triple = self._clean_triple(raw_src, raw_rel, raw_tgt)
            if triple is None:
                continue

            src, relation, tgt = triple
            edges.setdefault((src, tgt), []).append(
                Relationship(
                    src_id=src,
                    tgt_id=tgt,
                    source_id=chunk_id,
                    relation_name=relation,
                    weight=1.0,
                )
            )

            if src not in nodes:
                self._append_entity(nodes, src, chunk_id)
            if tgt not in nodes:
                self._append_entity(nodes, tgt, chunk_id)

    @staticmethod
    def _append_entity(nodes: Dict[str, List[Entity]], entity_name: str, chunk_id: str) -> None:
        nodes.setdefault(entity_name, []).append(Entity(entity_name=entity_name, source_id=chunk_id))

    @staticmethod
    def _clean_triple(raw_src: str, raw_rel: str, raw_tgt: str) -> Optional[Tuple[str, str, str]]:
        src = clean_entity_name(raw_src)
        relation = clean_entity_name(raw_rel)
        tgt = clean_entity_name(raw_tgt)
        if not src or not relation or not tgt:
            return None
        return src, relation, tgt

    async def _call_json_with_retry(self, prompt: str, stage: str, chunk_id: str) -> Tuple[str, dict]:
        last_response = ""
        last_error = ""

        for attempt in range(self.max_retries + 1):
            response = await asyncio.to_thread(self.ask, prompt)
            if inspect.isawaitable(response):
                response = await response

            last_response = "" if response is None else str(response)
            data = parse_json_object(response)
            if data:
                return last_response, data

            last_error = f"{stage} returned unparsable JSON for chunk={chunk_id}; raw[:200]={last_response[:200]!r}"
            logger.warning("%s (attempt %s/%s)", last_error, attempt + 1, self.max_retries + 1)

        raise ExtractionError(last_error)

    def _ner_prompt(self, passage: str) -> str:
        return self._prompt("hipporag_ner").replace("__USER_INPUT__", passage)

    def _openie_prompt(self, passage: str, entities: List[str]) -> str:
        entity_json = json.dumps({"named_entities": entities}, ensure_ascii=False)
        return (
            self._prompt("hipporag_openie")
            .replace("__PASSAGE__", passage)
            .replace("__NAMED_ENTITY_JSON__", entity_json)
        )

    def _prompt(self, name: str) -> str:
        return PromptManager.get_template(name).user_prompt

    def _validate_extraction(
        self,
        chunk_results: List[ChunkExtraction],
        nodes: Dict[str, List[Entity]],
        edges: Dict[tuple, List[Relationship]],
    ) -> None:
        if not nodes:
            previews = [result.ner_raw_response for result in chunk_results[:3]]
            raise ExtractionError(f"HippoRAG extraction produced no entities. Sample NER responses: {previews}")
        if not edges:
            previews = [result.openie_raw_response for result in chunk_results[:3]]
            raise ExtractionError(f"HippoRAG extraction produced no relations. Sample OpenIE responses: {previews}")


def _strip_json_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", text)


def _normalize_triple_shape(triple: Any) -> Optional[List[str]]:
    if isinstance(triple, list) and len(triple) == 1 and isinstance(triple[0], list):
        triple = triple[0]
    if not isinstance(triple, (list, tuple)) or len(triple) != 3:
        return None

    src, rel, tgt = triple
    if all(isinstance(item, str) and item.strip() for item in (src, rel, tgt)):
        return [src, rel, tgt]
    return None
