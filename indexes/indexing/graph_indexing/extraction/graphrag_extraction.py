import asyncio
import inspect
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from RoleRAG.config.types import Document
from RoleRAG.indexes.indexing.graph_indexing.community import clean_graph_str
from RoleRAG.indexes.indexing.graph_indexing.extraction_schema import (
    ChunkExtraction,
    Entity,
    ExtractionResult,
    Relationship,
)
from RoleRAG.rolerag.pipelines.prompts import PromptManager

DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"


class GraphRAGExtractionError(RuntimeError):
    pass


@dataclass
class GraphRAGExtractionProgress:
    chunks: Any


class GraphRAGExtraction:
    def __init__(
        self,
        ask_fn: Callable[[str], str],
        prompt_config: Optional[Dict[str, dict]] = None,
        max_concurrency: int = 1,
        max_retries: int = 2,
        fail_on_empty: bool = True,
        max_gleaning: int = 1,
    ):
        self.ask = ask_fn
        self.prompts = prompt_config or {}
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_retries = max(0, int(max_retries))
        self.fail_on_empty = fail_on_empty
        self.max_gleaning = max(0, int(max_gleaning))

    async def extract(self, chunks: List[Document]) -> ExtractionResult:
        chunk_results = await self._extract_chunks(chunks)
        nodes, edges = self._build_graph_items(chunk_results)
        if self.fail_on_empty and (not nodes or not edges):
            raise GraphRAGExtractionError(
                f"GraphRAG extraction produced nodes={len(nodes)}, edges={len(edges)}\n"
                f"{self._empty_extraction_debug(chunk_results)}"
            )
        return ExtractionResult(
            nodes=nodes,
            edges=edges,
            chunks=chunks,
            chunk_results=chunk_results,
            method="graphrag",
        )

    async def _extract_chunks(self, chunks: List[Document]) -> List[ChunkExtraction]:
        from tqdm import tqdm

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_one(chunk: Document, progress: GraphRAGExtractionProgress) -> ChunkExtraction:
            async with semaphore:
                result = await self._extract_chunk(chunk)
                progress.chunks.update(1)
                return result

        progress = GraphRAGExtractionProgress(
            chunks=tqdm(total=len(chunks), desc="GraphRAG extraction", unit="chunk")
        )
        try:
            tasks = [asyncio.create_task(run_one(chunk, progress)) for chunk in chunks]
            results = []
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Parsing graph records", unit="chunk"):
                results.append(await task)
        finally:
            progress.chunks.close()
        return results

    async def _extract_chunk(self, chunk: Document) -> ChunkExtraction:
        raw_response = ""
        try:
            raw_response = await self._call_with_retry(self._entity_extraction_prompt(chunk.content))
            raw_response = await self._glean(chunk.content, raw_response)
            entities, relationships = parse_graphrag_records(raw_response)
            return ChunkExtraction(
                chunk_id=chunk.id,
                chunk_content=chunk.content,
                named_entities=[entity.entity_name for entity in entities],
                triples=[[rel.src_id, rel.description or "", rel.tgt_id] for rel in relationships],
                openie_raw_response=raw_response,
            )
        except Exception as exc:
            return ChunkExtraction(
                chunk_id=chunk.id,
                chunk_content=chunk.content,
                openie_raw_response=raw_response,
                openie_error=str(exc),
            )

    async def _glean(self, passage: str, initial_result: str) -> str:
        final_result = initial_result or ""
        history = f"USER:\n{self._entity_extraction_prompt(passage)}\n\nASSISTANT:\n{final_result}"
        for idx in range(self.max_gleaning):
            continue_prompt = self._prompt("graphrag_continue_extraction")
            glean_prompt = f"{history}\n\nUSER:\n{continue_prompt}\n\nASSISTANT:\n"
            glean_result = await self._call_with_retry(glean_prompt)
            final_result += "\n" + (glean_result or "")
            history += f"\n\nUSER:\n{continue_prompt}\n\nASSISTANT:\n{glean_result}"
            if idx == self.max_gleaning - 1:
                break
            if_loop_prompt = self._prompt("graphrag_if_loop_extraction")
            if_loop_prompt_with_history = f"{history}\n\nUSER:\n{if_loop_prompt}\n\nASSISTANT:\n"
            if_loop_result = await self._call_with_retry(if_loop_prompt_with_history)
            if str(if_loop_result).strip().strip('"').strip("'").lower() != "yes":
                break
        return final_result

    def _build_graph_items(
        self,
        chunk_results: List[ChunkExtraction],
    ) -> Tuple[Dict[str, List[Entity]], Dict[tuple, List[Relationship]]]:
        nodes: Dict[str, List[Entity]] = defaultdict(list)
        edges: Dict[tuple, List[Relationship]] = defaultdict(list)

        for result in chunk_results:
            entities, relationships = parse_graphrag_records(result.openie_raw_response or "")
            for entity in entities:
                nodes[entity.entity_name].append(
                    Entity(
                        entity_name=entity.entity_name,
                        entity_type=entity.entity_type,
                        description=entity.description,
                        source_id=result.chunk_id,
                    )
                )
            for relationship in relationships:
                edges[(relationship.src_id, relationship.tgt_id)].append(
                    Relationship(
                        src_id=relationship.src_id,
                        tgt_id=relationship.tgt_id,
                        source_id=result.chunk_id,
                        weight=relationship.weight,
                        description=relationship.description,
                        relation_name="",
                        keywords="",
                    )
                )
        return dict(nodes), dict(edges)

    async def _call_with_retry(self, prompt: str) -> str:
        last_response = ""
        last_error = ""
        for _ in range(self.max_retries + 1):
            try:
                response = await asyncio.to_thread(self.ask, prompt)
                if inspect.isawaitable(response):
                    response = await response
                return "" if response is None else str(response)
            except Exception as exc:
                last_error = str(exc)
                last_response = ""
        raise GraphRAGExtractionError(last_error or f"empty response: {last_response!r}")

    def _entity_extraction_prompt(self, passage: str) -> str:
        template = self._prompt("graphrag_entity_extraction")
        return template.format(
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter=DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
            entity_types=",".join(DEFAULT_ENTITY_TYPES),
            input_text=passage,
        )

    def _prompt(self, name: str) -> str:
        return PromptManager.get_template(name).user_prompt

    def _empty_extraction_debug(self, chunk_results: List[ChunkExtraction]) -> str:
        samples = []
        for result in chunk_results[:3]:
            raw = result.openie_raw_response or ""
            entities, relationships = parse_graphrag_records(raw)
            snippet = raw[:800].replace("\n", "\\n")
            samples.append(
                "chunk_id={chunk_id}, parsed_entities={entities}, "
                "parsed_relationships={relationships}, error={error}, raw={raw!r}".format(
                    chunk_id=result.chunk_id,
                    entities=len(entities),
                    relationships=len(relationships),
                    error=result.openie_error,
                    raw=snippet,
                )
            )
        if not samples:
            return "No chunk extraction results were produced."
        return "Sample GraphRAG extraction diagnostics:\n" + "\n".join(samples)


@dataclass
class _ParsedEntity:
    entity_name: str
    entity_type: str = ""
    description: str = ""


@dataclass
class _ParsedRelationship:
    src_id: str
    tgt_id: str
    description: str = ""
    weight: float = 1.0


def parse_graphrag_records(text: str) -> Tuple[List[_ParsedEntity], List[_ParsedRelationship]]:
    entities: List[_ParsedEntity] = []
    relationships: List[_ParsedRelationship] = []
    for record in split_string_by_multi_markers(
        text or "",
        [DEFAULT_RECORD_DELIMITER, DEFAULT_COMPLETION_DELIMITER],
    ):
        match = re.search(r"\((.*)\)", record, flags=re.DOTALL)
        if not match:
            continue
        attrs = split_string_by_multi_markers(match.group(1), [DEFAULT_TUPLE_DELIMITER])
        if not attrs:
            continue
        kind = attrs[0].strip().strip('"').lower()
        if kind == "entity" and len(attrs) >= 4:
            entity_name = clean_graph_str(_strip_quotes(attrs[1]))
            if not entity_name:
                continue
            entities.append(
                _ParsedEntity(
                    entity_name=entity_name,
                    entity_type=clean_graph_str(_strip_quotes(attrs[2])) or "",
                    description=clean_description(_strip_quotes(attrs[3])),
                )
            )
        elif kind == "relationship" and len(attrs) >= 5:
            src = clean_graph_str(_strip_quotes(attrs[1]))
            tgt = clean_graph_str(_strip_quotes(attrs[2]))
            if not src or not tgt:
                continue
            relationships.append(
                _ParsedRelationship(
                    src_id=src,
                    tgt_id=tgt,
                    description=clean_description(_strip_quotes(attrs[3])),
                    weight=_parse_weight(attrs[-1]),
                )
            )
    return entities, relationships


def split_string_by_multi_markers(text: str, delimiters: List[str]) -> List[str]:
    if not delimiters:
        return [text]
    split_pattern = "|".join(re.escape(delimiter) for delimiter in delimiters)
    return [segment.strip() for segment in re.split(split_pattern, text) if segment.strip()]


def clean_description(text: Any) -> str:
    if not isinstance(text, str):
        return "" if text is None else str(text)
    text = text.strip()
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text


def _strip_quotes(text: Any) -> str:
    return str(text or "").strip().strip('"').strip("'")


def _parse_weight(value: Any) -> float:
    try:
        return float(_strip_quotes(value))
    except (TypeError, ValueError):
        return 1.0
