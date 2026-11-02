import json
import multiprocessing
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

from RoleRAG.indexes.indexing.chunk_indexing.chunk_indexing_base import BaseIndexing

try:
    from seal.index import FMIndex
except ImportError as exc:
    FMIndex = None
    SEAL_IMPORT_ERROR = exc
else:
    SEAL_IMPORT_ERROR = None


_TOKENIZER = None
_FORCED_BOS_TOKEN_ID = 128000


def _init_worker(model_name_or_path: str, forced_bos_token_id: int):
    global _TOKENIZER, _FORCED_BOS_TOKEN_ID
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    _FORCED_BOS_TOKEN_ID = forced_bos_token_id


def _tokenize_title(title: str) -> List[int]:
    text = " " + title.strip() + " </retrieve>"
    with _TOKENIZER.as_target_tokenizer():
        return (
            [_FORCED_BOS_TOKEN_ID]
            + _TOKENIZER(text, add_special_tokens=False)["input_ids"]
            + [_TOKENIZER.eos_token_id]
        )


class FMIndexing(BaseIndexing):

    def __init__(
        self,
        hf_model: str,
        forced_bos_token_id: int = 128000,
        jobs: int = 1,
        lowercase: bool = False,
        keep_duplicates: bool = False,
    ):
        if FMIndex is None:
            raise ImportError(
                "seal.index.FMIndex is unavailable. Install SEAL and set PYTHONPATH to include the SEAL repo."
            ) from SEAL_IMPORT_ERROR

        self.hf_model = hf_model
        self.forced_bos_token_id = forced_bos_token_id
        self.jobs = max(1, int(jobs))
        self.lowercase = lowercase
        self.keep_duplicates = keep_duplicates
        self.index = None
        self.labels: List[str] = []

    def build_index(self, documents=None, **kwargs):
        corpus_path = kwargs.get("corpus_path")
        if not corpus_path:
            raise ValueError("FMIndexing.build_index requires corpus_path")

        entries = list(
            iter_index_jsonl_entries(
                corpus_path,
                lowercase=self.lowercase,
                keep_duplicates=self.keep_duplicates,
            )
        )
        if not entries:
            raise ValueError(f"No id/title entries found in {corpus_path}")

        self.labels = [pid for pid, _ in entries]
        titles = [title for _, title in entries]

        self.index = FMIndex()
        if self.jobs == 1:
            _init_worker(self.hf_model, self.forced_bos_token_id)
            sequences = (_tokenize_title(title) for title in tqdm(titles, desc="Tokenizing FMIndex titles"))
            self.index.initialize(sequences)
        else:
            with multiprocessing.Pool(
                self.jobs,
                initializer=_init_worker,
                initargs=(self.hf_model, self.forced_bos_token_id),
            ) as pool:
                sequences = pool.imap(_tokenize_title, titles)
                self.index.initialize(tqdm(sequences, total=len(titles), desc="Tokenizing FMIndex titles"))

        self.index.labels = self.labels
        return self

    def save(self, path: str):
        if self.index is None:
            raise ValueError("FMIndex has not been built")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.index.save(path)

    def load(self, path: str):
        self.index = FMIndex.load(path)
        self.labels = list(getattr(self.index, "labels", []))
        return self

    def search(self, query_embedding, top_k: int = 5) -> tuple:
        raise NotImplementedError("FMIndexing is for constrained generation, not vector search.")


def iter_index_jsonl_entries(
    corpus_path: str,
    lowercase: bool = False,
    keep_duplicates: bool = False,
) -> Iterable[Tuple[str, str]]:
    seen_ids = set()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            psg_id = entry.get("id")
            title = entry.get("title")
            if psg_id is None or not title:
                continue

            psg_id = str(psg_id)
            if not keep_duplicates and psg_id in seen_ids:
                continue

            seen_ids.add(psg_id)
            title = str(title).replace(" [SEP] ", ": ").strip()
            if lowercase:
                title = title.lower()
            if title:
                yield psg_id, title


def make_fm_index_path(storage_dir: str, corpus_path: str, index_name: Optional[str] = None) -> str:
    corpus_name = Path(corpus_path).stem
    index_dir = os.path.join(storage_dir, f"FMindex_{corpus_name}")
    filename = index_name or "index.fm_index"
    return os.path.join(index_dir, filename)
