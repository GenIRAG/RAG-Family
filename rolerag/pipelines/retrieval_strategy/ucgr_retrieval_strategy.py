import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from RoleRAG.config.types import Document, RetrieverResult
from RoleRAG.rolerag.pipelines.prompts import PromptManager


class UCGRRetrievalStrategy:

    _fm_index_cache: Dict[str, Any] = {}

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        ucgr_cfg = (self.config.get("inference", {}) or {}).get("ucgr", {}) or {}
        self.second_stage_num_beams = int(ucgr_cfg.get("ucgr_second_stage_num_beams", 5))
        self.second_stage_top_k = int(ucgr_cfg.get("ucgr_second_stage_top_k", 5))
        self.second_stage_max_new_tokens = int(ucgr_cfg.get("ucgr_second_stage_max_new_tokens", 32))
        self.forced_bos_token_id = int(ucgr_cfg.get("fm_forced_bos_token_id", 128000))

    def retrieve(
        self,
        query: str,
        reasoner: Any,
    ) -> RetrieverResult:
        prompt_name = getattr(reasoner, "prompt_name", None)
        if not prompt_name:
            raise ValueError("reasoner.prompt_name is required for UCGR retrieval")
        prompt_template = PromptManager.get_template(prompt_name)
        prompt_messages = [
            {"role": "system", "content": prompt_template.system_prompt.format(question=query)},
            {"role": "user", "content": prompt_template.user_prompt.format(question=query)},
        ]

        raw_model = self._unwrap_reasoner_model(reasoner)
        fm_index = self._resolve_fm_index(reasoner)
        model = raw_model.model
        tokenizer = raw_model.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        constrained_inputs = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            add_special_tokens=True,
        )
        constrained_inputs = self._move_inputs_to_model_device(constrained_inputs, model)

        retrieval_texts: Dict[str, float] = {}
        retrievals: Dict[str, float] = {}
        raw_retrieval_texts: Dict[str, float] = {}
        documents: List[Document] = []
        scores: List[float] = []

        from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor

        trie_next_step = _UCGRNextStep(
            tokenizer=tokenizer,
            fm_index=fm_index,
            forced_bos_token_id=self.forced_bos_token_id,
        )
        trie_next_step.L_input = constrained_inputs["input_ids"].shape[1]
        logits_processor = [
            PrefixConstrainedLogitsProcessor(
                trie_next_step.allowed_tokens_fn,
                self.second_stage_num_beams,
            )
        ]

        constrained_outputs = model.generate(
            **constrained_inputs,
            logits_processor=logits_processor,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=self.second_stage_num_beams,
            num_return_sequences=self.second_stage_top_k,
            max_new_tokens=self.second_stage_max_new_tokens,
            early_stopping=False,
            renormalize_logits=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        input_len = constrained_inputs["input_ids"].shape[1]
        for sequence_ids, sequence_score in zip(
            constrained_outputs["sequences"].tolist(),
            constrained_outputs["sequences_scores"].tolist(),
        ):
            raw_retrieval_text = tokenizer.decode(
                sequence_ids[input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            retrieval_text = self._clean_retrieval_text(raw_retrieval_text)
            score = float(sequence_score)
            raw_retrieval_texts[raw_retrieval_text] = score
            retrieval_texts[retrieval_text] = score
            pid = self._map_retrieval_to_pid(
                retrieval_text=raw_retrieval_text,
                tokenizer=tokenizer,
                fm_index=fm_index,
                forced_bos_token_id=self.forced_bos_token_id,
            )
            if pid is None:
                continue

            retrievals[pid] = score
            doc = Document(
                id=pid,
                content=retrieval_text,
                metadata={
                    "retrieval_text": retrieval_text,
                    "raw_retrieval_text": raw_retrieval_text,
                    "retrieval_pid": pid,
                    "strategy": "ucgr",
                    "mode": "direct_constrained",
                },
                score=score,
            )
            documents.append(doc)
            scores.append(score)

        return RetrieverResult(
            documents=documents,
            scores=scores,
            metadata={
                "strategy": "ucgr",
                "mode": "direct_constrained",
                "query": query,
                "prompt_messages": prompt_messages,
                "retrieval_texts": retrieval_texts,
                "raw_retrieval_texts": raw_retrieval_texts,
                "retrieval_results": retrievals,
                "second_stage_num_beams": self.second_stage_num_beams,
                "second_stage_top_k": self.second_stage_top_k,
                "forced_bos_token_id": self.forced_bos_token_id,
            },
        )

    @staticmethod
    def load_fm_index(FM_index_path: str):
        from seal.index import FMIndex

        index_path = UCGRRetrievalStrategy._resolve_project_path(FM_index_path)
        return FMIndex.load(index_path)

    @classmethod
    def _resolve_fm_index(cls, reasoner: Any) -> Any:
        fm_index_path = getattr(reasoner, "fm_index_path", None)
        if not fm_index_path:
            raise ValueError("UCGR requires reasoner.fm_index_path")
        if fm_index_path not in cls._fm_index_cache:
            cls._fm_index_cache[fm_index_path] = cls.load_fm_index(fm_index_path)
        return cls._fm_index_cache[fm_index_path]

    @staticmethod
    def _resolve_project_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        project_root = Path(__file__).resolve().parents[3]
        return str(project_root / path)

    @staticmethod
    def _unwrap_reasoner_model(reasoner: Any) -> Any:
        raw_model = getattr(reasoner, "_model", reasoner)
        if not hasattr(raw_model, "model") or not hasattr(raw_model, "tokenizer"):
            raise TypeError("UCGR requires an HFDecoder reasoner with model/tokenizer access")
        return raw_model

    @staticmethod
    def _move_inputs_to_model_device(inputs: Any, model: Any) -> Any:
        model_device = getattr(model, "device", None)
        if model_device is not None and getattr(model_device, "type", None) != "meta":
            return inputs.to(model_device)
        return inputs

    @staticmethod
    def _clean_retrieval_text(text: str) -> str:
        text = str(text or "").strip()
        if text.startswith("<retrieve>"):
            text = text[len("<retrieve>"):]
        if "</retrieve>" in text:
            text = text.split("</retrieve>", 1)[0]
        return text.strip()

    @staticmethod
    def _map_retrieval_to_pid(
        retrieval_text: str,
        tokenizer: Any,
        fm_index: Any,
        forced_bos_token_id: int,
    ) -> Optional[str]:
        retrieval_ids = [forced_bos_token_id] + tokenizer(
            retrieval_text,
            add_special_tokens=False,
        )["input_ids"]
        for candidate_ids in (retrieval_ids, retrieval_ids[:16]):
            try:
                doc_indices = fm_index.get_doc_indices(candidate_ids)
                if doc_indices:
                    return str([fm_index.labels[dd] for dd in doc_indices][0])
            except Exception:
                continue
        return None


class _UCGRNextStep:

    def __init__(self, tokenizer, fm_index, forced_bos_token_id=None):
        self.tokenizer = tokenizer
        self.fm_index = fm_index
        self.L_input = None
        self.forced_bos_token_id = forced_bos_token_id

    def allowed_tokens_fn(self, batch_id: int, input_ids: List[int]):
        input_ids_list = input_ids.tolist()
        input_ids_list_ = input_ids_list[self.L_input:]

        if len(input_ids_list_) != 0 and input_ids_list_[-1] == self.tokenizer.eos_token_id:
            return [self.tokenizer.eos_token_id]

        if len(input_ids_list_) == 0:
            low, high = self.fm_index.get_range([self.forced_bos_token_id])
            fm_index_result = self.fm_index.get_distinct_count_multi([low], [high])
            allowed_tokens, _ = fm_index_result.pop()
        else:
            low, high = self.fm_index.get_range([self.forced_bos_token_id] + input_ids_list_)
            fm_index_result = self.fm_index.get_distinct_count_multi([low], [high])
            allowed_tokens, _ = fm_index_result.pop()

        if not allowed_tokens:
            allowed_tokens = [self.tokenizer.eos_token_id]

        return allowed_tokens
