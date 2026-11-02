from typing import List, Optional
import json
import os
import random

from RoleRAG.config.types import EvalItem


def _apply_sample_options(
    samples: List[EvalItem],
    sample_num: Optional[int] = None,
    random_sample: bool = False,
) -> List[EvalItem]:
    if sample_num is None or sample_num >= len(samples):
        return list(samples)
    if random_sample:
        return random.sample(samples, sample_num)
    return samples[:sample_num]


def load_eval_samples(
    data_file: str,
    sample_num: Optional[int] = None,
    random_sample: bool = False,
) -> List[EvalItem]:
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())

            question = item.get("question", "")
            golden_answers = item.get("golden_answers", [])

            if isinstance(golden_answers, str):
                golden_answers = [golden_answers]
            elif not isinstance(golden_answers, list):
                golden_answers = [str(golden_answers)]

            sample = EvalItem(
                query=question,
                golden_answers=golden_answers,
                metadata=item.get("metadata", {}),
                id=item.get("id"),
            )

            samples.append(sample)

    return _apply_sample_options(samples, sample_num=sample_num, random_sample=random_sample)
