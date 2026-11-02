#!/usr/bin/env python

import argparse
import json
from pathlib import Path


def _format_context(context) -> str:
    if isinstance(context, list):
        return "\n".join(str(item).strip() for item in context if str(item).strip())
    if context is None:
        return ""
    return str(context).strip()


def convert_topiocqa_eval(input_path: Path, output_path: Path) -> int:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for _, turns in data.items():
            for _, turn in turns.items():
                question = turn.get("Question", "")
                context = turn.get("Context", [])
                formatted_context = _format_context(context)
                target_psg = turn.get("Target_psg") or {}
                target_content = target_psg.get("content", "")
                if not question or not target_content:
                    continue

                item = {
                    "id": f"test_{count}",
                    "question": question,
                    "golden_answers": [target_content],
                    "metadata": {
                        "context": formatted_context,
                    },
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Convert TopiOCQA nested JSON to RoleRAG JSONL eval format.")
    parser.add_argument(
        "--input",
        default="data/evalute/topiocqa_grpo_test.json",
        help="Input TopiOCQA nested JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/evalute/topiocqa_grpo_test_converted.jsonl",
        help="Output JSONL file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    count = convert_topiocqa_eval(input_path, output_path)
    print(f"Wrote {count} samples to {output_path}")


if __name__ == "__main__":
    main()
