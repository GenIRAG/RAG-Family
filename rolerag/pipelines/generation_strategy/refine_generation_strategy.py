import re
from typing import Any

from RoleRAG.config.types import RetrieverResult
from RoleRAG.rolerag.pipelines.prompts.prompt_manager import PromptManager


class RefineGenerationStrategy:

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @staticmethod
    def format_retrieval_contexts(retrieval_result) -> list[str]:
        if retrieval_result is None:
            return []
        if hasattr(retrieval_result, "documents"):
            documents = retrieval_result.documents
        elif isinstance(retrieval_result, list):
            documents = retrieval_result
        else:
            documents = [retrieval_result]

        formatted = []
        for index, document in enumerate(documents):
            if hasattr(document, "content"):
                content = document.content
            elif isinstance(document, dict):
                content = document.get("content", document.get("contents", str(document)))
            else:
                content = str(document)
            formatted.append(f"Doc {index + 1}: {content}")
        return formatted

    @classmethod
    def format_retrieval_result(cls, retrieval_result) -> str:
        return "\n\n".join(cls.format_retrieval_contexts(retrieval_result))

    def generate(
        self,
        question: str,
        retrieval_result: RetrieverResult = None,
        reasoner: Any = None,
        refiner: Any = None,
    ) -> str:
        if reasoner is None:
            raise ValueError("reasoner is required")
        if refiner is None:
            raise ValueError("refiner is required")

        max_tokens = getattr(reasoner, "max_tokens", 256)
        temperature = getattr(reasoner, "temperature", 0.0)
        prompt_name = getattr(reasoner, "prompt_name", "naive_rag")
        contexts = self.format_retrieval_contexts(retrieval_result)
        refined_reference = refiner.refine(contexts, question=question)
        prompt_template = PromptManager.get_template(prompt_name)
        prompt_variables = {
            "question": question,
            "reference": refined_reference,
        }

        try:
            system_prompt = prompt_template.system_prompt.format(**prompt_variables)
            user_prompt = prompt_template.user_prompt.format(**prompt_variables)
        except Exception as exc:
            raise ValueError(f"Failed to format prompt template '{prompt_name}': {exc}") from exc

        prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        answer = reasoner.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        answer_prefix = getattr(reasoner, "answer_prefix", None)
        if answer_prefix:
            return _extract_after_prefix(answer, answer_prefix)
        return answer


def _extract_after_prefix(text: str, prefix: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    pattern = rf"(?:^|\n)\s*{re.escape(prefix)}\s*(.+)$"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text
