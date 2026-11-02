import logging
import re
from typing import Any

from RoleRAG.config.types import RetrieverResult
from RoleRAG.rolerag.pipelines.prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class NaiveGenerationStrategy:
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @staticmethod
    def format_retrieval_result(retrieval_result) -> str:
        if retrieval_result is None:
            return ""
        if hasattr(retrieval_result, "documents"):
            docs = retrieval_result.documents
        elif isinstance(retrieval_result, list):
            docs = retrieval_result
        else:
            docs = [retrieval_result]

        formatted = []
        for idx, doc in enumerate(docs):
            if hasattr(doc, "content"):
                content = doc.content
            elif isinstance(doc, dict):
                content = doc.get("content", doc.get("contents", str(doc)))
            else:
                content = str(doc)
            formatted.append(f"Doc {idx + 1}: {content}")

        return "\n\n".join(formatted)

    @staticmethod
    def get_first_document_content(retrieval_result) -> str:
        if retrieval_result is None or not hasattr(retrieval_result, "documents"):
            return ""
        if not retrieval_result.documents:
            return ""
        doc = retrieval_result.documents[0]
        if hasattr(doc, "content"):
            return doc.content
        if isinstance(doc, dict):
            return doc.get("content", doc.get("contents", str(doc)))
        return str(doc)

    def generate(
        self,
        question: str,
        retrieval_result: RetrieverResult = None,
        reasoner: Any = None,
    ) -> str:
        if reasoner is None:
            raise ValueError("reasoner is required")

        max_tokens = getattr(reasoner, 'max_tokens', 256)
        temperature = getattr(reasoner, 'temperature', 0.0)
        prompt_name = getattr(reasoner, "prompt_name", "naive_rag")
        prompt_template_obj = PromptManager.get_template(prompt_name)

        reference = self.format_retrieval_result(retrieval_result)
        prompt_variables = {
            "question": question,
            "reference": reference,
        }

        try:
            system_prompt = prompt_template_obj.system_prompt.format(**prompt_variables)
            user_prompt = prompt_template_obj.user_prompt.format(**prompt_variables)
        except Exception as exc:
            raise ValueError(f"Failed to format prompt template '{prompt_name}': {exc}") from exc

        if system_prompt:
            prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            prompt = user_prompt

        answer = reasoner.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        answer_prefix = getattr(reasoner, "answer_prefix", None)
        if answer_prefix:
            return _extract_after_prefix(answer, answer_prefix)
        return answer


def format_retrieval_result(retrieval_result) -> str:
    return NaiveGenerationStrategy.format_retrieval_result(retrieval_result)


def generate_answer(
    question: str,
    retrieval_result: RetrieverResult = None,
    reasoner: Any = None,
) -> str:
    return NaiveGenerationStrategy().generate(
        question=question,
        retrieval_result=retrieval_result,
        reasoner=reasoner,
    )


def _extract_after_prefix(text: str, prefix: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    pattern = rf"(?:^|\n)\s*{re.escape(prefix)}\s*(.+)$"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return text
