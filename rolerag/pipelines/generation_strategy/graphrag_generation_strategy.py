from typing import Any

from RoleRAG.config.types import RetrieverResult
from RoleRAG.rolerag.pipelines.prompts.prompt_manager import PromptManager
from RoleRAG.rolerag.pipelines.generation_strategy.naive_generation_strategy import NaiveGenerationStrategy, _extract_after_prefix


class _BaseGraphRAGGenerationStrategy:

    context_variable = ""

    def __init__(self, config: dict | None = None):
        graphrag_cfg = ((config or {}).get("inference", {}) or {}).get("graphrag", {}) or {}
        self.response_type = graphrag_cfg.get("response_type", "Multiple Paragraphs")

    def generate(
        self,
        question: str,
        retrieval_result: RetrieverResult = None,
        reasoner: Any = None,
    ) -> str:
        if reasoner is None:
            raise ValueError("reasoner is required")

        prompt_name = getattr(reasoner, "prompt_name", None)
        if not prompt_name:
            raise ValueError("reasoner.prompt_name is required for GraphRAG generation")

        context = NaiveGenerationStrategy.get_first_document_content(retrieval_result)
        prompt_template_obj = PromptManager.get_template(prompt_name)
        prompt_variables = {
            "question": question,
            "response_type": self.response_type,
            self.context_variable: context,
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
            max_tokens=getattr(reasoner, "max_tokens", 256),
            temperature=getattr(reasoner, "temperature", 0.0),
        )

        answer_prefix = getattr(reasoner, "answer_prefix", None)
        if answer_prefix:
            return _extract_after_prefix(answer, answer_prefix)
        return answer


class GraphRAGLocalGenerationStrategy(_BaseGraphRAGGenerationStrategy):

    context_variable = "context_data"


class GraphRAGGlobalGenerationStrategy(_BaseGraphRAGGenerationStrategy):

    context_variable = "report_data"
