from typing import Any, List, Optional

from RoleRAG.roles.reasoner import ReasonerRoleAdapter
from RoleRAG.roles.role_base import BaseRewriterContract


class HyDERewriter:

    def __init__(
        self,
        reasoner: Any,
        task: str = "web_search",
        language: str = "en",
        num_hypotheses: int = 8,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
    ):
        self.reasoner = reasoner
        self.task = task
        self.language = language
        self.num_hypotheses = num_hypotheses
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop = stop if stop is not None else ["\n\n\n"]

    def build_prompt(self, query: str) -> str:
        from RoleRAG.rolerag.pipelines.prompts import PromptManager

        template_name = f"hyde_{self.task}"
        template = PromptManager.get_template(template_name)
        if self.task == "mr_tydi":
            return template.user_prompt.format(language=self.language, query=query)
        return template.user_prompt.format(query=query)

    def generate(self, query: str, **kwargs) -> List[str]:
        from RoleRAG.rolerag.pipelines.prompts import PromptManager

        template_name = f"hyde_{self.task}"
        template = PromptManager.get_template(template_name)
        prompt = self.build_prompt(query)
        hypothesis_documents = []
        for _ in range(kwargs.pop("num_hypotheses", self.num_hypotheses)):
            text = self.reasoner.generate(
                prompt,
                system_prompt=template.system_prompt or None,
                max_tokens=kwargs.pop("max_tokens", self.max_tokens),
                temperature=kwargs.pop("temperature", self.temperature),
                stop=kwargs.pop("stop", self.stop),
                **kwargs,
            )
            if text:
                hypothesis_documents.append(str(text).strip())
        return hypothesis_documents

    def rewrite(self, query: str, **kwargs) -> str:
        hypothesis_documents = self.generate(query, **kwargs)
        if not hypothesis_documents:
            return query
        expanded_context = " ".join(doc for doc in hypothesis_documents if doc)
        if not expanded_context:
            return query
        return f"{query} {expanded_context}".strip()


class RewriterRoleAdapter(BaseRewriterContract):

    def __init__(
        self,
        model,
        method: str = "hyde",
        num_hypotheses: int = 4,
        max_tokens: int = 512,
        temperature: float = 0.7,
        task: str = "web_search",
        language: str = "en",
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        self.method = method
        self.num_hypotheses = num_hypotheses
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.task = task
        self.language = language
        self.stop = stop

        if self.method == "hyde":
            self._model = HyDERewriter(
                reasoner=ReasonerRoleAdapter(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                task=task,
                language=language,
                num_hypotheses=num_hypotheses,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
        else:
            raise ValueError(f"Unknown rewriter method: {self.method}")

    def rewrite(self, query: str, **kwargs) -> str:
        if hasattr(self._model, "rewrite"):
            return self._model.rewrite(query, **kwargs)
        raise TypeError(f"Model {type(self._model).__name__} does not support rewrite")
