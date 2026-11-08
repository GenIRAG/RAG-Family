import inspect
from typing import Any, Dict, List, Optional, Type

from RoleRAG.errors import error_context
from RoleRAG.roles.reasoner import ReasonerRoleAdapter
from RoleRAG.roles.registry import register_role
from RoleRAG.roles.role_base import BaseRewriterContract
from RoleRAG.roles.validation import validate_model_capability


_REWRITE_METHODS: Dict[str, Type] = {}


def register_rewrite_method(name: str):
    def decorator(cls: Type) -> Type:
        key = str(name).lower()
        if key in _REWRITE_METHODS:
            raise KeyError(
                f"Duplicate rewrite method registration: '{key}' "
                f"(already registered as {_REWRITE_METHODS[key].__name__})"
            )
        _REWRITE_METHODS[key] = cls
        return cls

    return decorator


def registered_rewrite_methods() -> List[str]:
    return sorted(_REWRITE_METHODS)


@register_rewrite_method("hyde")
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


@register_role("rewriter")
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
    ):
        validate_model_capability(model, "rewriter", "generate", "generate")
        self.method = str(method).lower()
        self.num_hypotheses = num_hypotheses
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.task = task
        self.language = language
        self.stop = stop

        impl_cls = _REWRITE_METHODS.get(self.method)
        if impl_cls is None:
            raise ValueError(
                f"Unknown rewriter method: {method}. "
                f"Available methods: {registered_rewrite_methods()}"
            )

        impl_params = inspect.signature(impl_cls.__init__).parameters
        base_kwargs = {
            "task": task,
            "language": language,
            "num_hypotheses": num_hypotheses,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop,
        }
        accepted_kwargs = {key: value for key, value in base_kwargs.items() if key in impl_params}

        self._model = impl_cls(
            reasoner=ReasonerRoleAdapter(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            **accepted_kwargs,
        )

    def rewrite(self, query: str, **kwargs) -> str:
        with error_context(f"role=rewriter, method={self.method}, operation=rewrite"):
            result = self._model.rewrite(query, **kwargs)
            if not isinstance(result, str):
                raise TypeError(f"rewrite() must return text; got {type(result).__name__}")
            return result
