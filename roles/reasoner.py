from typing import Any, Dict, Optional

from RoleRAG.errors import error_context
from RoleRAG.roles.registry import register_role
from RoleRAG.roles.role_base import BaseGeneratorContract
from RoleRAG.roles.validation import validate_model_capability


@register_role("reasoner")
class ReasonerRoleAdapter(BaseGeneratorContract):

    def __init__(
        self,
        model,
        max_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        prompt_name: str = "naive_rag",
        prompt_names: Optional[Dict[str, str]] = None,
        answer_prefix: Optional[str] = None,
        graph_index_path: Optional[str] = None,
        fm_index_path: Optional[str] = None,
    ):
        validate_model_capability(model, "reasoner", "generate", "generate")
        self._model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.prompt_name = prompt_name
        self.prompt_names = prompt_names or {}
        self.prompt_templates = self._load_prompt_templates(self.prompt_names)
        self.answer_prefix = answer_prefix
        self.graph_index_path = graph_index_path
        self.fm_index_path = fm_index_path

    @staticmethod
    def _load_prompt_templates(prompt_names: Dict[str, str]) -> Dict[str, Any]:
        if not prompt_names:
            return {}

        from RoleRAG.rolerag.pipelines.prompts import PromptManager

        return {
            key: PromptManager.get_template(prompt_name)
            for key, prompt_name in prompt_names.items()
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)
        system = system_prompt or self.system_prompt
        with error_context(f"role=reasoner, model={type(self._model).__name__}, operation=generate"):
            if system:
                prompt = f"{system}\n\n{prompt}"
            result = self._model.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
            if not isinstance(result, str):
                raise TypeError(f"generate() must return text; got {type(result).__name__}")
            return result
