from typing import Any, Dict, Optional

from RoleRAG.roles.role_base import BaseGeneratorContract


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
        **kwargs
    ):
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
        if hasattr(self._model, "generate"):
            if system:
                prompt = f"{system}\n\n{prompt}"
            return self._model.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
        raise TypeError(f"Model {type(self._model).__name__} does not support generate")
