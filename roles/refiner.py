from typing import List, Optional

from RoleRAG.roles.role_base import BaseRefinerContract


class RefinerRoleAdapter(BaseRefinerContract):

    def __init__(
        self,
        model,
        rate: float = 0.33,
        method: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs,
    ):
        self._model = model
        self.rate = rate
        self.method = method
        self.instruction = instruction
        self.model_name = getattr(model, "model_name", type(model).__name__)

    def refine(
        self,
        context: List[str],
        **kwargs,
    ) -> str:
        options = {
            "rate": kwargs.pop("rate", self.rate),
        }
        method = kwargs.pop("method", self.method)
        if method is not None:
            options["method"] = method

        question = kwargs.pop("question", None)
        if getattr(self._model, "requires_question", False):
            if not question:
                raise ValueError(f"{self.model_name} requires a non-empty question")
            options["question"] = question

        instruction = kwargs.pop("instruction", self.instruction)
        if instruction is not None:
            options["instruction"] = instruction
        options.update(kwargs)

        result = self._model.compress(context, **options)
        
        if isinstance(result, dict) and "compressed_prompt" in result:
            return result["compressed_prompt"]
        if isinstance(result, str):
            return result
