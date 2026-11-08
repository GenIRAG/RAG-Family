from typing import List, Optional

from RoleRAG.errors import error_context
from RoleRAG.roles.registry import register_role
from RoleRAG.roles.role_base import BaseRefinerContract
from RoleRAG.roles.validation import validate_model_capability


@register_role("refiner")
class RefinerRoleAdapter(BaseRefinerContract):

    def __init__(
        self,
        model,
        rate: float = 0.33,
        method: Optional[str] = None,
        instruction: Optional[str] = None,
    ):
        validate_model_capability(model, "refiner", "compress", "compress")
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

        with error_context(f"role=refiner, model={self.model_name}, operation=compress"):
            result = self._model.compress(context, **options)
            if isinstance(result, dict):
                result = result.get("compressed_prompt")
            if not isinstance(result, str):
                raise TypeError(
                    "compress() must return text or a dict containing a string 'compressed_prompt'; "
                    f"got {type(result).__name__}"
                )
            return result
