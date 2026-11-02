from __future__ import annotations

import importlib
import pkgutil
import re
from dataclasses import dataclass
from typing import Any, Dict


class PromptText(str):
    _pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

    def format(self, **kwargs: Any) -> str:  # type: ignore[override]
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key in kwargs:
                return str(kwargs[key])
            return match.group(0)

        return self._pattern.sub(replace, str(self))


@dataclass(frozen=True)
class PromptTemplate:

    system_prompt: PromptText = PromptText("Answer the question based on the given document.")
    user_prompt: PromptText = PromptText("Question: {question}\n\nThe following are given documents.\n\n{reference}")

    def __post_init__(self) -> None:
        object.__setattr__(self, "system_prompt", PromptText(self.system_prompt))
        object.__setattr__(self, "user_prompt", PromptText(self.user_prompt))


class PromptManager:

    _templates: Dict[str, PromptTemplate] = {}
    _loaded = False

    @classmethod
    def _load_templates(cls) -> None:
        if cls._loaded:
            return

        package = importlib.import_module("RoleRAG.rolerag.pipelines.prompts")
        for module_info in pkgutil.iter_modules(package.__path__):
            if module_info.name == "prompt_manager":
                continue
            module = importlib.import_module(f"{package.__name__}.{module_info.name}")
            prompt_map = getattr(module, "PROMPTS", None)
            if not prompt_map:
                continue
            if not isinstance(prompt_map, dict):
                raise TypeError(f"PROMPTS in {module.__name__} must be a dict")
            for name, template in prompt_map.items():
                if not isinstance(template, PromptTemplate):
                    raise TypeError(f"Prompt '{name}' in {module.__name__} must be a PromptTemplate")
                if name in cls._templates:
                    raise KeyError(f"Duplicate prompt template name: {name}")
                cls._templates[name] = template

        cls._loaded = True

    @classmethod
    def get_template(cls, name: str) -> PromptTemplate:
        
        cls._load_templates()
        if name not in cls._templates:
            raise KeyError(f"Prompt template '{name}' not found in prompts package")
        return cls._templates[name]
