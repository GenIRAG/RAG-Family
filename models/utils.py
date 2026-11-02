import os
from typing import Any, Dict, Optional

from RoleRAG.models.BM25_model import BM25Model
from RoleRAG.models.api_decoder import OpenAIGenerator, DeepSeekGenerator
from RoleRAG.models.hf_decoder import HFDecoder
from RoleRAG.models.llmlingua2_model import LLMLingua2Model
from RoleRAG.models.llmlingua_model import LLMLinguaModel
from RoleRAG.models.hf_encoder import (
    HFEncoder,
    HFReranker,
)


_MODEL_REGISTRY = {
    "BM25Model": BM25Model,
    "HFEncoder": HFEncoder,
    "OpenAIGenerator": OpenAIGenerator,
    "HFDecoder": HFDecoder,
    "HFReranker": HFReranker,
    "DeepSeekGenerator": DeepSeekGenerator,
    "LLMLingua2Model": LLMLingua2Model,
    "LLMLinguaModel": LLMLinguaModel,
}

_GLOBAL_MODEL_CONFIG_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _load_global_model_defaults() -> Dict[str, Dict[str, Any]]:
    global _GLOBAL_MODEL_CONFIG_CACHE
    if _GLOBAL_MODEL_CONFIG_CACHE is None:
        import yaml

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config", "global.yaml")
        if not os.path.exists(config_path):
            _GLOBAL_MODEL_CONFIG_CACHE = {}
        else:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
            model_config = ((config.get("global", {}) or {}).get("models", {}) or {})
            if not isinstance(model_config, dict):
                raise TypeError("global.models must be a mapping")
            _GLOBAL_MODEL_CONFIG_CACHE = {
                name: values.copy()
                for name, values in model_config.items()
                if isinstance(values, dict)
            }
    return _GLOBAL_MODEL_CONFIG_CACHE


def get_model(model_type: str, **kwargs) -> Any:
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(_MODEL_REGISTRY.keys())}")

    model_config = _load_global_model_defaults().get(model_type, {}).copy()
    model_config.update(kwargs)
    return _MODEL_REGISTRY[model_type](**model_config)
