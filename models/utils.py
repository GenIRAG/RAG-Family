import os
from typing import Any, Dict, Optional

from RoleRAG.errors import error_context, load_yaml_mapping, require_mapping, validate_constructor_kwargs
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
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config", "global.yaml")
        config = load_yaml_mapping(config_path)
        global_config = require_mapping(config, "global", str(config_path))
        model_config = require_mapping(global_config, "models", f"{config_path}: global")
        _GLOBAL_MODEL_CONFIG_CACHE = {
            name: require_mapping(model_config, name, f"{config_path}: global.models").copy()
            for name in model_config
        }
    return _GLOBAL_MODEL_CONFIG_CACHE


def get_model(model_type: str, **kwargs) -> Any:
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(_MODEL_REGISTRY.keys())}")

    with error_context(
        f"Creating model={model_type}",
        f"Check config/global.yaml: global.models.{model_type} and explicit get_model() arguments.",
    ):
        model_config = _load_global_model_defaults().get(model_type, {}).copy()
        model_config.update(kwargs)
        model_cls = _MODEL_REGISTRY[model_type]
        validate_constructor_kwargs(model_cls, model_config, f"global.models.{model_type}")
        return model_cls(**model_config)
