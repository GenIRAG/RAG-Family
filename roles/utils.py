import os
from collections.abc import Mapping
from typing import Any, Dict, Optional

from RoleRAG.errors import error_context, load_yaml_mapping, require_mapping, validate_constructor_kwargs
from RoleRAG.roles.reasoner import ReasonerRoleAdapter
from RoleRAG.roles.reranker import RerankerRoleAdapter
from RoleRAG.roles.retriever import RetrieverRoleAdapter
from RoleRAG.roles.rewriter import RewriterRoleAdapter
from RoleRAG.roles.refiner import RefinerRoleAdapter
from RoleRAG.roles.registry import get_role_class


_MODEL_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _load_yaml_config(filename: str) -> Dict[str, Any]:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", filename)
    return load_yaml_mapping(config_path)


def _load_model_config() -> Dict[str, Any]:
    global _MODEL_CONFIG_CACHE
    if _MODEL_CONFIG_CACHE is None:
        global_cfg = _load_yaml_config("global.yaml")
        global_block = require_mapping(global_cfg, "global", "config/global.yaml")
        model_cfg = require_mapping(global_block, "models", "config/global.yaml: global")
        _MODEL_CONFIG_CACHE = model_cfg
    return _MODEL_CONFIG_CACHE


def get_model_defaults(model_name: str) -> Dict[str, Any]:
    config = _load_model_config()
    if model_name not in config:
        raise KeyError(f"Model config not found in global.yaml global.models: {model_name}")
    model_cfg = config[model_name]
    if not isinstance(model_cfg, dict):
        raise TypeError(f"Model config must be a mapping: {model_name}")
    return model_cfg.copy()


def _format_available(keys) -> str:
    return ", ".join(sorted(str(key) for key in keys)) or "<none>"


def get_method_role_config(
    config: Dict[str, Any],
    role: str,
    model_name: str,
    instance_name: Optional[str] = None,
) -> Dict[str, Any]:
    pipeline_name = (config.get("global") or {}).get("pipeline_type", "<pipeline>")
    source = f"config/methods/{pipeline_name}.yaml"
    roles_cfg = config.get("roles", {})
    if not isinstance(roles_cfg, dict):
        raise TypeError(f"roles in {source} must be a mapping, got: {type(roles_cfg).__name__}")

    role_cfg = roles_cfg.get(role)
    if role_cfg is None:
        raise KeyError(
            f"Role config not found in {source}: roles.{role}. "
            f"Available roles in this method config: {_format_available(roles_cfg)}"
        )
    if not isinstance(role_cfg, dict):
        raise TypeError(f"Role config must be a mapping in {source}: roles.{role}")

    model_cfg = role_cfg.get(model_name)
    if model_cfg is None:
        raise KeyError(
            f"Model config not found in {source}: roles.{role}.{model_name}. "
            f"Available models under roles.{role}: {_format_available(role_cfg)}. "
            "The model selected by the pipeline must match a model key in this role's YAML configuration."
        )
    if not isinstance(model_cfg, dict):
        raise TypeError(f"Model config must be a mapping in {source}: roles.{role}.{model_name}")

    if not instance_name:
        return model_cfg.copy()

    instance_cfg = model_cfg.get(instance_name)
    if instance_cfg is None:
        raise KeyError(
            f"Role instance config not found in {source}: roles.{role}.{model_name}.{instance_name}. "
            f"Available instances: {_format_available(model_cfg)}"
        )
    if not isinstance(instance_cfg, dict):
        raise TypeError(
            f"Role instance config must be a mapping in {source}: "
            f"roles.{role}.{model_name}.{instance_name}"
        )
    return instance_cfg.copy()


def _configure_model(model, config: Dict[str, Any]) -> None:
    for key, value in config.items():
        if not hasattr(model, key):
            available = sorted(name for name in vars(model) if not name.startswith("_"))
            raise AttributeError(
                f"Unknown attribute '{key}' for model {type(model).__name__}. "
                f"Available attributes: {_format_available(available)}"
            )
        setattr(model, key, value)


def assist_role(
    model,
    role: str,
    role_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    role = str(role).lower()
    model_name = type(model).__name__
    with error_context(
        f"Creating role={role}, model={model_name}",
        f"Check roles.{role}.{model_name} in the method YAML and the model selected in the pipeline. "
        "Model construction settings belong in global.models, not in role parameters.",
    ):
        role_cls = get_role_class(role)
        if role_config is not None and not isinstance(role_config, Mapping):
            raise TypeError(f"role_config for '{role}' must be a mapping; got {type(role_config).__name__}")
        merged_config = dict(role_config or {})
        merged_config.pop("model", None)
        merged_config.update(kwargs)
        validate_constructor_kwargs(role_cls, {"model": model, **merged_config}, f"roles.{role}.{model_name}")

        if hasattr(model, "clone") and callable(model.clone):
            with error_context(f"Cloning model={model_name} for role={role}"):
                model = model.clone()
        return role_cls(model=model, **merged_config)
