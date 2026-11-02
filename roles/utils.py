import os
from typing import Any, Dict, Optional

from RoleRAG.roles.reasoner import ReasonerRoleAdapter
from RoleRAG.roles.reranker import RerankerRoleAdapter
from RoleRAG.roles.retriever import RetrieverRoleAdapter
from RoleRAG.roles.rewriter import RewriterRoleAdapter
from RoleRAG.roles.refiner import RefinerRoleAdapter


_MODEL_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_ALLOWED_ROLES = {"retriever", "reranker", "reasoner", "rewriter", "refiner"}


def _load_yaml_config(filename: str) -> Dict[str, Any]:
    import yaml

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", filename)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_model_config() -> Dict[str, Any]:
    global _MODEL_CONFIG_CACHE
    if _MODEL_CONFIG_CACHE is None:
        global_cfg = _load_yaml_config("global.yaml")
        global_block = global_cfg.get("global", {}) or {}
        model_cfg = global_block.get("models", {}) or {}
        if not isinstance(model_cfg, dict):
            raise TypeError("Model config must be a mapping: global.models")
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


def get_method_role_config(
    config: Dict[str, Any],
    role: str,
    model_name: str,
    instance_name: str = None,
) -> Dict[str, Any]:
    role_cfg = ((config.get("roles", {}) or {}).get(role, {}) or {}).get(model_name, {}) or {}
    if not isinstance(role_cfg, dict):
        raise TypeError(f"Role config must be a mapping: role={role}, model={model_name}")
    if instance_name:
        instance_cfg = role_cfg.get(instance_name, {}) or {}
        if not isinstance(instance_cfg, dict):
            raise TypeError(
                f"Role instance config must be a mapping: role={role}, model={model_name}, instance={instance_name}"
            )
        return instance_cfg.copy()
    return role_cfg.copy()


def _configure_model(model, config: Dict[str, Any]):
    for key, value in config.items():
        if hasattr(model, key):
            setattr(model, key, value)


def assist_role(
    model,
    role: str,
    role_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    role = role.lower()
    if role not in _ALLOWED_ROLES:
        raise ValueError(
            f"Unknown role: {role}. "
            f"Available roles: retriever, reranker, reasoner, rewriter, refiner"
        )

    if hasattr(model, "clone") and callable(model.clone):
        model = model.clone()

    merged_config = role_config.copy() if role_config else {}

    if role == "retriever":
        return RetrieverRoleAdapter(
            model=model,
            top_k=merged_config.get("top_k", 5),
            index_path=merged_config.get("index_path"),
            graph_index_path=merged_config.get("graph_index_path"),
            **kwargs
        )

    if role == "reranker":
        return RerankerRoleAdapter(
            model=model,
            top_k=merged_config.get("top_k", 10),
            **kwargs
        )

    if role == "reasoner":
        return ReasonerRoleAdapter(
            model=model,
            max_tokens=merged_config.get("max_tokens", 256),
            temperature=merged_config.get("temperature", 0.0),
            system_prompt=merged_config.get("system_prompt"),
            prompt_name=merged_config.get("prompt_name", "naive_rag"),
            prompt_names=merged_config.get("prompt_names"),
            answer_prefix=merged_config.get("answer_prefix"),
            graph_index_path=merged_config.get("graph_index_path"),
            fm_index_path=merged_config.get("fm_index_path"),
            **kwargs
        )

    if role == "rewriter":
        return RewriterRoleAdapter(
            model=model,
            method=merged_config.get("method", "hyde"),
            num_hypotheses=merged_config.get("num_hypotheses", 4),
            max_tokens=merged_config.get("max_tokens", 512),
            temperature=merged_config.get("temperature", 0.7),
            task=merged_config.get("task", "web_search"),
            language=merged_config.get("language", "en"),
            stop=merged_config.get("stop"),
            **kwargs
        )

    if role == "refiner":
        return RefinerRoleAdapter(
            model=model,
            rate=merged_config.get("rate", 0.33),
            method=merged_config.get("method"),
            instruction=merged_config.get("instruction"),
            **kwargs
        )

    raise ValueError(
        f"Unknown role: {role}. "
        f"Available roles: retriever, reranker, reasoner, rewriter, refiner"
    )
