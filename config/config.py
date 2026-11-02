import os
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    GLOBAL_FILE = "global.yaml"
    INDEX_FILE = "index.yaml"

    def __init__(self, config_dir_path: str, pipeline_name: str | None = None):
        self.final_config = self._load_all_configs(config_dir_path, pipeline_name=pipeline_name)
        self._setup_hf_endpoint()

    def _resolve_global_path(self, config_dir_path: str) -> Path:
        config_dir = Path(config_dir_path).expanduser().resolve()
        if not config_dir.is_dir():
            raise ValueError(f"Config directory not found: {config_dir}")
        return config_dir / self.GLOBAL_FILE

    def _resolve_index_path(self, config_dir_path: str) -> Path:
        config_dir = Path(config_dir_path).expanduser().resolve()
        if not config_dir.is_dir():
            raise ValueError(f"Config directory not found: {config_dir}")
        return config_dir / self.INDEX_FILE

    def _resolve_method_config_path(self, config_dir_path: str, pipeline_name: str) -> Path:
        config_dir = Path(config_dir_path).expanduser().resolve()
        return config_dir / "methods" / f"{pipeline_name}.yaml"

    def _load_yaml_file(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise ValueError(f"Required config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f.read())
        if not isinstance(data, dict):
            raise TypeError(f"Config file must contain a YAML mapping: {path}")
        return data

    def _build_config_from_method_yaml(
        self,
        global_cfg: Dict[str, Any],
        index_cfg: Dict[str, Any],
        method_cfg: Dict[str, Any],
        selected_pipeline: str,
    ) -> Dict[str, Any]:
        
        evaluation_defaults = global_cfg["evaluation"]
        method_evaluation = method_cfg["evaluation"]

        global_block = dict(global_cfg.get("global", {}) or {})
        global_block.update({
            "pipeline_type": selected_pipeline,
            "evaluation": deep_merge_dicts(evaluation_defaults, method_evaluation),
        })

        merged_index = deep_merge_dicts(index_cfg, method_cfg["index"])

        return {
            "global": global_block,
            "index": merged_index,
            "inference": method_cfg["inference"],
            "roles": method_cfg.get("roles", {}),
        }

    def _load_all_configs(self, config_dir_path: str, pipeline_name: str | None = None) -> Dict[str, Any]:
        global_path = self._resolve_global_path(config_dir_path)
        index_path = self._resolve_index_path(config_dir_path)
        global_cfg = self._load_yaml_file(global_path)
        index_cfg = self._load_yaml_file(index_path)
        selected_pipeline = pipeline_name or global_cfg["pipeline_type"]

        return self._build_config_from_method_yaml(
            global_cfg=global_cfg,
            index_cfg=index_cfg,
            method_cfg=self._load_yaml_file(self._resolve_method_config_path(config_dir_path, selected_pipeline)),
            selected_pipeline=selected_pipeline,
        )

    def get(self, key: str) -> Any:
        if not key:
            return self.final_config

        value: Any = self.final_config
        traversed = []
        for part in key.split("."):
            if not isinstance(value, dict):
                path = ".".join(traversed) or "<root>"
                raise TypeError(f"Config path is not a dict: {path}")
            if part not in value:
                missing_path = ".".join(traversed + [part])
                raise KeyError(f"Config key not found: {missing_path}")
            value = value[part]
            traversed.append(part)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        if "." not in key:
            self.final_config[key] = value
            return

        keys = key.split(".")
        current = self.final_config
        traversed = []
        for part in keys[:-1]:
            if part not in current:
                missing_path = ".".join(traversed + [part])
                raise KeyError(f"Config key not found: {missing_path}")
            if not isinstance(current[part], dict):
                path = ".".join(traversed + [part])
                raise TypeError(f"Config path is not a dict: {path}")
            current = current[part]
            traversed.append(part)
        current[keys[-1]] = value

    def _setup_hf_endpoint(self) -> None:
        hf_mirror = self.get("global.hf_mirror")
        if hf_mirror:
            os.environ["HF_ENDPOINT"] = hf_mirror
