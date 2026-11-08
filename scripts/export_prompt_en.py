#!/usr/bin/env python

import argparse
import ast
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from tqdm import tqdm


LLM_BASE_URL = "YOUR_BASE_URL"
LLM_API_KEY = "YOUR_API_KEY"
LLM_MODEL = "deepseek-v4-flash"


DEFAULT_INCLUDE_DIRS = {
    "config",
    "indexes",
    "models",
    "rolerag",
    "roles",
}

DEFAULT_INCLUDE_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".toml",
    ".ini",
    ".cfg",
}

DEFAULT_INCLUDE_ROOT_FILES = {
    "COMMANDS.md",
    "README.md",
    "main.py",
}

DEFAULT_REFERENCE_SOURCE_FILES = {
    "main.py",
    "config/config.py",
    "config/types.py",
    "config/global.yaml",
    "config/index.yaml",
    "models/__init__.py",
    "models/utils.py",
    "models/models_base.py",
    "roles/__init__.py",
    "roles/utils.py",
    "roles/role_base.py",
    "roles/retriever.py",
    "roles/reranker.py",
    "roles/reasoner.py",
    "roles/rewriter.py",
    "roles/refiner.py",
    "rolerag/pipelines/__init__.py",
    "rolerag/pipelines/pipelines_base.py",
    "rolerag/pipelines/naive_pipeline.py",
    "rolerag/pipelines/retrieval_strategy/__init__.py",
    "rolerag/pipelines/retrieval_strategy/basic_retrieval_strategy.py",
    "rolerag/pipelines/generation_strategy/__init__.py",
    "rolerag/pipelines/generation_strategy/naive_generation_strategy.py",
    "rolerag/pipelines/prompts/prompt_manager.py",
    "rolerag/pipelines/prompts/naive_prompt.py",
    "config/methods/naive.yaml",
}

SKIP_CACHE_DIRS = {
    "__pycache__",
}


@dataclass
class ProjectFile:
    path: str
    content: str


@dataclass
class CollectionStats:
    included_files: int
    skipped_files: int
    total_chars: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a project description file from the current project files."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root to scan. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output",
        default="project_description_en.md",
        help="Markdown file to write. Ignored when --stdout is set.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the generated prompt instead of writing a file.",
    )
    return parser.parse_args()


def is_included_text_file(path: Path) -> bool:
    return path.suffix in DEFAULT_INCLUDE_SUFFIXES


def has_included_file(directory: Path) -> bool:
    for path in directory.rglob("*"):
        if any(part in SKIP_CACHE_DIRS for part in path.parts):
            continue
        if path.is_file() and is_included_text_file(path):
            return True
    return False


def iter_tree_entries(
    root: Path,
) -> List[str]:
    entries: List[str] = []
    include_dirs = sorted(dirname for dirname in DEFAULT_INCLUDE_DIRS if (root / dirname).is_dir())

    def walk(directory: Path, depth: int) -> None:
        children = sorted(directory.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        for child in children:
            if child.is_dir() and child.name in SKIP_CACHE_DIRS:
                continue
            if child.is_dir() and not has_included_file(child):
                continue
            if child.is_file() and not is_included_text_file(child):
                continue
            rel = child.relative_to(root).as_posix()
            indent = "  " * max(depth - 1, 0)
            marker = "/" if child.is_dir() else ""
            entries.append(f"{indent}- {rel}{marker}")
            if child.is_dir():
                walk(child, depth + 1)

    for dirname in tqdm(include_dirs, desc="Building project tree", unit="dir"):
        directory = root / dirname
        if not has_included_file(directory):
            continue
        entries.append(f"- {dirname}/")
        walk(directory, 2)
    for filename in sorted(DEFAULT_INCLUDE_ROOT_FILES):
        path = root / filename
        if path.is_file():
            entries.append(f"- {filename}")
    return entries


def read_text_safely(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    except OSError:
        return None


def discover_project_files(root: Path) -> List[Path]:
    files = []
    include_dirs = sorted(dirname for dirname in DEFAULT_INCLUDE_DIRS if (root / dirname).is_dir())
    for dirname in tqdm(include_dirs, desc="Discovering project files", unit="dir"):
        directory = root / dirname
        for path in directory.rglob("*"):
            if any(part in SKIP_CACHE_DIRS for part in path.parts):
                continue
            if path.is_file() and is_included_text_file(path):
                files.append(path)
    for filename in sorted(DEFAULT_INCLUDE_ROOT_FILES):
        path = root / filename
        if path.is_file() and is_included_text_file(path):
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def collect_project_files(
    root: Path,
    paths: List[Path],
) -> tuple[List[ProjectFile], CollectionStats]:
    collected: List[ProjectFile] = []
    skipped_files = 0
    total_chars = 0

    with tqdm(total=len(paths), desc="Reading project files", unit="file") as progress:
        for path in paths:
            rel = path.relative_to(root).as_posix()
            text = read_text_safely(path)
            if text is None:
                skipped_files += 1
                progress.update(1)
                continue

            collected.append(ProjectFile(rel, text))
            total_chars += len(text)
            progress.update(1)

    stats = CollectionStats(
        included_files=len(collected),
        skipped_files=skipped_files,
        total_chars=total_chars,
    )
    return collected, stats


def detect_project_name(root: Path) -> str:
    for name in ("README.md", "READMEv2.md"):
        path = root / name
        text = read_text_safely(path)
        if not text:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped.lstrip("#").strip()
    return root.name


def format_code_block(content: str, language: str = "") -> str:
    fence = "```"
    if "```" in content:
        fence = "````"
    return f"{fence}{language}\n{content}\n{fence}"


def build_llm_context(
    project_name: str,
    tree_entries: List[str],
    project_files: List[ProjectFile],
    stats: CollectionStats,
) -> str:
    tree_text = "\n".join(tree_entries) if tree_entries else "(No files found.)"
    file_parts = [
        (
            f"===== FILE: {item.path} =====\n"
            f"{item.content}"
        )
        for item in project_files
    ]
    files_text = "\n\n".join(file_parts) or "(No project files collected.)"

    return f"""Project name:
{project_name}

Collection stats:
- Included files: {stats.included_files}
- Skipped files: {stats.skipped_files}
- Total collected characters: {stats.total_chars}

Project tree:
{tree_text}

Project file contents:
{files_text}
"""


def generate_project_file_description(
    project_name: str,
    tree_entries: List[str],
    project_files: List[ProjectFile],
    stats: CollectionStats,
) -> str:
    if not LLM_API_KEY:
        raise SystemExit(
            "LLM API key is required. Please set LLM_API_KEY at the top of scripts/export_prompt_en.py."
        )

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    llm_context = build_llm_context(project_name, tree_entries, project_files, stats)
    system_prompt = (
        "You are a senior software engineer. Your task is to generate a project description "
        "from the project tree and actual file contents, covering both the directory structure "
        "and the responsibilities of individual files. The description must be accurate and "
        "oriented toward collaborative code changes. Do not restate source code or invent files "
        "that do not exist. Write the entire response in English."
    )
    user_prompt = f"""Based on the project snapshot below, generate a Markdown section titled "Project Directory and File Guide."

Requirements:
1. The output must begin with the level-two heading "## Project Directory and File Guide".
2. Begin with one paragraph summarizing the project's overall responsibilities.
3. Describe the project structure, grouped by directory.
4. Cover as many collected source, configuration, and documentation files as possible.
5. For each file, explain its responsibilities and the types of changes for which it should be consulted.
6. Clearly explain key call relationships involving entry points, configuration, pipelines, models, roles, retrieval, evaluation, and Web/API components.
7. Do not include long code excerpts.
8. If a file was not collected, describe it cautiously and only from known information.

Project snapshot:

{llm_context}
"""

    llm_start_time = time.perf_counter()
    with tqdm(total=1, desc="Generating file description with LLM", unit="request") as progress:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        progress.update(1)
    llm_elapsed = time.perf_counter() - llm_start_time
    print(f"Generated file description with LLM in {llm_elapsed:.2f}s")
    content = response.choices[0].message.content
    if not content:
        raise SystemExit("LLM returned an empty project file description.")
    return content.strip()


def build_architecture_overview() -> str:
    return """## Architecture Overview

RoleRAG is a configuration-driven RAG experimentation framework. When an external model reads this document, it should first understand the relationships among the entry point, configuration, pipeline, model, role, strategy, and evaluator layers before deciding how to change the code.

### Core Execution Flow

```text
main.py
-> Config(config_dir, pipeline_name)
-> config/global.yaml + config/index.yaml + config/methods/<pipeline>.yaml
-> create_pipeline(config.final_config)
-> Pipeline.__init__ / create()
-> get_model(model_name)
-> assist_role(model, role, role_config)
-> pipeline.run(query) / pipeline.evaluate(dataset)
```

### Module Layers

- `main.py`: The command-line entry point. It dispatches index construction, single-query inference, and batch evaluation.
- `config/`: Configuration loading, configuration merging, and cross-module data structures.
- `models/`: Low-level model capabilities. This layer expresses only what a model can do, such as `encode`, `generate`, or `rerank`.
- `roles/`: The role adapter layer. It wraps low-level models in application-facing interfaces such as retriever, reasoner, reranker, rewriter, and refiner. The refiner compresses or cleans retrieved context before generation.
- `rolerag/pipelines/`: Composes roles and strategies to define the execution flow of each RAG method.
- `rolerag/pipelines/retrieval_strategy/`: Retrieval strategy implementations.
- `rolerag/pipelines/generation_strategy/`: Generation strategy implementations.
- `rolerag/pipelines/prompts/`: Prompt template registration and lookup.
- `indexes/`: Construction and loading of standard indexes, graph indexes, and FMIndex artifacts.
- `rolerag/evaluator/`: Evaluation data loading, metric computation, and log persistence.

### Registered Pipelines

- Core and context-processing pipelines: `naive`, `hybrid`, `rerank`, `llmlingua`, and `llmlingua2`.
- Multi-hop, graph, and generative-retrieval pipelines: `perqa`, `hipporag`, `graphrag_local`, `graphrag_global`, and `ucgr`.
- When changing a pipeline, treat its import in `rolerag/pipelines/__init__.py`, its trailing `register_pipeline()` call, and its matching `config/methods/<pipeline>.yaml` as authoritative. Do not infer role composition or strategy behavior from a pipeline name alone.
"""


def build_extension_manual() -> str:
    return """## Change Guide for External LLMs

After receiving a user request, an external LLM must not output code immediately. It should first classify the requested change, identify which files must change and why, and determine whether any required source files are still missing.

### Adding a New Model

The following files usually require attention:

- `models/`: Add the low-level model class. A generation model implements `generate()`, an embedding model implements `encode()`, and a reranking model implements `rerank()`.
- `models/utils.py`: Import the new model class and add it to `_MODEL_REGISTRY`.
- `models/__init__.py`: Add the import and update `__all__` if the model must be publicly exported.
- `config/global.yaml`: Add defaults under `global.models` using the same registered model name.
- `config/methods/<pipeline>.yaml`: If the model will serve a role, add the corresponding configuration under `roles`.
- First confirm that the model satisfies the target adapter's capability checks: a retriever requires `get_top_k()` or `encode()`, a reasoner or rewriter requires `generate()`, a reranker requires `rerank()`, and a refiner requires `compress()`. Use the corresponding adapter source as the authority for constructor arguments and invocation behavior.

### Adding a New Role

The following files usually require attention:

- `roles/role_base.py`: Add a role contract that defines the stable interface on which pipelines can depend.
- `roles/<new_role>.py`: Add an adapter that wraps a low-level model in the role interface.
- `roles/utils.py`: No change needed; decorate the adapter with `@register_role("<name>")` and `assist_role()` looks it up in the registry.
- `roles/__init__.py`: Add the import and update `__all__` if the role must be publicly exported.
- `config/methods/<pipeline>.yaml`: Add the role's configuration block.

### Adding a New Pipeline

The following files usually require attention:

- `rolerag/pipelines/<new_pipeline>_pipeline.py`: Add a pipeline that inherits from `BasePipeline` and implements `create()` and `_run_item()`.
- `rolerag/pipelines/pipelines_base.py`: Understand the interfaces of `BasePipeline`, `register_pipeline()`, and `create_pipeline()`; do not casually change the shared base class.
- `rolerag/pipelines/__init__.py`: Import the new pipeline module so that its registration code runs.
- `config/methods/<new_pipeline>.yaml`: Add method-level configuration, including `index.paths`, `inference`, `roles`, and `evaluation`.
- `config/global.yaml`: Change the top-level `pipeline_type` if the new pipeline should become the default.
- `README.md` / `COMMANDS.md`: Add usage commands if the pipeline should be exposed to users.

Pipeline registration must follow the actual signature of `register_pipeline()` in `pipelines_base.py`. When it is an ordinary two-argument registration function, call `register_pipeline("<name>", PipelineClass)` directly after the class definition. Do not use it as a decorator, and do not pass the class name as a string in the second argument.

If the user explicitly requests support for `@register_pipeline("<name>")`, first change `register_pipeline()` into a backward-compatible decorator factory: preserve direct registration when `pipeline_class` is provided; when only a name is supplied, return a decorator that accepts the class, registers it, and returns it unchanged. After the change, verify both existing direct calls and new decorator calls. Never write `@register_pipeline("<name>", "PipelineClass")`, because a string is not a pipeline class and the return value of an ordinary registration function cannot be used as a decorator.

### Configuring Role Parameters and Adding a Method YAML File

RoleRAG's runtime role parameters are primarily defined in `config/methods/<pipeline>.yaml`. Before changing configuration, an external LLM must determine whether it is modifying an existing pipeline or adding a new one:

- To change the model, prompt, or role parameters of an existing pipeline, edit the existing `config/methods/<pipeline>.yaml`.
- To add a pipeline, create `config/methods/<new_pipeline>.yaml`.
- `--pipeline <name>` loads `config/methods/<name>.yaml`.
- `<name>` must match the name registered by `register_pipeline("<name>", PipelineClass)`.
- A method YAML file typically contains `index`, `inference`, `roles`, and `evaluation`.
- `index.paths` is generally used to load existing index paths at runtime; index construction parameters primarily come from `config/index.yaml`.
- `inference` stores pipeline-specific inference parameters, such as top-k values, fusion parameters, graph retrieval parameters, and generation-stage parameters.
- The exact `roles` hierarchy is `roles.<role>.<registered_model_name>.<optional_instance_name>`. The registered model name must be a key in `_MODEL_REGISTRY` in `models/utils.py` and must have defaults in `global.models`; it is not the `model_name` field.
- Single-instance example:

  ```yaml
  roles:
    retriever:
      BM25Model:
        top_k: 5
    reasoner:
      OpenAIGenerator:
        prompt_name: "naive_rag"
        max_tokens: 256
  ```

- Multi-instance example: when one model serves several purposes within the same role, add an instance-name level. The pipeline must call `get_method_role_config()` with the matching instance name.

  ```yaml
  roles:
    reasoner:
      OpenAIGenerator:
        planner_reasoner:
          prompt_names:
            type_system: "perqa_type_system"
        executor_reasoner:
          prompt_name: "perqa_rag_qa"
  ```

- `get_model()` creates the underlying model from `global.models[model_name]`; `assist_role()` clones that model and uses only role parameters from the method YAML file to configure the adapter.
- Consult the corresponding adapter source to determine exactly which fields a role configuration supports, for example `roles/reasoner.py`, `roles/retriever.py`, `roles/reranker.py`, `roles/rewriter.py`, or `roles/refiner.py`. Do not infer fields from a responsibility summary.
- Adding YAML configuration alone does not activate it. The corresponding pipeline's `create()` method must read the role configuration and call `get_model()` and `assist_role()`, and `_run_item()` or a strategy must actually use the role.
- If a new role serves only an existing pipeline, edit that pipeline's method YAML file. If it belongs to a new pipeline, place it in the new method YAML file.

### Adding a Retrieval or Generation Strategy

The following files usually require attention:

- This project has no retrieval or generation strategy registry. A concrete pipeline imports a strategy directly and instantiates it in `__init__`; adding a strategy therefore requires changing the imports and instantiation code of every pipeline that uses it.
- `rolerag/pipelines/retrieval_strategy/`: Add or change a retrieval strategy. It should accept role objects supplied by the pipeline, return a `RetrieverResult`, and read pipeline-specific parameters from `config["inference"][<pipeline_name>]`.
- `rolerag/pipelines/generation_strategy/`: Add or change a generation strategy. It usually accepts `question`, `retrieval_result`, and `reasoner`, and returns an answer string. A complex method may define additional return values, but the pipeline must unpack them and construct `RAGOutput`.
- `rolerag/pipelines/prompts/`: If the strategy requires a prompt, add a `PROMPTS` template and ensure that `PromptManager` can load it.
- When adding a prompt file, imports such as `PromptTemplate` and `PromptManager` must exactly follow the real module paths demonstrated by the provided `prompt_manager.py`. Do not infer or abbreviate import paths from directory names.
- In each pipeline that uses the strategy, instantiate it in `__init__`, invoke it in `_run_item()`, and preserve the existing retrieval-only branch, logging behavior, and `RAGOutput` conventions.
- Update the corresponding strategy package's `__init__.py` if the new strategy must be imported through the package path. This export is not required when a pipeline imports the strategy directly from its module.

### Backward Compatibility and Fallback Rules

Unless the user explicitly requests otherwise, new or modified functionality must not:

- Switch to another strategy, role, model, or prompt when results are empty, configuration is missing, or an exception occurs.
- Return a substitute business result to conceal an error.
- Support aliases for legacy configuration fields or interfaces, or automatically migrate legacy configuration.
- Automatically select another existing implementation as a substitute.

A component may use its own explicit, stable parameter defaults, such as `max_tokens`, `temperature`, or `top_k`. Missing required configuration must raise a clear error. This rule does not require refactoring existing project behavior.

### Changing Configuration Behavior

The following files usually require attention:

- `config/config.py`: Configuration loading, merging, and dot-path access.
- `config/global.yaml`: Global defaults and default model parameters.
- `config/index.yaml`: Index construction defaults.
- `config/methods/<pipeline>.yaml`: Runtime configuration for one pipeline.
- `main.py`: CLI argument override logic.

### Requirements When Outputting Code

- Group output by file path.
- Explain why each file must change.
- If the full contents of a required file have not been provided, ask the user for it first; do not invent it.
- Prefer targeted replacements. Output a complete file only when it is short or when changes are widely dispersed.
- Provide at least one verification command after the change.

### Mandatory Self-Check Before Outputting Code

Before outputting any code, verify every item below:

- Do not output code, pseudocode, patches, or complete file contents until the user explicitly confirms in a subsequent turn.
- For every new YAML field, list its configuration path, the class or function that reads it, and where it takes effect.
- For every new prompt, list the template name, template file, the `PromptManager` lookup site, and where it is rendered before the model call.
- For any new module that depends on a template, adapter, low-level model, index, or external component, explicitly document the complete call chain from input to final invocation. Configuring a name or object does not mean it is executed. Before calling a dependency directly, confirm its method signature and parameter semantics.
- Do not call an instance method, function, or attribute whose existence has not been confirmed in the provided source.
- A new role's method name must be identical across its contract, adapter, strategy, and pipeline.
- Before adding a strategy, determine whether an existing strategy can be reused. If the user explicitly asks for a new strategy, explain why it must differ from existing strategies.
- Do not switch to another strategy, role, model, or prompt when results are empty, configuration is missing, or an exception occurs, and do not conceal errors with substitute business results.
- The list of files to add or modify must include every new module, prompt, and configuration file referenced by the code.
- Before output, verify the reference closure in this order: file -> import -> caller -> configuration field.
- For algorithms, data transformations, sorting, fusion, or scoring, confirm input structures, field semantics, and ordering conventions from the provided source. Explicitly list the invariants on which the algorithm depends and the actual fields it uses. Do not replace business-semantic fields with iteration positions, display fields, or guessed fields. Cover boundary inputs and result ordering in the verification plan.
- Verification commands may use only arguments defined by `argparse` in `main.py`; argument names and value forms must match exactly. Configuration without a CLI override must be changed through YAML or another provided configuration entry point. Do not invent CLI arguments.

### Implementation Constraint Checklist

The plan for every new or modified pipeline, role, retrieval strategy, or generation strategy must include the following checklists. If any checklist is missing, complete the plan before outputting code.

#### Required Configuration Checklist

| Configuration path | Required? | Reading class or function | Where it takes effect | Behavior when missing |
| --- | --- | --- | --- | --- |

When required configuration is missing, `Behavior when missing` must specify a clear error. Optional configuration may use explicit, stable component-level defaults, but the default value, its use site, and why it does not alter the business path must be documented. Do not use legacy field aliases, substitute configuration, silent skipping, or automatic fallback.

#### Prompt Rendering Responsibility Matrix

Every new or used prompt must be listed:

| Template name | Template file | Template variables | Rendering class or function | Final `generate()` call site |
| --- | --- | --- | --- | --- |

Runtime parameters in templates, including counts and top-k values, must be rendered from configuration variables rather than hard-coded. Do not assume that setting `prompt_name` alone activates or renders a template.

For a new module that calls another component, the responsibility matrix must also state whether the final call targets an adapter, a low-level implementation, or an external service. If the dependency is called directly, list the confirmed method signature, parameter-passing behavior, and return-value semantics.

#### No-Fallback and Compatibility Audit

Before outputting code, answer each question with `Yes` or `No`; every answer should be `No`:

- Does the implementation fall back to another strategy, role, model, or prompt, or return a substitute business result when results are empty, configuration is missing, or an exception occurs?
- Does it include aliases for legacy configuration fields or interfaces, or automatic compatibility with legacy configuration?
- Does it automatically switch prompts, models, roles, or strategies?
- Does it call any method, function, or attribute whose existence has not been confirmed in the provided source?
"""


def build_common_change_entrypoints() -> str:
    return """## Quick Reference for Common Changes

| Change | Files to inspect first |
| --- | --- |
| Add a CLI argument | `main.py` |
| Change configuration merging | `config/config.py` |
| Add a data field | `config/types.py` |
| Add a low-level model | `models/`, `models/utils.py`, `config/global.yaml`, and the target role adapter / pipeline |
| Add a role | `roles/role_base.py`, `roles/utils.py`, `roles/<role>.py` |
| Add a pipeline | `rolerag/pipelines/pipelines_base.py`, `rolerag/pipelines/__init__.py`, `rolerag/pipelines/<pipeline>_pipeline.py`, `config/methods/<pipeline>.yaml` |
| Change retrieval logic | `rolerag/pipelines/retrieval_strategy/` and the corresponding pipeline |
| Change generation logic | `rolerag/pipelines/generation_strategy/`, `rolerag/pipelines/prompts/`, and the corresponding pipeline |
| Change index construction | `main.py`, `indexes/indexing/`, `indexes/loading/`, `config/index.yaml` |
| Change evaluation metrics | `rolerag/evaluator/metrics/` |
"""


def build_cli_configuration_guide(project_files: List[ProjectFile]) -> str:
    """Build an authoritative CLI-to-config table from main.py's AST."""
    main_file = next((item for item in project_files if item.path == "main.py"), None)
    if main_file is None:
        return ""

    try:
        tree = ast.parse(main_file.content)
    except SyntaxError:
        return ""

    arguments: dict[str, str] = {}
    config_paths: dict[str, List[str]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("--")
        ):
            continue
        option = node.args[0].value
        dest = option[2:].replace("-", "_")
        for keyword in node.keywords:
            if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                dest = keyword.value.value
        arguments[dest] = option

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "config"
            and isinstance(node.targets[0].slice, ast.Constant)
            and isinstance(node.targets[0].slice.value, str)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "args"
        ):
            continue
        config_paths.setdefault(node.value.attr, []).append(node.targets[0].slice.value)

    # Boolean flags are commonly written as `if args.flag: config["path"] = True`.
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute)
            and isinstance(node.test.value, ast.Name)
            and node.test.value.id == "args"
        ):
            continue
        dest = node.test.attr
        for statement in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Subscript)
                and isinstance(statement.targets[0].value, ast.Name)
                and statement.targets[0].value.id == "config"
                and isinstance(statement.targets[0].slice, ast.Constant)
                and isinstance(statement.targets[0].slice.value, str)
            ):
                continue
            config_paths.setdefault(dest, []).append(statement.targets[0].slice.value)

    if not arguments:
        return ""

    rows = []
    for dest, option in sorted(arguments.items(), key=lambda item: item[1]):
        paths = list(dict.fromkeys(config_paths.get(dest, [])))
        target = "<br>".join(f"`{path}`" for path in paths) if paths else "Does not write configuration directly (runtime action or dedicated workflow)"
        rows.append(f"| `{option}` | {target} |")

    return "\n".join(
        [
            "## Authoritative CLI Argument Table",
            "",
            "The table below is derived automatically from the current `argparse` definitions and direct configuration assignments in `main.py`. It is the only set of CLI arguments that may be assumed to exist.",
            "",
            "| CLI argument | Configuration path written directly |",
            "| --- | --- |",
            *rows,
            "",
            "Do not assume that a configuration field absent from this table has a corresponding CLI argument, especially an `inference` field in `config/methods/<pipeline>.yaml`. Unless `main.py` explicitly defines an entry point, change it through YAML or another listed configuration entry point.",
        ]
    )


def build_external_file_policy() -> str:
    return """## Policy for Requesting Additional Files

To keep the context manageable, this document embeds the shared contract source, every role adapter including the refiner, one runnable simple flow (Naive), and one multi-instance role flow (PERQA). Concrete LLMLingua, graph-indexing, UCGR, HippoRAG, and GraphRAG pipeline or strategy implementations must still be supplied on demand.

An external LLM must follow these rules:

- If the current task requires the full contents of a file that is not in the "Key Source Appendix," explicitly request that file from the user first.
- Give exact paths when requesting files, for example `roles/reasoner.py`, `rolerag/pipelines/perqa_pipeline.py`, or `README.md`.
- Do not merely say "please provide the relevant files." List the required files and explain why each is needed.
- To understand project background only, you may request `README.md`. For code changes, prioritize the actual source files.
- Until the user supplies the required files, do not invent interfaces, function signatures, class names, configuration fields, registry formats, or complete code.
- Without the full target file, output only a project understanding, a change plan, and the list of additional files required. Do not output a patch.
- Do not infer callable methods, parameter names, YAML fields, or return structures solely from a file responsibility summary.
- Request only the smallest file set needed for the current decision. If an interface remains uncertain, then request the next specific file.

Examples of task-specific file requests:

- To add an OpenAI-compatible API generation model, request `models/api_decoder.py` to confirm the existing API model wrapper and parameter conventions.
- To add a Hugging Face embedding or reranking model, request `models/hf_encoder.py` to confirm the conventions for `encode()` and `rerank()`.
- To add a Hugging Face decoder model, request `models/hf_decoder.py` to confirm local generation model loading and generation parameters.
- To add a retrieval role or change retrieval role parameters, use the embedded `roles/retriever.py`; for graph retrieval, also request the closest graph retrieval strategy.
- To add a reranking role or change reranking parameters, use the embedded `roles/reranker.py`; also request the target pipeline and the closest reranking method YAML file.
- To add reasoner parameters or change a prompt call, use the embedded `roles/reasoner.py` and `prompt_manager.py`; if the template variables differ, request a similar prompt file.
- To add a rewriting role, use the embedded `roles/rewriter.py`; when changing an existing method, also request the pipeline and strategy that use it.
- To add or change a context-refinement role, use the embedded `roles/refiner.py`; for an LLMLingua or LLMLingua2 execution flow, also request the target pipeline and `rolerag/pipelines/generation_strategy/refine_generation_strategy.py` to confirm `refine()` inputs, question passing, and prompt-rendering order before generation.
- To add a role that reuses a generation model, read `roles/reasoner.py`, `rolerag/pipelines/prompts/prompt_manager.py`, and a similar prompt file in addition to the role adapter, target pipeline, and strategy. Confirm whether the adapter or strategy renders the prompt. Do not assume that `ReasonerRoleAdapter` automatically renders a template from `prompt_name`.
- To add a prompt, request `rolerag/pipelines/prompts/prompt_manager.py` and a similar prompt file to confirm the `PROMPTS` structure and template variables.
- To add a retrieval strategy, request the target pipeline, the closest retrieval strategy, and the target method YAML file. Confirm the `retrieve()` arguments, `RetrieverResult.metadata`, and `inference` configuration keys.
- To add a generation strategy, request the target pipeline, the closest generation strategy, a similar prompt file, and the target method YAML file. Confirm the `generate()` arguments, template variables, and return-value unpacking.
- To modify an existing pipeline, request the target `rolerag/pipelines/<pipeline>_pipeline.py` and every strategy file it imports to confirm `create()`, `_run_item()`, and role configuration lookup. Complete Naive and PERQA examples are already embedded.
- To add a standard RAG pipeline, the Naive implementation, `rolerag/pipelines/__init__.py`, `rolerag/pipelines/pipelines_base.py`, and `naive.yaml` are already embedded. If the index or role composition differs, request the closest pipeline and method YAML file.
- To follow a complex pipeline, request the corresponding pipeline and its retrieval and generation strategy files, such as the relevant PERQA, HippoRAG, GraphRAG, or UCGR files.
- To adjust an existing method YAML file, request the target `config/methods/<pipeline>.yaml` to confirm its current `index`, `inference`, `roles`, and `evaluation` structure.
- To add a method YAML file, request the closest `config/methods/*.yaml` and the target pipeline file to confirm the filename, registration name, role configuration, and pipeline lookup logic.
"""


def build_reference_source_appendix(project_files: List[ProjectFile]) -> str:
    files_by_path = {item.path: item for item in project_files}
    selected_files = [
        files_by_path[path]
        for path in sorted(DEFAULT_REFERENCE_SOURCE_FILES)
        if path in files_by_path
    ]
    if not selected_files:
        return ""

    parts = [
        "## Key Source Appendix",
        "",
        "The files below contain shared contracts, every role adapter including the refiner, and runnable Naive and PERQA reference flows. For LLMLingua, graph indexing, UCGR, HippoRAG, GraphRAG, or another implementation not embedded here, request the complete target files first.",
    ]
    for item in selected_files:
        language = item.path.rsplit(".", 1)[-1] if "." in item.path else ""
        if language in {"yaml", "yml"}:
            language = "yaml"
        elif language == "py":
            language = "python"
        parts.extend(
            [
                "",
                f"### File: `{item.path}`",
                "",
                format_code_block(item.content, language),
            ]
        )
    return "\n".join(parts)


def build_prompt(
    project_name: str,
    tree_entries: List[str],
    project_file_description: str,
    project_files: List[ProjectFile],
) -> str:
    tree_text = "\n".join(tree_entries) if tree_entries else "(No files found.)"

    # Overall architecture and core execution flow.
    architecture_overview = build_architecture_overview()
    # Extension rules for external LLMs adding or modifying functionality.
    extension_manual = build_extension_manual()
    # Quick reference mapping common changes to the files to inspect first.
    common_change_entrypoints = build_common_change_entrypoints()
    # CLI arguments and direct configuration assignments extracted from main.py.
    cli_configuration_guide = build_cli_configuration_guide(project_files)
    # Rules for requesting source files that are not embedded.
    external_file_policy = build_external_file_policy()
    # Complete source appendix for the selected key files.
    reference_source_appendix = build_reference_source_appendix(project_files)

    # Assemble fixed sections, the LLM-generated guide, and the project snapshot.
    return f"""# Project Description

The following content is generated automatically from the project's current files. Its purpose is to help an external LLM that cannot access the local repository understand RoleRAG and assist with code changes after the user confirms a plan.

## Instructions

You are a senior code modification assistant. You cannot access my local project directly and must work only from the project context and file contents I provide.

Follow these rules strictly:

1. Understand the project before offering code.
2. First provide a "Project Understanding Summary" and a "Change Plan."
3. Wait for my confirmation before outputting code.
4. Group code by file path so I can place it back into the local project.
5. If required file contents are missing, explicitly ask me for the relevant paths and explain why they are needed. Do not invent them.
6. Prefer the project's existing structure, naming, configuration conventions, and dependencies.
7. Do not perform unrelated refactoring.
8. If the change spans several files, explain why each file must change.
9. After outputting code, provide executable verification commands.
10. Unless explicitly required, do not add compatibility for legacy configuration or interfaces, automatically switch strategies, roles, models, or prompts, or fall back to another implementation on exceptions or empty results. Explicit parameter defaults within the same component may remain.
11. Unless the user explicitly requests reuse, do not use HyDE, the naive prompt, `NaiveGenerationStrategy`, or another existing implementation as a default substitute. Any reuse must be described in the plan with its purpose and necessity, and must wait for user confirmation.

## Project Name

{project_name}

## Current Project Tree

{format_code_block(tree_text)}

{architecture_overview}

{project_file_description}

{extension_manual}

{common_change_entrypoints}

{cli_configuration_guide}

{external_file_policy}

{reference_source_appendix}

## Recommended Collaboration Workflow

I will append a specific request after this context. Respond in the following format when you receive it:

### 1. Project Understanding Summary

- Modules involved in the current request
- Relevant call chains or data flows
- Files that must be read or supplied; if additional files are required, list their exact paths and explain why

### 2. Change Plan

- Target functionality
- Files to modify
- Files to add
- Configuration changes
- Risks
- Verification approach

### 3. Await Confirmation

End the response here. Do not output code, pseudocode, patches, or complete file contents until I explicitly confirm the plan and request code in a subsequent turn.

### 4. Output Code After Confirmation

Use the following format for code:

```text
### File: path/to/file.py

```python
Complete code or a precise targeted replacement
```
```

For a targeted replacement, specify:

- Which section to replace
- The function, class, or location where it should be inserted
- The surrounding code to use as an anchor

## My Specific Request

Describe the functionality to change, the error, and the expected behavior here. Include the full contents of additional relevant files when necessary.
"""


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Project root not found: {root}")

    # Scan the project, build its tree, and collect exportable text files.
    project_name = detect_project_name(root)
    tree_entries = iter_tree_entries(root)
    project_paths = discover_project_files(root)
    project_files, stats = collect_project_files(
        root=root,
        paths=project_paths,
    )

    # Ask the LLM to generate a dynamic directory and file guide from the snapshot.
    project_file_description = generate_project_file_description(
        project_name=project_name,
        tree_entries=tree_entries,
        project_files=project_files,
        stats=stats,
    )

    # Combine the dynamic guide with the built-in architecture, collaboration rules, and source appendix.
    prompt = build_prompt(project_name, tree_entries, project_file_description, project_files)

    if args.stdout:
        print(prompt)
        elapsed = time.perf_counter() - start_time
        print(f"\nGenerated project description in {elapsed:.2f}s")
        return

    # By default, write the result to the output file in the project root.
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")
    print(f"Wrote project description to {output_path}")
    elapsed = time.perf_counter() - start_time
    print(f"Generated project description in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
