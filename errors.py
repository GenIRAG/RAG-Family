from collections.abc import Mapping
from contextlib import contextmanager
from difflib import get_close_matches
import inspect
from pathlib import Path


@contextmanager
def error_context(location: str, hint: str = ""):
    try:
        yield
    except Exception as exc:
        note = f"Context: {location}"
        if hint:
            note += f"\nHint: {hint}"
        if note not in getattr(exc, "__notes__", ()):
            exc.add_note(note)
        raise


def validate_constructor_kwargs(cls, kwargs: Mapping, location: str) -> None:
    if any(not isinstance(name, str) for name in kwargs):
        raise TypeError(f"Parameter names at {location} must be strings; check the YAML mapping keys.")
    signature = inspect.signature(cls)
    try:
        signature.bind(**kwargs)
    except TypeError as exc:
        allowed = [
            name for name, param in signature.parameters.items()
            if name != "model" and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        unknown = set(kwargs) - set(signature.parameters)
        suggestions = []
        for name in sorted(unknown):
            matches = get_close_matches(name, allowed, n=1, cutoff=0.65)
            if matches:
                suggestions.append(f"{name!r} -> {matches[0]!r}")
        message = (
            f"Invalid constructor parameters at {location}: {exc}. "
            f"Accepted parameters: {', '.join(allowed) or '<none>'}."
        )
        if suggestions:
            message += f" Did you mean: {', '.join(suggestions)}?"
        raise TypeError(message) from exc


def load_yaml_mapping(path) -> dict:
    import yaml

    path = Path(path)
    with error_context(f"Reading configuration: {path}", "Check the file path, YAML indentation and mapping keys."):
        with path.open("r", encoding="utf-8") as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                mark = getattr(exc, "problem_mark", None)
                position = f":{mark.line + 1}:{mark.column + 1}" if mark else ""
                problem = getattr(exc, "problem", None) or "invalid YAML syntax"
                raise ValueError(f"Invalid YAML in {path}{position}: {problem}") from exc
        if not isinstance(data, dict):
            raise TypeError(f"Configuration must be a YAML mapping: {path}; got {type(data).__name__}")
        return data


def require_mapping(data: Mapping, key: str, source: str) -> dict:
    if key not in data:
        raise KeyError(f"Missing configuration block '{key}' in {source}")
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError(f"Configuration block '{key}' in {source} must be a mapping; got {type(value).__name__}")
    return value


def format_cli_error(exc: Exception) -> str:
    lines = [f"{type(exc).__name__}: {exc}"]
    seen_errors = set()
    seen_notes = set()
    current = exc
    while current is not None and id(current) not in seen_errors:
        seen_errors.add(id(current))
        for note in getattr(current, "__notes__", ()):
            if note not in seen_notes:
                lines.append(note)
                seen_notes.add(note)
        if isinstance(current, ModuleNotFoundError):
            module = (current.name or "").split(".")[0]
            packages = {"faiss": "faiss-cpu", "Stemmer": "PyStemmer", "yaml": "PyYAML", "sentence_transformers": "sentence-transformers"}
            if module == "seal":
                lines.append("Hint: UCGR requires SEAL from source; follow README's SEAL installation instructions.")
            else:
                lines.append(
                    f"Hint: Missing dependency '{packages.get(module, module)}'. "
                    "Install requirements.txt in the Python environment used to run this command."
                )
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return "\n".join(lines)
