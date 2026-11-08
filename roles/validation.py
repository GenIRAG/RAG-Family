def model_capabilities(model, role: str) -> frozenset:
    capabilities = getattr(model, "capabilities", None)
    if not isinstance(capabilities, (set, frozenset, list, tuple)) or any(
        not isinstance(value, str) for value in capabilities
    ):
        raise TypeError(
            f"Cannot bind model '{type(model).__name__}' to role '{role}': "
            "capabilities must be a collection of strings. "
            "Declare capabilities on the model class, for example frozenset({'generate'})."
        )
    return frozenset(capabilities)


def validate_model_capability(model, role: str, capability: str, method: str = None) -> None:
    capabilities = model_capabilities(model, role)
    if capability not in capabilities:
        raise TypeError(
            f"Cannot bind model '{type(model).__name__}' to role '{role}': "
            f"required capability '{capability}'; declared capabilities: {sorted(capabilities)}. "
            "Select a compatible model in the pipeline, or implement and declare the required capability."
        )
    if method and not callable(getattr(model, method, None)):
        raise TypeError(
            f"Cannot bind model '{type(model).__name__}' to role '{role}': "
            f"capability '{capability}' is declared but {method}() is not callable. "
            "Implement the corresponding model interface before assigning this role."
        )
