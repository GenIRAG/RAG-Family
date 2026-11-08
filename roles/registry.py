from typing import Callable, Dict, List, Type


_ROLE_REGISTRY: Dict[str, Type] = {}


def register_role(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        key = str(name).lower()
        if key in _ROLE_REGISTRY:
            raise KeyError(
                f"Duplicate role registration: '{key}' "
                f"(already registered as {_ROLE_REGISTRY[key].__name__})"
            )
        _ROLE_REGISTRY[key] = cls
        return cls

    return decorator


def get_role_class(name: str) -> Type:
    key = str(name).lower()
    if key not in _ROLE_REGISTRY:
        raise ValueError(
            f"Unknown role: '{name}'. Available roles: {registered_roles()}"
        )
    return _ROLE_REGISTRY[key]


def registered_roles() -> List[str]:
    return sorted(_ROLE_REGISTRY)
