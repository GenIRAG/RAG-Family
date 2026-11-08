from RoleRAG.roles.registry import (
    get_role_class,
    register_role,
    registered_roles,
)
from RoleRAG.roles.reasoner import ReasonerRoleAdapter
from RoleRAG.roles.reranker import RerankerRoleAdapter
from RoleRAG.roles.retriever import RetrieverRoleAdapter
from RoleRAG.roles.rewriter import (
    HyDERewriter,
    RewriterRoleAdapter,
    register_rewrite_method,
    registered_rewrite_methods,
)
from RoleRAG.roles.refiner import RefinerRoleAdapter
from RoleRAG.roles.role_base import (
    BaseGeneratorContract,
    BaseRerankerContract,
    BaseRetrieverContract,
    BaseRewriterContract,
    BaseRefinerContract,
)
from RoleRAG.roles.utils import (
    _configure_model,
    assist_role,
    get_method_role_config,
    get_model_defaults,
)

__all__ = [
    "BaseGeneratorContract",
    "BaseRerankerContract",
    "BaseRetrieverContract",
    "BaseRewriterContract",
    "BaseRefinerContract",
    "ReasonerRoleAdapter",
    "HyDERewriter",
    "RerankerRoleAdapter",
    "RetrieverRoleAdapter",
    "RewriterRoleAdapter",
    "RefinerRoleAdapter",
    "get_role_class",
    "register_role",
    "registered_roles",
    "register_rewrite_method",
    "registered_rewrite_methods",
    "_configure_model",
    "assist_role",
    "get_method_role_config",
    "get_model_defaults",
]
