"""HTTP gateway exposing `SemanticCacheSystem` as an OpenAI-compatible API."""

from mlcache.api.server import API_DEPENDENCIES_ERROR, create_app

__all__ = ["API_DEPENDENCIES_ERROR", "create_app"]
