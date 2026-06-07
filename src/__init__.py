"""Compatibility exports for the semantic cache package."""

try:
    from mlcache import *  # noqa: F401,F403
    from mlcache import __all__  # noqa: F401
except ModuleNotFoundError:
    from .mlcache import *  # noqa: F401,F403
    from .mlcache import __all__  # noqa: F401
