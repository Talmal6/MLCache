"""Compatibility wrapper for the old flat policy module."""

from mlcache.policies.base import CachePolicy, PolicyAction, PolicyContext, PolicyDecision

__all__ = ["CachePolicy", "PolicyAction", "PolicyContext", "PolicyDecision"]

