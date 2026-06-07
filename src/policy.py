"""Compatibility wrapper for the old flat policy module."""

from mlcache.policies import (
    CachePolicy,
    FileQueryLevelShadowDecisionStore,
    InMemoryQueryLevelShadowDecisionStore,
    PolicyAction,
    PolicyContext,
    PolicyDecision,
    QueryLevelLearnedPolicy,
    QueryLevelPolicyConfig,
    QueryLevelPolicyDecision,
    QueryLevelPolicyMode,
    QueryLevelShadowDecisionStore,
)

__all__ = [
    "CachePolicy",
    "FileQueryLevelShadowDecisionStore",
    "InMemoryQueryLevelShadowDecisionStore",
    "PolicyAction",
    "PolicyContext",
    "PolicyDecision",
    "QueryLevelLearnedPolicy",
    "QueryLevelPolicyConfig",
    "QueryLevelPolicyDecision",
    "QueryLevelPolicyMode",
    "QueryLevelShadowDecisionStore",
]
