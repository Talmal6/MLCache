"""Compatibility wrapper for the old flat policy module."""

from mlcache.policies import (
    CachePolicy,
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
