"""Fallback-first policy placeholders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class FallbackFirstPolicy:
    """Placeholder for future fallback-first serving behavior."""

    metadata: dict[str, Any] = field(default_factory=dict)

