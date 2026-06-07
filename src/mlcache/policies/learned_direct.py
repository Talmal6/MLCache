"""Learned direct-threshold policy placeholders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class LearnedDirectPolicy:
    """Placeholder for the current pair-level direct-threshold strategy."""

    metadata: dict[str, Any] = field(default_factory=dict)

