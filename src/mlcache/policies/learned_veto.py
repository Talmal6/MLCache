"""Learned veto policy placeholders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class LearnedVetoPolicy:
    """Placeholder for future learned veto serving behavior."""

    metadata: dict[str, Any] = field(default_factory=dict)

