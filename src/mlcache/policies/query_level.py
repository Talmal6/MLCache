"""Query-level learned serving policy skeletons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class QueryLevelLearnedPolicy:
    """Placeholder for future query-level calibrated serving."""

    metadata: dict[str, Any] = field(default_factory=dict)

