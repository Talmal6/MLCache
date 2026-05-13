"""Online updater interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from online_properties.types import FeedbackEvent, OnlineBatch


class OnlineUpdater(ABC):
    """Consumes feedback and updates oracle-owned semantic state."""

    @abstractmethod
    def update(self, event: FeedbackEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_batch(self, batch: OnlineBatch) -> None:
        raise NotImplementedError
