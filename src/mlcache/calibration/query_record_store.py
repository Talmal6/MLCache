"""Stores query-level calibration records."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mlcache.calibration.query_level import QueryCalibrationRecord
from mlcache.calibration.query_records import QueryCalibrationRecordBuilder


class QueryCalibrationRecordStore(ABC):
    """Stores query-level calibration records for future calibration jobs."""

    @abstractmethod
    def add(self, record: QueryCalibrationRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def records(self) -> tuple[QueryCalibrationRecord, ...]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


class InMemoryQueryCalibrationRecordStore(QueryCalibrationRecordStore):
    """Bounded FIFO in-memory query calibration record store."""

    def __init__(self, *, max_records: int = 100_000) -> None:
        if int(max_records) <= 0:
            raise ValueError("max_records must be positive")
        self.max_records = int(max_records)
        self._records: list[QueryCalibrationRecord] = []

    def add(self, record: QueryCalibrationRecord) -> None:
        if len(self._records) >= self.max_records:
            self._records.pop(0)
        self._records.append(QueryCalibrationRecordBuilder.copy_record(record))

    def records(self) -> tuple[QueryCalibrationRecord, ...]:
        return tuple(QueryCalibrationRecordBuilder.copy_record(record) for record in self._records)

    def clear(self) -> None:
        self._records.clear()


__all__ = ["InMemoryQueryCalibrationRecordStore", "QueryCalibrationRecordStore"]
