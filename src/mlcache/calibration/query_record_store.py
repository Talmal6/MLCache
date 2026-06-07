"""Stores query-level calibration records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from mlcache.calibration.query_level import QueryCalibrationRecord
from mlcache.calibration.query_records import QueryCalibrationRecordBuilder
from mlcache.persistence import atomic_write_json, decode_query_record, encode_query_record, read_json_or_default


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


class FileQueryCalibrationRecordStore(InMemoryQueryCalibrationRecordStore):
    """Bounded FIFO JSON-backed query calibration record store."""

    def __init__(self, path: str | Path, *, max_records: int = 100_000) -> None:
        self.path = Path(path)
        super().__init__(max_records=max_records)
        data = read_json_or_default(self.path, {"records": []})
        decoded = tuple(decode_query_record(item) for item in data.get("records", ()))
        self._records = [
            QueryCalibrationRecordBuilder.copy_record(record)
            for record in decoded[-self.max_records :]
        ]

    def add(self, record: QueryCalibrationRecord) -> None:
        super().add(record)
        self._persist()

    def clear(self) -> None:
        super().clear()
        self._persist()

    def _persist(self) -> None:
        atomic_write_json(
            self.path,
            {
                "format": "mlcache.file_query_calibration_record_store.v1",
                "max_records": self.max_records,
                "records": [encode_query_record(record) for record in self._records],
            },
        )


__all__ = [
    "FileQueryCalibrationRecordStore",
    "InMemoryQueryCalibrationRecordStore",
    "QueryCalibrationRecordStore",
]
