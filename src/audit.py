"""Compatibility wrapper for the old flat audit module."""

from mlcache.observability.audit import AuditEvent, AuditLogger

__all__ = ["AuditEvent", "AuditLogger"]

