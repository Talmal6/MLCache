"""Local persistence helpers."""

from mlcache.persistence.json_utils import (
    atomic_write_json,
    decode_cache_metadata,
    decode_query_level_policy_decision,
    decode_query_record,
    encode_cache_metadata,
    encode_query_level_policy_decision,
    encode_query_record,
    json_safe,
    read_json_or_default,
)

__all__ = [
    "atomic_write_json",
    "decode_cache_metadata",
    "decode_query_level_policy_decision",
    "decode_query_record",
    "encode_cache_metadata",
    "encode_query_level_policy_decision",
    "encode_query_record",
    "json_safe",
    "read_json_or_default",
]
