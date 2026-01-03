"""Compression utilities for SimpleMem MCP.

Provides gzip compression for large payloads (code files, traces)
sent between MCP thin layer and backend API.
"""

import base64
import gzip
import json
from typing import Any


def compress_payload(data: Any) -> str:
    """Gzip compress and base64 encode data for JSON transport.

    Args:
        data: Any JSON-serializable data (dict, list, str, etc.)

    Returns:
        Base64-encoded gzip-compressed string
    """
    json_bytes = json.dumps(data).encode("utf-8")
    compressed = gzip.compress(json_bytes, compresslevel=6)
    return base64.b64encode(compressed).decode("ascii")


def decompress_payload(data: str) -> Any:
    """Decode base64 and gunzip data.

    Args:
        data: Base64-encoded gzip-compressed string

    Returns:
        Original JSON-deserialized data
    """
    compressed = base64.b64decode(data.encode("ascii"))
    json_bytes = gzip.decompress(compressed)
    return json.loads(json_bytes.decode("utf-8"))


def compress_if_large(data: Any, threshold_bytes: int = 1024) -> tuple[Any, bool]:
    """Compress data only if it exceeds threshold size.

    Args:
        data: Any JSON-serializable data
        threshold_bytes: Minimum size to trigger compression (default 1KB)

    Returns:
        Tuple of (data or compressed_data, was_compressed)
    """
    json_bytes = json.dumps(data).encode("utf-8")
    if len(json_bytes) > threshold_bytes:
        return compress_payload(data), True
    return data, False
