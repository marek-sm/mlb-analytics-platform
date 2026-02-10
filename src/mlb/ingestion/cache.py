"""Shared TTL-based HTTP response cache for ingestion providers."""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class CacheEntry:
    """Cache entry with TTL."""

    key: str
    payload: bytes
    fetched_at: datetime  # UTC
    ttl_seconds: int

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        now = datetime.now(timezone.utc)
        age_seconds = (now - self.fetched_at).total_seconds()
        return age_seconds >= self.ttl_seconds


class Cache:
    """In-memory TTL-based cache for HTTP responses."""

    def __init__(self):
        """Initialize empty cache."""
        self._store: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[bytes]:
        """
        Get cached value if exists and not expired.

        Args:
            key: Cache key

        Returns:
            Cached payload bytes or None if miss/expired
        """
        entry = self._store.get(key)
        if entry is None:
            return None

        if entry.is_expired():
            # Clean up expired entry
            del self._store[key]
            return None

        return entry.payload

    def set(self, key: str, payload: bytes, ttl_seconds: int) -> None:
        """
        Store value in cache with TTL.

        Args:
            key: Cache key
            payload: Data to cache
            ttl_seconds: Time-to-live in seconds
        """
        entry = CacheEntry(
            key=key,
            payload=payload,
            fetched_at=datetime.now(timezone.utc),
            ttl_seconds=ttl_seconds,
        )
        self._store[key] = entry

    def clear(self) -> None:
        """Clear all cache entries."""
        self._store.clear()

    def prune_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._store.items() if entry.is_expired()
        ]
        for key in expired_keys:
            del self._store[key]
        return len(expired_keys)


# Global cache instance
_cache: Optional[Cache] = None


def get_cache() -> Cache:
    """Get or create the singleton cache instance."""
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache
