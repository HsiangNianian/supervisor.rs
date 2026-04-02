"""State management for the supervisor agent framework.

Provides pluggable state backends and a dict-like ``State`` container
with change-tracking, checkpoint/restore, and persistence support.
"""

from __future__ import annotations

import copy
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional


class StateBackend(ABC):
    """Abstract base class for state storage backends.

    Subclass this to implement a custom persistence layer.
    """

    @abstractmethod
    def save(self, key: str, data: dict[str, Any]) -> None:
        """Persist *data* under *key*.

        Args:
            key: Unique identifier for the stored data.
            data: Dictionary payload to persist.
        """

    @abstractmethod
    def load(self, key: str) -> dict[str, Any]:
        """Load previously-stored data for *key*.

        Args:
            key: Identifier used when the data was saved.

        Returns:
            The stored dictionary payload.

        Raises:
            KeyError: If *key* does not exist.
        """

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove *key* and its associated data.

        Args:
            key: Identifier to remove.

        Raises:
            KeyError: If *key* does not exist.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether *key* is present.

        Args:
            key: Identifier to check.

        Returns:
            ``True`` if the key exists, ``False`` otherwise.
        """

    @abstractmethod
    def list_keys(self) -> list[str]:
        """Return all stored keys.

        Returns:
            A list of key strings.
        """


# ------------------------------------------------------------------
# Concrete backends
# ------------------------------------------------------------------


class MemoryBackend(StateBackend):
    """In-memory dict-based state backend.

    Useful for testing or ephemeral state that does not need to survive
    a process restart.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def save(self, key: str, data: dict[str, Any]) -> None:
        """Persist *data* in memory."""
        self._store[key] = copy.deepcopy(data)

    def load(self, key: str) -> dict[str, Any]:
        """Load data from memory.

        Raises:
            KeyError: If *key* does not exist.
        """
        if key not in self._store:
            raise KeyError(key)
        return copy.deepcopy(self._store[key])

    def delete(self, key: str) -> None:
        """Delete *key* from memory.

        Raises:
            KeyError: If *key* does not exist.
        """
        if key not in self._store:
            raise KeyError(key)
        del self._store[key]

    def exists(self, key: str) -> bool:
        """Check existence in memory."""
        return key in self._store

    def list_keys(self) -> list[str]:
        """List all keys stored in memory."""
        return list(self._store.keys())


class FileBackend(StateBackend):
    """JSON file-based state backend.

    Each key is stored as a separate JSON file inside *directory*.

    Args:
        directory: Filesystem path for the storage directory.
            Created automatically if it does not exist.
    """

    def __init__(self, directory: str) -> None:
        self._directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path_for(self, key: str) -> str:
        """Return the file path for *key*."""
        safe_name = key.replace(os.sep, "_") + ".json"
        return os.path.join(self._directory, safe_name)

    def save(self, key: str, data: dict[str, Any]) -> None:
        """Persist *data* as a JSON file."""
        path = self._path_for(key)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)

    def load(self, key: str) -> dict[str, Any]:
        """Load data from a JSON file.

        Raises:
            KeyError: If the file for *key* does not exist.
        """
        path = self._path_for(key)
        if not os.path.isfile(path):
            raise KeyError(key)
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def delete(self, key: str) -> None:
        """Remove the JSON file for *key*.

        Raises:
            KeyError: If the file does not exist.
        """
        path = self._path_for(key)
        if not os.path.isfile(path):
            raise KeyError(key)
        os.remove(path)

    def exists(self, key: str) -> bool:
        """Check whether a JSON file exists for *key*."""
        return os.path.isfile(self._path_for(key))

    def list_keys(self) -> list[str]:
        """List all keys by scanning the storage directory."""
        keys: list[str] = []
        for name in os.listdir(self._directory):
            if name.endswith(".json"):
                keys.append(name[: -len(".json")])
        return keys


class RedisBackend(StateBackend):
    """Redis-compatible state backend.

    When the ``redis`` package is available and a *url* is provided,
    real Redis is used.  Otherwise an in-memory dictionary emulates
    Redis behaviour, which is useful for testing without infrastructure.

    Args:
        url: Optional Redis connection URL (e.g. ``redis://localhost``).
            When ``None``, the backend falls back to in-memory storage.
        prefix: Key prefix applied to all operations so that multiple
            ``RedisBackend`` instances can share the same Redis database
            without collisions.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        prefix: str = "state:",
    ) -> None:
        self._prefix = prefix
        self._redis: Any = None
        self._memory: dict[str, str] = {}

        if url is not None:
            try:
                import redis as _redis_mod  # type: ignore[import-untyped]

                self._redis = _redis_mod.Redis.from_url(url)
            except ImportError:
                pass

    def _prefixed(self, key: str) -> str:
        return self._prefix + key

    # -- helpers for the two modes ------------------------------------

    def _mem_set(self, key: str, raw: str) -> None:
        self._memory[key] = raw

    def _mem_get(self, key: str) -> Optional[str]:
        return self._memory.get(key)

    def _mem_del(self, key: str) -> bool:
        if key in self._memory:
            del self._memory[key]
            return True
        return False

    def _mem_exists(self, key: str) -> bool:
        return key in self._memory

    def _mem_keys(self, pattern: str) -> list[str]:
        return [k for k in self._memory if k.startswith(pattern)]

    # -- public API ---------------------------------------------------

    def save(self, key: str, data: dict[str, Any]) -> None:
        """Persist *data* as a JSON string in Redis (or memory)."""
        pkey = self._prefixed(key)
        raw = json.dumps(data, ensure_ascii=False)
        if self._redis is not None:
            self._redis.set(pkey, raw)
        else:
            self._mem_set(pkey, raw)

    def load(self, key: str) -> dict[str, Any]:
        """Load data from Redis (or memory).

        Raises:
            KeyError: If *key* does not exist.
        """
        pkey = self._prefixed(key)
        if self._redis is not None:
            raw = self._redis.get(pkey)
            if raw is None:
                raise KeyError(key)
            return json.loads(raw)
        else:
            raw_mem = self._mem_get(pkey)
            if raw_mem is None:
                raise KeyError(key)
            return json.loads(raw_mem)

    def delete(self, key: str) -> None:
        """Remove *key* from Redis (or memory).

        Raises:
            KeyError: If *key* does not exist.
        """
        pkey = self._prefixed(key)
        if self._redis is not None:
            deleted = self._redis.delete(pkey)
            if not deleted:
                raise KeyError(key)
        else:
            if not self._mem_del(pkey):
                raise KeyError(key)

    def exists(self, key: str) -> bool:
        """Check existence in Redis (or memory)."""
        pkey = self._prefixed(key)
        if self._redis is not None:
            return bool(self._redis.exists(pkey))
        return self._mem_exists(pkey)

    def list_keys(self) -> list[str]:
        """List all keys under the configured prefix."""
        if self._redis is not None:
            raw_keys = self._redis.keys(self._prefix + "*")
            prefix_len = len(self._prefix)
            return [k.decode("utf-8")[prefix_len:] for k in raw_keys]
        else:
            prefix_len = len(self._prefix)
            return [k[prefix_len:] for k in self._mem_keys(self._prefix)]


# ------------------------------------------------------------------
# State container
# ------------------------------------------------------------------

# Attribute names that live on the object itself, not in ``_data``.
_INTERNAL_ATTRS = frozenset(
    {"_data", "_changes", "_checkpoints", "_backend", "_key"}
)


class State:
    """Dict-like container with change-tracking and checkpoint/restore.

    ``State`` stores arbitrary attributes and tracks which ones have
    been modified since the last checkpoint.  Snapshots can be taken
    via :meth:`checkpoint` and rolled-back via :meth:`restore`.  When a
    :class:`StateBackend` is attached, :meth:`save` and :meth:`load`
    persist the state externally.

    Args:
        backend: Optional persistence backend.
        key: Key used for ``save``/``load`` when interacting with the
            backend.  Defaults to ``"default"``.
        initial: Optional mapping of initial key/value pairs.

    Examples:
        >>> s = State()
        >>> s.counter = 0
        >>> s.counter += 1
        >>> s.checkpoint()
        0
        >>> s.counter = 99
        >>> s.restore(0)
        >>> s.counter
        1
    """

    def __init__(
        self,
        backend: Optional[StateBackend] = None,
        key: str = "default",
        initial: Optional[dict[str, Any]] = None,
    ) -> None:
        object.__setattr__(self, "_data", {})
        object.__setattr__(self, "_changes", set())
        object.__setattr__(self, "_checkpoints", [])
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_key", key)
        if initial:
            self._data.update(initial)
            self._changes.update(initial.keys())

    # -- attribute access ---------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Return the value of a state attribute.

        Raises:
            AttributeError: If the attribute has not been set.
        """
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Set a state attribute and record the change."""
        if name in _INTERNAL_ATTRS:
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value
            self._changes.add(name)

    def __delattr__(self, name: str) -> None:
        """Delete a state attribute.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        if name not in self._data:
            raise AttributeError(name)
        del self._data[name]
        self._changes.discard(name)

    # -- dict-like helpers --------------------------------------------

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"State({self._data!r})"

    def keys(self):
        """Return state keys."""
        return self._data.keys()

    def values(self):
        """Return state values."""
        return self._data.values()

    def items(self):
        """Return state items."""
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Return value for *key*, or *default* if missing."""
        return self._data.get(key, default)

    def update(self, mapping: dict[str, Any]) -> None:
        """Merge *mapping* into the state, tracking changes."""
        self._data.update(mapping)
        self._changes.update(mapping.keys())

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the state data as a plain dict."""
        return copy.deepcopy(self._data)

    # -- change tracking ----------------------------------------------

    @property
    def changes(self) -> set[str]:
        """Return the set of attribute names changed since last checkpoint."""
        return set(self._changes)

    def clear_changes(self) -> None:
        """Reset the change tracker."""
        self._changes.clear()

    @property
    def dirty(self) -> bool:
        """``True`` if any attributes have changed since the last checkpoint."""
        return bool(self._changes)

    # -- checkpoint / restore -----------------------------------------

    def checkpoint(self) -> int:
        """Take a snapshot of the current state.

        Returns:
            The version index of this checkpoint (zero-based).
        """
        self._checkpoints.append(copy.deepcopy(self._data))
        self._changes.clear()
        return len(self._checkpoints) - 1

    def restore(self, version: int = -1) -> None:
        """Restore the state from a previous checkpoint.

        Args:
            version: Checkpoint index to restore.  Negative indices are
                supported (e.g. ``-1`` for the latest checkpoint).

        Raises:
            IndexError: If *version* is out of range.
        """
        if not self._checkpoints:
            raise IndexError("no checkpoints available")
        snapshot = self._checkpoints[version]
        object.__setattr__(self, "_data", copy.deepcopy(snapshot))
        self._changes.clear()

    @property
    def checkpoint_count(self) -> int:
        """Number of checkpoints currently stored."""
        return len(self._checkpoints)

    # -- persistence --------------------------------------------------

    def save(self, key: Optional[str] = None) -> None:
        """Persist the current state via the configured backend.

        Args:
            key: Override the default key for this operation.

        Raises:
            RuntimeError: If no backend is configured.
        """
        if self._backend is None:
            raise RuntimeError("no backend configured")
        self._backend.save(key or self._key, copy.deepcopy(self._data))

    def load(self, key: Optional[str] = None) -> None:
        """Load state from the configured backend, replacing current data.

        Args:
            key: Override the default key for this operation.

        Raises:
            RuntimeError: If no backend is configured.
        """
        if self._backend is None:
            raise RuntimeError("no backend configured")
        loaded = self._backend.load(key or self._key)
        object.__setattr__(self, "_data", loaded)
        self._changes.clear()
