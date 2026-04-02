"""Comprehensive tests for supervisor.state module."""

from __future__ import annotations

import json
import os
import shutil

import pytest

from supervisor.state import (
    FileBackend,
    MemoryBackend,
    RedisBackend,
    State,
    StateBackend,
)


# ====================================================================
# StateBackend ABC
# ====================================================================


class TestStateBackendABC:
    """Verify that StateBackend cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            StateBackend()  # type: ignore[abstract]

    def test_subclass_must_implement_all(self):
        class Incomplete(StateBackend):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


# ====================================================================
# MemoryBackend
# ====================================================================


class TestMemoryBackend:
    def test_save_and_load(self):
        b = MemoryBackend()
        b.save("k", {"a": 1})
        assert b.load("k") == {"a": 1}

    def test_load_missing_key_raises(self):
        b = MemoryBackend()
        with pytest.raises(KeyError):
            b.load("missing")

    def test_exists(self):
        b = MemoryBackend()
        assert b.exists("k") is False
        b.save("k", {})
        assert b.exists("k") is True

    def test_delete(self):
        b = MemoryBackend()
        b.save("k", {"x": 1})
        b.delete("k")
        assert b.exists("k") is False

    def test_delete_missing_raises(self):
        b = MemoryBackend()
        with pytest.raises(KeyError):
            b.delete("nope")

    def test_list_keys(self):
        b = MemoryBackend()
        b.save("a", {})
        b.save("b", {})
        assert sorted(b.list_keys()) == ["a", "b"]

    def test_isolation_deepcopy(self):
        """Stored data must be independent of the original dict."""
        b = MemoryBackend()
        data: dict = {"nested": [1, 2]}
        b.save("k", data)
        data["nested"].append(3)
        assert b.load("k") == {"nested": [1, 2]}

    def test_load_returns_independent_copy(self):
        b = MemoryBackend()
        b.save("k", {"v": [1]})
        loaded = b.load("k")
        loaded["v"].append(2)
        assert b.load("k") == {"v": [1]}


# ====================================================================
# FileBackend
# ====================================================================


_FILE_BACKEND_DIR = os.path.join(
    os.path.dirname(__file__), "_state_test_files"
)


class TestFileBackend:
    @pytest.fixture(autouse=True)
    def _setup_teardown(self):
        """Create and clean up the test directory."""
        os.makedirs(_FILE_BACKEND_DIR, exist_ok=True)
        yield
        shutil.rmtree(_FILE_BACKEND_DIR, ignore_errors=True)

    def test_save_creates_file(self):
        b = FileBackend(_FILE_BACKEND_DIR)
        b.save("mykey", {"val": 42})
        path = os.path.join(_FILE_BACKEND_DIR, "mykey.json")
        assert os.path.isfile(path)
        with open(path, "r", encoding="utf-8") as fh:
            assert json.load(fh) == {"val": 42}

    def test_save_and_load(self):
        b = FileBackend(_FILE_BACKEND_DIR)
        b.save("k", {"a": "b"})
        assert b.load("k") == {"a": "b"}

    def test_load_missing_raises(self):
        b = FileBackend(_FILE_BACKEND_DIR)
        with pytest.raises(KeyError):
            b.load("nope")

    def test_exists(self):
        b = FileBackend(_FILE_BACKEND_DIR)
        assert b.exists("k") is False
        b.save("k", {})
        assert b.exists("k") is True

    def test_delete(self):
        b = FileBackend(_FILE_BACKEND_DIR)
        b.save("k", {})
        b.delete("k")
        assert b.exists("k") is False

    def test_delete_missing_raises(self):
        b = FileBackend(_FILE_BACKEND_DIR)
        with pytest.raises(KeyError):
            b.delete("gone")

    def test_list_keys(self):
        b = FileBackend(_FILE_BACKEND_DIR)
        b.save("alpha", {})
        b.save("beta", {})
        assert sorted(b.list_keys()) == ["alpha", "beta"]

    def test_creates_directory_if_missing(self):
        nested = os.path.join(_FILE_BACKEND_DIR, "sub", "dir")
        b = FileBackend(nested)
        b.save("k", {"ok": True})
        assert b.load("k") == {"ok": True}

    def test_overwrite(self):
        b = FileBackend(_FILE_BACKEND_DIR)
        b.save("k", {"v": 1})
        b.save("k", {"v": 2})
        assert b.load("k") == {"v": 2}


# ====================================================================
# RedisBackend (in-memory stub mode)
# ====================================================================


class TestRedisBackend:
    def test_save_and_load(self):
        b = RedisBackend()
        b.save("k", {"x": 10})
        assert b.load("k") == {"x": 10}

    def test_load_missing_raises(self):
        b = RedisBackend()
        with pytest.raises(KeyError):
            b.load("missing")

    def test_exists(self):
        b = RedisBackend()
        assert b.exists("k") is False
        b.save("k", {})
        assert b.exists("k") is True

    def test_delete(self):
        b = RedisBackend()
        b.save("k", {})
        b.delete("k")
        assert b.exists("k") is False

    def test_delete_missing_raises(self):
        b = RedisBackend()
        with pytest.raises(KeyError):
            b.delete("nope")

    def test_list_keys(self):
        b = RedisBackend()
        b.save("a", {})
        b.save("b", {})
        assert sorted(b.list_keys()) == ["a", "b"]

    def test_prefix_isolation(self):
        b1 = RedisBackend(prefix="ns1:")
        b2 = RedisBackend(prefix="ns2:")
        b1.save("k", {"from": "b1"})
        b2.save("k", {"from": "b2"})
        assert b1.load("k") == {"from": "b1"}
        assert b2.load("k") == {"from": "b2"}

    def test_default_prefix(self):
        b = RedisBackend()
        b.save("k", {})
        assert "state:k" in b._memory

    def test_fallback_without_redis_package(self):
        """Passing a url when redis is not installed falls back gracefully."""
        b = RedisBackend(url="redis://localhost")
        # If redis package is absent, _redis stays None and we use memory.
        # If redis package is present, it may connect; either way it should
        # not blow up during construction.
        assert isinstance(b, RedisBackend)


# ====================================================================
# State - dot-notation access
# ====================================================================


class TestStateDotAccess:
    def test_set_and_get(self):
        s = State()
        s.name = "alice"
        assert s.name == "alice"

    def test_missing_attr_raises(self):
        s = State()
        with pytest.raises(AttributeError):
            _ = s.missing

    def test_del_attr(self):
        s = State()
        s.x = 1
        del s.x
        with pytest.raises(AttributeError):
            _ = s.x

    def test_del_missing_raises(self):
        s = State()
        with pytest.raises(AttributeError):
            del s.no_such

    def test_overwrite(self):
        s = State()
        s.val = 1
        s.val = 2
        assert s.val == 2


# ====================================================================
# State - dict-like interface
# ====================================================================


class TestStateDictLike:
    def test_contains(self):
        s = State()
        s.k = 1
        assert "k" in s
        assert "z" not in s

    def test_iter(self):
        s = State()
        s.a = 1
        s.b = 2
        assert sorted(s) == ["a", "b"]

    def test_len(self):
        s = State()
        assert len(s) == 0
        s.x = 1
        assert len(s) == 1

    def test_repr(self):
        s = State()
        s.x = 1
        assert repr(s) == "State({'x': 1})"

    def test_keys_values_items(self):
        s = State()
        s.a = 1
        s.b = 2
        assert sorted(s.keys()) == ["a", "b"]
        assert sorted(s.values()) == [1, 2]
        assert sorted(s.items()) == [("a", 1), ("b", 2)]

    def test_get_with_default(self):
        s = State()
        assert s.get("missing") is None
        assert s.get("missing", 42) == 42
        s.x = 10
        assert s.get("x") == 10

    def test_update(self):
        s = State()
        s.update({"a": 1, "b": 2})
        assert s.a == 1
        assert s.b == 2

    def test_to_dict(self):
        s = State()
        s.x = [1, 2]
        d = s.to_dict()
        assert d == {"x": [1, 2]}
        d["x"].append(3)
        assert s.x == [1, 2]

    def test_initial_data(self):
        s = State(initial={"a": 1, "b": 2})
        assert s.a == 1
        assert s.b == 2
        assert "a" in s.changes


# ====================================================================
# State - change tracking
# ====================================================================


class TestStateChangeTracking:
    def test_changes_tracked(self):
        s = State()
        s.x = 1
        assert "x" in s.changes

    def test_clear_changes(self):
        s = State()
        s.x = 1
        s.clear_changes()
        assert s.changes == set()

    def test_dirty_flag(self):
        s = State()
        assert s.dirty is False
        s.x = 1
        assert s.dirty is True
        s.clear_changes()
        assert s.dirty is False

    def test_multiple_changes(self):
        s = State()
        s.a = 1
        s.b = 2
        s.a = 10
        assert s.changes == {"a", "b"}


# ====================================================================
# State - checkpoint / restore
# ====================================================================


class TestStateCheckpoint:
    def test_checkpoint_returns_version(self):
        s = State()
        s.x = 1
        v0 = s.checkpoint()
        assert v0 == 0
        s.x = 2
        v1 = s.checkpoint()
        assert v1 == 1

    def test_restore_latest(self):
        s = State()
        s.x = 1
        s.checkpoint()
        s.x = 99
        s.restore(-1)
        assert s.x == 1

    def test_restore_specific_version(self):
        s = State()
        s.x = "v0"
        s.checkpoint()
        s.x = "v1"
        s.checkpoint()
        s.x = "v2"
        s.restore(0)
        assert s.x == "v0"

    def test_restore_clears_changes(self):
        s = State()
        s.x = 1
        s.checkpoint()
        s.x = 2
        assert s.dirty is True
        s.restore(0)
        assert s.dirty is False

    def test_restore_no_checkpoints_raises(self):
        s = State()
        with pytest.raises(IndexError):
            s.restore()

    def test_restore_out_of_range_raises(self):
        s = State()
        s.checkpoint()
        with pytest.raises(IndexError):
            s.restore(5)

    def test_checkpoint_clears_changes(self):
        s = State()
        s.x = 1
        assert s.dirty is True
        s.checkpoint()
        assert s.dirty is False

    def test_checkpoint_count(self):
        s = State()
        assert s.checkpoint_count == 0
        s.checkpoint()
        assert s.checkpoint_count == 1
        s.checkpoint()
        assert s.checkpoint_count == 2

    def test_restore_is_deep_copy(self):
        """Modifying state after restore must not affect the checkpoint."""
        s = State()
        s.items_list = [1, 2, 3]
        s.checkpoint()
        s.items_list.append(4)
        s.restore(0)
        assert s.items_list == [1, 2, 3]

    def test_multiple_restore_same_checkpoint(self):
        s = State()
        s.x = 10
        s.checkpoint()
        s.x = 20
        s.restore(0)
        assert s.x == 10
        s.x = 30
        s.restore(0)
        assert s.x == 10


# ====================================================================
# State - persistence via backend
# ====================================================================


class TestStatePersistence:
    def test_save_without_backend_raises(self):
        s = State()
        with pytest.raises(RuntimeError, match="no backend"):
            s.save()

    def test_load_without_backend_raises(self):
        s = State()
        with pytest.raises(RuntimeError, match="no backend"):
            s.load()

    def test_save_and_load_memory(self):
        backend = MemoryBackend()
        s = State(backend=backend, key="mystate")
        s.x = 42
        s.save()
        s2 = State(backend=backend, key="mystate")
        s2.load()
        assert s2.x == 42

    def test_save_with_key_override(self):
        backend = MemoryBackend()
        s = State(backend=backend)
        s.x = 1
        s.save(key="custom")
        assert backend.exists("custom")

    def test_load_with_key_override(self):
        backend = MemoryBackend()
        backend.save("custom", {"y": 99})
        s = State(backend=backend)
        s.load(key="custom")
        assert s.y == 99

    def test_load_replaces_state(self):
        backend = MemoryBackend()
        backend.save("k", {"a": 1})
        s = State(backend=backend, key="k")
        s.z = "old"
        s.load()
        assert s.a == 1
        assert "z" not in s

    def test_load_clears_changes(self):
        backend = MemoryBackend()
        backend.save("k", {"a": 1})
        s = State(backend=backend, key="k")
        s.dirty_field = True
        assert s.dirty is True
        s.load()
        assert s.dirty is False

    def test_roundtrip_file_backend(self):
        d = os.path.join(os.path.dirname(__file__), "_state_persist_test")
        os.makedirs(d, exist_ok=True)
        try:
            backend = FileBackend(d)
            s = State(backend=backend, key="test")
            s.name = "alice"
            s.score = 100
            s.save()

            s2 = State(backend=backend, key="test")
            s2.load()
            assert s2.name == "alice"
            assert s2.score == 100
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_roundtrip_redis_backend(self):
        backend = RedisBackend()
        s = State(backend=backend, key="rtest")
        s.val = [1, 2, 3]
        s.save()

        s2 = State(backend=backend, key="rtest")
        s2.load()
        assert s2.val == [1, 2, 3]


# ====================================================================
# Integration: checkpoint + persistence
# ====================================================================


class TestStateIntegration:
    def test_checkpoint_then_persist(self):
        backend = MemoryBackend()
        s = State(backend=backend, key="int")
        s.x = 1
        s.checkpoint()
        s.x = 2
        s.save()

        s2 = State(backend=backend, key="int")
        s2.load()
        assert s2.x == 2

    def test_restore_after_load(self):
        backend = MemoryBackend()
        s = State(backend=backend, key="k")
        s.x = "original"
        s.checkpoint()
        s.x = "modified"
        s.save()

        s2 = State(backend=backend, key="k")
        s2.load()
        assert s2.x == "modified"

    def test_save_does_not_mutate_internal_data(self):
        backend = MemoryBackend()
        s = State(backend=backend, key="k")
        s.items_list = [1, 2]
        s.save()
        s.items_list.append(3)
        s2 = State(backend=backend, key="k")
        s2.load()
        assert s2.items_list == [1, 2]

    def test_default_key(self):
        backend = MemoryBackend()
        s = State(backend=backend)
        s.x = 1
        s.save()
        assert backend.exists("default")
