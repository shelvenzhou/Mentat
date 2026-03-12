import json
from datetime import datetime, timedelta, timezone

import pytest

from mentat.storage.collections import CollectionStore


@pytest.fixture
def store_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def store(store_dir):
    return CollectionStore(store_dir=store_dir)


# ── Basic doc membership ────────────────────────────────────────────────


class TestCollectionStoreBasic:
    def test_empty_initially(self, store):
        assert store.list_collections() == []

    def test_add_doc_creates_collection(self, store):
        store.add_doc("papers", "doc-1")
        assert "papers" in store.list_collections()
        assert store.get_doc_ids("papers") == {"doc-1"}

    def test_add_multiple_docs(self, store):
        store.add_doc("papers", "doc-1")
        store.add_doc("papers", "doc-2")
        store.add_doc("papers", "doc-3")
        assert store.get_doc_ids("papers") == {"doc-1", "doc-2", "doc-3"}

    def test_add_doc_idempotent(self, store):
        store.add_doc("papers", "doc-1")
        store.add_doc("papers", "doc-1")
        assert store.get_doc_ids("papers") == {"doc-1"}

    def test_multiple_collections(self, store):
        store.add_doc("papers", "doc-1")
        store.add_doc("notes", "doc-2")
        assert set(store.list_collections()) == {"papers", "notes"}

    def test_doc_in_multiple_collections(self, store):
        store.add_doc("papers", "doc-1")
        store.add_doc("physics", "doc-1")
        store.add_doc("physics", "doc-2")
        assert "doc-1" in store.get_doc_ids("papers")
        assert "doc-1" in store.get_doc_ids("physics")

    def test_get_doc_ids_nonexistent(self, store):
        assert store.get_doc_ids("nope") == set()

    def test_remove_doc(self, store):
        store.add_doc("papers", "doc-1")
        store.add_doc("papers", "doc-2")
        store.remove_doc("papers", "doc-1")
        assert store.get_doc_ids("papers") == {"doc-2"}

    def test_remove_doc_nonexistent_collection(self, store):
        store.remove_doc("nope", "doc-1")

    def test_remove_doc_not_in_collection(self, store):
        store.add_doc("papers", "doc-1")
        store.remove_doc("papers", "doc-999")
        assert store.get_doc_ids("papers") == {"doc-1"}

    def test_delete_collection(self, store):
        store.add_doc("papers", "doc-1")
        assert store.delete_collection("papers") is True
        assert store.list_collections() == []
        assert store.get_doc_ids("papers") == set()

    def test_delete_nonexistent_collection(self, store):
        assert store.delete_collection("nope") is False

    def test_delete_leaves_other_collections(self, store):
        store.add_doc("papers", "doc-1")
        store.add_doc("notes", "doc-2")
        store.delete_collection("papers")
        assert store.list_collections() == ["notes"]
        assert store.get_doc_ids("notes") == {"doc-2"}

    def test_doc_collections(self, store):
        store.add_doc("papers", "doc-1")
        store.add_doc("physics", "doc-1")
        store.add_doc("notes", "doc-2")
        assert set(store.doc_collections("doc-1")) == {"papers", "physics"}
        assert store.doc_collections("doc-2") == ["notes"]
        assert store.doc_collections("doc-999") == []


# ── Persistence ─────────────────────────────────────────────────────────


class TestCollectionStorePersistence:
    def test_persistence(self, store_dir):
        s1 = CollectionStore(store_dir=store_dir)
        s1.add_doc("papers", "doc-1")
        s1.add_doc("papers", "doc-2")
        s1.add_doc("notes", "doc-3")

        s2 = CollectionStore(store_dir=store_dir)
        assert set(s2.list_collections()) == {"papers", "notes"}
        assert s2.get_doc_ids("papers") == {"doc-1", "doc-2"}
        assert s2.get_doc_ids("notes") == {"doc-3"}

    def test_persistence_after_remove(self, store_dir):
        s1 = CollectionStore(store_dir=store_dir)
        s1.add_doc("papers", "doc-1")
        s1.add_doc("papers", "doc-2")
        s1.remove_doc("papers", "doc-1")

        s2 = CollectionStore(store_dir=store_dir)
        assert s2.get_doc_ids("papers") == {"doc-2"}

    def test_persistence_after_delete(self, store_dir):
        s1 = CollectionStore(store_dir=store_dir)
        s1.add_doc("papers", "doc-1")
        s1.delete_collection("papers")

        s2 = CollectionStore(store_dir=store_dir)
        assert s2.list_collections() == []

    def test_corrupted_file_starts_fresh(self, store_dir):
        path = CollectionStore(store_dir=store_dir)._path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json{{{")

        store = CollectionStore(store_dir=store_dir)
        assert store.list_collections() == []


# ── Create / metadata / watch config ───────────────────────────────────


class TestCollectionStoreCreate:
    def test_create_new_collection(self, store):
        rec = store.create("memory", metadata={"type": "system"})
        assert rec["metadata"] == {"type": "system"}
        assert rec["doc_ids"] == []
        assert "memory" in store.list_collections()

    def test_create_with_full_config(self, store):
        rec = store.create(
            "code:proj",
            metadata={"type": "project"},
            watch_paths=["/src"],
            watch_ignore=["node_modules", "*.lock"],
            auto_add_sources=["openclaw:Read"],
        )
        assert rec["watch_paths"] == ["/src"]
        assert rec["watch_ignore"] == ["node_modules", "*.lock"]
        assert rec["auto_add_sources"] == ["openclaw:Read"]
        assert "created_at" in rec

    def test_create_updates_existing(self, store):
        store.create("memory", metadata={"type": "system"})
        store.add_doc("memory", "doc-1")
        rec = store.create("memory", metadata={"type": "system", "v": 2})
        assert rec["metadata"] == {"type": "system", "v": 2}
        assert "doc-1" in rec["doc_ids"]

    def test_create_partial_update(self, store):
        store.create(
            "memory",
            metadata={"type": "system"},
            watch_paths=["/old"],
            auto_add_sources=["src:*"],
        )
        rec = store.create("memory", watch_paths=["/new"])
        assert rec["watch_paths"] == ["/new"]
        assert rec["metadata"] == {"type": "system"}
        assert rec["auto_add_sources"] == ["src:*"]

    def test_get_returns_none_for_missing(self, store):
        assert store.get("nope") is None

    def test_get_returns_full_record(self, store):
        store.create("memory", metadata={"type": "system"})
        store.add_doc("memory", "doc-1")
        rec = store.get("memory")
        assert rec is not None
        assert "doc-1" in rec["doc_ids"]
        assert rec["metadata"] == {"type": "system"}


class TestCollectionStoreMetadata:
    def test_get_metadata(self, store):
        store.create("memory", metadata={"type": "system"})
        assert store.get_metadata("memory") == {"type": "system"}

    def test_get_metadata_missing(self, store):
        assert store.get_metadata("nope") is None

    def test_update_metadata_merges(self, store):
        store.create("memory", metadata={"type": "system", "v": 1})
        store.update_metadata("memory", {"v": 2, "extra": True})
        assert store.get_metadata("memory") == {"type": "system", "v": 2, "extra": True}

    def test_update_metadata_noop_for_missing(self, store):
        store.update_metadata("nope", {"v": 1})


class TestCollectionStoreWatchConfig:
    def test_get_watch_config(self, store):
        store.create("code", watch_paths=["/src"], watch_ignore=["*.lock"])
        cfg = store.get_watch_config("code")
        assert cfg == {"watch_paths": ["/src"], "watch_ignore": ["*.lock"]}

    def test_get_watch_config_missing(self, store):
        assert store.get_watch_config("nope") is None

    def test_get_all_watch_configs(self, store):
        store.create("code", watch_paths=["/src"])
        store.create("memory", watch_paths=["/mem"])
        store.create("empty")
        configs = store.get_all_watch_configs()
        assert set(configs.keys()) == {"code", "memory"}

    def test_get_all_watch_configs_empty(self, store):
        assert store.get_all_watch_configs() == {}


# ── Auto-routing ────────────────────────────────────────────────────────


class TestCollectionStoreAutoRouting:
    def test_exact_match(self, store):
        store.create("files", auto_add_sources=["web_fetch"])
        assert store.get_auto_route_targets("web_fetch") == ["files"]

    def test_glob_match(self, store):
        store.create("files", auto_add_sources=["openclaw:*"])
        assert store.get_auto_route_targets("openclaw:Read") == ["files"]
        assert store.get_auto_route_targets("openclaw:WebFetch") == ["files"]
        assert store.get_auto_route_targets("web_fetch") == []

    def test_multiple_collections_match(self, store):
        store.create("files", auto_add_sources=["openclaw:*"])
        store.create("memory", auto_add_sources=["openclaw:memory"])
        targets = store.get_auto_route_targets("openclaw:memory")
        assert set(targets) == {"files", "memory"}

    def test_no_match(self, store):
        store.create("files", auto_add_sources=["openclaw:*"])
        assert store.get_auto_route_targets("composio:gmail") == []

    def test_empty_source(self, store):
        store.create("files", auto_add_sources=["*"])
        assert store.get_auto_route_targets("") == []

    def test_no_auto_add_sources(self, store):
        store.create("empty")
        assert store.get_auto_route_targets("anything") == []


# ── Garbage collection ──────────────────────────────────────────────────


class TestCollectionStoreGC:
    def test_gc_removes_expired(self, store):
        store.create("session", metadata={"ttl": 10})
        rec = store.get("session")
        rec["created_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=20)
        ).isoformat()
        store._save()

        deleted = store.gc()
        assert deleted == ["session"]
        assert store.list_collections() == []

    def test_gc_keeps_unexpired(self, store):
        store.create("session", metadata={"ttl": 3600})
        deleted = store.gc()
        assert deleted == []
        assert "session" in store.list_collections()

    def test_gc_ignores_no_ttl(self, store):
        store.create("permanent", metadata={"type": "system"})
        deleted = store.gc()
        assert deleted == []
        assert "permanent" in store.list_collections()

    def test_gc_mixed(self, store):
        store.create("keep", metadata={"type": "system"})
        store.create("expire", metadata={"ttl": 1})
        rec = store.get("expire")
        rec["created_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        ).isoformat()
        store._save()

        deleted = store.gc()
        assert deleted == ["expire"]
        assert store.list_collections() == ["keep"]

    def test_gc_with_explicit_now(self, store):
        store.create("session", metadata={"ttl": 100})
        rec = store.get("session")
        created = datetime.fromisoformat(rec["created_at"]).timestamp()
        assert store.gc(now=created + 50) == []
        assert store.gc(now=created + 200) == ["session"]
