import json
import pytest
from mentat.storage.collections import CollectionStore


@pytest.fixture
def store_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def store(store_dir):
    return CollectionStore(store_dir=store_dir)


class TestCollectionStore:
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
        # Should not raise
        store.remove_doc("nope", "doc-1")

    def test_remove_doc_not_in_collection(self, store):
        store.add_doc("papers", "doc-1")
        # Should not raise
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
