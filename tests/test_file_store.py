from mentat.storage.file_store import LocalFileStore


def test_save_and_get_path(tmp_path):
    src = tmp_path / "source.txt"
    src.write_text("hello")

    store = LocalFileStore(storage_dir=str(tmp_path / "store"))
    saved_path = store.save(str(src), "doc1")

    assert store.exists("doc1") is True
    assert store.get_path("doc1").as_posix() == saved_path


def test_save_preserves_suffix(tmp_path):
    src = tmp_path / "data.json"
    src.write_text('{"a": 1}')

    store = LocalFileStore(storage_dir=str(tmp_path / "store"))
    saved_path = store.save(str(src), "doc2")

    assert saved_path.endswith(".json")


def test_exists(tmp_path):
    src = tmp_path / "source.txt"
    src.write_text("hello")

    store = LocalFileStore(storage_dir=str(tmp_path / "store"))
    assert store.exists("doc-x") is False

    store.save(str(src), "doc-x")
    assert store.exists("doc-x") is True


def test_get_size(tmp_path):
    src = tmp_path / "source.txt"
    src.write_text("hello world")

    store = LocalFileStore(storage_dir=str(tmp_path / "store"))
    store.save(str(src), "doc-size")

    assert store.get_size("doc-size") == 11


def test_total_size(tmp_path):
    src1 = tmp_path / "a.txt"
    src1.write_text("aaaa")
    src2 = tmp_path / "b.txt"
    src2.write_text("bbbbbb")

    store = LocalFileStore(storage_dir=str(tmp_path / "store"))
    store.save(str(src1), "doc-a")
    store.save(str(src2), "doc-b")

    assert store.total_size() == 10


def test_get_path_nonexistent(tmp_path):
    store = LocalFileStore(storage_dir=str(tmp_path / "store"))
    path = store.get_path("missing-doc")

    assert path.exists() is False
