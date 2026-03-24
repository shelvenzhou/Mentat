import json
import pytest
from mentat.storage.cache import ContentHashCache, compute_file_hash


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def cache(cache_dir):
    return ContentHashCache(cache_dir=cache_dir)


@pytest.fixture
def sample_file(tmp_path):
    """Create a temp file with known content."""
    f = tmp_path / "sample.txt"
    f.write_text("hello world")
    return str(f)


@pytest.fixture
def sample_file_copy(tmp_path):
    """Another file with identical content (same hash)."""
    f = tmp_path / "copy.txt"
    f.write_text("hello world")
    return str(f)


@pytest.fixture
def different_file(tmp_path):
    """File with different content (different hash)."""
    f = tmp_path / "other.txt"
    f.write_text("different content")
    return str(f)


class TestComputeFileHash:
    def test_deterministic(self, sample_file):
        h1 = compute_file_hash(sample_file)
        h2 = compute_file_hash(sample_file)
        assert h1 == h2

    def test_same_content_same_hash(self, sample_file, sample_file_copy):
        assert compute_file_hash(sample_file) == compute_file_hash(sample_file_copy)

    def test_different_content_different_hash(self, sample_file, different_file):
        assert compute_file_hash(sample_file) != compute_file_hash(different_file)

    def test_returns_hex_string(self, sample_file):
        h = compute_file_hash(sample_file)
        assert len(h) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in h)


class TestContentHashCache:
    def test_empty_cache_returns_none(self, cache, sample_file):
        assert cache.get(sample_file) is None

    def test_put_and_get(self, cache, sample_file):
        cache.put(sample_file, "doc-123")
        assert cache.get(sample_file) == "doc-123"

    def test_same_content_hits_cache(self, cache, sample_file, sample_file_copy):
        cache.put(sample_file, "doc-123")
        assert cache.get(sample_file_copy) == "doc-123"

    def test_different_content_misses(self, cache, sample_file, different_file):
        cache.put(sample_file, "doc-123")
        assert cache.get(different_file) is None

    def test_len(self, cache, sample_file, different_file):
        assert len(cache) == 0
        cache.put(sample_file, "doc-1")
        assert len(cache) == 1
        cache.put(different_file, "doc-2")
        assert len(cache) == 2

    def test_remove(self, cache, sample_file, different_file):
        cache.put(sample_file, "doc-1")
        cache.put(different_file, "doc-2")
        cache.remove("doc-1")
        assert cache.get(sample_file) is None
        assert cache.get(different_file) == "doc-2"
        assert len(cache) == 1

    def test_clear(self, cache, sample_file, different_file):
        cache.put(sample_file, "doc-1")
        cache.put(different_file, "doc-2")
        cache.clear()
        assert len(cache) == 0
        assert cache.get(sample_file) is None

    def test_overwrite_same_file(self, cache, sample_file):
        cache.put(sample_file, "doc-old")
        cache.put(sample_file, "doc-new")
        assert cache.get(sample_file) == "doc-new"
        assert len(cache) == 1

    def test_persistence(self, cache_dir, sample_file):
        c1 = ContentHashCache(cache_dir=cache_dir)
        c1.put(sample_file, "doc-123")

        c2 = ContentHashCache(cache_dir=cache_dir)
        assert c2.get(sample_file) == "doc-123"

    def test_persistence_after_remove(self, cache_dir, sample_file, different_file):
        c1 = ContentHashCache(cache_dir=cache_dir)
        c1.put(sample_file, "doc-1")
        c1.put(different_file, "doc-2")
        c1.remove("doc-1")

        c2 = ContentHashCache(cache_dir=cache_dir)
        assert c2.get(sample_file) is None
        assert c2.get(different_file) == "doc-2"

    def test_corrupted_file_starts_fresh(self, cache_dir):
        path = ContentHashCache(cache_dir=cache_dir)._path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{invalid json")

        cache = ContentHashCache(cache_dir=cache_dir)
        assert len(cache) == 0
