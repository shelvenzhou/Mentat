import pytest

from mentat.probes._utils import (
    estimate_tokens,
    should_bypass,
    extract_preview,
    truncate_string,
    format_size,
    safe_read_text,
)


def test_estimate_tokens_basic():
    text = "one two three four"
    assert estimate_tokens(text) == 5


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_should_bypass_small_file():
    assert should_bypass("small file", threshold=10) is True


def test_should_bypass_large_file():
    text = "word " * 40
    assert should_bypass(text, threshold=20) is False


def test_extract_preview_normal():
    text = "\n\n# Title\nFirst meaningful line.\nSecond line"
    assert extract_preview(text) == "First meaningful line."


def test_extract_preview_empty():
    assert extract_preview("\n   \n# Only header\n") is None


def test_extract_preview_truncation():
    text = "This is a very long line that should be truncated at a word boundary for preview extraction"
    preview = extract_preview(text, max_len=30)
    assert preview is not None
    assert preview.endswith("...")
    assert "boundary" not in preview


def test_truncate_string():
    s = "abcdefghijklmnopqrstuvwxyz"
    out = truncate_string(s, head=4, tail=4)
    assert out.startswith("abcd...")
    assert out.endswith("wxyz (Len: 26)")


def test_format_size():
    assert format_size(512) == "512 B"
    assert format_size(2048) == "2.0 KB"
    assert format_size(1024 * 1024) == "1.0 MB"


def test_safe_read_text_utf8(tmp_path):
    p = tmp_path / "utf8.txt"
    p.write_text("hello utf8", encoding="utf-8")
    assert safe_read_text(str(p)) == "hello utf8"


def test_safe_read_text_latin1_fallback(tmp_path):
    p = tmp_path / "latin1.txt"
    p.write_bytes("caf\xe9".encode("latin-1"))
    assert safe_read_text(str(p)) == "café"

