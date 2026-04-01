"""Tests for mentat.probes.record_filter — in-memory record filtering."""

import pytest
from mentat.probes.record_filter import (
    RecordFilter,
    RecordFilterSet,
    _resolve_dot_path,
    _MISSING,
)


# ── dot-path resolution ────────────────────────────────────────────────


class TestResolveDotPath:
    def test_simple_key(self):
        assert _resolve_dot_path({"type": "message"}, "type") == "message"

    def test_nested(self):
        obj = {"message": {"role": "user"}}
        assert _resolve_dot_path(obj, "message.role") == "user"

    def test_deep_nested(self):
        obj = {"a": {"b": {"c": 42}}}
        assert _resolve_dot_path(obj, "a.b.c") == 42

    def test_missing_key(self):
        assert _resolve_dot_path({"a": 1}, "b") is _MISSING

    def test_missing_nested(self):
        assert _resolve_dot_path({"a": {"b": 1}}, "a.c") is _MISSING

    def test_list_traversal(self):
        obj = {"items": [{"name": "a"}, {"name": "b"}]}
        assert _resolve_dot_path(obj, "items.name") == ["a", "b"]

    def test_list_traversal_with_missing(self):
        obj = {"items": [{"name": "a"}, {"other": "b"}]}
        result = _resolve_dot_path(obj, "items.name")
        assert result == ["a"]

    def test_list_traversal_empty(self):
        obj = {"items": [{"other": "a"}]}
        assert _resolve_dot_path(obj, "items.name") is _MISSING

    def test_content_block_pattern(self):
        """Simulates OpenClaw message.content.text extraction."""
        obj = {
            "message": {
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image", "url": "..."},
                    {"type": "text", "text": "world"},
                ]
            }
        }
        result = _resolve_dot_path(obj, "message.content.text")
        assert result == ["hello", "world"]

    def test_non_dict_value(self):
        assert _resolve_dot_path({"a": 42}, "a.b") is _MISSING

    def test_empty_dict(self):
        assert _resolve_dot_path({}, "a") is _MISSING

    def test_dunder_rejected(self):
        with pytest.raises(ValueError, match="Python internal"):
            _resolve_dot_path({"__class__": "bad"}, "__class__")

    def test_dunder_in_nested_path(self):
        with pytest.raises(ValueError, match="Python internal"):
            _resolve_dot_path({"a": {}}, "a.__init__")

    def test_empty_segment_rejected(self):
        with pytest.raises(ValueError, match="empty segment"):
            _resolve_dot_path({"a": 1}, "a..b")

    def test_list_flattening(self):
        """Lists within list-of-dicts traversal are flattened."""
        obj = {"items": [{"tags": ["a", "b"]}, {"tags": ["c"]}]}
        result = _resolve_dot_path(obj, "items.tags")
        assert result == ["a", "b", "c"]


# ── RecordFilter construction ──────────────────────────────────────────


class TestRecordFilterConstruction:
    def test_valid_ops(self):
        for op in ("eq", "neq", "in", "nin", "gt", "gte", "lt", "lte", "exists", "regex", "contains"):
            val = ["a"] if op in ("in", "nin") else "x"
            RecordFilter(field="f", op=op, value=val)

    def test_invalid_op(self):
        with pytest.raises(ValueError, match="Invalid filter op"):
            RecordFilter(field="f", op="nope", value=1)

    def test_in_requires_list(self):
        with pytest.raises(ValueError, match="requires a list"):
            RecordFilter(field="f", op="in", value="notalist")

    def test_dunder_field_rejected(self):
        with pytest.raises(ValueError, match="Python internal"):
            RecordFilter(field="__class__", op="eq", value="x")


# ── RecordFilter matching ──────────────────────────────────────────────


class TestRecordFilterMatch:
    def test_eq(self):
        f = RecordFilter(field="type", op="eq", value="message")
        assert f.match({"type": "message"}) is True
        assert f.match({"type": "other"}) is False

    def test_neq(self):
        f = RecordFilter(field="type", op="neq", value="system")
        assert f.match({"type": "message"}) is True
        assert f.match({"type": "system"}) is False

    def test_in(self):
        f = RecordFilter(field="role", op="in", value=["user", "assistant"])
        assert f.match({"role": "user"}) is True
        assert f.match({"role": "system"}) is False

    def test_nin(self):
        f = RecordFilter(field="role", op="nin", value=["system"])
        assert f.match({"role": "user"}) is True
        assert f.match({"role": "system"}) is False

    def test_gt_lt(self):
        f_gt = RecordFilter(field="count", op="gt", value=5)
        f_lt = RecordFilter(field="count", op="lt", value=5)
        assert f_gt.match({"count": 10}) is True
        assert f_gt.match({"count": 3}) is False
        assert f_lt.match({"count": 3}) is True

    def test_exists_true(self):
        f = RecordFilter(field="data", op="exists", value=True)
        assert f.match({"data": "x"}) is True
        assert f.match({"other": "x"}) is False

    def test_exists_false(self):
        f = RecordFilter(field="data", op="exists", value=False)
        assert f.match({"data": "x"}) is False
        assert f.match({"other": "x"}) is True

    def test_regex(self):
        f = RecordFilter(field="msg", op="regex", value=r"^hello")
        assert f.match({"msg": "hello world"}) is True
        assert f.match({"msg": "world hello"}) is False

    def test_contains_str(self):
        f = RecordFilter(field="msg", op="contains", value="ell")
        assert f.match({"msg": "hello"}) is True
        assert f.match({"msg": "world"}) is False

    def test_contains_list(self):
        f = RecordFilter(field="tags", op="contains", value="a")
        assert f.match({"tags": ["a", "b"]}) is True
        assert f.match({"tags": ["c"]}) is False

    def test_nested_field(self):
        f = RecordFilter(field="message.role", op="eq", value="user")
        assert f.match({"message": {"role": "user"}}) is True
        assert f.match({"message": {"role": "system"}}) is False

    def test_missing_field(self):
        f = RecordFilter(field="missing", op="eq", value="x")
        assert f.match({"other": "x"}) is False

    def test_list_traversal_any_match(self):
        """When dot-path resolves to list, ANY element match suffices."""
        f = RecordFilter(field="items.status", op="eq", value="active")
        obj = {"items": [{"status": "inactive"}, {"status": "active"}]}
        assert f.match(obj) is True


# ── RecordFilterSet ────────────────────────────────────────────────────


class TestRecordFilterSet:
    def test_empty_matches_all(self):
        fs = RecordFilterSet()
        assert fs.match({"anything": True}) is True

    def test_and_logic(self):
        fs = RecordFilterSet(
            filters=[
                RecordFilter(field="type", op="eq", value="message"),
                RecordFilter(field="role", op="in", value=["user", "assistant"]),
            ],
            logic="AND",
        )
        assert fs.match({"type": "message", "role": "user"}) is True
        assert fs.match({"type": "message", "role": "system"}) is False
        assert fs.match({"type": "other", "role": "user"}) is False

    def test_or_logic(self):
        fs = RecordFilterSet(
            filters=[
                RecordFilter(field="type", op="eq", value="message"),
                RecordFilter(field="type", op="eq", value="event"),
            ],
            logic="OR",
        )
        assert fs.match({"type": "message"}) is True
        assert fs.match({"type": "event"}) is True
        assert fs.match({"type": "other"}) is False

    def test_from_dicts(self):
        fs = RecordFilterSet.from_dicts([
            {"field": "type", "op": "eq", "value": "message"},
            {"field": "role", "op": "in", "value": ["user"]},
        ])
        assert fs.match({"type": "message", "role": "user"}) is True
        assert fs.match({"type": "message", "role": "bot"}) is False

    def test_invalid_logic(self):
        with pytest.raises(ValueError, match="Invalid logic"):
            RecordFilterSet(logic="XOR")
