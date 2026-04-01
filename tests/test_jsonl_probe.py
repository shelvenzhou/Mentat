"""Tests for mentat.probes.jsonl_probe — JSONL file probing."""

import json
import pytest
from mentat.probes.jsonl_probe import JSONLProbe


@pytest.fixture
def probe():
    return JSONLProbe()


@pytest.fixture
def session_jsonl(tmp_path):
    """Create a mock OpenClaw session JSONL file."""
    lines = [
        {"type": "session", "version": 3, "id": "abc-123", "timestamp": "2026-03-28T06:00:00Z"},
        {"type": "model_change", "id": "m1", "timestamp": "2026-03-28T06:00:01Z"},
        {
            "type": "message", "id": "msg1", "timestamp": "2026-03-28T06:00:02Z",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "What is the weather?"}],
            },
        },
        {
            "type": "message", "id": "msg2", "timestamp": "2026-03-28T06:00:03Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "..."},
                    {"type": "text", "text": "It looks sunny today."},
                ],
            },
        },
        {
            "type": "message", "id": "msg3", "timestamp": "2026-03-28T06:00:04Z",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Thanks!"}],
            },
        },
        {
            "type": "message", "id": "msg4", "timestamp": "2026-03-28T06:00:05Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "You're welcome!"}],
            },
        },
    ]
    f = tmp_path / "test-session.jsonl"
    f.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return str(f)


@pytest.fixture
def simple_jsonl(tmp_path):
    """Simple JSONL with flat records."""
    lines = [
        {"name": "alice", "age": 30},
        {"name": "bob", "age": 25},
        {"name": "charlie", "age": 35},
    ]
    f = tmp_path / "simple.jsonl"
    f.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return str(f)


# ── can_handle ──────────────────────────────────────────────────────────


class TestCanHandle:
    def test_jsonl(self, probe):
        assert probe.can_handle("data.jsonl", "") is True

    def test_ndjson(self, probe):
        assert probe.can_handle("data.ndjson", "") is True

    def test_json_rejected(self, probe):
        assert probe.can_handle("data.json", "") is False

    def test_case_insensitive(self, probe):
        assert probe.can_handle("DATA.JSONL", "") is True


# ── Basic probing (no config) ──────────────────────────────────────────


class TestBasicProbe:
    def test_all_records_chunked(self, probe, simple_jsonl):
        result = probe.run(simple_jsonl)
        assert result.file_type == "jsonl"
        assert result.stats["total_records"] == 3
        assert result.stats["filtered_records"] == 3
        assert len(result.chunks) == 3

    def test_chunk_content_is_json(self, probe, simple_jsonl):
        result = probe.run(simple_jsonl)
        # Each chunk should be valid JSON
        for chunk in result.chunks:
            parsed = json.loads(chunk.content)
            assert "name" in parsed

    def test_session_file_all_records(self, probe, session_jsonl):
        result = probe.run(session_jsonl)
        assert result.stats["total_records"] == 6
        assert len(result.chunks) == 6

    def test_schema_from_first_record(self, probe, simple_jsonl):
        result = probe.run(simple_jsonl)
        assert result.structure.schema_tree is not None
        assert "name" in result.structure.schema_tree

    def test_empty_file(self, probe, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        result = probe.run(str(f))
        assert result.stats["total_records"] == 0
        assert len(result.chunks) == 0


# ── With filters ────────────────────────────────────────────────────────


class TestFiltered:
    def test_filter_by_type(self, probe, session_jsonl):
        result = probe.run(session_jsonl, probe_config={
            "filters": [{"field": "type", "op": "eq", "value": "message"}],
        })
        assert result.stats["total_records"] == 6
        assert result.stats["filtered_records"] == 4  # 4 messages

    def test_filter_by_role(self, probe, session_jsonl):
        result = probe.run(session_jsonl, probe_config={
            "filters": [
                {"field": "type", "op": "eq", "value": "message"},
                {"field": "message.role", "op": "in", "value": ["user"]},
            ],
        })
        assert result.stats["filtered_records"] == 2  # 2 user messages

    def test_combined_filters(self, probe, session_jsonl):
        result = probe.run(session_jsonl, probe_config={
            "filters": [
                {"field": "type", "op": "eq", "value": "message"},
                {"field": "message.role", "op": "in", "value": ["user", "assistant"]},
            ],
        })
        assert result.stats["filtered_records"] == 4


# ── Text extraction ─────────────────────────────────────────────────────


class TestTextExtraction:
    def test_text_fields(self, probe, session_jsonl):
        result = probe.run(session_jsonl, probe_config={
            "filters": [{"field": "type", "op": "eq", "value": "message"}],
            "text_fields": ["message.content.text"],
        })
        # All chunks should contain extracted text, not raw JSON
        for chunk in result.chunks:
            assert not chunk.content.startswith("{")

    def test_thinking_blocks_excluded(self, probe, session_jsonl):
        """The 'thinking' content block has no 'text' field, so it's excluded."""
        result = probe.run(session_jsonl, probe_config={
            "filters": [
                {"field": "type", "op": "eq", "value": "message"},
                {"field": "message.role", "op": "eq", "value": "assistant"},
            ],
            "text_fields": ["message.content.text"],
        })
        # First assistant chunk should be "It looks sunny today." not "..."
        assert "sunny" in result.chunks[0].content
        assert "thinking" not in result.chunks[0].content.lower()

    def test_label_field(self, probe, session_jsonl):
        result = probe.run(session_jsonl, probe_config={
            "filters": [{"field": "type", "op": "eq", "value": "message"}],
            "text_fields": ["message.content.text"],
            "label_field": "message.role",
        })
        # First message chunk should have role prefix
        assert result.chunks[0].content.startswith("user:")


# ── Grouping ────────────────────────────────────────────────────────────


class TestGrouping:
    def test_group_size_2(self, probe, session_jsonl):
        result = probe.run(session_jsonl, probe_config={
            "filters": [
                {"field": "type", "op": "eq", "value": "message"},
                {"field": "message.role", "op": "in", "value": ["user", "assistant"]},
            ],
            "text_fields": ["message.content.text"],
            "label_field": "message.role",
            "group_size": 2,
        })
        # 4 messages / 2 = 2 turn chunks
        assert len(result.chunks) == 2
        # Each turn should contain both user and assistant text
        assert "user:" in result.chunks[0].content
        assert "assistant:" in result.chunks[0].content

    def test_group_size_1(self, probe, simple_jsonl):
        result = probe.run(simple_jsonl, probe_config={"group_size": 1})
        assert len(result.chunks) == 3

    def test_group_size_larger_than_records(self, probe, simple_jsonl):
        result = probe.run(simple_jsonl, probe_config={"group_size": 10})
        # All 3 records in one chunk
        assert len(result.chunks) == 1


# ── Malformed input ─────────────────────────────────────────────────────


class TestMalformed:
    def test_invalid_json_lines_skipped(self, probe, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text('{"valid": true}\nnot json\n{"also": "valid"}\n')
        result = probe.run(str(f))
        assert result.stats["total_records"] == 2
        assert result.stats["parse_errors"] == 1
        assert len(result.chunks) == 2

    def test_non_dict_lines_skipped(self, probe, tmp_path):
        f = tmp_path / "arrays.jsonl"
        f.write_text('[1,2,3]\n{"name": "ok"}\n"just a string"\n')
        result = probe.run(str(f))
        assert result.stats["total_records"] == 1
        assert result.stats["parse_errors"] == 2

    def test_single_line_file(self, probe, tmp_path):
        f = tmp_path / "single.jsonl"
        f.write_text('{"key": "value"}\n')
        result = probe.run(str(f))
        assert result.stats["total_records"] == 1
        assert len(result.chunks) == 1
