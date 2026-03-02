"""Tests for the Skill Integration Layer."""

from mentat.skill import get_tool_schemas, get_system_prompt, export_skill


def test_tool_schemas_structure():
    schemas = get_tool_schemas()
    assert isinstance(schemas, list)
    assert len(schemas) == 6

    names = {s["function"]["name"] for s in schemas}
    assert names == {
        "search_memory",
        "read_segment",
        "get_summary",
        "index_memory",
        "memory_status",
        "get_doc_meta",
    }


def test_tool_schemas_valid_format():
    for schema in get_tool_schemas():
        assert schema["type"] == "function"
        func = schema["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        params = func["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
        # All required fields should exist in properties
        for req in params["required"]:
            assert req in params["properties"]


def test_system_prompt_content():
    prompt = get_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 100
    assert "two-step" in prompt.lower() or "two-step" in prompt
    assert "search_memory" in prompt
    assert "read_segment" in prompt
    assert "index_memory" in prompt


def test_export_skill_complete():
    skill = export_skill()
    assert "tools" in skill
    assert "system_prompt" in skill
    assert "version" in skill
    assert "protocol" in skill
    assert len(skill["tools"]) == 6
    assert isinstance(skill["system_prompt"], str)
    assert skill["version"] == "1.0"
    assert skill["protocol"] == "two-step-retrieval"


def test_search_memory_schema_has_toc_only():
    schemas = get_tool_schemas()
    search_schema = next(s for s in schemas if s["function"]["name"] == "search_memory")
    props = search_schema["function"]["parameters"]["properties"]
    assert "toc_only" in props
    assert props["toc_only"]["type"] == "boolean"


def test_get_doc_meta_schema():
    schemas = get_tool_schemas()
    meta_schema = next(s for s in schemas if s["function"]["name"] == "get_doc_meta")
    required = meta_schema["function"]["parameters"]["required"]
    assert "doc_id" in required


def test_system_prompt_mentions_get_doc_meta():
    prompt = get_system_prompt()
    assert "get_doc_meta" in prompt


def test_read_segment_schema_requires_doc_id_and_section():
    schemas = get_tool_schemas()
    read_schema = next(s for s in schemas if s["function"]["name"] == "read_segment")
    required = read_schema["function"]["parameters"]["required"]
    assert "doc_id" in required
    assert "section_path" in required
