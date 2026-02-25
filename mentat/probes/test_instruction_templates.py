"""Tests for instruction template placeholders.

Validates that all template format strings use correct placeholder names
and that all placeholders can be filled with example data.
"""

import re
from mentat.probes import instruction_templates as tpl


def extract_placeholders(template: str) -> set:
    """Extract all {placeholder} names from a format string."""
    return set(re.findall(r'\{(\w+)(?:[^}]*)?\}', template))


def test_archive_templates():
    """Test archive probe templates have valid placeholders."""
    # Test ARCHIVE_BRIEF_INTRO
    placeholders = extract_placeholders(tpl.ARCHIVE_BRIEF_INTRO)
    assert placeholders == {'format', 'module'}

    # Test formatting works
    intro = tpl.ARCHIVE_BRIEF_INTRO.format(format='ZIP', module='zipfile')
    assert 'ZIP' in intro

    # Test ARCHIVE_INSTRUCTIONS
    placeholders = extract_placeholders(tpl.ARCHIVE_INSTRUCTIONS)
    assert 'format' in placeholders
    assert 'extraction_code' in placeholders

    # Test extraction templates
    assert '{filename}' in tpl.ARCHIVE_EXTRACTION_ZIP
    assert '{filename}' in tpl.ARCHIVE_EXTRACTION_TAR


def test_csv_templates():
    """Test CSV probe templates have valid placeholders."""
    # CSV_BRIEF_INTRO is a static string (no placeholders)
    assert isinstance(tpl.CSV_BRIEF_INTRO, str)
    assert 'pandas' in tpl.CSV_BRIEF_INTRO

    # Test CSV_INSTRUCTIONS has sampling_strategy placeholder
    placeholders = extract_placeholders(tpl.CSV_INSTRUCTIONS)
    assert 'sampling_strategy' in placeholders

    # Test sampling notes are static strings
    assert isinstance(tpl.CSV_SAMPLING_NOTE_FULL, str)
    assert isinstance(tpl.CSV_SAMPLING_NOTE_SAMPLED, str)


def test_code_templates():
    """Test code probe templates have valid placeholders."""
    placeholders = extract_placeholders(tpl.CODE_BRIEF_INTRO)
    assert placeholders == {'language'}

    placeholders = extract_placeholders(tpl.CODE_INSTRUCTIONS)
    assert 'language' in placeholders


def test_pdf_templates():
    """Test PDF probe templates have valid placeholders."""
    assert '{toc_method}' in tpl.PDF_BRIEF_INTRO

    assert '{toc_source}' in tpl.PDF_INSTRUCTIONS


def test_all_templates_exist():
    """Ensure all 13 probe types have templates defined."""
    required_prefixes = [
        'ARCHIVE', 'CSV', 'CODE', 'PDF', 'IMAGE', 'DOCX', 'PPTX',
        'CALENDAR', 'JSON', 'CONFIG', 'LOG', 'MARKDOWN', 'WEB'
    ]

    for prefix in required_prefixes:
        brief = f"{prefix}_BRIEF_INTRO"
        instr = f"{prefix}_INSTRUCTIONS"
        assert hasattr(tpl, brief), f"Missing {brief}"
        assert hasattr(tpl, instr), f"Missing {instr}"
        assert isinstance(getattr(tpl, brief), str)
        assert isinstance(getattr(tpl, instr), str)


if __name__ == '__main__':
    test_archive_templates()
    test_csv_templates()
    test_code_templates()
    test_pdf_templates()
    test_all_templates_exist()
    print("✅ All template tests passed!")
