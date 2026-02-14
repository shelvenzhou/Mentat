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
    assert placeholders == {'format', 'file_count', 'size', 'dir_count'}

    # Test formatting works
    intro = tpl.ARCHIVE_BRIEF_INTRO.format(
        format='ZIP', file_count=100, size='10 MB', dir_count=5
    )
    assert 'ZIP' in intro

    # Test ARCHIVE_INSTRUCTIONS
    placeholders = extract_placeholders(tpl.ARCHIVE_INSTRUCTIONS)
    assert 'extraction_code' in placeholders

    # Test extraction templates
    assert '{filename}' in tpl.ARCHIVE_EXTRACTION_ZIP
    assert '{filename}' in tpl.ARCHIVE_EXTRACTION_TAR


def test_csv_templates():
    """Test CSV probe templates have valid placeholders."""
    placeholders = extract_placeholders(tpl.CSV_BRIEF_INTRO)
    assert 'row_count' in placeholders and 'col_count' in placeholders

    # Test outlier note
    assert '{outlier_columns}' in tpl.CSV_OUTLIER_NOTE

    # Test sampling notes
    assert '{row_count' in tpl.CSV_SAMPLING_NOTE_SAMPLED


def test_code_templates():
    """Test code probe templates have valid placeholders."""
    placeholders = extract_placeholders(tpl.CODE_BRIEF_INTRO)
    assert placeholders == {'language', 'class_count', 'function_count'}

    placeholders = extract_placeholders(tpl.CODE_INSTRUCTIONS)
    assert 'import_list' in placeholders


def test_pdf_templates():
    """Test PDF probe templates have valid placeholders."""
    assert '{page_count}' in tpl.PDF_BRIEF_INTRO
    assert '{title_part}' in tpl.PDF_BRIEF_INTRO

    assert '{toc_source}' in tpl.PDF_INSTRUCTIONS
    assert '{section_count}' in tpl.PDF_INSTRUCTIONS


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
