"""Instruction templates for probe-generated guidance.

This file is AI-managed and contains all format-specific instruction templates.
Each probe imports its template constants from this module.

IMPORTANT: Instructions explain PROBE BEHAVIOR, not file content.
- Describe the extraction method (how the probe analyzes the file)
- Document chunking strategy (how content is divided)
- List data limitations (what was NOT extracted or was truncated)

Template Guidelines:
- Use Python format strings with named placeholders: {field_name}
- Brief intros: describe the analysis method (e.g., "analyzed via pandas")
- Instructions: explain extraction method, chunking strategy, and limitations
- Focus on probe behavior, not data summary (LLM already has ToC/Stats)
- Keep templates concise but informative (10-20 lines)
"""

# ==============================================================================
# ARCHIVE PROBE - ZIP/TAR archives
# ==============================================================================

ARCHIVE_BRIEF_INTRO = "{format} archive analyzed via {module} module."

ARCHIVE_INSTRUCTIONS = """Extraction Method:
- Format detected: {format}
- Directory tree built to max depth 3
- File type distribution computed from file extensions
- ToC limited to first 30 directory entries
- File listing limited to first 100 files

Data Limitations:
- Individual file metadata (permissions, timestamps, compression ratios) not extracted
- Archive integrity not validated
- Nested archives not recursively analyzed

Note: {extraction_code} can extract individual files for detailed inspection."""

ARCHIVE_EXTRACTION_ZIP = """Use Python's zipfile module:
  import zipfile
  with zipfile.ZipFile('{filename}') as z:
      z.extractall('output_dir/')"""

ARCHIVE_EXTRACTION_TAR = """Use Python's tarfile module:
  import tarfile
  with tarfile.open('{filename}') as t:
      t.extractall('output_dir/')"""

# ==============================================================================
# CSV PROBE - Tabular data files
# ==============================================================================

CSV_BRIEF_INTRO = "CSV dataset analyzed via pandas with dtype inference and statistical profiling."

CSV_INSTRUCTIONS = """Extraction Method:
- Column types inferred via pandas dtype detection (Int, Float, DateTime, String, Bool)
- Date columns detected via pd.to_datetime() with format='mixed'
- Cardinality computed per column (unique value count)
- Null rates and counts calculated for all columns
- Outlier detection: Z-score > 3 threshold for numeric columns
- String length statistics (min/max/mean) for text columns
- Sampling strategy: {sampling_strategy}

Data Limitations:
- Correlation analysis between columns not computed
- Categorical value distributions not extracted (only cardinality)
- Text patterns (emails, URLs, phone numbers) not detected
- Duplicate rows not identified"""

CSV_SAMPLING_NOTE_FULL = "full dataset included (< 50 total cells)"
CSV_SAMPLING_NOTE_SAMPLED = "representative sample (first, middle, last row only)"

# ==============================================================================
# CODE PROBE - Python/JavaScript/TypeScript source code
# ==============================================================================

CODE_BRIEF_INTRO = "{language} source code analyzed via tree-sitter AST parsing."

CODE_INSTRUCTIONS = """Extraction Method:
- Language: {language}
- AST parsing via tree-sitter (syntax-aware, not regex-based)
- Hierarchical ToC: Level 1 = classes/top-level functions, Level 2 = methods
- Import statements extracted from module-level nodes
- Function signatures captured (parameters + return types when available)
- Docstrings extracted (first line only, max 120 chars)
- Decorated definitions tracked separately

Chunking Strategy:
- One chunk per top-level definition (class or function)
- Function bodies omitted for brevity
- Small files (< 1000 tokens) return full content

Data Limitations:
- Function bodies not included (implementation details omitted)
- Type hints captured in signatures but not separately indexed
- TODO/FIXME comments not extracted
- Test coverage not analyzed
- Import list capped at 20 entries"""

# ==============================================================================
# PDF PROBE - PDF documents
# ==============================================================================

PDF_BRIEF_INTRO = "PDF document analyzed via PyMuPDF with {toc_method} ToC extraction."

PDF_TOC_METHOD_METADATA = "metadata-based"
PDF_TOC_METHOD_VISUAL = "font-size-based"

PDF_INSTRUCTIONS = """Extraction Method:
- ToC source: {toc_source}
- Font analysis: body text detected via histogram (most common font size)
- Visual header detection: font size > 1.1x body font, length 2-100 chars
- Caption detection: regex pattern for "Figure/Table N"
- Title extraction: max 100 chars from PDF metadata or largest font on page 1
- First paragraph: 500 char cutoff, body-font-sized text only

Chunking Strategy:
- One chunk per page
- Each chunk tagged with current section from ToC
- Page boundaries preserved (no cross-page merging)

Data Limitations:
- Form fields not extracted
- Table cell contents not captured (captions only)
- Bookmark depth not analyzed
- Image content not described (OCR not performed)
- Rotation and aspect ratio metadata not included"""

PDF_TOC_SOURCE_METADATA = "extracted from PDF metadata (native outline)"
PDF_TOC_SOURCE_VISUAL = "inferred from font size analysis (bold text > 1.1x body font)"

# ==============================================================================
# IMAGE PROBE - Image files (JPEG, PNG, etc.)
# ==============================================================================

IMAGE_BRIEF_INTRO = "Image file analyzed via Pillow (PIL) for metadata extraction."

IMAGE_INSTRUCTIONS = """Extraction Method:
- Format and dimensions detected via Pillow
- Color mode identified (RGB, RGBA, Grayscale, etc.)
- EXIF metadata extracted from JPEG/TIFF images
- Interesting EXIF tags: DateTime, Make, Model, Software, GPS, ExposureTime, FNumber, ISO
- EXIF values truncated at 200 chars per field
- GPS coordinates parsed to Lat/Lon format

Chunking Strategy:
- Single metadata-only chunk (no visual content)

Data Limitations:
- Visual content not described (requires vision model/multimodal LLM)
- OCR text not extracted
- Object/face detection not performed
- Image quality metrics (sharpness, blur) not computed
- Perceptual hashing not generated"""

# ==============================================================================
# DOCX PROBE - Microsoft Word documents
# ==============================================================================

DOCX_BRIEF_INTRO = "Word document analyzed via python-docx with style-based hierarchy detection."

DOCX_INSTRUCTIONS = """Extraction Method:
- Heading detection via style regex: "Heading 1", "Heading 2", etc.
- List detection via style names containing "List"
- Table headers extracted (first 5 tables only, pipe-separated)
- Metadata from document properties (title, author, created, modified, revision)
- Paragraph count and word count computed
- First non-heading paragraph used as topic (300 char limit)
- "Preamble" section synthesized for content before first heading

Chunking Strategy:
- One chunk per heading section (heading + paragraphs until next heading)
- Small chunks merged via merge_small_chunks() (< 300 tokens → merge, max 1200 merged)
- H1/H2 boundaries respected (no merging across major sections)
- Annotations: "List, N items | M paragraphs"

Data Limitations:
- Table cell contents not extracted (only headers from first 5 tables)
- List hierarchy not captured (indentation depth, bullets vs. numbered)
- Images/embedded objects counted but not analyzed
- Tracked changes and comments not extracted
- Hyperlink targets not captured
- Text styles (bold, italic, colors) not recorded beyond heading styles"""

# ==============================================================================
# PPTX PROBE - PowerPoint presentations
# ==============================================================================

PPTX_BRIEF_INTRO = "PowerPoint presentation analyzed via python-pptx."

PPTX_INSTRUCTIONS = """Extraction Method:
- Slide-level ToC: title + first 3 bullets per slide
- Bullet text truncated at 80 chars in ToC
- Speaker notes extracted if present (used as preview, max 120 chars)
- Media counts per slide: bullets, images, tables
- Total counts aggregated: slides, bullets, images, tables, notes

Chunking Strategy:
- One chunk per slide (title + body text + notes)
- Slide-centric organization

Data Limitations:
- Table contents not extracted (only count per slide)
- Animations and transitions not captured
- Embedded media (video/audio) presence not detected
- Slide layouts and master slides not analyzed
- Shape geometry and positioning not recorded
- Text box positioning not captured"""

# ==============================================================================
# CALENDAR PROBE - iCalendar (.ics) files
# ==============================================================================

CALENDAR_BRIEF_INTRO = "iCalendar file analyzed via icalendar library."

CALENDAR_INSTRUCTIONS = """Extraction Method:
- Events sorted chronologically by start time
- Recurring detection via RRULE presence (boolean only, not frequency)
- Attendees extracted from ATTENDEE property, deduplicated, mailto: prefix stripped
- Description truncated at 200 chars per event
- Time range computed from earliest/latest start times
- Calendar name from X-WR-CALNAME property

Chunking Strategy:
- One chunk per event (summary + time + location + description)
- Event-level organization

Data Limitations:
- ToC limited to first 30 events (sorted chronologically)
- Attendee list capped at top 20 in stats
- Time zone info not extracted from DTSTART/DTEND
- Recurrence rules simplified to boolean (not frequency: daily/weekly/monthly)
- Organizer not distinguished from attendees
- Alarms/reminders (VALARM) not captured
- Exceptions to recurrence (RECURRENCE-ID) not tracked
- Free/busy status not extracted"""

# ==============================================================================
# JSON PROBE - JSON data files
# ==============================================================================

JSON_BRIEF_INTRO = "JSON document analyzed via Python json module with schema inference."

JSON_INSTRUCTIONS = """Extraction Method:
- Schema tree inferred recursively with max depth 3
- String values truncated at 40 chars in schema representation
- Value previews: 30 head + 15 tail chars with ellipsis
- Array structure inferred from first item only
- Root type detected: object/array/primitive

Chunking Strategy:
- Root is dict: one chunk per top-level key (re-serialized JSON)
- Root is array: single chunk with first item as sample
- Root is primitive: single chunk with complete value
- Small chunks merged (< 300 tokens) for efficiency

Data Limitations:
- Schema depth limited to 3 levels for ToC (actual depth tracked in stats)
- Long string values truncated in schema (40 chars)
- Arrays assumed homogeneous (structure from first item only)
- Circular references not detected
- JSON Schema validation not performed
- Deep value paths not mapped"""

# ==============================================================================
# CONFIG PROBE - YAML/TOML/INI configuration files
# ==============================================================================

CONFIG_BRIEF_INTRO = "{format} configuration analyzed via {parser} with recursive key traversal."

CONFIG_INSTRUCTIONS = """Extraction Method:
- Format detected: {format}
- Parser: {parser_info}
- Key hierarchy walked recursively with max depth 3
- Value types inferred from parsed data
- Top-level keys shown (first 5 + "..." if more)
- INI handling: sections extracted + DEFAULT section synthesized

Chunking Strategy:
- Full content returned for most configs (< 1000 tokens typical)
- Large files chunked at 2000 char boundaries

Data Limitations:
- Variable interpolation not analyzed (YAML anchors, env var references like ${{VAR}})
- Secrets not flagged (recommend manual review for API keys/passwords)
- Include directives not followed (external file references in configs)
- Conditional sections not evaluated (feature flags, environment-based config)
- Schema validation not performed"""

CONFIG_PARSER_YAML = "PyYAML with safe_load()"
CONFIG_PARSER_TOML = "tomli parser"
CONFIG_PARSER_INI = "configparser"

# ==============================================================================
# LOG PROBE - Log files
# ==============================================================================

LOG_BRIEF_INTRO = "Log file analyzed via regex-based format detection and keyword extraction."

LOG_INSTRUCTIONS = """Extraction Method:
- Format detection via regex sampling (50%+ match threshold)
- Formats supported: JSONL, syslog, Apache CLF, ISO timestamp, custom
- Timestamp extraction from first/last 20 lines (multiple patterns tried)
- Error level detection: case-insensitive regex for FATAL/CRITICAL/ERROR/WARN/INFO/DEBUG/TRACE
- Keyword extraction: stop-word filtered, min 3 chars, top 10 by frequency
- Level statistics computed across all lines

Chunking Strategy:
- Small files (< 1000 tokens): full log
- Large files: head (5 lines) + tail (5 lines) + error sample (5 error lines)

Data Limitations:
- Process/thread IDs not extracted
- Stack traces not parsed (multiline errors truncated to single line)
- Request tracing/correlation IDs not captured
- Anomaly detection not performed
- Rate analysis (entries per second) not computed"""

# ==============================================================================
# MARKDOWN PROBE - Markdown documents
# ==============================================================================

MARKDOWN_BRIEF_INTRO = "Markdown document analyzed via regex-based heading detection with code-fence filtering."

MARKDOWN_INSTRUCTIONS = """Extraction Method:
- Heading detection: regex for H1-H6 (# through ######)
- Code fence filtering: headers inside ``` blocks excluded from ToC
- Section annotations: lists, code blocks, links, line counts
- Preview extraction: first non-empty, non-header line per section (max 120 chars)
- Heading density computed: ratio of heading lines to total lines

Chunking Strategy:
- Small files (< 1000 tokens): full content
- High heading density (> 25% AND < 3000 tokens): full content for context
- Otherwise: split by headers with merge_small_chunks() applied
- H1/H2 boundaries respected (no merging across major sections)
- Small chunks merged (< 300 tokens → merge, max 1200 tokens merged)

Data Limitations:
- Blockquotes not separately indexed
- Footnotes/references not extracted
- YAML frontmatter not parsed (metadata headers ignored)
- Task lists (checkboxes) not detected
- Tables parsed as text (not structured/columnar)
- Image alt text in links not captured separately
- Cross-document link validation not performed"""

# ==============================================================================
# WEB PROBE - HTML web pages
# ==============================================================================

WEB_BRIEF_INTRO = "HTML page analyzed via trafilatura (content extraction) and regex (structure parsing)."

WEB_INSTRUCTIONS = """Extraction Method:
- Dual extraction: trafilatura for clean text, regex for HTML structure
- Heading hierarchy: H1-H6 detected via regex
- Semantic elements: nav, header, main, article, section, footer, aside (presence noted)
- Metadata: title from <title> tag, meta description/keywords, trafilatura metadata (author, date, categories)
- Navigation links: text extracted from <nav> element
- Section annotations: paragraphs, links, list items, images, tables counted per section
- Content length and word count computed from trafilatura output

Chunking Strategy:
- Small files (< 1000 tokens clean text): full content
- Otherwise: split by heading sections with merge_small_chunks() applied
- Fallback: paragraph splitting on \\n\\n if heading-based fails

Data Limitations:
- JavaScript-generated content not extracted (static HTML only)
- Structured data (JSON-LD, schema.org microdata) not parsed
- Open Graph metadata (og:image, og:type) not extracted
- Forms and input fields not detected
- Video/audio embeds not analyzed (only image/table counts)
- CSS media queries and responsive design not analyzed
- Accessibility metadata (ARIA roles, alt text) not separately indexed"""
