import abc
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel


class TopicInfo(BaseModel):
    """Core extraction: content representing the overall theme of the file."""

    title: Optional[str] = None
    abstract: Optional[str] = None
    first_paragraph: Optional[str] = None


class TocEntry(BaseModel):
    """A single entry in a table-of-contents or header hierarchy."""

    level: int = 1
    title: str = ""
    page: Optional[int] = None  # For page-based formats (PDF)
    preview: Optional[str] = None  # First sentence / key phrase of the section
    annotation: Optional[str] = None  # Structural features, e.g. "List, 8 items"


class Caption(BaseModel):
    """A figure or table caption."""

    text: str
    page: Optional[int] = None
    kind: str = "figure"  # "figure" | "table"


class StructureInfo(BaseModel):
    """Core extraction: content representing the file's structure."""

    toc: List[TocEntry] = []  # For docs: ToC / headers
    captions: List[Caption] = []  # For docs: figure/table captions
    schema_tree: Optional[Any] = None  # For JSON: key structure tree
    columns: List[str] = []  # For CSV: header row
    definitions: List[str] = []  # For code: function/class names


class Chunk(BaseModel):
    """A format-aware chunk of content with structural context."""

    content: str
    index: int
    section: Optional[str] = None  # Which ToC/header section this belongs to
    page: Optional[int] = None  # For page-based formats
    metadata: Dict[str, Any] = {}


class ProbeResult(BaseModel):
    """Output of a probe run. Provides structured data for the Librarian layer."""

    doc_id: str = ""
    filename: str
    file_type: str

    # --- Core extractions (design requirement) ---
    topic: TopicInfo = TopicInfo()
    structure: StructureInfo = StructureInfo()
    stats: Dict[str, Any] = {}

    # --- Format-aware chunks ---
    chunks: List[Chunk] = []

    # --- Probe-generated instructions (optional, falls back to librarian templates) ---
    brief_intro: Optional[str] = None
    instructions: Optional[str] = None

    # --- Raw sample for small files ---
    raw_snippet: Optional[str] = None


class BaseProbe(abc.ABC):
    @abc.abstractmethod
    def can_handle(self, filename: str, content_type: str) -> bool:
        """Check if this probe can handle the file."""
        pass

    @abc.abstractmethod
    def run(self, file_path: str, **kwargs) -> ProbeResult:
        """Extract statistical and structural data from the file.

        Subclasses may accept additional keyword arguments (e.g.
        ``probe_config``) for format-specific configuration.
        """
        pass

    def generate_instructions(self, probe_result: ProbeResult) -> Tuple[str, str]:
        """Generate format-specific instructions from probe result.

        Default implementation returns empty strings, signaling that the librarian
        should use fallback template generation. Override in subclasses for
        format-specific guidance.

        Args:
            probe_result: The ProbeResult from run(), with all stats/structure populated

        Returns:
            (brief_intro, instructions) tuple of strings
        """
        return "", ""
