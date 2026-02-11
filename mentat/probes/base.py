import abc
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ProbeResult(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    structure: Dict[str, Any]  # e.g., ToC, headers, function names
    stats: Dict[str, Any]  # e.g., null rate, mean, outlier count
    summary_hint: str  # Preliminary info for the Librarian
    raw_snippet: Optional[str] = None  # For small files or samples


class BaseProbe(abc.ABC):
    @abc.abstractmethod
    def can_handle(self, filename: str, content_type: str) -> bool:
        """Check if this probe can handle the file."""
        pass

    @abc.abstractmethod
    def run(self, file_path: str) -> ProbeResult:
        """Extract statistical and structural data from the file."""
        pass
