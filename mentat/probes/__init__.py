from pathlib import Path
from typing import Optional

from mentat.probes.base import BaseProbe, ProbeResult

# --- Always-available probes ---
from mentat.probes.pdf_probe import PDFProbe
from mentat.probes.archive_probe import ArchiveProbe
from mentat.probes.csv_probe import CSVProbe
from mentat.probes.json_probe import JSONProbe
from mentat.probes.config_probe import ConfigProbe
from mentat.probes.code_probe import CodeProbe
from mentat.probes.log_probe import LogProbe
from mentat.probes.markdown_probe import MarkdownProbe
from mentat.probes.web_probe import WebProbe

# Registry: order matters — first match wins.
# Most specific formats first; broadest fallbacks last.
_REGISTERED_PROBES: list = [
    PDFProbe(),       # .pdf
]

# --- Optional probes (graceful degradation if deps missing) ---
try:
    from mentat.probes.image_probe import ImageProbe
    _REGISTERED_PROBES.append(ImageProbe())      # .jpg/.png/.gif/...
except ImportError:
    pass

try:
    from mentat.probes.docx_probe import DOCXProbe
    _REGISTERED_PROBES.append(DOCXProbe())        # .docx
except ImportError:
    pass

try:
    from mentat.probes.pptx_probe import PPTXProbe
    _REGISTERED_PROBES.append(PPTXProbe())        # .pptx
except ImportError:
    pass

try:
    from mentat.probes.calendar_probe import CalendarProbe
    _REGISTERED_PROBES.append(CalendarProbe())    # .ics
except ImportError:
    pass

# --- Always-available probes (continued) ---
_REGISTERED_PROBES.extend([
    ArchiveProbe(),   # .zip/.tar.*
    CSVProbe(),       # .csv
    JSONProbe(),      # .json
    ConfigProbe(),    # .yaml/.toml/.ini/.conf
    CodeProbe(),      # .py/.js/.ts
    LogProbe(),       # .log
    MarkdownProbe(),  # .md/.markdown
    WebProbe(),       # .html/.htm (broadest text fallback)
])


def get_probe(file_path: str) -> Optional[BaseProbe]:
    """Return the first probe that can handle the given file."""
    filename = Path(file_path).name
    content_type = ""  # TODO: content-type detection via python-magic if needed

    for probe in _REGISTERED_PROBES:
        if probe.can_handle(filename, content_type):
            return probe
    return None


def run_probe(file_path: str) -> ProbeResult:
    """Convenience: find the right probe and run it."""
    probe = get_probe(file_path)
    if not probe:
        raise ValueError(f"No probe found for file: {file_path}")
    return probe.run(file_path)
