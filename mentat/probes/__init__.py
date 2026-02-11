from pathlib import Path
from typing import Optional

from mentat.probes.base import BaseProbe, ProbeResult
from mentat.probes.code_probe import CodeProbe
from mentat.probes.csv_probe import CSVProbe
from mentat.probes.json_probe import JSONProbe
from mentat.probes.markdown_probe import MarkdownProbe
from mentat.probes.pdf_probe import PDFProbe
from mentat.probes.web_probe import WebProbe

# Registry: order matters — first match wins
_REGISTERED_PROBES = [
    PDFProbe(),
    CSVProbe(),
    JSONProbe(),
    CodeProbe(),
    MarkdownProbe(),
    WebProbe(),
]


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
