from pathlib import Path
from typing import Optional

from mentat.probes.base import BaseProbe
from mentat.probes.code_probe import CodeProbe
from mentat.probes.csv_probe import CSVProbe
from mentat.probes.json_probe import JSONProbe
from mentat.probes.markdown_probe import MarkdownProbe
from mentat.probes.pdf_probe import PDFProbe
from mentat.probes.web_probe import WebProbe

_REGISTERED_PROBES = [
    CodeProbe(),
    PDFProbe(),
    CSVProbe(),
    JSONProbe(),
    MarkdownProbe(),
    WebProbe(),
]


def get_probe(file_path: str) -> Optional[BaseProbe]:
    """
    Return the first probe that can handle the given file.
    """
    path = Path(file_path)
    filename = path.name
    # TODO: implement real content-type detection if needed
    content_type = ""

    for probe in _REGISTERED_PROBES:
        if probe.can_handle(filename, content_type):
            return probe
    return None
