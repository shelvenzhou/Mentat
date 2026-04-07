"""Shell out to external agents that maintain the synthesized wiki."""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

logger = logging.getLogger("mentat.wiki")

WIKI_DRIVERS = {
    "codex": ["codex", "exec", "--cd", "{wiki_dir}", "-"],
    "claude": ["claude", "-p", "@CLAUDE.md"],
    "openclaw": ["openclaw", "task", "--workdir", "{wiki_dir}", "--prompt-file", "CLAUDE.md"],
}


class WikiAgentRunner:
    """Write the active template to CLAUDE.md and invoke an agent driver."""

    def __init__(self, wiki_dir: str | Path, default_driver: str = "codex"):
        self.wiki_dir = Path(wiki_dir).resolve()
        self.default_driver = default_driver or "codex"
        self.templates_dir = Path(__file__).resolve().parent / "templates"

    def run(self, mode: str, driver: str | None = None) -> int:
        chosen_driver = driver or self.default_driver
        if chosen_driver not in WIKI_DRIVERS:
            raise ValueError(
                f"Unknown wiki driver: {chosen_driver}. Expected one of {sorted(WIKI_DRIVERS)}"
            )

        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = self.write_prompt(mode)
        prompt_text = prompt_path.read_text("utf-8")
        command = [
            part.format(wiki_dir=str(self.wiki_dir))
            for part in WIKI_DRIVERS[chosen_driver]
        ]
        logger.info("Running wiki %s via %s: %s", mode, chosen_driver, shlex.join(command))
        stdin_text = prompt_text if chosen_driver == "codex" else None
        result = subprocess.run(
            command,
            cwd=self.wiki_dir,
            check=False,
            input=stdin_text,
            text=True,
        )
        logger.info("Wiki %s via %s exited with code %s", mode, chosen_driver, result.returncode)
        return result.returncode

    def write_prompt(self, mode: str) -> Path:
        template_name = f"{mode}.claude.md"
        template_path = self.templates_dir / template_name
        if not template_path.exists():
            raise ValueError(f"Unknown wiki mode: {mode}")

        prompt_text = template_path.read_text("utf-8")
        target_path = self.wiki_dir / "CLAUDE.md"
        if target_path.exists():
            existing = target_path.read_text("utf-8")
            if existing != prompt_text:
                logger.warning("Overwriting existing CLAUDE.md with wiki %s template", mode)
        target_path.write_text(prompt_text, "utf-8")
        return target_path
