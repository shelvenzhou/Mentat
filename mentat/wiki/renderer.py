"""Wiki HTML renderer — renders deterministic and agent-owned wiki pages."""

from __future__ import annotations

import json
import logging
import re
from html import escape
from pathlib import Path
from typing import Any, Optional

import markdown as md

logger = logging.getLogger("mentat.wiki")

_MD_EXTENSIONS = ["toc", "tables", "fenced_code"]
_BADGE_BY_VERDICT = {
    "supported": ("✓", "badge-supported"),
    "partial": ("~", "badge-partial"),
    "contradicted": ("✗", "badge-contradicted"),
    "not_in_source": ("?", "badge-missing"),
}

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Mentat Wiki</title>
<style>
  :root {{
    --bg: #ffffff; --fg: #1a1a1a; --muted: #666; --border: #e0e0e0;
    --link: #2563eb; --code-bg: #f4f4f5; --quote-border: #d0d0d0;
    --ok: #166534; --warn: #92400e; --bad: #991b1b; --missing: #475569;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #111; --fg: #e0e0e0; --muted: #999; --border: #333;
      --link: #60a5fa; --code-bg: #1e1e1e; --quote-border: #444;
      --ok: #86efac; --warn: #fcd34d; --bad: #fca5a5; --missing: #cbd5e1;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 820px; margin: 0 auto; padding: 1.5rem 1rem;
    line-height: 1.65; color: var(--fg); background: var(--bg);
  }}
  nav {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem; margin-bottom: 1.5rem;
    display: flex; gap: 1rem; flex-wrap: wrap;
  }}
  nav a {{ color: var(--link); text-decoration: none; font-size: 0.9rem; }}
  nav a:hover {{ text-decoration: underline; }}
  h1 {{ margin-top: 0; }}
  h2 {{ border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }}
  a {{ color: var(--link); }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{
    border: 1px solid var(--border); padding: 0.4rem 0.6rem;
    text-align: left; font-size: 0.9rem;
  }}
  th {{ background: var(--code-bg); }}
  code {{
    background: var(--code-bg); padding: 0.15em 0.35em;
    border-radius: 3px; font-size: 0.88em;
  }}
  pre {{ background: var(--code-bg); padding: 1rem; overflow-x: auto; border-radius: 4px; }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{
    border-left: 3px solid var(--quote-border);
    margin: 1rem 0; padding: 0.2rem 1rem; color: var(--muted);
  }}
  .copy-link {{
    opacity: 0; font-size: 0.75em; margin-left: 0.3em;
    cursor: pointer; color: var(--muted); text-decoration: none;
  }}
  h2:hover .copy-link, h3:hover .copy-link {{ opacity: 1; }}
  .verify-summary {{
    margin: 1rem 0; padding: 0.75rem 1rem; border: 1px solid var(--border);
    background: var(--code-bg); border-radius: 6px; font-size: 0.92rem;
  }}
  .verify-badge {{
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 1.2rem; height: 1.2rem; margin-left: 0.25rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; border: 1px solid currentColor;
  }}
  .badge-supported {{ color: var(--ok); }}
  .badge-partial {{ color: var(--warn); }}
  .badge-contradicted {{ color: var(--bad); }}
  .badge-missing {{ color: var(--missing); }}
  .sources {{ margin-top: 2rem; padding-top: 0.75rem; border-top: 1px solid var(--border); }}
  .sources ol {{ padding-left: 1.4rem; }}
  .sources li {{ margin: 0.35rem 0; }}
  .citation {{
    white-space: nowrap; margin-left: 0.08rem; font-size: 0.78em; vertical-align: super;
  }}
  .citation a {{ text-decoration: none; }}
</style>
<script>
function copyAnchor(el) {{
  const id = el.parentElement.id;
  const url = location.origin + location.pathname + '#' + id;
  navigator.clipboard.writeText(url).then(() => {{
    el.textContent = ' ✓';
    setTimeout(() => el.textContent = ' #', 1200);
  }});
}}
</script>
</head>
<body>
<nav>
  <a href="/wiki/">Index</a>
  <a href="/wiki/topics/">Topics</a>
  <a href="/wiki/memories">Memories</a>
  <a href="/wiki/conversations">Conversations</a>
</nav>
{content}
</body>
</html>"""


def render_markdown(text: str) -> str:
    return md.markdown(text, extensions=_MD_EXTENSIONS)


def _strip_frontmatter(md_text: str) -> str:
    if not md_text.startswith("---\n"):
        return md_text
    end = md_text.find("\n---\n", 4)
    if end == -1:
        return md_text
    return md_text[end + 5 :]


def _add_copy_links(html: str) -> str:
    def _inject(match: re.Match) -> str:
        tag = match.group(1)
        id_attr = match.group(2)
        content = match.group(3)
        copy_btn = '<a class="copy-link" onclick="copyAnchor(this)"> #</a>'
        return f'<{tag} id="{id_attr}">{content}{copy_btn}</{tag}>'

    return re.sub(r"<(h[23]) id=\"([^\"]+)\">(.+?)</\1>", _inject, html)


def _read_markdown_title(md_text: str, fallback: str) -> str:
    for line in md_text.split("\n"):
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _load_verification(topic_path: Path) -> dict[str, Any] | None:
    sidecar = topic_path.with_suffix(".verified.json")
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Invalid topic verification sidecar: %s", sidecar, exc_info=True)
        return None


def _verification_summary(verification: dict[str, Any] | None) -> str:
    if not verification:
        return ""
    claims = verification.get("claims", [])
    total = len(claims)
    supported = sum(1 for claim in claims if _claim_verdict(claim) == "supported")
    partial = sum(1 for claim in claims if _claim_verdict(claim) == "partial")
    contradicted = sum(1 for claim in claims if _claim_verdict(claim) == "contradicted")
    missing = sum(1 for claim in claims if _claim_verdict(claim) == "not_in_source")
    return (
        '<div class="verify-summary">'
        f"<strong>verified:</strong> {supported}/{total}"
        f" · supported {supported}"
        f" · partial {partial}"
        f" · contradicted {contradicted}"
        f" · missing {missing}"
        "</div>"
    )


def _inject_verification_badges(
    html: str, verification: dict[str, Any] | None
) -> str:
    if not verification:
        return html

    citation_verdicts: dict[str, str] = {}
    for claim in verification.get("claims", []):
        verdict = _claim_verdict(claim)
        citations = claim.get("citations") or []
        single_citation = claim.get("citation")
        if single_citation:
            citations = [single_citation, *citations]
        if not verdict:
            continue
        for citation in citations:
            if citation:
                citation_verdicts[str(citation)] = verdict

    def _replace(match: re.Match) -> str:
        raw = match.group(0)
        verdict = citation_verdicts.get(raw)
        if not verdict:
            return raw
        icon, css_class = _BADGE_BY_VERDICT.get(verdict, ("?", "badge-missing"))
        badge = f'<span class="verify-badge {css_class}" title="{verdict}">{icon}</span>'
        return f"{raw}{badge}"

    return re.sub(r"\[\^[^\]]+\]", _replace, html)


def _claim_verdict(claim: dict[str, Any]) -> str | None:
    verdict = claim.get("verdict")
    if verdict:
        return str(verdict)
    status = claim.get("status")
    if status:
        return str(status)
    return None


def _split_topic_sources(md_text: str) -> tuple[str, dict[str, str]]:
    body_lines: list[str] = []
    sources: dict[str, str] = {}
    for line in md_text.splitlines():
        match = re.match(r"^\[\^([^\]]+)\]:\s*(.+)$", line)
        if match:
            sources[match.group(1)] = match.group(2).strip()
        else:
            body_lines.append(line)
    return "\n".join(body_lines).strip(), sources


def _render_source_label(raw: str) -> str:
    if raw.startswith("[") and raw.endswith(")"):
        sep = raw.rfind("](")
        if sep > 0:
            label = raw[1:sep]
            href = raw[sep + 2 : -1]
            return f'<a href="{escape(href, quote=True)}">{escape(label)}</a>'
    return escape(raw)


def _render_sources_list(ordered_keys: list[str], source_defs: dict[str, str]) -> str:
    if not ordered_keys:
        return ""
    items = []
    for key in ordered_keys:
        raw = source_defs.get(key, "")
        items.append(
            f'<li id="source-{escape(key, quote=True)}"><code>{escape(key)}</code>: '
            f"{_render_source_label(raw)}</li>"
        )
    return (
        '<section class="sources">'
        "<h2>Sources</h2>"
        "<ol>"
        + "".join(items)
        + "</ol>"
        "</section>"
    )


def _render_topic_body(
    body_md: str,
    source_defs: dict[str, str],
    verification: dict[str, Any] | None,
) -> str:
    citation_order: list[str] = []
    citation_numbers: dict[str, int] = {}
    citation_verdicts: dict[str, str] = {}
    if verification:
        for claim in verification.get("claims", []):
            verdict = _claim_verdict(claim)
            if not verdict:
                continue
            citations = claim.get("citations") or []
            single_citation = claim.get("citation")
            if single_citation:
                citations = [single_citation, *citations]
            for citation in citations:
                if citation:
                    citation_verdicts[str(citation).removeprefix("[^").removesuffix("]")] = verdict

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in citation_numbers:
            citation_numbers[key] = len(citation_order) + 1
            citation_order.append(key)
        number = citation_numbers[key]
        verdict = citation_verdicts.get(key)
        badge = ""
        if verdict:
            icon, css_class = _BADGE_BY_VERDICT.get(verdict, ("?", "badge-missing"))
            badge = f'<span class="verify-badge {css_class}" title="{verdict}">{icon}</span>'
        if key in source_defs:
            href = f"#source-{escape(key, quote=True)}"
        else:
            href = "#"
        return (
            f'<sup class="citation"><a href="{href}" title="Source {number}">[{number}]</a>'
            f"{badge}</sup>"
        )

    body_with_citations = re.sub(r"\[\^([^\]]+)\]", _replace, body_md)
    html = render_markdown(body_with_citations)
    html = _add_copy_links(html)
    html += _render_sources_list(citation_order, source_defs)
    return html


def render_wiki_page(wiki_dir: str, page_id: str) -> Optional[str]:
    page_path = Path(wiki_dir) / "pages" / f"{page_id}.md"
    if not page_path.exists():
        return None

    md_text = page_path.read_text("utf-8")
    html_content = _add_copy_links(render_markdown(_strip_frontmatter(md_text)))
    title = _read_markdown_title(md_text, page_id)
    return _HTML_TEMPLATE.format(title=title, content=html_content)


def render_wiki_file(wiki_dir: str, filename: str, title: str | None = None) -> Optional[str]:
    file_path = Path(wiki_dir) / filename
    if not file_path.exists():
        return None

    md_text = file_path.read_text("utf-8")
    html_content = _add_copy_links(render_markdown(_strip_frontmatter(md_text)))
    final_title = title or _read_markdown_title(md_text, filename)
    return _HTML_TEMPLATE.format(title=final_title, content=html_content)


def render_topic_page(wiki_dir: str, slug: str) -> Optional[str]:
    topic_path = Path(wiki_dir) / "topics" / f"{slug}.md"
    if not topic_path.exists():
        return None

    md_text = topic_path.read_text("utf-8")
    verification = _load_verification(topic_path)
    body_md, source_defs = _split_topic_sources(_strip_frontmatter(md_text))
    html_content = _verification_summary(verification) + _render_topic_body(
        body_md, source_defs, verification
    )
    title = _read_markdown_title(md_text, slug)
    return _HTML_TEMPLATE.format(title=title, content=html_content)


def render_topic_index(wiki_dir: str) -> Optional[str]:
    topics_dir = Path(wiki_dir) / "topics"
    if not topics_dir.exists():
        return None

    topic_paths = sorted(topics_dir.glob("*.md"))
    lines = ["# Topics", ""]
    if not topic_paths:
        lines.append("_No topics yet. Run `mentat wiki sync` to generate them._")
    else:
        for topic_path in topic_paths:
            slug = topic_path.stem
            title = _read_markdown_title(topic_path.read_text("utf-8"), slug)
            lines.append(f"- [{title}](/wiki/topics/{slug})")

    html_content = _add_copy_links(render_markdown("\n".join(lines)))
    return _HTML_TEMPLATE.format(title="Topics", content=html_content)
