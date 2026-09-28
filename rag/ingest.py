from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_ATTR_LIST_RE = re.compile(r"\s*\{[^{}]*\}\s*$")
_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


def _strip_frontmatter(text: str) -> str:
    """Strip a leading YAML frontmatter block (---...---), if present."""
    return _FRONTMATTER_RE.sub("", text, count=1)


@dataclass
class Chunk:
    id: str
    source: str
    heading: str
    text: str


def chunk_markdown(text: str, source: str) -> list[Chunk]:
    """Split markdown into one chunk per heading (any level 1-6).

    Content before the first heading is dropped if empty, kept otherwise
    under heading "".
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            sections.append((current_heading, current_lines))
            current_heading = _ATTR_LIST_RE.sub("", match.group(2).strip()).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    sections.append((current_heading, current_lines))

    chunks: list[Chunk] = []
    for i, (heading, section_lines) in enumerate(sections):
        body_lines = section_lines[1:] if heading else section_lines
        if not "\n".join(body_lines).strip():
            continue
        full_text = "\n".join(section_lines).strip()
        chunks.append(Chunk(id=f"{source}::{i}", source=source, heading=heading, text=full_text))
    return chunks


def load_docs(data_dir: Path) -> list[Chunk]:
    """Load and chunk every .md file in data_dir, sorted by filename."""
    chunks: list[Chunk] = []
    for path in sorted(Path(data_dir).glob("*.md")):
        text = _strip_frontmatter(path.read_text(encoding="utf-8"))
        chunks.extend(chunk_markdown(text, source=path.name))
    return chunks
