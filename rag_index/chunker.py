import json
import re
from pathlib import Path

from . import config
from .logging_config import logger

HEADING3_RE = re.compile(r"^###\s+(.+)$")
HEADING2_RE = re.compile(r"^##\s+(.+)$")


def load_index_records(index_jsonl: str | Path) -> list[dict]:
    records = []
    with open(index_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_chunks(records: list[dict]) -> list[dict]:
    """Yield one dict per (chunk_file, index record) with full metadata + content."""
    base = Path(config.CHUNKS_DIR)
    for rec in records:
        rel = rec.get("chunk_file")
        if not rel:
            continue
        path = base / rel
        if not path.exists():
            continue
        yield {
            **rec,
            "content": path.read_text(encoding="utf-8"),
        }


def split_section_at_h3(section: dict) -> list[dict]:
    """
    Hierarchical split of a level-2 section chunk into level-3 sub-chunks.

    Each `### ` heading starts a new child chunk. Content runs until the next
    heading of level <= 3. Metadata is inherited from the parent section.
    Returns a list of dicts; a single-element list if no `### ` headings exist.
    """
    lines = section["content"].split("\n")
    parent_heading = section.get("heading")  # e.g. "## 1. SAFETY INSTRUCTIONS"
    parent_chunk_file = section.get("chunk_file")

    starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = HEADING3_RE.match(line)
        if m:
            starts.append((i, m.group(1).strip()))

    if not starts:
        return [section]

    children = []
    for idx, (start_i, h3) in enumerate(starts):
        end_i = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        body_lines = lines[start_i + 1:end_i]
        content = "\n".join(body_lines).strip()

        child = {
            **section,
            "chunk_type": "section",
            "heading_level": 3,
            "heading": h3,
            "parent_heading": parent_heading,
            "parent_chunk_file": parent_chunk_file,
            "chunk_file": f"{parent_chunk_file}#{h3}",
            "content": content,
        }
        children.append(child)

    return children


def build_all_chunks() -> list[dict]:
    """Build the full chunk set: special chunks kept as-is, sections split at ###."""
    records = load_index_records(config.INDEX_JSONL)
    logger.info("Loaded %d index records from %s", len(records), config.INDEX_JSONL)
    chunks: list[dict] = []
    for doc in load_chunks(records):
        if doc.get("chunk_type") == "section":
            parts = split_section_at_h3(doc)
            if len(parts) > 1:
                logger.debug("Split %s -> %d h3 sub-chunks", doc.get("chunk_file"), len(parts))
            chunks.extend(parts)
        else:
            chunks.append(doc)

    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        logger.info(
            "Chunk %d/%d: %s (type=%s, level=%s)",
            i,
            total,
            chunk.get("chunk_file"),
            chunk.get("chunk_type"),
            chunk.get("heading_level"),
        )
    return chunks