"""Hierarchical chunk splitting with parent heading context.

New chunking rules:
  1. For each ## chunk (file), count tokens
  2. If tokens <= MAX_CHUNK_TOKENS: keep as single chunk
  3. If tokens > MAX_CHUNK_TOKENS: split at ### headings
  4. For each ### chunk:
     a. Add parent ## heading as prefix
     b. Add parent ## heading + next ### heading as suffix
     c. Count tokens
     d. If tokens > MAX_CHUNK_TOKENS: split at #### headings or recursive
"""

import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .logging_config import logger
from .noise_filter import filter_noise
from .token_counter import count_tokens

HEADING4_RE = re.compile(r"^####\s+(.+)$")
HEADING3_RE = re.compile(r"^###\s+(.+)$")
HEADING2_RE = re.compile(r"^##\s+(.+)$")


def _get_heading_level(heading: str) -> int:
    """Determine heading level from heading string prefix."""
    if heading.startswith("####"):
        return 4
    elif heading.startswith("###"):
        return 3
    elif heading.startswith("##"):
        return 2
    elif heading.startswith("#"):
        return 1
    return 2  # default


# ---------------------------------------------------------------------------
# Index / chunk I/O
# ---------------------------------------------------------------------------

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


def _sanitize_filename(name: str) -> str:
    """Sanitize a string to be used as a filename."""
    invalid = '<>:"/\\|?*\x00-\x1f'
    for c in invalid:
        name = name.replace(c, "_")
    name = re.sub(r"_+", "_", name)
    if len(name) > 100:
        name = name[:100]
    return name


# ---------------------------------------------------------------------------
# Noise pre-filtering
# ---------------------------------------------------------------------------

def prefilter_content(chunk: dict) -> dict:
    """Apply noise filter to chunk content in-place and return it."""
    if config.NOISE_FILTER_ENABLED:
        chunk["content"] = filter_noise(chunk["content"])
    return chunk


# ---------------------------------------------------------------------------
# Recursive text splitter for token overflow
# ---------------------------------------------------------------------------


def _split_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    size = chunk_size or config.RECURSIVE_CHUNK_SIZE
    ov = overlap or config.RECURSIVE_OVERLAP
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=ov,
        length_function=count_tokens,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def recursive_split_chunk(chunk: dict) -> list[dict]:
    """Split a chunk's content with RecursiveCharacterTextSplitter.

    Uses a token-aware length function so chunk_size is in tokens, not chars.
    Accounts for parent context overhead when calculating effective chunk size.
    Preserves parent_headings structure across splits.
    """
    parent_headings = chunk.get("parent_headings", [])
    next_heading = chunk.get("next_heading") or ""
    # Calculate overhead from all parent headings
    overhead = sum(count_tokens(p["content"]) for p in parent_headings) + count_tokens(next_heading) + 10

    effective_size = max(256, config.RECURSIVE_CHUNK_SIZE - overhead)

    parts = _split_text(chunk["content"], chunk_size=effective_size)
    if len(parts) <= 1:
        return [chunk]

    result = []
    for i, text in enumerate(parts):
        full_text = _build_full_text_with_parent(parent_headings, text, next_heading)
        c = {
            **chunk,
            "content": text,
            "full_text": full_text,
            "chunk_file": f"{chunk['chunk_file']}_part{i}",
            "split_from": chunk["chunk_file"],
        }
        result.append(c)
    logger.debug(
        "Recursive split %s -> %d parts (effective_size=%d, overhead=%d)",
        chunk["chunk_file"], len(parts), effective_size, overhead,
    )
    return result


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------

def _chunk_text(chunk: dict) -> str:
    """Build the full text that will be embedded for a chunk."""
    parts = [p for p in (chunk.get("heading"), chunk.get("content")) if p]
    return "\n\n".join(parts).strip()


def _build_full_text_with_parent(parent_headings: list[dict], content: str, next_heading: str | None = None) -> str:
    """Build full text for a chunk with parent heading context.
    
    parent_headings: list of dicts like [{"heading_level": 2, "content": "## 2. SAFE OPERATION..."}, ...]
    
    Format: prefix (parent headings) + content. Suffix (next heading) is intentionally
    omitted to avoid duplication at file boundaries.
    """
    # Build prefix from all parent headings
    prefix = "\n\n".join([p["content"] for p in parent_headings])
    parts = [prefix, "", content]
    # Suffix (next_heading + parent context) is omitted to prevent duplication
    # when writing to .md files. If needed for embedding, use _chunk_text() instead.
    return "\n\n".join(parts).strip()


# ---------------------------------------------------------------------------
# #### (level-4) splitting
# ---------------------------------------------------------------------------

def split_at_h4(chunk: dict) -> list[dict]:
    """Split a level-3 chunk at #### headings into level-4 sub-chunks."""
    lines = chunk["content"].split("\n")
    h3_heading = chunk.get("heading")
    h3_chunk_file = chunk.get("chunk_file")
    # parent_headings is a list of dicts from the level-3 chunk (contains level-2 heading)
    parent_headings = chunk.get("parent_headings", [])

    # Append the current level-3 heading to the parent headings list for level-4 children
    h3_prefixed = h3_heading or ""
    # Add level-3 heading as the latest parent
    updated_parent_headings = parent_headings + [{"heading_level": 3, "content": h3_prefixed}]

    starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = HEADING4_RE.match(line)
        if m:
            starts.append((i, m.group(1).strip()))

    if not starts:
        return [chunk]

    children = []
    for idx, (start_i, h4_heading) in enumerate(starts):
        end_i = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        body_lines = lines[start_i + 1:end_i]
        content = "\n".join(body_lines).strip()

        # Determine next heading for suffix
        next_heading = None
        if idx + 1 < len(starts):
            next_heading = f"#### {starts[idx + 1][1]}"

        # Build full_text with parent context (now includes both level-2 and level-3)
        child_content_with_heading = f"#### {h4_heading}\n\n{content}"
        h4_next_suffix = None
        if next_heading:
            h4_next_suffix = f"{h3_prefixed}\n{next_heading}"
        child_full_text = _build_full_text_with_parent(
            updated_parent_headings, child_content_with_heading, h4_next_suffix
        )

        child = {
            **chunk,
            "chunk_type": "section",
            "heading_level": 4,
            "heading": f"#### {h4_heading}" if h4_heading else h4_heading,
            "parent_headings": updated_parent_headings,
            "parent_chunk_file": h3_chunk_file,
            "chunk_file": f"{h3_chunk_file}_part{idx}",
            "content": content,
            "full_text": child_full_text,
            "next_heading": next_heading,
        }
        children.append(child)

    if len(children) > 1:
        logger.debug(
            "Split %s at #### -> %d sub-chunks", h3_chunk_file, len(children)
        )
    return children


# ---------------------------------------------------------------------------
# Token overflow handling
# ---------------------------------------------------------------------------

def handle_token_overflow(chunk: dict) -> list[dict]:
    """If a chunk exceeds MAX_CHUNK_TOKENS, split it.

    Checks full_text (with parent context) if available, else heading+content.
    For level-3 (###) chunks: first try splitting at #### headings.
    If that doesn't help or no #### headings exist, fall back to recursive.
    For level-4 (####) and special/level-1/level-2 chunks: recursive split.
    """
    text = chunk.get("full_text") or _chunk_text(chunk)
    tokens = count_tokens(text)

    if tokens <= config.MAX_CHUNK_TOKENS:
        return [chunk]

    level = chunk.get("heading_level")

    # Level-3: try #### split first
    if level == 3:
        h4_parts = split_at_h4(chunk)
        if len(h4_parts) > 1:
            result = []
            for part in h4_parts:
                result.extend(handle_token_overflow(part))
            return result

    # All cases (level-3 without ####, level-4, special): recursive split
    logger.info(
        "Chunk %s tokens=%d > %d, splitting recursively",
        chunk["chunk_file"], tokens, config.MAX_CHUNK_TOKENS,
    )
    return recursive_split_chunk(chunk)


# ---------------------------------------------------------------------------
# ### (level-3) splitting with parent context
# ---------------------------------------------------------------------------

def split_section_at_h3(section: dict) -> list[dict]:
    """Split a level-2 section chunk at ### headings into level-3 sub-chunks.
    
    Each sub-chunk includes parent heading context as list of dicts:
      - parent_headings: [{"heading_level": 2, "content": "## 1. SAFETY INSTRUCTIONS"}]
    """
    lines = section["content"].split("\n")
    # Root level-2 chunk has no grandparent; use empty list for top-level
    initial_parent_headings = section.get("parent_headings", [])
    # For root ## chunks, the heading IS the parent heading
    parent_heading = section.get("heading")  # e.g. "## 1. SAFETY INSTRUCTIONS"
    parent_chunk_file = section.get("chunk_file")

    # Build initial parent_headings: if this is a root ## chunk, use its heading as level-2
    if initial_parent_headings:
        parent_headings = initial_parent_headings
    elif parent_heading and parent_heading.startswith("##"):
        parent_headings = [{"heading_level": 2, "content": parent_heading}]
    else:
        parent_headings = []

    # Find ### headings
    starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = HEADING3_RE.match(line)
        if m:
            starts.append((i, m.group(1).strip()))

    if not starts:
        # No ### headings: check if this section itself needs token splitting
        return handle_token_overflow(section, parent_headings=parent_headings, parent_chunk_file=parent_chunk_file)

    # Collect content before first ### (if any)
    first_h3_line = starts[0][0]
    preamble = "\n".join(lines[:first_h3_line]).strip()

    children = []
    for idx, (start_i, h3) in enumerate(starts):
        end_i = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        body_lines = lines[start_i + 1:end_i]
        content = "\n".join(body_lines).strip()

        h3_prefixed = f"### {h3}"

        # Determine next heading for suffix
        next_heading = None
        if idx + 1 < len(starts):
            next_heading = f"### {starts[idx + 1][1]}"

        # Build full text with parent context for token counting
        full_text = _build_full_text_with_parent(parent_headings, f"{h3_prefixed}\n\n{content}", next_heading)

        child = {
            **section,
            "chunk_type": "section",
            "heading_level": 3,
            "heading": h3_prefixed,
            "parent_headings": parent_headings,
            "parent_chunk_file": parent_chunk_file,
            "chunk_file": f"{parent_chunk_file}_part{idx}",
            "content": content,
            "full_text": full_text,
        }

        # Check token overflow (may split at #### or recursively)
        children.extend(handle_token_overflow(child))

    # If there's preamble content, add it as a separate chunk
    if preamble:
        preamble_tokens = count_tokens(_build_full_text_with_parent(parent_headings, preamble))
        preamble_chunk = {
            **section,
            "chunk_type": "section",
            "heading_level": 3,
            "heading": None,
            "parent_headings": parent_headings,
            "parent_chunk_file": parent_chunk_file,
            "chunk_file": f"{parent_chunk_file}_preamble",
            "content": preamble,
            "full_text": _build_full_text_with_parent(parent_headings, preamble),
        }
        children.insert(0, preamble_chunk)

    if len(children) > 1:
        logger.debug(
            "Split %s -> %d h3 sub-chunks", parent_chunk_file, len(children)
        )
    return children


# ---------------------------------------------------------------------------
# Main chunk builder
# ---------------------------------------------------------------------------

def build_all_chunks() -> list[dict]:
    """Build the full chunk set with new hierarchical logic.
    
    Pipeline:
      1. Load index records and chunk file contents
      2. Convert old parent_heading (string) to new parent_headings (list of dicts) for backward compatibility
      3. Pre-filter OCR noise from ALL chunk content (special + section)
      4. For each ## chunk:
         - Count tokens
         - If tokens <= MAX_CHUNK_TOKENS: keep as single chunk
         - If tokens > MAX_CHUNK_TOKENS: split at ### headings
      5. For each ### chunk:
         - Add parent heading context
         - If tokens > MAX_CHUNK_TOKENS: split at #### or recursive
      6. Special chunks: same logic
    """
    records = load_index_records(config.INDEX_JSONL)
    logger.info("Loaded %d index records from %s", len(records), config.INDEX_JSONL)

    # Backward compatibility: convert parent_heading (string) to parent_headings (list of dicts)
    converted_records = []
    for rec in records:
        rec = dict(rec)  # shallow copy
        if "parent_heading" in rec and not rec.get("parent_headings"):
            # Convert string parent_heading to list format
            old_parent = rec.pop("parent_heading")
            rec["parent_headings"] = [{"heading_level": _get_heading_level(old_parent), "content": old_parent}]
        converted_records.append(rec)
    records = converted_records

    chunks: list[dict] = []
    for doc in load_chunks(records):
        # Step 1: Pre-filter noise from ALL chunks
        prefilter_content(doc)

        # Step 2: Check token count for the ## chunk
        full_text = _chunk_text(doc)
        tokens = count_tokens(full_text)

        if tokens <= config.MAX_CHUNK_TOKENS:
            # Within limit: keep as single chunk
            chunks.append(doc)
        else:
            # Over limit: split based on chunk type
            if doc.get("chunk_type") == "section":
                # Split section at ### headings
                parts = split_section_at_h3(doc)
                chunks.extend(parts)
            else:
                # Special chunks: split recursively
                chunks.extend(handle_token_overflow(doc))

    # Build full_text for all chunks and update metadata
    for chunk in chunks:
        if "full_text" not in chunk:
            chunk["full_text"] = _chunk_text(chunk)

    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        tokens = count_tokens(chunk.get("full_text", _chunk_text(chunk)))
        logger.info(
            "Chunk %d/%d: %s (type=%s, level=%s, tokens=%d)",
            i,
            total,
            chunk.get("chunk_file"),
            chunk.get("chunk_type"),
            chunk.get("heading_level"),
            tokens,
        )

    # Summary stats
    over_limit = sum(1 for c in chunks if count_tokens(c.get("full_text", _chunk_text(c))) > config.MAX_CHUNK_TOKENS)
    if over_limit:
        logger.warning("%d chunks still exceed %d tokens after splitting", over_limit, config.MAX_CHUNK_TOKENS)
    else:
        logger.info("All %d chunks within %d token limit", total, config.MAX_CHUNK_TOKENS)

    # Write processed chunks to disk
    try:
        write_processed_chunks(chunks)
    except Exception as e:
        logger.error("Failed to write processed chunks: %s", e)

    return chunks


# ---------------------------------------------------------------------------
# Write processed chunks
# ---------------------------------------------------------------------------

def write_processed_chunks(chunks: list[dict], base_dir: Path | None = None) -> None:
    """Write all processed chunks to .md files with parent heading context.
    
    Creates directory structure:
      - sections/*.md
      - special/*.md
    
    Uses _partN suffix for sub-chunks.
    Generates new index.jsonl in chunks_processed/.
    """
    if base_dir is None:
        base_dir = Path(config.PROCESSED_CHUNKS_DIR)

    if not config.WRITE_PROCESSED_CHUNKS:
        logger.info("WRITE_PROCESSED_CHUNKS is disabled, skipping file write")
        return

    base_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    index_records = []

    for chunk in chunks:
        # Determine output subdirectory
        chunk_type = chunk.get("chunk_type", "section")
        if chunk_type == "special":
            out_subdir = base_dir / "special"
        else:
            out_subdir = base_dir / "sections"

        # Build output filename
        chunk_file = chunk.get("chunk_file", "")
        parts = chunk_file.replace("\\", "/").split("/")
        basename = parts[-1] if parts else "chunk.md"

        # Remove .md extension for processing
        if basename.endswith(".md"):
            basename = basename[:-3]

        # Sanitize the basename
        basename = _sanitize_filename(basename)

        # Determine part number from chunk_file
        part_num = ""
        if "_part" in basename:
            # Extract part number from existing name
            pm = re.search(r"_part(\d+)$", basename)
            if pm:
                part_num = f"_part{pm.group(1)}"
                basename = basename[:pm.start()]
        elif chunk.get("split_from"):
            # This chunk was split from another - use _partN
            sf = chunk.get("chunk_file", "")
            pm = re.search(r"_part(\d+)$", sf)
            if pm:
                part_num = f"_part{pm.group(1)}"
            else:
                # Assign sequential part number
                parent_file = chunk.get("split_from")
                matching = [c for c in chunks if c.get("split_from") == parent_file]
                idx = matching.index(chunk) if chunk in matching else 0
                part_num = f"_part{idx}"

        out_filename = basename + part_num + ".md"
        out_path = out_subdir / out_filename

        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content with parent heading context
        content = chunk.get("full_text", chunk.get("content", ""))
        out_path.write_text(content + "\n", encoding="utf-8")
        written += 1

        # Build index record
        rel_path = str(out_path.relative_to(base_dir)).replace("\\", "/")
        index_records.append({
            "chunk_file": rel_path,
            "chunk_type": chunk.get("chunk_type"),
            "heading": chunk.get("heading"),
            "heading_level": chunk.get("heading_level"),
            "parent_headings": chunk.get("parent_headings", []),
            "parent_chunk_file": chunk.get("parent_chunk_file"),
            "pages": chunk.get("pages", []),
            "headers": chunk.get("headers", []),
            "footers": chunk.get("footers", []),
            "page_numbers": chunk.get("page_numbers", []),
            "sources": chunk.get("sources", []),
            "page_sizes": chunk.get("page_sizes", []),
            "token_count": count_tokens(content),
        })

    # Write new index.jsonl
    index_path = base_dir / "index.jsonl"
    with open(index_path, "w", encoding="utf-8") as f:
        for rec in index_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Wrote %d processed chunks to %s", written, base_dir)
    logger.info("Wrote %d index records to %s", len(index_records), index_path)
