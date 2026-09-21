"""Noise line detection and filtering for chunk content."""

import re

from . import config
from .logging_config import logger

# Pre-compiled patterns for common noise
_NOISE_PATTERNS = [
    re.compile(r"^[0\s.]+$"),           # Only zeros, dots, spaces
    re.compile(r"^[•\s|]+$"),           # Only bullets, pipes, spaces
    re.compile(r"^[\s•|0.]+$"),         # Mixed noise chars
    re.compile(r"^\(?\d+\)?\s*\.\.\."), # "(1)......" patterns
    re.compile(r"^\.\.\.\.\.\.\s*$"),   # "......" lines
]


def is_noise_line(line: str, threshold: float, noise_chars: set, min_len: int) -> bool:
    """Return True if a line is primarily noise characters (OCR artifacts).

    Noise patterns include repeated '0', '.', '•', '|', spaces, etc.
    Lines shorter than min_len are never considered noise (preserves short text).
    """
    if len(line) < min_len:
        return False

    # Quick check: if line matches any known noise pattern, skip threshold check
    for pat in _NOISE_PATTERNS:
        if pat.match(line):
            return True

    # Fallback: threshold-based check
    noise_count = sum(1 for c in line if c in noise_chars or c.isspace())
    return (noise_count / len(line)) >= threshold


def filter_noise(
    content: str,
    threshold: float | None = None,
    noise_chars: set | None = None,
    min_len: int | None = None,
) -> str:
    """Remove noise lines from content.

    Uses config defaults when parameters are None.
    Preserves heading lines (##, ###, ####) and empty lines for structure.
    """
    if threshold is None:
        threshold = config.NOISE_CHAR_THRESHOLD
    if noise_chars is None:
        noise_chars = config.NOISE_CHARS
    if min_len is None:
        min_len = config.NOISE_MIN_LINE_LENGTH

    lines = content.split("\n")
    filtered = []
    removed = 0
    for line in lines:
        stripped = line.strip()
        # Always keep headings (preserve markdown structure)
        if stripped.startswith("#"):
            filtered.append(line)
            continue
        # Always keep empty lines (paragraph separators)
        if not stripped:
            filtered.append(line)
            continue
        if is_noise_line(stripped, threshold, noise_chars, min_len):
            removed += 1
            continue
        filtered.append(line)

    if removed:
        logger.debug("Noise filter removed %d lines", removed)

    # Collapse multiple consecutive empty lines into one
    result = []
    prev_empty = False
    for line in filtered:
        is_empty = not line.strip()
        if is_empty and prev_empty:
            continue
        result.append(line)
        prev_empty = is_empty

    return "\n".join(result)
