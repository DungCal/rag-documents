#!/usr/bin/env python
"""Test noise filter, token counting, and chunk splitting on sample files.

Usage:
    python -m rag_index.test_noise_filter
"""

from pathlib import Path

from . import config
from .chunker import build_all_chunks, _chunk_text, _build_full_text_with_parent
from .noise_filter import filter_noise, is_noise_line
from .token_counter import count_tokens

SAMPLE_FILES = [
    Path("output/parsed/chunks/sections/12_5_universal_symbols.md"),
    Path("output/parsed/chunks/sections/57_1_standard_for_farmwork.md"),
    Path("output/parsed/chunks/sections/11_4_safety_decals.md"),
]


def test_noise_filter():
    """Test noise line detection and filtering on sample chunk files."""
    print("=" * 60)
    print("TEST 1: Noise Filter")
    print("=" * 60)

    import sys
    def safe_print(text):
        try:
            sys.stdout.buffer.write((str(text) + "\n").encode("utf-8", errors="replace"))
        except Exception:
            print(repr(text))

    for path in SAMPLE_FILES:
        if not path.exists():
            safe_print(f"\nSKIP: {path} not found")
            continue

        content = path.read_text(encoding="utf-8")
        orig_tokens = count_tokens(content)
        orig_lines = len(content.splitlines())

        filtered = filter_noise(content)
        filt_tokens = count_tokens(filtered)
        filt_lines = len(filtered.splitlines())

        reduction = 100 * (1 - filt_tokens / orig_tokens) if orig_tokens else 0

        safe_print(f"\n--- {path.name} ---")
        safe_print(f"  Original:   {orig_tokens} tokens, {orig_lines} lines")
        safe_print(f"  Filtered:   {filt_tokens} tokens, {filt_lines} lines")
        safe_print(f"  Reduction:  {reduction:.1f}%")


def test_token_counter():
    """Verify token counting works correctly."""
    print("\n" + "=" * 60)
    print("TEST 2: Token Counter")
    print("=" * 60)

    test_texts = [
        ("Hello world", 2),
        ("0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0", 50),
    ]

    for text, expected_approx in test_texts:
        tokens = count_tokens(text)
        print(f"  Text: '{text[:50]}...' -> {tokens} tokens (expected ~{expected_approx})")


def test_parent_context():
    """Test parent heading context building."""
    print("\n" + "=" * 60)
    print("TEST 3: Parent Heading Context")
    print("=" * 60)

    parent = "## 4. SAFETY DECALS"
    content = "### ▶ GENERAL INFORMATION OF DECALS\n\nIn order to work with the machine safely..."
    next_heading = "### ▶ DECALS ON CHASSIS"

    full_text = _build_full_text_with_parent(parent, content, next_heading)
    tokens = count_tokens(full_text)

    # Use encode to handle Unicode in Windows console
    import sys
    def safe_print(text):
        try:
            sys.stdout.buffer.write((str(text) + "\n").encode("utf-8", errors="replace"))
        except Exception:
            print(repr(text))

    safe_print(f"\nParent: {parent}")
    safe_print(f"Content: {content[:60]}...")
    safe_print(f"Next heading: {next_heading}")
    safe_print(f"\nFull text with context:\n{full_text}")
    safe_print(f"\nTokens: {tokens}")


def test_full_chunker():
    """Test the full chunking pipeline with new hierarchical logic."""
    print("\n" + "=" * 60)
    print("TEST 4: Full Chunker Pipeline (New Logic)")
    print("=" * 60)

    import sys
    def safe_print(text):
        try:
            sys.stdout.buffer.write((str(text) + "\n").encode("utf-8", errors="replace"))
        except Exception:
            print(repr(text))

    try:
        chunks = build_all_chunks()
    except Exception as e:
        safe_print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    safe_print(f"\n  Total chunks: {len(chunks)}")

    # Count by type
    type_counts = {}
    level_counts = {}
    for c in chunks:
        ct = c.get("chunk_type", "unknown")
        hl = c.get("heading_level", "?")
        type_counts[ct] = type_counts.get(ct, 0) + 1
        level_counts[hl] = level_counts.get(hl, 0) + 1

    safe_print(f"  By type: {type_counts}")
    safe_print(f"  By level: {level_counts}")

    # Token stats
    token_counts = []
    for c in chunks:
        t = count_tokens(c.get("full_text", _chunk_text(c)))
        token_counts.append(t)

    if token_counts:
        safe_print(f"  Token stats:")
        safe_print(f"    Min: {min(token_counts)}")
        safe_print(f"    Max: {max(token_counts)}")
        safe_print(f"    Avg: {sum(token_counts) / len(token_counts):.0f}")
        safe_print(f"    Total: {sum(token_counts)}")

    # Over-limit chunks
    over = [(c, count_tokens(c.get("full_text", _chunk_text(c)))) for c in chunks if count_tokens(c.get("full_text", _chunk_text(c))) > config.MAX_CHUNK_TOKENS]
    if over:
        safe_print(f"\n  WARNING: {len(over)} chunks exceed {config.MAX_CHUNK_TOKENS} tokens:")
        for c, t in over[:10]:
            safe_print(f"    {c.get('chunk_file')}: {t} tokens")
    else:
        safe_print(f"\n  All chunks within {config.MAX_CHUNK_TOKENS} token limit")

    # Show some split examples
    split_chunks = [c for c in chunks if c.get("split_from") or "_part" in c.get("chunk_file", "")]
    if split_chunks:
        safe_print(f"\n  Split examples (from hierarchical splitting):")
        for c in split_chunks[:10]:
            safe_print(f"    {c.get('chunk_file')}: {count_tokens(c.get('full_text', _chunk_text(c)))} tokens")

    # Show chunks with parent context
    parent_chunks = [c for c in chunks if c.get("parent_heading")]
    if parent_chunks:
        safe_print(f"\n  Chunks with parent context: {len(parent_chunks)}")
        for c in parent_chunks[:5]:
            safe_print(f"    {c.get('chunk_file')}: parent={c.get('parent_heading')}")


def main():
    test_noise_filter()
    test_token_counter()
    test_parent_context()
    test_full_chunker()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
