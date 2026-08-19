import json
import re
from pathlib import Path

OUT_DIR = Path("/home/dungx/LGI/rag-documents/output/parsed")
JSONL_PATH = OUT_DIR / "documents.jsonl"
MERGED_PATH = OUT_DIR / "merged.md"
CHUNKS_DIR = OUT_DIR / "chunks"
SPECIAL_DIR = CHUNKS_DIR / "special"
SECTIONS_DIR = CHUNKS_DIR / "sections"
INDEX_PATH = CHUNKS_DIR / "index.jsonl"

SPECIAL_KEYWORDS = [
    ("foreword", "FOREWORD"),
    ("warning_signs", "WARNING SIGNS"),
    ("table_of_contents", "TABLE OF CONTENTS"),
]

SECTION_RE = re.compile(r"^##\s")


def load_pages(jsonl_path: Path) -> list[dict]:
    pages = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            md = rec["metadata"]
            pages.append(
                {
                    "page_idx": md["page_idx"],
                    "content": rec["page_content"],
                    "header": md.get("header"),
                    "footer": md.get("footer"),
                    "page_number": md.get("page_number"),
                    "source": md.get("source"),
                    "page_size": md.get("page_size"),
                }
            )
    pages.sort(key=lambda p: p["page_idx"])
    return pages


def slugify(text: str) -> str:
    t = re.sub(r"^#{1,4}\s+", "", text.strip())
    t = re.sub(r"^[•\-\*▶]\s*", "", t)
    t = re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_").lower()
    return t or "section"


def page_meta(page: dict) -> dict:
    return {
        "header": page.get("header"),
        "footer": page.get("footer"),
        "page_number": page.get("page_number"),
        "source": page.get("source"),
        "page_size": page.get("page_size"),
    }


def index_record(path: str, chunk_type: str, heading, level, pages: list[dict]) -> dict:
    metas = [page_meta(p) for p in pages]
    return {
        "chunk_file": path,
        "chunk_type": chunk_type,
        "heading": heading,
        "heading_level": level,
        "pages": [p["page_idx"] for p in pages],
        "headers": [m["header"] for m in metas if m["header"]],
        "footers": [m["footer"] for m in metas if m["footer"]],
        "page_numbers": [m["page_number"] for m in metas if m["page_number"]],
        "sources": [m["source"] for m in metas if m["source"]],
        "page_sizes": [m["page_size"] for m in metas if m["page_size"]],
    }


def build_special_chunks(pages: list[dict]) -> tuple[list[dict], set[int]]:
    special_pages = set()
    chunks = []
    for slug, keyword in SPECIAL_KEYWORDS:
        group = [p for p in pages if keyword.upper() in (p.get("header") or "").upper()]
        if not group:
            continue
        special_pages.update(p["page_idx"] for p in group)
        path = f"special/{slug}.md"
        content = "\n\n".join(p["content"].strip() for p in group)
        (SPECIAL_DIR / f"{slug}.md").write_text(content)
        chunks.append(index_record(path, "special", keyword, 1, group))
    return chunks, special_pages


def build_section_chunks(pages: list[dict]) -> list[dict]:
    chunks = []
    cur = None
    for page in pages:
        for line in page["content"].split("\n"):
            if SECTION_RE.match(line):
                if cur is not None and cur["lines"]:
                    chunks.append(cur)
                cur = {"heading": line.strip(), "lines": [line], "pages": [page["page_idx"]]}
            else:
                if cur is None:
                    cur = {"heading": None, "lines": [], "pages": []}
                cur["lines"].append(line)
                cur["pages"].append(page["page_idx"])
    if cur is not None and cur["lines"]:
        chunks.append(cur)

    records = []
    for i, chunk in enumerate(chunks):
        if chunk["heading"] is None:
            path = "sections/00_front_matter.md"
            level = None
        else:
            level = len(chunk["heading"]) - len(chunk["heading"].lstrip("#"))
            slug = slugify(chunk["heading"])
            path = f"sections/{i:02d}_{slug}.md"
        content = "\n".join(chunk["lines"]).strip()
        (SECTIONS_DIR / Path(path).name).write_text(content + "\n")
        uniq_pages = []
        for idx in chunk["pages"]:
            if not uniq_pages or uniq_pages[-1]["page_idx"] != idx:
                uniq_pages.append(pages_by_idx[idx])
        records.append(index_record(path, "section", chunk["heading"], level, uniq_pages))
    return records


pages_by_idx = {}


def main() -> None:
    global pages_by_idx
    pages = load_pages(JSONL_PATH)
    pages_by_idx = {p["page_idx"]: p for p in pages}

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    SPECIAL_DIR.mkdir(parents=True, exist_ok=True)
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    merged = "\n\n---\n\n".join(p["content"].strip() for p in pages) + "\n"
    MERGED_PATH.write_text(merged)

    records, special_pages = build_special_chunks(pages)
    rest = [p for p in pages if p["page_idx"] not in special_pages]
    records.extend(build_section_chunks(rest))

    with open(INDEX_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Merged {len(pages)} pages -> {MERGED_PATH}")
    print(f"Wrote {len(records)} chunks -> {CHUNKS_DIR}")


if __name__ == "__main__":
    main()