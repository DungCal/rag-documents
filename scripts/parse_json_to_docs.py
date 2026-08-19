import json
import pickle
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

from langchain_core.documents import Document

HEADER_RE = re.compile(r"^[A-Z]\.?\s")
SECTION_RE = re.compile(r"^\d+\.\s")
CAPTION_RE = re.compile(r"^▶\s")
BULLET_RE = re.compile(r"^[•\-\*]\s")
ADMONITION_WORDS = {"IMPORTANT", "WARNING", "DANGER", "CAUTION", "NOTE"}


def detect_header_level(text: str, prev_level: int = 2) -> int:
    t = text.strip()
    if HEADER_RE.match(t):
        return 1
    if SECTION_RE.match(t):
        return 2
    if CAPTION_RE.match(t):
        return 3
    if BULLET_RE.match(t):
        return 4
    return min(prev_level + 1, 4)


def block_header(text: str, level: int = 2) -> str:
    return "#" * level + " " + text.strip()


def extract_spans(block: dict) -> list[str]:
    out = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            content = span.get("content")
            if content:
                out.append(content.strip())
    return out


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self._row = []
        self._cell = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
            self._in_cell = True
        elif tag == "img" and self._in_cell:
            src = dict(attrs).get("src", "")
            if src:
                self._cell.append(f"![]({src})")

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._row.append("".join(self._cell).strip())
            self._in_cell = False
        elif tag == "tr":
            if self._row:
                self.rows.append(self._row)

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)


def html_table_to_md(html: str) -> str:
    p = _TableParser()
    p.feed(html)
    rows = p.rows
    if not rows:
        return html
    widths = max(len(r) for r in rows)
    out = []
    for i, row in enumerate(rows):
        row = [c.replace("|", "\\|") for c in row]
        row = row + [""] * (widths - len(row))
        out.append("| " + " | ".join(row) + " |")
        if i == 0:
            out.append("| " + " | ".join(["---"] * widths) + " |")
    return "\n".join(out)


def extract_lines(block: dict, state: dict) -> list[str]:
    texts = []
    btype = block.get("type")
    if btype == "title":
        for t in extract_spans(block):
            level = detect_header_level(t, state["last_level"])
            state["last_level"] = level
            texts.append(block_header(t, level))
    elif btype == "text":
        texts.extend(extract_spans(block))
    elif btype == "list":
        for sub in block.get("blocks", []):
            texts.extend(extract_lines(sub, state))
    elif btype == "table":
        for sub in block.get("blocks", []):
            for line in sub.get("lines", []):
                for span in line.get("spans", []):
                    html = span.get("html")
                    if html:
                        texts.append(html_table_to_md(html))
                    elif span.get("content"):
                        texts.append(span["content"].strip())
    elif btype == "chart":
        for sub in block.get("blocks", []):
            for line in sub.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("content"):
                        texts.append(span["content"].strip())
                    img = span.get("image_path")
                    if img:
                        texts.append(f"![]({img})")
    elif btype == "image":
        for sub in block.get("blocks", []):
            stype = sub.get("type")
            if stype == "image_caption":
                for t in extract_spans(sub):
                    level = detect_header_level(t, state["last_level"])
                    state["last_level"] = level
                    texts.append(block_header(t, level))
            elif stype == "image_footnote":
                texts.extend(extract_spans(sub))
            elif stype == "image_body":
                for line in sub.get("lines", []):
                    for span in line.get("spans", []):
                        img = span.get("image_path")
                        if img:
                            texts.append(f"![]({img})")
    return texts


def extract_metadata(page: dict, source: str) -> dict:
    md = {
        "page_idx": page.get("page_idx"),
        "source": source,
        "page_size": page.get("page_size"),
    }
    headers, footers, page_numbers = [], [], []
    for b in page.get("discarded_blocks", []):
        content = " ".join(extract_spans(b)).strip()
        if not content:
            continue
        t = b.get("type")
        if t == "header":
            headers.append(content)
        elif t == "footer":
            footers.append(content)
        elif t == "page_number":
            page_numbers.append(content)
    if headers:
        md["header"] = " ".join(headers)
    if footers:
        md["footer"] = " ".join(footers)
    if page_numbers:
        md["page_number"] = " ".join(page_numbers)
    return md


def page_to_document(page: dict, source: str, state: dict) -> Document:
    blocks = page.get("para_blocks") or page.get("preproc_blocks") or []
    parts = []
    for b in blocks:
        parts.extend(extract_lines(b, state))
    content = "\n\n".join(p for p in parts if p.strip())
    return Document(
        page_content=content,
        metadata=extract_metadata(page, source),
    )


def json_to_documents(json_path: str, limit: int | None = None) -> list[Document]:
    json_path = Path(json_path)
    source = json_path.stem
    with open(json_path) as f:
        data = json.load(f)
    pages = data["pdf_info"]
    if limit is not None:
        pages = pages[:limit]
    state = {"last_level": 2}
    docs = [page_to_document(p, source, state) for p in pages]
    return docs


def save_documents(docs: list[Document], out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "documents.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps({"page_content": d.page_content, "metadata": d.metadata}) + "\n")
    with open(out_dir / "documents.pkl", "wb") as f:
        pickle.dump(docs, f)
    md_dir = out_dir / "markdown"
    md_dir.mkdir(parents=True, exist_ok=True)
    for d in docs:
        idx = d.metadata.get("page_idx", 0)
        (md_dir / f"page_{idx:03d}.md").write_text(d.page_content)
    print(f"Saved {len(docs)} docs to {out_dir}")


if __name__ == "__main__":
    default = (
        "/home/dungx/LGI/rag-documents/output/pipeline_0812_effort-high/"
        "t130sp_na_operator_manual/hybrid_ocr/t130sp_na_operator_manual_middle.json"
    )
    src = sys.argv[1] if len(sys.argv) > 1 else default
    out = "/home/dungx/LGI/rag-documents/output/parsed"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    docs = json_to_documents(src, limit=limit)
    save_documents(docs, out)
    print(f"Parsed {len(docs)} documents")
