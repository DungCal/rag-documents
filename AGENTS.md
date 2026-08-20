# AGENTS.md

## System Overview

This repository implements a **document intelligence pipeline** that converts PDF technical manuals (currently a TYM tractor operator manual) into structured, chunked, indexable knowledge assets ready for Retrieval-Augmented Generation (RAG) workflows.

The system is composed of two layers that collaborate in a linear pipeline:

1. **Ingestion & Serving Layer (Dockerized Minero services)** — A set of four GPU-backed container services built from a single image (`mineru:latest`) that wrap the Mineru document-understanding engine. They expose four distinct interfaces (OpenAI-compatible API, batch conversion API, router, and Gradio UI) and produce raw parsed JSON for each processed PDF.

2. **Processing & Indexing Layer (Python scripts)** — Two offline scripts that transform Mineru's raw JSON output into LangChain `Document` objects, then merge and split those documents into semantic chunks (front matter, special pages, and `##`-section-based chunks), emitting a searchable index with preserved page metadata.

Data flows strictly forward: **PDF → Mineru parse → JSON → LangChain Documents → Chunks + Index**. There is no feedback loop or shared state store; each node consumes the previous node's artifacts from disk (JSONL / JSON files) or via bind-mounted directories.

All model inference runs fully offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) against a locally bundled HuggingFace model cache, and all services require an NVIDIA GPU.

---

## Agent Profiles

### 1. `mineru-openai-server` (Docker Service)

- **Name:** `mineru-openai-server`
- **Role:** Exposes Mineru's document conversion and comprehension capabilities through an OpenAI-compatible HTTP API, allowing external agents/LLM tooling to submit documents and receive structured results using a familiar chat/endpoint contract.
- **Tools & Capabilities:**
  - Mineru OpenAI-compatible server binary (`mineru-openai-server` entrypoint)
  - GPU-accelerated model inference (NVIDIA device `0`, configurable via `device_ids`)
  - Offline local model loading (`MINERU_MODEL_SOURCE=local`, HF/Transformers offline flags)
  - Health check endpoint on `:30000/health`
- **Data Flow / Dependencies:**
  - **Input:** HTTP requests from clients; local model weights from `/root/.cache/huggingface` (bind-mounted, read-only); tool configuration from `/root/mineru.json` (bind-mounted, read-only).
  - **Output:** OpenAI-compatible HTTP responses containing parsed document structure.
  - **Interactions:** Standalone entrypoint — not consumed by other services in this repo; intended as a client-facing server. Shares the same offline model volume as all sibling services.

### 2. `mineru-api` (Docker Service)

- **Name:** `mineru-api`
- **Role:** The primary batch document conversion API. Ingests PDF files from a host input directory and writes parsed Mineru JSON output to a host output directory.
- **Tools & Capabilities:**
  - Mineru API binary (`mineru-api` entrypoint)
  - GPU-accelerated conversion pipeline (vLLM engine, `--gpu-memory-utilization 0.40`)
  - Health check endpoint on `:8000/health` (30s interval, 180s start period)
  - Bind-mounted I/O directories (input read-only, output read-write)
- **Data Flow / Dependencies:**
  - **Input:** PDFs placed in `/home/victor/mineru/input` → mounted at `/data/input` (read-only); model weights and `mineru.json` config from shared offline volume.
  - **Output:** Parsed results written to `/data/output` (→ host `/home/victor/mineru/output`).
  - **Interactions:** Produces the raw JSON consumed by the `parse_json_to_docs.py` processing node. Serves as the model-inference engine that the `mineru-router` can optionally aggregate.

### 3. `mineru-router` (Docker Service)

- **Name:** `mineru-router`
- **Role:** Orchestration and load-balancing node. Routes conversion requests across a fleet of Mineru backends — either by spawning local GPU workers or by proxying to existing `mineru-api` upstreams.
- **Tools & Capabilities:**
  - Mineru router binary (`mineru-router` entrypoint)
  - Local GPU worker pool management (`--local-gpus auto`)
  - Upstream aggregation mode (commented-out `--upstream-url http://mineru-api:8000` for horizontal scaling)
  - Optional public HTTP client backend (disabled by default to mitigate SSRF risk)
  - Health check endpoint on `:8002/health`
- **Data Flow / Dependencies:**
  - **Input:** Client conversion requests on `:8002`; local model weights from the shared offline volume.
  - **Output:** Routed requests dispatched to either locally spawned workers or configured upstream `mineru-api` instances.
  - **Interactions:** Sits above the conversion layer; directs work to `mineru-api` backends (or local equivalents) and returns aggregated results to callers.

### 4. `mineru-gradio` (Docker Service)

- **Name:** `mineru-gradio`
- **Role:** Interactive human-in-the-loop interface. Provides a Gradio web UI for uploading documents and inspecting/chatting with conversion results without writing code.
- **Tools & Capabilities:**
  - Mineru Gradio binary (`mineru-gradio` entrypoint)
  - GPU-accelerated inference (vLLM engine, configurable `--gpu-memory-utilization`)
  - Optional API endpoint (`--enable-api`) and page-limit controls (`--max-convert-pages`)
  - Web UI served on `:7860`
- **Data Flow / Dependencies:**
  - **Input:** User-uploaded documents via the web UI; model weights and `mineru.json` config from the shared offline volume.
  - **Output:** Converted documents rendered in the UI and exposed via the optional HTTP API.
  - **Interactions:** Standalone front-end over the same Mineru engine; no coupling to the other services beyond shared model/config volumes.

### 5. `parse_json_to_docs.py` (Processing Node / Script)

- **Name:** `parse_json_to_docs.py`
- **Role:** Converts Mineru's raw per-page JSON (`pdf_info` records) into LangChain `Document` objects, reconstructing readable Markdown and capturing page-level metadata.
- **Tools & Capabilities:**
  - Mineru block-type dispatch (`title`, `text`, `list`, `table`, `chart`, `image`) → Markdown rendering
  - Custom `HTMLParser`-based table → Markdown table converter (`html_table_to_md`)
  - Heuristic heading-level detection (`detect_header_level`) via regex patterns for section numbers, captions (`▶`), bullets, and admonitions
  - Admonition awareness (`IMPORTANT`, `WARNING`, `DANGER`, `CAUTION`, `NOTE`)
  - Image/chart span extraction and embedding as Markdown image links
  - Metadata extraction of headers, footers, and page numbers from `discarded_blocks`
  - Serialization to `documents.jsonl`, pickled `list[Document]`, and per-page `markdown/` files
- **Data Flow / Dependencies:**
  - **Input:** A Mineru JSON file (e.g. `t130sp_na_operator_manual_middle.json`) — output of the `mineru-api` service.
  - **Output:** `documents.jsonl` + `documents.pkl` + `markdown/page_XXX.md` in `output/parsed/`.
  - **Interactions:** Feeds `merge_and_chunk.py` via `documents.jsonl`. Relies on `langchain_core.documents.Document`.

### 6. `merge_and_chunk.py` (Processing Node / Script)

- **Name:** `merge_and_chunk.py`
- **Role:** The indexing/structuring node. Merges parsed pages into a single document, extracts special front-matter pages as dedicated chunks, splits the remainder into heading-based sections, and writes a searchable chunk index with full provenance metadata.
- **Tools & Capabilities:**
  - Page merge → `merged.md`
  - Special-chunk builder (`build_special_chunks`) for `FOREWORD`, `WARNING SIGNS`, and `TABLE OF CONTENTS` (matched via page headers)
  - Section-chunk builder (`build_section_chunks`) splitting content on `##` headings, excluding pages already captured as special chunks
  - Slug generation (`slugify`) for deterministic chunk filenames
  - Index record assembly (`index_record`) capturing chunk type, heading level, page indices, headers, footers, page numbers, source, and page sizes
- **Data Flow / Dependencies:**
  - **Input:** `output/parsed/documents.jsonl` (produced by `parse_json_to_docs.py`).
  - **Output:** `output/parsed/merged.md`, `output/parsed/chunks/special/*.md`, `output/parsed/chunks/sections/*.md`, and `output/parsed/chunks/index.jsonl`.
  - **Interactions:** Terminal node of the pipeline — its `index.jsonl` and chunk files are the final RAG-ready assets consumed downstream by retrieval systems.

---

## Data Flow & Interaction Map

```text
                    ┌─────────────────────────────────────────────┐
                    │            INGESTION / SERVING LAYER        │
                    │              (Docker services, GPU)         │
                    └─────────────────────────────────────────────┘
   PDF input ─────────────────────────────────────────────────────────────┐
      │                                                                   │
      ▼                                                                   │
  mineru-api (port 8000) ◀──────────────────────────┐                     │
   bind mount: /home/victor/mineru/input (ro)       │ (optional upstream) │
      │  parsed Mineru JSON  ───────────────────────┼─────────────────────┤
      ▼                                             ▼                     │
  parse_json_to_docs.py                     mineru-router (port 8002)     │
      │  documents.jsonl/.pkl/markdown      ── routes to local workers    │
      ▼                                             or upstream APIs       │
  merge_and_chunk.py                          mineru-openai-server (30000)│
      │  chunks/ + index.jsonl                 mineru-gradio (7860)        │
      ▼                                          (client-facing, standalone)
  RAG-ready assets
  (output/parsed/chunks, merged.md, index.jsonl)
```

**Key contract points:**

- `mineru-api` output (JSON) → `parse_json_to_docs.py` input (CLI arg, default `output/pipeline_0812_effort-high/.../t130sp_na_operator_manual_middle.json`)
- `parse_json_to_docs.py` output (`documents.jsonl`) → `merge_and_chunk.py` input (`output/parsed/documents.jsonl`)
- `merge_and_chunk.py` output (`index.jsonl`) → downstream retrieval/indexing consumers
- Shared model/config volume (`./mineru-models/mineru-download/...` → `/root/.cache/huggingface`, `/root/mineru.json`) is common to all four services; all run with offline mode enabled.