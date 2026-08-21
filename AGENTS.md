# AGENTS.md

## System Overview

This repository implements a **document intelligence pipeline** that converts PDF technical manuals (currently a TYM tractor operator manual) into structured, chunked, indexable knowledge assets ready for Retrieval-Augmented Generation (RAG) workflows.

The system is composed of three layers that collaborate in a linear pipeline:

1. **Ingestion & Serving Layer (Dockerized Minero services)** — A set of four GPU-backed container services built from a single image (`mineru:latest`) that wrap the Mineru document-understanding engine. They expose four distinct interfaces (OpenAI-compatible API, batch conversion API, router, and Gradio UI) and produce raw parsed JSON for each processed PDF. Each service is gated behind its own Compose profile (`openai-server`, `api`, `router`, `gradio`).

2. **Processing & Indexing Layer (Python scripts)** — Two offline scripts that transform Mineru's raw JSON output into LangChain `Document` objects, then merge and split those documents into semantic chunks (front matter, special pages, and `##`-section-based chunks), emitting a searchable index with preserved page metadata.

3. **Retrieval & Vector-Indexing Layer (`rag_index/` Python package)** — A cloud-connected module that re-splits section chunks hierarchically at `###` headings, embeds them with BGE-M3 (dense, 1024-dim) plus SPLADE (sparse), upserts both into a Pinecone hybrid index, and serves hybrid search queries with parent-section result merging.

Data flows strictly forward: **PDF → Mineru parse → JSON → LangChain Documents → Chunks + Index → Pinecone vectors → hybrid retrieval**. There is no feedback loop or shared state store; each node consumes the previous node's artifacts from disk (JSONL / JSON files) or via bind-mounted directories.

GPU model inference for document parsing runs fully offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) against a locally bundled HuggingFace model cache, and all Docker services require an NVIDIA GPU. The `rag_index/` layer is the exception: it calls hosted services (Pinecone and the HuggingFace Inference API) and needs no GPU.

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
  - **Interactions:** Terminal node of the offline pipeline — its `index.jsonl` and chunk files are the input assets consumed by the `rag_index/` retrieval layer.

### 7. `rag_index/` (Retrieval & Vector-Indexing Package)

A Python package (see `rag_index/requirements.txt`, configured via `.env`; see `.env.example`) that turns the offline chunks into queryable Pinecone vector indexes in two flavors: **hybrid** (dense + SPLADE sparse) and **dense-only**. Entrypoints are run as modules: `python -m rag_index.index [--index-type {hybrid,dense}] [--limit N]` and `python -m rag_index.query [--index-type {hybrid,dense}] "..."`. Both indexes must be created (indexed) before queries can target them.

#### 7a. `rag_index/config.py`

- **Role:** Central, env-driven configuration. Loads `.env` via `python-dotenv`.
- **Settings:** `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, `PINECONE_INDEX_NAME` (hybrid index), `PINECONE_DENSE_INDEX_NAME` (dense-only index, default `rag-documents-ubuntu-dense`), `HF_TOKEN`, `BGE_M3_MODEL` (default `BAAI/bge-m3`), chunk paths (`output/parsed/chunks`, `index.jsonl`), `EMBEDDING_DIM=1024`, `METRIC=dotproduct`, and logging options (`LOG_LEVEL`, `LOG_FILE=logs/indexing.log`, `LOG_TO_FILE`).

#### 7b. `rag_index/chunker.py`

- **Role:** Hierarchical re-splitting of the offline chunks.
- **Tools & Capabilities:**
  - Loads `index.jsonl` records and their chunk files (`load_index_records`, `load_chunks`)
  - `split_section_at_h3`: splits each level-2 section at `### ` headings into level-3 child chunks that inherit parent metadata (`parent_heading`, `parent_chunk_file`) with composite IDs (`parent.md#heading`)
  - `build_all_chunks`: special chunks pass through unchanged; sections get the h3 split
- **Data Flow / Dependencies:** Input `index.jsonl` + `chunks/` from `merge_and_chunk.py`; output: list of chunk dicts consumed by the embedder/indexer.

#### 7c. `rag_index/embedder.py`

- **Role:** Dense embeddings via BGE-M3 through the HuggingFace Inference API (`InferenceClient.feature_extraction`).
- **Tools & Capabilities:**
  - `BGE_M3_Embedder.embed/embed_documents` producing 1024-dim vectors (unwraps batch-shaped responses; empty text → zero vector)
  - Exponential-backoff retry with jitter on transient HF errors (429/502/503/504)
- **Data Flow / Dependencies:** Requires `HF_TOKEN`; text per chunk = heading + content.

#### 7d. `rag_index/indexer.py`

- **Role:** Pinecone upsert in two flavors.
- **Tools & Capabilities:**
  - `PineconeIndexer.ensure_index`: creates the hybrid serverless index if missing (AWS, default region `us-east-1`, `dotproduct`, dense+sparse capable)
  - `DenseIndexer.ensure_index`: same spec, targeting `PINECONE_DENSE_INDEX_NAME`
  - SPLADE sparse encoding via `pinecone_text.SpladeEncoder` (hybrid only)
  - `upsert_chunks` (both classes): one vector per chunk — slugified ID from `chunk_file`, dense values, cleaned metadata (drops `None`s; content, headings, page numbers, sources); the hybrid indexer adds sparse values, the dense indexer omits them
- **Data Flow / Dependencies:** Requires `PINECONE_API_KEY`; consumes chunk dicts + dense embeddings.

#### 7e. `rag_index/searcher.py`

- **Role:** Retrieval with hierarchical result merging, in two flavors.
- **Tools & Capabilities:**
  - `HybridSearcher.search`: single Pinecone query carrying both dense (BGE-M3) and sparse (SPLADE) query vectors against `PINECONE_INDEX_NAME`
  - `DenseSearcher.search`: dense-only Pinecone query against `PINECONE_DENSE_INDEX_NAME`
  - `merge_by_parent` (module-level helper; both searchers expose it as a static method): groups level-3 hits by `parent_chunk_file` into full-section results (max score, concatenated content, unioned pages/sources) — the parent-document retrieval strategy
- **Data Flow / Dependencies:** Queries the index created by the matching indexer class; returns ranked merged results.

#### 7f. CLI entrypoints

- **`rag_index/index.py`** — Full indexing pipeline: build chunks → filter empties → embed → upsert. `--index-type {hybrid,dense}` selects the target index (default `hybrid`). Usage: `python -m rag_index.index --index-type dense [--limit N]`.
- **`rag_index/query.py`** — Search CLI with optional parent merging (`--merge/--no-merge`, `--top-k`). `--index-type {hybrid,dense}` selects the source index (default `hybrid`). Usage: `python -m rag_index.query --index-type dense "how do I adjust the seat belt"`.
- **`rag_index/logging_config.py`** — Shared `rag_index` logger: console always, optional rotating file handler at `logs/indexing.log`.

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
┌───────────────────────────────────────────────────────────────────────┐
│                 RETRIEVAL / VECTOR-INDEXING LAYER                     │
│                       (rag_index/, cloud APIs)                        │
└───────────────────────────────────────────────────────────────────────┘
  rag_index.index: chunker (h3 split) → BGE-M3 embedder
      │  dense vectors (+ SPLADE sparse for hybrid)
      ▼
  ┌─── --index-type hybrid ─────────────┐  ┌─── --index-type dense ──────────┐
  │ PineconeIndexer.upsert_chunks       │  │ DenseIndexer.upsert_chunks      │
  ▼                                     │  ▼                                 │
  PINECONE_INDEX_NAME                   │  PINECONE_DENSE_INDEX_NAME         │
  (rag-documents, hybrid)               │  (rag-documents-ubuntu-dense)      │
      ▲ query (dense+sparse)            │      ▲ query (dense only)          │
      │                                 │      │                             │
  HybridSearcher.search                 │  DenseSearcher.search              │
      └──────── rag_index.query ────────┴──────┘                             │
                    │                                                        │
                    ▼                                                        │
          merge_by_parent → ranked results ◄─────────────────────────────────┘
```

**Key contract points:**

- `mineru-api` output (JSON) → `parse_json_to_docs.py` input (CLI arg, default `output/pipeline_0812_effort-high/.../t130sp_na_operator_manual_middle.json`)
- `parse_json_to_docs.py` output (`documents.jsonl`) → `merge_and_chunk.py` input (`output/parsed/documents.jsonl`)
- `merge_and_chunk.py` output (`index.jsonl`) → `rag_index/chunker.py` input (`config.INDEX_JSONL`)
- `rag_index/index.py` upserts into one of two Pinecone indexes selected by `--index-type`: hybrid (`PINECONE_INDEX_NAME`) or dense-only (`PINECONE_DENSE_INDEX_NAME`, default `rag-documents-ubuntu-dense`); both must be indexed before queries can target them
- `rag_index/query.py` retrieves from the index matching its `--index-type`; both searchers apply the same `merge_by_parent` strategy
- Shared model/config volume (`./mineru-models/mineru-download/...` → `/root/.cache/huggingface`, `/root/mineru.json`) is common to all four services; all run with offline mode enabled.
- Docker services are activated per profile: `docker compose --profile api up` (etc.); only one service runs at a time by default since all share GPU device 0.