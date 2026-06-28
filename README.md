# RAG Question-Answering

A local Retrieval-Augmented Generation system that answers questions about your
own documents, with source citations. Runs **fully offline** on a consumer GPU
(built/tested on an RTX 4060).

## Architecture

```
Indexing (once):   data/ ──> extract+OCR ──> chunk ──> embed ──> Chroma
Query (per ask):   question ──> [rewrite] ──> embed ──> hybrid retrieve (vector + BM25, RRF)
                            ──> cross-encoder re-rank ──> LLM (streamed) ──> answer + sources
```

Framework-light by design — each pipeline step is a small, readable module rather
than hidden behind LangChain, so every stage is inspectable.

| Step | Module | Tool |
|------|--------|------|
| Load + chunk | `ingest.py` | PyMuPDF text + EasyOCR fallback for image/slide pages |
| Embed | `embeddings.py` | sentence-transformers (`bge-base-en-v1.5`, 768-dim) |
| Store + search | `vectorstore.py` | Chroma (cosine) |
| Hybrid retrieve | `retrieval.py` | BM25 (`rank_bm25`) + vector, fused with RRF |
| Query rewrite (opt) | `rewrite.py` | rephrasings + HyDE (off by default) |
| Re-rank | `reranker.py` | cross-encoder (`bge-reranker-base`) |
| Generate | `llm.py` | Ollama + Mistral 7B (streamed) |
| Orchestrate | `rag.py` | `retrieve()` + `answer()` |
| Evaluate | `evaluate.py` | retrieval metrics + LLM-judge faithfulness |

Every stage is toggleable in `config.py` and per-query via `rag.retrieve(...)` /
the Streamlit sidebar.

## What you need running

Two independent pieces:

1. **Ollama** — the local LLM server. Installs as a Windows app that auto-starts in
   the tray and listens on `localhost:11434`. Check with `ollama list`; it must have
   the `mistral` model (`ollama pull mistral` once).
2. **The Python env `rag`** — conda env at `%USERPROFILE%\miniconda3\envs\rag` with
   all dependencies (torch+CUDA, sentence-transformers, chromadb, PyMuPDF, EasyOCR…).

Models (bge embedder/reranker, EasyOCR weights) download themselves on first use and
are cached. Everything after that is offline.

## Operations — start / stop / kill

**Easiest: double-click `start.bat`.** It checks Ollama is running (starts it if not),
makes sure the `mistral` model is present, launches the UI, and opens the browser.
**Closing that window shuts everything down** — the UI (and Ollama, if the launcher
started it) are tied to the window via a Job Object, so no manual cleanup is needed.
Ctrl+C also unloads the model from VRAM. (`start.bat` just runs `run.ps1` with the
execution policy relaxed.)

For manual control, the commands below run from the project folder
(`%USERPROFILE%\Desktop\RAG`). Activate the env first with `conda activate rag` (from an
Anaconda Prompt), or call the env's Python directly:
`%USERPROFILE%\miniconda3\envs\rag\python.exe`.

```powershell
# --- START THE UI ---
streamlit run src/app.py
#   -> open http://localhost:8501   (Ctrl+C in this terminal to stop it)

# --- ASK FROM THE TERMINAL (no UI) ---
python src/query.py "your question here"

# --- (RE)BUILD THE INDEX  (after changing files in data/) ---
python src/index.py            # incremental: only new/changed files
python src/index.py --rebuild  # wipe and rebuild from scratch

# --- STOP THINGS ---
#   UI:            Ctrl+C in its terminal, or:
taskkill /F /IM streamlit.exe
#   stuck job:     kills every Python process (use when something hangs)
taskkill /F /IM python.exe
#   free the LLM from VRAM early (it auto-unloads after 30 min idle):
ollama stop mistral
```

Notes:
- **Don't run `src/index.py` while the UI is open** — both open the same Chroma DB and a
  concurrent write can clash. Stop the UI first (`taskkill /F /IM streamlit.exe`).
- The UI keeps the embed/re-rank models warm; the **first** query after a fresh start
  is slower (models load into VRAM), then it's fast and the answer streams live.

## Setup (already done on this machine)

```powershell
conda create -n rag python=3.10 -y
conda activate rag
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA build
pip install -r requirements.txt
# Install Ollama from https://ollama.com/download, then:
ollama pull mistral
```

## Usage

Drop `.txt` / `.md` / `.pdf` files into `data/`, run `python src/index.py`, then ask via
the UI or `src/query.py`. Indexing is **incremental** — a `manifest.json` content-hash
tracks each file, so re-running after adding one document only embeds that document;
removed files are dropped automatically.

## Evaluation

```powershell
python src/evaluate.py          # retrieval metrics: hit-rate@k + MRR,
                                #   vector / vector+rerank / hybrid+rerank
python src/evaluate.py --judge  # + faithfulness & answer-relevancy (local LLM judge, slow)
```

Labeled questions live in `eval_set.json`. Faithfulness target: > 0.85 (currently 1.0
on text content).

## Design notes (dataset-agnostic)

Choices that keep quality/performance solid on *any* corpus, not just the test data:

- **Token-aware chunking** — chunks are bounded by the embedding model's token limit,
  so no chunk is silently truncated at embed time (word-count windows can't guarantee
  this and lose the tail of long chunks).
- **OCR fallback** — slide/diagram PDFs hide content in images the text layer can't
  see. Pages with sparse text (< `OCR_MIN_CHARS`) are rendered and OCR'd (EasyOCR,
  GPU); text-rich pages skip OCR, so cost is paid only where needed.
- **Strong embedder + reranker** — `bge-base-en-v1.5` (768-dim) for retrieval recall,
  `bge-reranker-base` for sharp final ordering.
- **Hybrid retrieval** — BM25 catches exact terms (names, acronyms) vector search
  misses; RRF fuses the two rankings.
- **Query rewriting (optional)** — rephrasings + HyDE to bridge vocabulary gaps; off
  by default (costs extra LLM calls), toggle on when a query returns weak sources.
- **Streaming generation + keep-alive** — the answer streams token-by-token, and the
  LLM stays resident in VRAM between queries.
- **Incremental indexing, batched inserts, tuned HNSW, cached client/models** — for
  scale and speed on larger corpora.

## Known issues / gotchas (Windows)

- **EasyOCR and the embedder can't share a process** (native segfault). Indexing runs
  OCR/extraction in a **subprocess** (`ingest.py` as a worker) and embeds in the main
  process. Don't merge them.
- **pyarrow must load before torch** or it segfaults via the sklearn→pandas import
  chain. `config.py` pre-imports `pyarrow` at the top; keep that line.
- Segfaults give no Python traceback — debug with `PYTHONFAULTHANDLER=1`.
- **Restarting the UI:** Streamlit runs as `python.exe` (not `streamlit.exe`). To
  fully restart, `taskkill /F /IM python.exe` **and** clear `__pycache__` — otherwise
  a hot-reload can keep a stale module cached and throw an `ImportError`.
- **Launcher internals (`run.ps1`):** check service ports with a TCP probe to
  `127.0.0.1`, not `Invoke-WebRequest localhost` (cold-shell false-negatives); launch
  Streamlit hidden + redirected, not `-NoNewWindow` (which needs a console the detached
  launcher lacks).

## Limits

OCR of dense diagrams is noisy, so content that lives *only* inside a labeled diagram
(e.g. LSTM gate internals) can be hard to retrieve unless the query happens to use the
source's exact wording. The real fix would be a vision-language model captioning
diagrams — a separate, larger project. Everything with real text is solid.

## Tuning

All knobs live in `config.py`: chunk size/overlap, top-k, candidate count, HNSW
params, OCR threshold, embedding/reranker/LLM models, keep-alive. See
`Project Notes/RAG_Project_Notes.md` in the Obsidian vault for the build log and the
full decision history.
