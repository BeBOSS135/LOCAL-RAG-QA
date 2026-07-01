# RAG Question-Answering

A local Retrieval-Augmented Generation system that answers questions about your
own documents, with source citations. Runs **fully offline** on a consumer GPU
(built/tested on an RTX 4060).

## Architecture

```
Indexing (once):  data/ + your sources ──> extract+OCR ──> structure-aware chunk ──> embed ──> Chroma
Query (per ask):  question ──> [rewrite] ──> embed ──> retrieve (vector; +BM25 hybrid optional)
                           ──> cross-encoder re-rank ──> dedup ──> abstention gate
                           ──> LLM (streamed) ──> grounding check ──> answer + sources
```

Framework-light by design — each pipeline step is a small, readable module rather
than hidden behind LangChain, so every stage is inspectable. All modules live in `src/`.

| Step | Module | Tool |
|------|--------|------|
| Load (per file type) | `loaders.py` | registry: markdown / pdf / txt + text fallback; PyMuPDF + EasyOCR for slide/image PDFs |
| Chunk | `ingest.py` | structure-aware, token-bounded (splits on headings / pages) |
| Embed | `embeddings.py` | sentence-transformers (`bge-base-en-v1.5`, 768-dim) |
| Store + search | `vectorstore.py` | Chroma (cosine); folder/tag metadata for scoped retrieval |
| Retrieve | `retrieval.py` | vector; optional BM25 (`rank_bm25`) hybrid fused with RRF |
| Query rewrite (opt) | `rewrite.py` | rephrasings + HyDE (off by default) |
| Re-rank | `reranker.py` | cross-encoder (`bge-reranker-base`) |
| Generate | `llm.py` | Ollama + Mistral 7B (streamed) + grounding check |
| Orchestrate | `rag.py` | `retrieve()` / `answer()`: dedup, abstention gate, scope/tag |
| Log usage | `feedback.py` | query + 👍/👎 logs (JSONL) |
| Evaluate | `evaluate.py` | retrieval + abstention + LLM-judge faithfulness |

Every stage is toggleable in `src/config.py` and per-query via `rag.retrieve(...)` /
the Streamlit sidebar.

## Answer quality controls

- **Abstention gate** — when retrieval is weak the system refuses ("not in my indexed
  sources") instead of answering from loosely-related chunks, and skips the LLM entirely.
  Two-signal: it answers if *either* the re-ranker *or* the embedding similarity clears its
  floor, so terse questions aren't wrongly refused.
- **Post-generation grounding check** — after generating, an LLM pass checks the answer is
  supported by the retrieved context; if not, the answer is flagged with a ⚠️ caveat (not
  deleted). Runs only in the "gray zone" (skipped when the top score is already high). Fail-open.
- **Scope filtering** — restrict retrieval to a source root / folder (Chroma metadata
  pre-filter) or to a tag (post-filter), via the sidebar or `rag.retrieve(scope=…, tag=…)`.
- **Near-duplicate dedup** — repeated content across files won't consume top-k slots.
- **Feedback log** — 👍/👎 in the UI and every query append to `logs/*.jsonl` (gitignored),
  growing a real eval set from actual use.

## What you need running

Two independent pieces:

1. **Ollama** — the local LLM server. Installs as a Windows app that auto-starts in
   the tray and listens on `localhost:11434`. Check with `ollama list`; it must have
   the `mistral` model (`ollama pull mistral` once).
2. **A Python environment with the dependencies installed** — torch+CUDA,
   sentence-transformers, chromadb, PyMuPDF, EasyOCR, etc. See [Setup](#setup) below;
   a conda env named `rag` is assumed by the launcher but any environment works.

Models (bge embedder/reranker, EasyOCR weights) download themselves on first use and
are cached. Everything after that is offline.

## Operations — start / stop / kill

**Easiest: double-click `start.bat`.** It checks Ollama is running (starts it if not),
makes sure the `mistral` model is present, launches the UI, and opens the browser.
**Closing that window shuts everything down** — the UI (and Ollama, if the launcher
started it) are tied to the window via a Job Object, so no manual cleanup is needed.
Ctrl+C also unloads the model from VRAM. (`start.bat` just runs `run.ps1` with the
execution policy relaxed.)

For manual control, the commands below run from the project root (the cloned repo
folder). Activate your environment first (e.g. `conda activate rag` from an Anaconda
Prompt), or call that env's Python directly.

```powershell
# --- START THE UI ---
streamlit run src/app.py
#   -> open http://localhost:8501   (Ctrl+C in this terminal to stop it)

# --- ASK FROM THE TERMINAL (no UI) ---
python src/query.py "your question here"

# --- (RE)BUILD THE INDEX  (after changing sources) ---
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

## Setup

One-time, on a fresh machine:

```powershell
git clone https://github.com/BeBOSS135/LOCAL-RAG-QA.git
cd LOCAL-RAG-QA

conda create -n rag python=3.10 -y
conda activate rag
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA build (omit --index-url for CPU)
pip install -r requirements.txt

# Install Ollama from https://ollama.com/download, then:
ollama pull mistral
```

The model weights (bge embedder/reranker, EasyOCR) download on first run and are then
cached; everything after that is offline.

## Usage

Two ways to index content:

1. **Drop files in `data/`** — `.txt` / `.md` / `.pdf`. The repo ships one `data/sample.md`
   so it runs out of the box.
2. **Point at folders anywhere** (e.g. a notes vault) without editing tracked code — create a
   gitignored `sources.local.json` at the repo root:
   ```json
   [{"name": "vault", "path": "C:/path/to/your/vault", "exclude": ["Archive"]}]
   ```
   Each entry is indexed under its `name`; `exclude` skips folders by name (hidden dirs like
   `.obsidian` / `.git` are skipped automatically).

After changing sources, run `python src/index.py` — indexing is **incremental** (a
`manifest.json` content-hash embeds only new/changed files and drops removed ones) — then
ask via the UI or `src/query.py`.

## Evaluation

```powershell
python src/evaluate.py           # retrieval: hit-rate@k + MRR (vector / vector+rerank / hybrid+rerank)
python src/evaluate.py --abstain # + abstention gate: false-refusal vs correct-abstention rates
python src/evaluate.py --judge   # + faithfulness & answer-relevancy (local LLM judge, slow)
```

Questions live in `eval_set.json` (a generic set over `data/sample.md`); drop a gitignored
`eval_set.local.json` to evaluate against your own corpus instead. Answerable items carry
`expected` source tags; an `"answerable": false` slice measures refusal on out-of-corpus
questions. Faithfulness target: > 0.85.

## Design notes (dataset-agnostic)

Choices that keep quality/performance solid on *any* corpus, not just the test data:

- **Token-aware chunking** — chunks are bounded by the embedding model's token limit,
  so no chunk is silently truncated at embed time (word-count windows can't guarantee
  this and lose the tail of long chunks).
- **Structure-aware chunking** — markdown splits on headings (carrying `H1 > H2` ancestry),
  PDFs one section per page, so citations are human (`[Title › Heading]` / `[file › p.3]`).
- **OCR fallback** — slide/diagram PDFs hide content in images the text layer can't
  see. Pages with sparse text (< `OCR_MIN_CHARS`) are rendered and OCR'd (EasyOCR,
  GPU); text-rich pages skip OCR, so cost is paid only where needed.
- **Vector-first retrieval + reranker** — `bge-base-en-v1.5` (768-dim) + `bge-reranker-base`,
  measured best on clean prose. **Hybrid (BM25) is opt-in** — enable it for keyword/exact-match
  corpora (code, acronyms, IDs) where sparse retrieval catches tokens dense embeddings miss.
- **Abstention over hallucination** — refuse when both retrieval signals are weak, and flag
  post-generation answers that aren't supported by the context. Prefer "I don't know" to invention.
- **Query rewriting (optional)** — rephrasings + HyDE to bridge vocabulary gaps; off
  by default (costs extra LLM calls), toggle on when a query returns weak sources.
- **Streaming generation + keep-alive** — the answer streams token-by-token, and the
  LLM stays resident in VRAM between queries.
- **Incremental indexing, batched inserts, tuned HNSW, cached client/models** — for
  scale and speed on larger corpora.

## Known issues / gotchas (Windows)

- **EasyOCR and the embedder can't share a process** (native segfault). Indexing runs
  OCR/extraction in a **subprocess** (`src/ingest.py` as a worker) and embeds in the main
  process. Don't merge them.
- **pyarrow must load before torch** or it segfaults via the sklearn→pandas import
  chain. `src/config.py` pre-imports `pyarrow` at the top; keep that line.
- Segfaults give no Python traceback — debug with `PYTHONFAULTHANDLER=1`.
- **Restarting the UI:** Streamlit runs as `python.exe` (not `streamlit.exe`). To
  fully restart, `taskkill /F /IM python.exe` **and** clear `src/__pycache__` — otherwise
  a hot-reload can keep a stale module cached and throw an `ImportError`.
- **Launcher internals (`run.ps1`):** check service ports with a TCP probe to
  `127.0.0.1`, not `Invoke-WebRequest localhost` (cold-shell false-negatives); launch
  Streamlit hidden + redirected, not `-NoNewWindow` (which needs a console the detached
  launcher lacks).

## Limits

OCR of dense diagrams is noisy, so content that lives *only* inside a labeled diagram
can be hard to retrieve unless the query happens to use the source's exact wording. The
real fix would be a vision-language model captioning diagrams — a separate, larger
project. Everything with real text is solid. Comparison/multi-hop questions across
documents are answered from what single-shot retrieval surfaces, so they lean on good
cross-references in the source notes rather than reasoning across whole documents.

## Tuning

All knobs live in `src/config.py`: chunk size/overlap, top-k, candidate count, HNSW
params, OCR threshold, embedding/reranker/LLM models, keep-alive, hybrid on/off,
abstention floors (`ABSTAIN_MIN_RERANK` / `ABSTAIN_MIN_SIMILARITY`), the grounding-check
gate (`VERIFY_SKIP_ABOVE`), and dedup (`DEDUP_JACCARD`).
