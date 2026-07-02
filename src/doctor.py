# Environment / Setup Health Check
# Run:  python src/doctor.py
# Verifies Deps, CUDA, Ollama, Model Cache, Source Roots and the Index WITHOUT
# Loading Any Model - Fast Enough to Run Before Every Debugging Session.
# Exit Code 0 = No Failures (Warnings Allowed), 1 = Something Needs Fixing
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import config

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_results: list[str] = []


def report(status: str, name: str, msg: str = "") -> None:
    _results.append(status)
    print(f"  [{status:<4}] {name:<28} {msg}")


# --- Environment -------------------------------------------------------------

def check_python() -> None:
    v = sys.version_info
    status = OK if (v.major, v.minor) >= (3, 10) else FAIL
    report(status, "Python", f"{v.major}.{v.minor}.{v.micro}"
           + ("" if status == OK else " - needs >= 3.10 (uses modern type syntax)"))


# Presence via find_spec (No Import) - a Broken Native Dep Can't Crash the Doctor
def check_deps() -> None:
    required = ["chromadb", "sentence_transformers", "transformers", "torch",
                "fitz", "easyocr", "rank_bm25", "streamlit", "ollama"]
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if missing:
        report(FAIL, "Dependencies", f"missing: {', '.join(missing)} - pip install -r requirements.txt")
    else:
        report(OK, "Dependencies", f"all {len(required)} importable")
    # pyarrow Is Only the Windows DLL-Order Guard (See config) - Warn, Don't Fail
    if importlib.util.find_spec("pyarrow") is None:
        report(WARN, "pyarrow", "absent - watch for torch/pyarrow DLL segfaults on Windows")


# The One Real Import - CUDA Availability Can't Be Checked Without torch
def check_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            report(OK, "CUDA", torch.cuda.get_device_name(0))
        else:
            report(WARN, "CUDA", "not available - embedding/OCR fall back to CPU (slower)")
    except Exception as e:
        report(FAIL, "CUDA", f"torch import failed: {e}")


# --- Models ------------------------------------------------------------------

def check_ollama() -> None:
    try:
        import ollama
        models = [m.model for m in ollama.Client(host=config.OLLAMA_HOST).list().models]
    except Exception:
        report(FAIL, "Ollama", f"unreachable at {config.OLLAMA_HOST} - start it (ollama serve / run.ps1)")
        return
    # "mistral" in Config Matches "mistral:latest" on the Server
    if any(m == config.OLLAMA_MODEL or m.startswith(config.OLLAMA_MODEL + ":") for m in models):
        report(OK, "Ollama", f"up, model '{config.OLLAMA_MODEL}' present")
    else:
        report(WARN, "Ollama", f"up, but '{config.OLLAMA_MODEL}' missing - ollama pull {config.OLLAMA_MODEL}")


# Pinned Revision Present in the HF Hub Cache? Pure Directory Check - Never Touches
# the Network and Never Loads Weights
def check_hf_cache() -> None:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        report(WARN, "HF model cache", "huggingface_hub not importable - skipped")
        return
    for label, repo, rev in [("Embedder", config.EMBED_MODEL, config.EMBED_REVISION),
                             ("Re-ranker", config.RERANK_MODEL, config.RERANK_REVISION)]:
        snap = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
        if snap.is_dir():
            report(OK, label, f"{repo} cached @ pinned revision")
        else:
            report(WARN, label, f"{repo} not cached - downloads on first use")


# --- Sources + Index ---------------------------------------------------------

def check_sources() -> None:
    local = config.PROJECT_ROOT / "sources.local.json"
    report(OK if local.exists() else WARN, "sources.local.json",
           "merged" if local.exists() else "absent - only the tracked data/ root is indexed")
    try:
        from ingest import list_documents
        docs = list_documents()
    except Exception as e:
        report(FAIL, "Source listing", f"{e}")
        return
    for src in config.SOURCES:
        root, name = Path(src["path"]), src["name"]
        if not root.exists():
            report(FAIL, f"Source '{name}'", f"path missing: {root}")
        else:
            n = sum(1 for _, sp in docs if sp.startswith(name + "/"))
            report(OK if n else WARN, f"Source '{name}'",
                   f"{n} indexable file(s)" + ("" if n else " - nothing matches INDEX_EXTENSIONS"))


def check_index() -> None:
    manifest_file = config.CHROMA_DIR / "manifest.json"
    if not manifest_file.exists():
        report(WARN, "Index", "not built yet - python src/index.py")
        return
    try:
        import chromadb
        count = (chromadb.PersistentClient(path=str(config.CHROMA_DIR))
                 .get_collection(config.COLLECTION_NAME).count())
        report(OK, "Index", f"{count} chunks in '{config.COLLECTION_NAME}'")
    except Exception as e:
        report(FAIL, "Index", f"chroma_db exists but unreadable: {e}")
        return

    # Manifest vs Disk - How Many Files Changed/Appeared/Vanished Since Last Index
    try:
        from ingest import list_documents
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        current = {sp: hashlib.sha256(p.read_bytes()).hexdigest() for p, sp in list_documents()}
        pending = sum(1 for sp, h in current.items() if manifest.get(sp) != h)
        removed = sum(1 for sp in manifest if sp not in current)
        if pending or removed:
            report(WARN, "Index freshness",
                   f"{pending} changed/new, {removed} removed since last index - python src/index.py")
        else:
            report(OK, "Index freshness", f"up to date ({len(manifest)} files)")
    except Exception as e:
        report(WARN, "Index freshness", f"could not compare: {e}")

    import lock
    if lock.held():
        report(WARN, "Index lock", f"held ({lock.describe()}) - indexing in progress or a crashed run")


def main() -> int:
    print("RAG doctor\n")
    print("Environment")
    check_python()
    check_deps()
    check_cuda()
    print("\nModels")
    check_ollama()
    check_hf_cache()
    print("\nSources + index")
    check_sources()
    check_index()

    fails = _results.count(FAIL)
    warns = _results.count(WARN)
    print(f"\n{len(_results)} checks: {fails} failed, {warns} warning(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
