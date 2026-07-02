# Index Build / Update CLI
# Run:  python index.py            # Incremental Update
#       python index.py --rebuild  # Full Rebuild
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import config
import embeddings
import lock
import retrieval
import vectorstore
from ingest import Chunk, list_documents

MANIFEST = config.CHROMA_DIR / "manifest.json"
INGEST = Path(__file__).resolve().parent / "ingest.py"


# Content Hash of a File, So Identical Bytes Are Recognised as Unchanged
def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def _save_manifest(manifest: dict) -> None:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# Extract + Chunk the Given Files in a Separate Process (OCR Lives There). The
# File Spec Goes in via stdin (Not argv) So It Scales Past the Windows ~32k Limit
def _extract(spec: list[dict]) -> list[Chunk]:
    if not spec:
        return []
    proc = subprocess.run(
        [sys.executable, str(INGEST)],
        input=json.dumps(spec),
        capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if proc.stderr.strip():
        print(proc.stderr.strip())  # OCR-Failure Notices Etc.
    if proc.returncode != 0:
        raise RuntimeError(f"extraction subprocess failed (exit {proc.returncode})")
    return [Chunk(**d) for d in json.loads(proc.stdout)]


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    docs = list_documents()  # [(abs_path, source_path)]
    if not docs:
        print("No documents found. Add files under a configured source (see config.SOURCES).")
        return

    # Exclusive Write Lock - a Second Indexer (or Indexing While the UI Reads) Would
    # Race Chroma Writes; Released in the finally Below Even When Extraction Fails
    lock.acquire()

    try:
        _run(docs, rebuild)
    finally:
        lock.release()


def _run(docs: list, rebuild: bool) -> None:
    if rebuild:
        print("Full rebuild: clearing existing index")
        vectorstore.reset()
        manifest = {}
    else:
        manifest = _load_manifest()

    # Everything Keyed by source_path (Collision-Free Across Roots), Not Filename
    abs_by_sp = {sp: p for p, sp in docs}
    current = {sp: _hash(p) for p, sp in docs}
    mtimes = {sp: p.stat().st_mtime for p, sp in docs}

    # Drop Chunks for Files That Vanished or Changed (Skip When Rebuilding - Already Wiped)
    removed = 0
    for sp in list(manifest):
        if manifest[sp] != current.get(sp):
            if not rebuild:
                vectorstore.delete_source(sp)
            removed += sp not in current
            manifest.pop(sp, None)

    # Re-Extract Only New/Changed Files; Pass hash/mtime So the Worker Needn't Re-Stat
    to_index = [sp for _, sp in docs if manifest.get(sp) != current[sp]]
    print(f"Extracting {len(to_index)} file(s) (OCR in subprocess)…")
    spec = [
        {"abs_path": str(abs_by_sp[sp]), "source_path": sp,
         "content_hash": current[sp], "mtime": mtimes[sp]}
        for sp in to_index
    ]
    chunks = _extract(spec)

    if chunks:
        print(f"Embedding {len(chunks)} chunks")
        vectorstore.add(chunks, embeddings.embed([c.text for c in chunks]))

    per_file = Counter(c.source_path for c in chunks)
    for sp in to_index:
        print(f"  {'~' if sp in manifest else '+'} {sp}: {per_file[sp]} chunks")
        manifest[sp] = current[sp]

    _save_manifest(manifest)
    retrieval.reset_cache()  # BM25 Index Is Now Stale; Rebuilds on Next Query
    skipped = len(docs) - len(to_index)
    print(f"Done. indexed={len(to_index)} removed={removed} unchanged={skipped} "
          f"-> {config.CHROMA_DIR}")


if __name__ == "__main__":
    try:
        main()
    except lock.IndexLocked as e:
        print(f"Refusing to index: {e}")
        sys.exit(1)
