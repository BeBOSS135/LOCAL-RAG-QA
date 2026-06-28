"""CLI: build / update the vector index from data/.

Two-process design: text extraction + OCR runs in a **subprocess** (ingest.py as a
worker), and embedding runs here in the main process. They must not share a process
— EasyOCR's native stack and the SentenceTransformer model segfault together.

Incremental by default: a manifest records each file's content hash, so unchanged
files are skipped and only new/changed files are re-extracted+embedded (removed
files are dropped). Use --rebuild to wipe and start clean.

Run:  python index.py            # incremental update
      python index.py --rebuild  # full rebuild
"""
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import config
import embeddings
import retrieval
import vectorstore
from ingest import Chunk, list_documents

MANIFEST = config.CHROMA_DIR / "manifest.json"
INGEST = Path(__file__).resolve().parent / "ingest.py"


def _hash(path: Path) -> str:
    """Content hash of a file, so identical bytes are recognised as unchanged."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def _save_manifest(manifest: dict) -> None:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _extract(paths: list[Path]) -> list[Chunk]:
    """Extract+chunk the given files in a separate process (OCR lives there)."""
    if not paths:
        return []
    proc = subprocess.run(
        [sys.executable, str(INGEST), *[str(p) for p in paths]],
        capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if proc.stderr.strip():
        print(proc.stderr.strip())  # OCR-failure notices etc.
    if proc.returncode != 0:
        raise RuntimeError(f"extraction subprocess failed (exit {proc.returncode})")
    return [Chunk(**d) for d in json.loads(proc.stdout)]


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    docs = list_documents()
    if not docs:
        print("No documents found. Drop .txt/.md/.pdf files into data/ and retry.")
        return

    if rebuild:
        print("Full rebuild: clearing existing index")
        vectorstore.reset()
        manifest = {}
    else:
        manifest = _load_manifest()

    current = {p.name: _hash(p) for p in docs}

    # Drop chunks for files that vanished or changed (skip when rebuilding — already wiped)
    removed = 0
    for name in list(manifest):
        if manifest[name] != current.get(name):
            if not rebuild:
                vectorstore.delete_source(name)
            removed += name not in current
            manifest.pop(name, None)

    # Re-extract only new/changed files
    to_index = [p for p in docs if manifest.get(p.name) != current[p.name]]
    print(f"Extracting {len(to_index)} file(s) (OCR in subprocess)…")
    chunks = _extract(to_index)

    if chunks:
        print(f"Embedding {len(chunks)} chunks")
        vectorstore.add(chunks, embeddings.embed([c.text for c in chunks]))

    per_file = Counter(c.source for c in chunks)
    for p in to_index:
        print(f"  {'~' if p.name in manifest else '+'} {p.name}: {per_file[p.name]} chunks")
        manifest[p.name] = current[p.name]

    _save_manifest(manifest)
    retrieval.reset_cache()  # BM25 index is now stale; rebuilds on next query
    skipped = len(docs) - len(to_index)
    print(f"Done. indexed={len(to_index)} removed={removed} unchanged={skipped} "
          f"-> {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
