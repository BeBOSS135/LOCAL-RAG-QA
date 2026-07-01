# Chunking + Document Listing (Also the Extraction Worker: reads a JSON spec on
# stdin, emits Chunk dicts on stdout - OCR Runs Here, Isolated From the Embedder)
import sys
from dataclasses import dataclass
from pathlib import Path

import config
import embeddings
import loaders


@dataclass
class Chunk:
    text: str
    source_path: str    # "<source name>/<relpath>" - Collision-Free Id Base
    source_type: str    # "markdown" | "pdf" | "text"
    title: str          # Note Title (Frontmatter) or Filename Stem
    heading_path: str   # "H1 > H2" (md), "p.3" (pdf), "" (txt)
    tags: list[str]     # Frontmatter + Inline #tags (md); [] Otherwise
    mtime: float        # File st_mtime - for Recency Ranking Later
    content_hash: str   # sha256 of File Bytes (Carried for Reference)
    chunk_index: int    # Position Within the File


# Token Length of a String Under the Embedder's Tokenizer (Decides Merge/Window)
def _token_len(text: str) -> int:
    return len(embeddings.get_tokenizer()(text, add_special_tokens=False)["input_ids"])


# Sliding Token Window Over One Section's Text, So Every Chunk Fits the Embedder's
# Max Tokens (Word Windows Can't Bound Tokens). Tokenize Once With Offset Mapping,
# Window Over Offsets, Slice Text by Char So There's No Sub-Word Breakage
def _window(text: str) -> list[str]:
    tok = embeddings.get_tokenizer()
    offsets = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    if not offsets:
        return []
    size = config.CHUNK_TOKENS
    step = max(1, size - config.CHUNK_OVERLAP_TOKENS)  # Guard Misconfig (Overlap >= Size)
    out = []
    for start in range(0, len(offsets), step):
        window = offsets[start:start + size]
        if not window:
            break
        snippet = text[window[0][0]:window[-1][1]].strip()
        if snippet:
            out.append(snippet)
        if start + size >= len(offsets):
            break  # Last Window Reached; Avoid Trailing Duplicates
    return out


# Turn a LoadedDoc's Sections Into Chunks. Structure-Aware: Greedily Merge Small
# Adjacent Sections Under the Same Top Heading (Avoids One-Line Chunks From
# Heading-Heavy Markdown), and Token-Window Anything Still Too Big. Each Chunk
# Inherits Its Section's heading_path Plus the Doc's Metadata
def _chunk_doc(doc: loaders.LoadedDoc, source_path: str, mtime: float, content_hash: str) -> list[Chunk]:
    chunks: list[Chunk] = []

    def emit(text: str, heading_path: str):
        for piece in _window(text):
            chunks.append(Chunk(
                text=piece, source_path=source_path, source_type=doc.source_type,
                title=doc.title, heading_path=heading_path, tags=list(doc.tags),
                mtime=mtime, content_hash=content_hash, chunk_index=len(chunks),
            ))

    top = lambda hp: hp.split(" > ")[0] if hp else ""
    buf_text, buf_hp, buf_tokens = "", "", 0

    def flush():
        nonlocal buf_text, buf_hp, buf_tokens
        if buf_text.strip():
            emit(buf_text, buf_hp)
        buf_text, buf_hp, buf_tokens = "", "", 0

    for sec in doc.sections:
        n = _token_len(sec.text)
        if n > config.CHUNK_TOKENS:        # Too Big to Merge - Flush, Then Window Alone
            flush()
            emit(sec.text, sec.heading_path)
            continue
        # New Group on Top-Heading Change or When Adding Would Overflow the Window
        if buf_text and (top(sec.heading_path) != top(buf_hp) or buf_tokens + n > config.CHUNK_TOKENS):
            flush()
        if not buf_text:
            buf_hp = sec.heading_path      # Group Anchored at Its First Section
        else:
            buf_text += "\n\n"
        buf_text += sec.text
        buf_tokens += n
    flush()
    return chunks


# Load + Chunk One File Into Fully-Populated Chunks. hash/mtime Come From the
# Caller (index.py Already Stats/Hashes for the Manifest - Don't Re-Read the File)
def chunk_file(abs_path: Path, source_path: str, content_hash: str, mtime: float) -> list[Chunk]:
    return _chunk_doc(loaders.load(abs_path), source_path, mtime, content_hash)


# Every Indexable File Across All Configured Sources, as (abs_path, source_path).
# source_path = "<source name>/<relpath>" Keeps Files Unique Across Roots
def list_documents() -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for src in config.SOURCES:
        root = Path(src["path"])
        if not root.exists():
            continue
        exclude = set(src.get("exclude", []))  # Folder Names to Skip Under This Root
        for p in sorted(root.rglob("*")):
            if not (p.is_file() and p.suffix.lower() in config.INDEX_EXTENSIONS):
                continue
            parts = p.relative_to(root).parts
            # Skip Hidden Dirs (.obsidian/.git) and Any Excluded Ancestor Folder
            if any(part.startswith(".") for part in parts) or (exclude & set(parts[:-1])):
                continue
            out.append((p, f"{src['name']}/{p.relative_to(root).as_posix()}"))
    return out


if __name__ == "__main__":
    # Extraction Worker: Read [{abs_path, source_path, content_hash, mtime}] on stdin
    # (Avoids the Windows ~32k argv Limit as the Corpus Grows), Emit Chunk Dicts
    import json
    spec = json.load(sys.stdin)
    out = [
        vars(c)
        for item in spec
        for c in chunk_file(Path(item["abs_path"]), item["source_path"],
                            item["content_hash"], item["mtime"])
    ]
    json.dump(out, sys.stdout)
