# Document Loading + Chunking (Also an Extraction Worker: python ingest.py <file>...)
import sys
from dataclasses import dataclass
from pathlib import Path

import config
import embeddings

SUPPORTED = {".txt", ".md", ".pdf"}


@dataclass
class Chunk:
    text: str
    source: str       # Filename the Chunk Came From
    chunk_index: int  # Position Within That File


_ocr_reader = None  # Lazy Singleton - Loading EasyOCR Weights Is Slow


def _get_ocr_reader():
    import torch
    import easyocr
    global _ocr_reader
    if _ocr_reader is None:
        # GPU When Available; EasyOCR Shares the Same CUDA Torch as the Embedder
        # verbose=False: Its Progress Bar Prints Block Chars That Crash the cp1252 Console
        _ocr_reader = easyocr.Reader(
            config.OCR_LANGS, gpu=torch.cuda.is_available(), verbose=False
        )
    return _ocr_reader


# Render a PDF Page to an Image and OCR It (for Sparse/Diagram Pages)
def _ocr_page(page) -> str:
    import io
    import numpy as np
    from PIL import Image
    pix = page.get_pixmap(dpi=config.OCR_DPI)
    img = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
    # paragraph=True Groups Nearby Words Into Lines; detail=0 Returns Plain Strings
    return "\n".join(_get_ocr_reader().readtext(img, detail=0, paragraph=True))


# Extract Text per Page via PyMuPDF, Falling Back to OCR on Sparse Pages
def _read_pdf(path: Path) -> str:
    import fitz  # PyMuPDF - Better Text Extraction Than pypdf, and Renders Pages
    parts = []
    ocr_failures = 0
    with fitz.open(str(path)) as doc:
        for page in doc:
            text = page.get_text().strip()
            # Near-Empty Text Layer Means Content Lives in the Slide Image -> OCR
            # One Unreadable Page Must Not Abort the Whole Index, So OCR Is Guarded
            if config.USE_OCR and len(text) < config.OCR_MIN_CHARS:
                try:
                    ocr_text = _ocr_page(page).strip()
                    if len(ocr_text) > len(text):
                        text = ocr_text
                except Exception:
                    ocr_failures += 1
            if text:
                parts.append(text)
    if ocr_failures:
        # stderr So It Never Corrupts the JSON This Worker Writes to stdout
        print(f"    ({ocr_failures} page(s) failed OCR, used text layer)", file=sys.stderr)
    return "\n".join(parts)


# Extract Raw Text From One File - Supports .txt, .md, .pdf
def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"Unsupported file type: {path.name}")


# Sliding Window Over the Tokenizer's Tokens, So Every Chunk Fits the Model
# Word-Count Windows Can't Bound Tokens (a 400-Word Chunk Can Be 500+ and Get Truncated)
# Tokenize Once With Offset Mapping, Window Over Offsets, Then Slice Text by Character -
# Keeps Real Text (No Sub-Word Breakage) While Guaranteeing Each Chunk <= CHUNK_TOKENS
def _chunk_tokens(text: str, source: str) -> list[Chunk]:
    tok = embeddings.get_tokenizer()
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if not offsets:
        return []
    size = config.CHUNK_TOKENS
    step = max(1, size - config.CHUNK_OVERLAP_TOKENS)  # Guard Misconfig (Overlap >= Size)

    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(offsets), step)):
        window = offsets[start:start + size]
        if not window:
            break
        # Slice From the First Token's Start Char to the Last Token's End Char
        snippet = text[window[0][0]:window[-1][1]].strip()
        if snippet:
            chunks.append(Chunk(text=snippet, source=source, chunk_index=i))
        if start + size >= len(offsets):
            break  # Last Window Reached; Avoid Emitting Trailing Duplicates
    return chunks


# Read and Chunk a Single File
def chunk_file(path: Path) -> list[Chunk]:
    return _chunk_tokens(_read_file(path), source=path.name)


# Every Supported Document Under data_dir, in Stable Order
def list_documents(data_dir: Path = config.DATA_DIR) -> list[Path]:
    return sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )


# Read Every Supported File in data_dir and Return All Chunks
def load_chunks(data_dir: Path = config.DATA_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in list_documents(data_dir):
        file_chunks = chunk_file(path)
        chunks.extend(file_chunks)
        print(f"  {path.name}: {len(file_chunks)} chunks", file=sys.stderr)
    return chunks


if __name__ == "__main__":
    # Extraction Worker: Emit Chunks for the Given Files as JSON on stdout
    import json
    out = [
        {"text": c.text, "source": c.source, "chunk_index": c.chunk_index}
        for arg in sys.argv[1:]
        for c in chunk_file(Path(arg))
    ]
    json.dump(out, sys.stdout)
