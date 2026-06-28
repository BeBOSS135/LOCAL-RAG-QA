"""Load documents from data/ and split them into overlapping, token-bounded chunks.

Also runnable as an extraction worker: `python ingest.py <file>...` prints the
chunks as JSON on stdout. index.py uses this to run OCR in a separate process from
the embedder (EasyOCR + the SentenceTransformer model segfault if they share one).
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import config
import embeddings

SUPPORTED = {".txt", ".md", ".pdf"}


@dataclass
class Chunk:
    text: str
    source: str       # filename the chunk came from
    chunk_index: int  # position within that file


_ocr_reader = None  # lazy singleton; loading EasyOCR weights is slow


def _get_ocr_reader():
    import torch
    import easyocr
    global _ocr_reader
    if _ocr_reader is None:
        # GPU when available; EasyOCR runs on the same CUDA torch as the embedder.
        # verbose=False: its progress bar prints block chars that crash the cp1252 console
        _ocr_reader = easyocr.Reader(
            config.OCR_LANGS, gpu=torch.cuda.is_available(), verbose=False
        )
    return _ocr_reader


def _ocr_page(page) -> str:
    """Render a PDF page to an image and OCR it (used for sparse/diagram pages)."""
    import io
    import numpy as np
    from PIL import Image
    pix = page.get_pixmap(dpi=config.OCR_DPI)
    img = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
    # paragraph=True groups nearby words into lines; detail=0 returns plain strings
    return "\n".join(_get_ocr_reader().readtext(img, detail=0, paragraph=True))


def _read_pdf(path: Path) -> str:
    """Extract text per page via PyMuPDF, falling back to OCR on sparse pages."""
    import fitz  # PyMuPDF — better text extraction than pypdf, and renders pages
    parts = []
    ocr_failures = 0
    with fitz.open(str(path)) as doc:
        for page in doc:
            text = page.get_text().strip()
            # A near-empty text layer means the content is in the slide image -> OCR.
            # A single unreadable page must not abort the whole index, so OCR is guarded.
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
        # stderr so it never corrupts the JSON this worker writes to stdout
        print(f"    ({ocr_failures} page(s) failed OCR, used text layer)", file=sys.stderr)
    return "\n".join(parts)


def _read_file(path: Path) -> str:
    """Extract raw text from one file. Supports .txt, .md, .pdf."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def _chunk_tokens(text: str, source: str) -> list[Chunk]:
    """Sliding window over the *tokenizer's* tokens, so every chunk fits the model.

    Word-count windows can't bound tokens (a 400-word chunk can be 500+ tokens and
    get truncated at embed time). We tokenize once with offset mapping, window over
    token offsets, then slice the original text by character — keeping real text
    (no sub-word breakage) while guaranteeing each chunk is <= CHUNK_TOKENS.
    """
    tok = embeddings.get_tokenizer()
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if not offsets:
        return []
    size = config.CHUNK_TOKENS
    step = max(1, size - config.CHUNK_OVERLAP_TOKENS)  # guard misconfig (overlap >= size)

    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(offsets), step)):
        window = offsets[start:start + size]
        if not window:
            break
        # Slice from the first token's start char to the last token's end char
        snippet = text[window[0][0]:window[-1][1]].strip()
        if snippet:
            chunks.append(Chunk(text=snippet, source=source, chunk_index=i))
        if start + size >= len(offsets):
            break  # last window reached; avoid emitting trailing duplicates
    return chunks


def chunk_file(path: Path) -> list[Chunk]:
    """Read and chunk a single file."""
    return _chunk_tokens(_read_file(path), source=path.name)


def list_documents(data_dir: Path = config.DATA_DIR) -> list[Path]:
    """Every supported document under data_dir, in stable order."""
    return sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )


def load_chunks(data_dir: Path = config.DATA_DIR) -> list[Chunk]:
    """Read every supported file in data_dir and return all chunks."""
    chunks: list[Chunk] = []
    for path in list_documents(data_dir):
        file_chunks = chunk_file(path)
        chunks.extend(file_chunks)
        print(f"  {path.name}: {len(file_chunks)} chunks", file=sys.stderr)
    return chunks


if __name__ == "__main__":
    # Extraction worker: emit chunks for the given files as JSON on stdout.
    import json
    out = [
        {"text": c.text, "source": c.source, "chunk_index": c.chunk_index}
        for arg in sys.argv[1:]
        for c in chunk_file(Path(arg))
    ]
    json.dump(out, sys.stdout)
