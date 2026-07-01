# Document Loaders - One per File Type, Registered by Extension
# Each Returns a LoadedDoc: Doc-Level Metadata + Structure-Split Sections, So the
# Chunker Downstream Is Uniform. Adding a New File Type = Write One Loader and
# Register It in LOADERS; the Pipeline Never Changes
import sys
from dataclasses import dataclass
from pathlib import Path
import re

import config  # Pre-Imports pyarrow Before torch (Windows DLL-Order Fix)


@dataclass
class Section:
    text: str
    heading_path: str = ""  # "H1 > H2" (markdown) or "p.3" (pdf); "" When Flat


@dataclass
class LoadedDoc:
    source_type: str          # "markdown" | "pdf" | "text"
    title: str
    tags: list[str]
    sections: list[Section]


# --- Markdown ---------------------------------------------------------------

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]*`")
# Obsidian Inline Tag: #tag Containing at Least One Letter (Purely Numeric "#1",
# "#2/" Are List/Footnote Markers, Not Tags). Nested Tags (#area/ml) Keep the Slash
_INLINE_TAG = re.compile(r"(?:^|\s)#([A-Za-z0-9_/-]*[A-Za-z][A-Za-z0-9_/-]*)")


# Collect Inline #tags From Markdown Body. Strip Code First (Fenced + Inline) So a
# "#comment" or "#CONST" in a Snippet Never Becomes a Tag - Code Is Full of Stray #
def _inline_tags(body: str) -> list[str]:
    clean = _FENCED_CODE.sub(" ", body)
    clean = _INLINE_CODE.sub(" ", clean)
    return [t.strip("/-") for t in _INLINE_TAG.findall(clean) if t.strip("/-")]


# Minimal YAML Frontmatter Reader - We Only Need title/tags/aliases, So a Small
# Line Parser Beats Pulling in a YAML Dependency. Handles "key: value", Inline
# "[a, b]" Lists, and "- item" Block Lists
def _parse_frontmatter(block: str) -> dict:
    meta: dict = {}
    key = None
    for line in block.splitlines():
        if not line.strip():
            continue
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if m:
            key, val = m.group(1).lower(), m.group(2).strip()
            if val.startswith("[") and val.endswith("]"):
                meta[key] = [v.strip().strip("\"'") for v in val[1:-1].split(",") if v.strip()]
            elif val:
                meta[key] = val.strip("\"'")
            else:
                meta[key] = []
        elif line.lstrip().startswith("- ") and key and isinstance(meta.get(key), list):
            meta[key].append(line.lstrip()[2:].strip().strip("\"'"))
    return meta


# Split Markdown on ATX Headings, Tracking the Heading Stack So Each Block Carries
# Its Full Ancestry ("H1 > H2"). Frontmatter Feeds title/tags; Inline #tags Are Collected
def load_markdown(path: Path) -> LoadedDoc:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    title, tags = path.stem, []

    fm = _FRONTMATTER.match(raw)
    if fm:
        meta = _parse_frontmatter(fm.group(1))
        body = raw[fm.end():]
        if isinstance(meta.get("title"), str):
            title = meta["title"]
        t = meta.get("tags", [])
        tags += t if isinstance(t, list) else [t]
    else:
        body = raw

    tags += _inline_tags(body)
    tags = sorted({t for t in tags if t})

    sections: list[Section] = []
    stack: list[str] = []  # Heading Text by Level; stack[i] == Level i+1 Heading
    buf: list[str] = []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(text, " > ".join(s for s in stack if s)))
        buf.clear()

    for line in body.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            stack[:] = stack[:level - 1]          # Pop Deeper/Sibling Levels
            while len(stack) < level - 1:
                stack.append("")                  # Fill Skipped Levels
            stack.append(m.group(2).strip())
        else:
            buf.append(line)
    flush()

    if not sections:                              # Heading-Free Note
        body = body.strip()
        if body:
            sections = [Section(body)]
    return LoadedDoc("markdown", title, tags, sections)


# --- PDF (PyMuPDF + OCR Fallback) -------------------------------------------

_ocr_reader = None  # Lazy Singleton - Loading EasyOCR Weights Is Slow


def _get_ocr_reader():
    import torch
    import easyocr
    global _ocr_reader
    if _ocr_reader is None:
        # GPU When Available. verbose=False: Its Progress Bar Prints Block Chars
        # That Crash the Windows cp1252 Console
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
    # paragraph=True Groups Words Into Lines; detail=0 Returns Plain Strings
    return "\n".join(_get_ocr_reader().readtext(img, detail=0, paragraph=True))


# One Section per Page (heading_path "p.N" -> Page-Level Citations). Sparse Text
# Layers Fall Back to OCR; One Unreadable Page Must Not Abort the Whole Index
def load_pdf(path: Path) -> LoadedDoc:
    import fitz  # PyMuPDF - Better Extraction Than pypdf, and Renders Pages
    sections: list[Section] = []
    ocr_failures = 0
    with fitz.open(str(path)) as doc:
        for n, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if config.USE_OCR and len(text) < config.OCR_MIN_CHARS:
                try:
                    ocr_text = _ocr_page(page).strip()
                    if len(ocr_text) > len(text):
                        text = ocr_text
                except Exception:
                    ocr_failures += 1
            if text:
                sections.append(Section(text, f"p.{n}"))
    if ocr_failures:
        # stderr So It Never Corrupts the JSON the Extraction Worker Writes to stdout
        print(f"    ({ocr_failures} page(s) failed OCR, used text layer)", file=sys.stderr)
    return LoadedDoc("pdf", path.stem, [], sections)


# --- Plain Text (and Fallback for Unregistered Extensions) ------------------

def load_text(path: Path) -> LoadedDoc:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return LoadedDoc("text", path.stem, [], [Section(text)] if text else [])


# Registry. Unknown Extensions Fall Back to the Text Loader (Covers Code as Plain Text)
LOADERS = {".md": load_markdown, ".pdf": load_pdf, ".txt": load_text}


def load(path: Path) -> LoadedDoc:
    return LOADERS.get(path.suffix.lower(), load_text)(path)
