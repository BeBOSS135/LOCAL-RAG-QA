# Loader Tests - Frontmatter, Inline Tags, Heading Ancestry
from pathlib import Path

import loaders


def _md(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "note.md"
    p.write_text(text, encoding="utf-8")
    return p


# --- Frontmatter --------------------------------------------------------------

def test_frontmatter_title_and_inline_tags_list(tmp_path):
    doc = loaders.load_markdown(_md(tmp_path, '---\ntitle: My Note\ntags: [ml, "rag"]\n---\nBody.'))
    assert doc.title == "My Note"
    assert doc.tags == ["ml", "rag"]


def test_frontmatter_block_list_tags(tmp_path):
    doc = loaders.load_markdown(_md(tmp_path, "---\ntags:\n- alpha\n- beta\n---\nBody."))
    assert doc.tags == ["alpha", "beta"]


def test_no_frontmatter_title_falls_back_to_stem(tmp_path):
    doc = loaders.load_markdown(_md(tmp_path, "Just a body."))
    assert doc.title == "note"


# --- Inline #tags ---------------------------------------------------------------

def test_inline_tags_require_a_letter(tmp_path):
    # "#1"/"#2" Are List/Footnote Markers, Not Tags - the Bug the Regex Fix Targeted
    doc = loaders.load_markdown(_md(tmp_path, "Steps #1 and #2 relate to #ml and #area/nlp."))
    assert doc.tags == ["area/nlp", "ml"]


def test_inline_tags_ignore_code(tmp_path):
    text = "Real #tag here.\n```py\nx = 1  #comment\n#CONST\n```\nAnd `#inline` too."
    doc = loaders.load_markdown(_md(tmp_path, text))
    assert doc.tags == ["tag"]


def test_frontmatter_and_inline_tags_merge_dedup(tmp_path):
    doc = loaders.load_markdown(_md(tmp_path, "---\ntags: [ml]\n---\nAlso #ml and #new."))
    assert doc.tags == ["ml", "new"]


# --- Heading Structure ---------------------------------------------------------

def test_heading_ancestry(tmp_path):
    text = "# Top\nintro\n## Sub\ndetail\n## Sib\nmore"
    doc = loaders.load_markdown(_md(tmp_path, text))
    assert [(s.heading_path, s.text) for s in doc.sections] == [
        ("Top", "intro"), ("Top > Sub", "detail"), ("Top > Sib", "more"),
    ]


def test_skipped_heading_level_omitted_from_path(tmp_path):
    doc = loaders.load_markdown(_md(tmp_path, "# Top\n### Deep\ntext"))
    assert doc.sections[0].heading_path == "Top > Deep"


def test_text_before_first_heading_is_flat(tmp_path):
    doc = loaders.load_markdown(_md(tmp_path, "preamble\n# Top\nbody"))
    assert doc.sections[0] == loaders.Section("preamble", "")


def test_heading_free_note_is_one_section(tmp_path):
    doc = loaders.load_markdown(_md(tmp_path, "line one\nline two"))
    assert len(doc.sections) == 1
    assert doc.sections[0].heading_path == ""


def test_empty_note_has_no_sections(tmp_path):
    assert loaders.load_markdown(_md(tmp_path, "")).sections == []


# --- Plain Text ------------------------------------------------------------------

def test_load_text(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("  hello  ", encoding="utf-8")
    doc = loaders.load_text(p)
    assert doc.source_type == "text"
    assert doc.sections == [loaders.Section("hello")]


def test_load_text_empty(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("", encoding="utf-8")
    assert loaders.load_text(p).sections == []


def test_unknown_extension_falls_back_to_text(tmp_path):
    p = tmp_path / "script.py"
    p.write_text("print('hi')", encoding="utf-8")
    assert loaders.load(p).source_type == "text"
