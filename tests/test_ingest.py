# Chunking + Document Listing Tests (Fake Word Tokenizer: 1 Token == 1 Word)
import config
import ingest
from loaders import LoadedDoc, Section


# Small Window Sizes So Tests Stay Readable: 8-Token Chunks, 2-Token Overlap
def _small_chunks(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_TOKENS", 8)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_TOKENS", 2)


def _words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


# --- _window --------------------------------------------------------------------

def test_window_short_text_is_one_chunk(word_tokenizer, monkeypatch):
    _small_chunks(monkeypatch)
    assert ingest._window("a b c") == ["a b c"]


def test_window_splits_with_overlap_no_word_breakage(word_tokenizer, monkeypatch):
    _small_chunks(monkeypatch)
    text = _words(20)
    out = ingest._window(text)
    # Step = 8 - 2 = 6 -> Starts 0/6/12; Window at 12 Reaches the End, So No 4th
    assert len(out) == 3
    assert out[0].split() == text.split()[0:8]
    assert out[1].split() == text.split()[6:14]   # 2-Word Overlap With the Previous
    assert out[2].split() == text.split()[12:20]  # Final Window Covers Through the End
    # Whole-Word Slicing - Every Piece Is Made of Intact Input Words
    vocab = set(text.split())
    assert all(w in vocab for piece in out for w in piece.split())


def test_window_empty_text(word_tokenizer, monkeypatch):
    _small_chunks(monkeypatch)
    assert ingest._window("   ") == []


# --- _chunk_doc -------------------------------------------------------------------

def _doc(sections):
    return LoadedDoc("markdown", "T", ["t1"], sections)


def _chunk(monkeypatch, sections):
    _small_chunks(monkeypatch)
    return ingest._chunk_doc(_doc(sections), "src/f.md", 1.0, "hash")


def test_small_same_top_heading_sections_merge(word_tokenizer, monkeypatch):
    chunks = _chunk(monkeypatch, [Section("a b", "H1 > A"), Section("c d", "H1 > B")])
    assert len(chunks) == 1
    assert chunks[0].text == "a b\n\nc d"
    assert chunks[0].heading_path == "H1 > A"  # Group Anchored at Its First Section


def test_top_heading_change_starts_new_chunk(word_tokenizer, monkeypatch):
    chunks = _chunk(monkeypatch, [Section("a b", "H1"), Section("c d", "H2")])
    assert [(c.text, c.heading_path) for c in chunks] == [("a b", "H1"), ("c d", "H2")]


def test_merge_stops_at_token_budget(word_tokenizer, monkeypatch):
    # 5 + 5 Words > 8-Token Window -> Two Chunks Despite the Same Top Heading
    chunks = _chunk(monkeypatch, [Section(_words(5), "H1"), Section(_words(5, "x"), "H1")])
    assert len(chunks) == 2


def test_oversized_section_windows_alone(word_tokenizer, monkeypatch):
    chunks = _chunk(monkeypatch, [Section("a b", "H1"), Section(_words(20), "H1")])
    # Buffer Flushed First, Then the Big Section Split Into Its Own Windows
    assert chunks[0].text == "a b"
    assert len(chunks) == 1 + 3


def test_chunk_metadata_and_index_sequence(word_tokenizer, monkeypatch):
    chunks = _chunk(monkeypatch, [Section(_words(20), "H1")])
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.source_path == "src/f.md" and c.title == "T" and c.tags == ["t1"]
               for c in chunks)


# --- list_documents ----------------------------------------------------------------

def test_list_documents_filters(monkeypatch, tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "b.exe").write_text("x", encoding="utf-8")           # Not in the Allowlist
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "c.md").write_text("x", encoding="utf-8")  # Hidden Dir
    (tmp_path / "Archive").mkdir()
    (tmp_path / "Archive" / "d.md").write_text("x", encoding="utf-8")    # Excluded
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "e.txt").write_text("x", encoding="utf-8")

    monkeypatch.setattr(config, "SOURCES",
                        [{"name": "t", "path": tmp_path, "exclude": ["Archive"]}])
    assert [sp for _, sp in ingest.list_documents()] == ["t/a.md", "t/sub/e.txt"]


def test_list_documents_skips_missing_root(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SOURCES", [{"name": "gone", "path": tmp_path / "nope"}])
    assert ingest.list_documents() == []
