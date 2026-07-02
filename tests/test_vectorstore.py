# Metadata Derivation + Scope-Filter Build Tests (No Chroma Client Touched)
import vectorstore
from ingest import Chunk


def _chunk(source_path, tags=()):
    return Chunk(text="t", source_path=source_path, source_type="markdown", title="T",
                 heading_path="", tags=list(tags), mtime=1.0, content_hash="h", chunk_index=0)


# --- _meta -------------------------------------------------------------------------

def test_meta_derives_source_name_and_folder():
    m = vectorstore._meta(_chunk("vault/Project Notes/x.md"))
    assert m["source_name"] == "vault"
    assert m["folder"] == "vault/Project Notes"


def test_meta_flat_source_path():
    m = vectorstore._meta(_chunk("x.md"))
    assert m["source_name"] == "x.md"
    assert m["folder"] == "x.md"


def test_meta_tags_bar_delimited():
    # Bars Wrap Whole Tags So a Substring Post-Filter Can't Prefix-Match ("|ml|" vs "mlops")
    assert vectorstore._meta(_chunk("v/x.md", ["a", "b"]))["tags"] == "|a|b|"
    assert vectorstore._meta(_chunk("v/x.md"))["tags"] == ""


# --- scope_where --------------------------------------------------------------------

def _fake_folders(monkeypatch, folders):
    monkeypatch.setattr(vectorstore, "list_folders", lambda: folders)


def test_scope_where_none_is_unscoped():
    assert vectorstore.scope_where(None) is None
    assert vectorstore.scope_where("") is None


def test_scope_where_expands_prefix_to_descendants(monkeypatch):
    # "vaulty" Must NOT Match the "vault" Prefix (Boundary Is prefix + "/")
    _fake_folders(monkeypatch, ["data", "vault", "vault/Project Notes", "vaulty"])
    assert vectorstore.scope_where("vault") == {"folder": {"$in": ["vault", "vault/Project Notes"]}}


def test_scope_where_subfolder_only(monkeypatch):
    _fake_folders(monkeypatch, ["vault", "vault/Project Notes"])
    assert vectorstore.scope_where("vault/Project Notes") == {"folder": {"$in": ["vault/Project Notes"]}}


def test_scope_where_no_match_uses_impossible_sentinel(monkeypatch):
    # Empty $in Would Make Chroma Error - the Sentinel Returns Nothing so the Gate Abstains
    _fake_folders(monkeypatch, ["data"])
    where = vectorstore.scope_where("nope")
    assert where["folder"]["$in"] == ["\x00__no_such_folder__"]
