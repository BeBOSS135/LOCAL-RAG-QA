# Query-Path Logic Tests - Dedup, Abstention Gate, Citations
import config
import rag


# --- _dedup ----------------------------------------------------------------------

def _hits(*texts):
    return [{"text": t} for t in texts]


def test_dedup_drops_exact_duplicate_keeps_first(monkeypatch):
    same = "alpha beta gamma delta"
    kept = rag._dedup(_hits(same, same, "totally different words here"))
    assert [h["text"] for h in kept] == [same, "totally different words here"]


def test_dedup_keeps_below_floor(monkeypatch):
    monkeypatch.setattr(config, "DEDUP_JACCARD", 0.85)
    kept = rag._dedup(_hits("a b c d e f", "a b c x y z"))  # Jaccard 3/9 = 0.33
    assert len(kept) == 2


def test_dedup_near_duplicate_dropped(monkeypatch):
    monkeypatch.setattr(config, "DEDUP_JACCARD", 0.85)
    base = [f"w{i}" for i in range(20)]
    near = base[:19] + ["different"]  # Jaccard 19/21 = 0.90
    kept = rag._dedup(_hits(" ".join(base), " ".join(near)))
    assert len(kept) == 1


def test_dedup_empty_texts_never_divide_by_zero():
    assert len(rag._dedup(_hits("", "", "words"))) == 3


# --- _abstain ---------------------------------------------------------------------
# Two-Signal Gate: Answer if EITHER Rerank or Cosine Clears Its Floor (See config)

def _hit(rerank=None, sim=None):
    h = {"score": sim}
    if rerank is not None:
        h["rerank_score"] = rerank
    return h


def test_abstain_gate_off_answers(monkeypatch):
    monkeypatch.setattr(config, "USE_ABSTAIN", False)
    assert rag._abstain([_hit(rerank=0.0, sim=0.1)], True) == (False, None, None, "none")


def test_abstain_no_hits_always_abstains(monkeypatch):
    monkeypatch.setattr(config, "USE_ABSTAIN", False)
    assert rag._abstain([], True)[0] is True


def test_abstain_rerank_signal_passes(monkeypatch):
    monkeypatch.setattr(config, "USE_ABSTAIN", True)
    abstain, rr, sim, signal = rag._abstain([_hit(rerank=0.4, sim=0.2)], True)
    assert (abstain, rr, sim, signal) == (False, 0.4, 0.2, "rerank|sim")


def test_abstain_terse_query_cosine_rescues(monkeypatch):
    # The Design Case: Rerank Collapses (~0) on Terse Questions but Cosine Holds
    monkeypatch.setattr(config, "USE_ABSTAIN", True)
    abstain, *_ = rag._abstain([_hit(rerank=0.004, sim=0.6)], True)
    assert abstain is False


def test_abstain_both_weak_refuses(monkeypatch):
    monkeypatch.setattr(config, "USE_ABSTAIN", True)
    abstain, *_ = rag._abstain([_hit(rerank=0.004, sim=0.3)], True)
    assert abstain is True


def test_abstain_uses_max_cosine_across_hits(monkeypatch):
    monkeypatch.setattr(config, "USE_ABSTAIN", True)
    hits = [_hit(rerank=0.0, sim=0.2), _hit(sim=0.9)]
    assert rag._abstain(hits, True)[0] is False


def test_abstain_bm25_only_hits_skip_none_scores(monkeypatch):
    # BM25-Only Chunks Carry score=None - Must Not Crash and Must Count as 0 Cosine
    monkeypatch.setattr(config, "USE_ABSTAIN", True)
    abstain, _, sim, _ = rag._abstain([_hit(rerank=0.004, sim=None)], True)
    assert (abstain, sim) == (True, 0.0)


def test_abstain_sim_only_path_without_rerank(monkeypatch):
    monkeypatch.setattr(config, "USE_ABSTAIN", True)
    assert rag._abstain([_hit(sim=0.6)], False) == (False, None, 0.6, "sim")
    assert rag._abstain([_hit(sim=0.3)], False)[0] is True


# --- _ref (Citations) ---------------------------------------------------------------

def test_ref_title_and_heading():
    assert rag._ref({"title": "Guide", "heading_path": "Setup > GPU"}) == "Guide › Setup › GPU"


def test_ref_drops_heading_repeating_title():
    assert rag._ref({"title": "Guide", "heading_path": "Guide > Setup"}) == "Guide › Setup"


def test_ref_flat_doc_is_just_title():
    assert rag._ref({"title": "Guide", "heading_path": ""}) == "Guide"


def test_ref_falls_back_to_source_path():
    assert rag._ref({"title": "", "source_path": "data/f.txt"}) == "data/f.txt"
