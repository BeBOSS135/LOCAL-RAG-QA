# Eval-Set Parsing + Rank Scoring Tests
import evaluate


def test_eval_set_splits_by_answerable_flag():
    # Every Item Lands in Exactly One Slice; Answerable Items Must Carry Expected Sources
    assert len(evaluate.ANSWERABLE) + len(evaluate.UNANSWERABLE) == len(evaluate.EVAL_SET)
    assert all("expected" in it for it in evaluate.ANSWERABLE)


def test_first_hit_rank_is_one_based():
    sources = ["vault/a.md", "vault/b.md", "data/c.pdf"]
    assert evaluate._first_hit_rank(sources, ["b.md"]) == 2
    assert evaluate._first_hit_rank(sources, ["a.md", "c.pdf"]) == 1  # Earliest Match Wins


def test_first_hit_rank_no_match_is_zero():
    assert evaluate._first_hit_rank(["vault/a.md"], ["nope.md"]) == 0
