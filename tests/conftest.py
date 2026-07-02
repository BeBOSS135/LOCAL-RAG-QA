# Shared Test Setup - Puts src/ on the Path and Fakes the HF Tokenizer So Chunking
# Tests Are Deterministic and Never Load a Model
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# Whitespace "Tokenizer" With the Same Interface ingest Uses (input_ids +
# offset_mapping) - One Token per Word Makes Window/Merge Sizes Easy to Reason About
class WordTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        out = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


@pytest.fixture
def word_tokenizer(monkeypatch):
    import embeddings
    monkeypatch.setattr(embeddings, "get_tokenizer", lambda: WordTokenizer())
