import numpy as np
import pytest

from rag.index import build_index
from rag.ingest import Chunk


class FakeEmbedder:
    """Deterministic fake embedder — no network, no model download."""

    def encode(self, texts):
        return np.array([[float(len(t)), float(t.count("a"))] for t in texts])


def test_build_index_creates_queryable_collection():
    chunks = [
        Chunk(id="a", source="a.md", heading="A", text="aaaa"),
        Chunk(id="b", source="b.md", heading="B", text="bb"),
    ]
    index = build_index(chunks, embedder=FakeEmbedder())
    assert index.collection.count() == 2
    assert len(index.bm25.doc_freqs) == 2


def test_build_index_rejects_empty_chunks():
    with pytest.raises(ValueError):
        build_index([], embedder=FakeEmbedder())
