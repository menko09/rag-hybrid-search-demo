import numpy as np

from rag.index import build_index
from rag.ingest import Chunk
from rag.retrieve import reciprocal_rank_fusion, retrieve


def test_reciprocal_rank_fusion_favors_items_ranked_high_in_both_lists():
    dense = ["a", "b", "c"]
    bm25 = ["b", "a", "c"]
    fused = reciprocal_rank_fusion([dense, bm25])
    assert fused[0] in {"a", "b"}
    assert fused[-1] == "c"


def test_reciprocal_rank_fusion_rewards_consensus_over_single_list_top_rank():
    # "x" ranks #1 in dense but is entirely absent from bm25; "a" and "b" rank
    # lower individually but appear in both lists. A passthrough of either
    # input list alone would put "x" first (or drop it) — real fusion should
    # rank the two-list consensus above the single-list top pick.
    dense = ["x", "a", "b"]
    bm25 = ["a", "b"]
    fused = reciprocal_rank_fusion([dense, bm25])
    assert fused == ["a", "b", "x"]


class FakeEmbedder:
    def encode(self, texts):
        return np.array([[float(len(t)), float(t.count("a"))] for t in texts])


class FakeReranker:
    """Scores a candidate higher when the query's first word appears in its text."""

    def predict(self, pairs):
        return [float(query.split()[0] in text) for query, text in pairs]


def test_retrieve_returns_top_n_reranked_chunks():
    chunks = [
        Chunk(id="1", source="a.md", heading="A", text="uvicorn runs the app"),
        Chunk(id="2", source="b.md", heading="B", text="totally unrelated text"),
        Chunk(id="3", source="c.md", heading="C", text="another unrelated chunk"),
    ]
    index = build_index(chunks, embedder=FakeEmbedder())
    results = retrieve(index, "uvicorn", top_n=1, reranker=FakeReranker())
    assert len(results) == 1
    assert results[0].id == "1"
