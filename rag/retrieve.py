from __future__ import annotations

from typing import Any

from rag.index import Index
from rag.ingest import Chunk

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """Merge several ranked id lists into one ranking via reciprocal rank fusion."""
    scores: dict[str, float] = {}
    for ranked_ids in rank_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)


def _dense_rank(index: Index, query: str, k: int) -> list[str]:
    query_embedding = index.embedder.encode([query]).tolist()
    result = index.collection.query(
        query_embeddings=query_embedding, n_results=min(k, len(index.chunks))
    )
    return result["ids"][0]


def _bm25_rank(index: Index, query: str, k: int) -> list[str]:
    scores = index.bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [index.chunks[i].id for i in ranked[:k]]


def load_reranker():
    """Load the cross-encoder reranker. Callers that run many queries against
    the same index (e.g. app.py) should load this once and pass it in via
    `retrieve(..., reranker=...)` rather than relying on the per-call default."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL_NAME)


def retrieve(
    index: Index,
    query: str,
    k_dense: int = 20,
    k_bm25: int = 20,
    top_n: int = 5,
    reranker: Any | None = None,
) -> list[Chunk]:
    dense_ids = _dense_rank(index, query, k_dense)
    bm25_ids = _bm25_rank(index, query, k_bm25)
    fused_ids = reciprocal_rank_fusion([dense_ids, bm25_ids])

    by_id = {c.id: c for c in index.chunks}
    candidates = [by_id[doc_id] for doc_id in fused_ids if doc_id in by_id]
    if not candidates:
        return []

    reranker = reranker if reranker is not None else load_reranker()
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_n]]
