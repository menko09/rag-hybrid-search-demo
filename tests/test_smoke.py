from pathlib import Path

from rag.index import build_index
from rag.ingest import load_docs
from rag.retrieve import retrieve

DATA_DIR = Path(__file__).parent.parent / "data" / "fastapi-docs"


def test_retrieval_cites_the_right_source_file():
    """Uses the real bge-small-en-v1.5 embedder and ms-marco-MiniLM-L-6-v2
    reranker against the bundled corpus — no fakes, unlike every other test
    in this suite. Downloads ~220MB of model weights on first run (cold run
    took ~2 minutes); the only test here that checks retrieval quality
    against real data, not just wiring."""
    chunks = load_docs(DATA_DIR)
    index = build_index(chunks)
    results = retrieve(
        index, "How do I declare a path parameter in FastAPI?", top_n=3
    )
    sources = {c.source for c in results}
    assert "path-params.md" in sources
