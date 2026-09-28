# RAG Hybrid Search HF Space Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a Hugging Face Space that answers questions over a bundled set of FastAPI docs using hybrid (dense + BM25) retrieval, cross-encoder reranking, and a grounded local LLM answer with source citations.

**Architecture:** Single Streamlit process. Startup builds an in-memory index (Chroma + BM25) from vendored markdown docs, cached via `st.cache_resource`. Each question runs dense + BM25 retrieval, reciprocal-rank-fusion merge, cross-encoder rerank to top 5, then Qwen2.5-3B-Instruct generates a grounded answer citing those chunks.

**Tech Stack:** Python, Streamlit, `sentence-transformers` (bge-small-en-v1.5 + ms-marco-MiniLM-L-6-v2 cross-encoder), ChromaDB (ephemeral/in-memory), `rank_bm25`, `transformers` (Qwen2.5-3B-Instruct), pytest.

Spec: `docs/superpowers/specs/2026-09-28-rag-hybrid-search-hf-space-design.md`

---

### Task 1: Project scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `rag/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write `requirements.txt`**

```
streamlit>=1.38
transformers>=4.45
sentence-transformers>=3.1
chromadb>=0.5
rank_bm25>=0.2.2
torch>=2.4
numpy>=1.26
pytest>=8.3
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
.pytest_cache/
.cache/
```

- [ ] **Step 3: Create empty package markers**

`rag/__init__.py` and `tests/__init__.py` — both empty files.

- [ ] **Step 4: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: all packages install without error (this pulls torch, ~a few hundred MB — no model weights yet).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore rag/__init__.py tests/__init__.py
git commit -m "chore: project scaffold"
```

---

### Task 2: Vendor the sample corpus

**Files:**
- Create: `data/fastapi-docs/first-steps.md`
- Create: `data/fastapi-docs/path-params.md`
- Create: `data/fastapi-docs/query-params.md`
- Create: `data/fastapi-docs/body.md`
- Create: `data/fastapi-docs/query-params-str-validations.md`
- Create: `data/fastapi-docs/path-params-numeric-validations.md`
- Create: `data/fastapi-docs/body-multiple-params.md`
- Create: `data/fastapi-docs/body-fields.md`
- Create: `data/fastapi-docs/extra-models.md`
- Create: `data/fastapi-docs/response-status-code.md`
- Create: `data/fastapi-docs/request-forms.md`
- Create: `data/fastapi-docs/handling-errors.md`
- Create: `data/fastapi-docs/SOURCE.md`

- [ ] **Step 1: Download the 12 bundled doc pages**

Each file below is fetched verbatim from the FastAPI repo (MIT license), master branch, and saved under `data/fastapi-docs/` with the same filename:

```bash
mkdir -p data/fastapi-docs
base="https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/tutorial"
for f in first-steps path-params query-params body query-params-str-validations \
         path-params-numeric-validations body-multiple-params body-fields \
         extra-models response-status-code request-forms handling-errors; do
  curl -sL "$base/$f.md" -o "data/fastapi-docs/$f.md"
done
```

Expected: 12 `.md` files in `data/fastapi-docs/`, each non-empty (`wc -l data/fastapi-docs/*.md` shows real line counts, not 0).

- [ ] **Step 2: Write the attribution file**

`data/fastapi-docs/SOURCE.md`:

```markdown
# Source

These files are vendored, unmodified, from the FastAPI project's
documentation (MIT license):

https://github.com/fastapi/fastapi/tree/master/docs/en/docs/tutorial

Fetched 2026-09-28. Used here as sample content for a retrieval-augmented
generation demo — not affiliated with or endorsed by the FastAPI project.
```

- [ ] **Step 3: Verify no download failed silently**

Run: `grep -L . data/fastapi-docs/*.md || echo "all files non-empty"`
Expected: `all files non-empty` (grep -L lists files with zero matching lines; a 404 page saved as a file would still have content, so also spot check one file's first line looks like markdown, e.g. `head -3 data/fastapi-docs/first-steps.md` should show a `#` heading, not an HTML error page).

- [ ] **Step 4: Commit**

```bash
git add data/
git commit -m "content: vendor FastAPI tutorial docs as sample corpus"
```

---

### Task 3: Ingestion and heading-aware chunking

**Files:**
- Create: `rag/ingest.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_ingest.py`:

```python
from rag.ingest import chunk_markdown, load_docs


def test_chunk_markdown_splits_on_headings():
    text = (
        "# Title\n\nIntro text.\n\n"
        "## Section One\n\nContent one.\n\n"
        "## Section Two\n\nContent two.\n"
    )
    chunks = chunk_markdown(text, source="doc.md")
    assert [c.heading for c in chunks] == ["Title", "Section One", "Section Two"]
    assert "Content one." in chunks[1].text
    assert all(c.source == "doc.md" for c in chunks)


def test_chunk_markdown_drops_empty_sections():
    text = "# Title\n\n## Empty\n\n## Filled\n\nSome text.\n"
    chunks = chunk_markdown(text, source="doc.md")
    assert [c.heading for c in chunks] == ["Title", "Filled"]


def test_load_docs_reads_all_markdown_files(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nBody A.\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n\nBody B.\n", encoding="utf-8")
    chunks = load_docs(tmp_path)
    assert {c.source for c in chunks} == {"a.md", "b.md"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError` or `ImportError: cannot import name 'chunk_markdown'` (`rag/ingest.py` doesn't exist yet).

- [ ] **Step 3: Write `rag/ingest.py`**

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    id: str
    source: str
    heading: str
    text: str


def chunk_markdown(text: str, source: str) -> list[Chunk]:
    """Split markdown into one chunk per heading (any level 1-6).

    Content before the first heading is dropped if empty, kept otherwise
    under heading "".
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            sections.append((current_heading, current_lines))
            current_heading = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    sections.append((current_heading, current_lines))

    chunks: list[Chunk] = []
    for i, (heading, section_lines) in enumerate(sections):
        body = "\n".join(section_lines).strip()
        if not body:
            continue
        chunks.append(Chunk(id=f"{source}::{i}", source=source, heading=heading, text=body))
    return chunks


def load_docs(data_dir: Path) -> list[Chunk]:
    """Load and chunk every .md file in data_dir, sorted by filename."""
    chunks: list[Chunk] = []
    for path in sorted(Path(data_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(text, source=path.name))
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/ingest.py tests/test_ingest.py
git commit -m "feat: heading-aware markdown chunking and doc loading"
```

---

### Task 4: Index building (embeddings + Chroma + BM25)

**Files:**
- Create: `rag/index.py`
- Test: `tests/test_index.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_index.py` (uses a fake embedder so this test never downloads a model — fast and offline):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_index.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.index'`.

- [ ] **Step 3: Write `rag/index.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from rag.ingest import Chunk

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dataclass
class Index:
    chunks: list[Chunk]
    collection: Any
    bm25: BM25Okapi
    embedder: Any


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _default_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def build_index(chunks: list[Chunk], embedder: Any | None = None) -> Index:
    if not chunks:
        raise ValueError("build_index requires at least one chunk")

    embedder = embedder or _default_embedder()
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="docs")

    embeddings = embedder.encode([c.text for c in chunks]).tolist()
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
    )

    bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    return Index(chunks=chunks, collection=collection, bm25=bm25, embedder=embedder)
```

Note: `_default_embedder` imports `sentence_transformers` lazily inside the function so importing `rag.index` (and running tests with a fake embedder) never triggers a model download.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_index.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/index.py tests/test_index.py
git commit -m "feat: build in-memory Chroma + BM25 index from chunks"
```

---

### Task 5: Hybrid retrieval, fusion, and reranking

**Files:**
- Create: `rag/retrieve.py`
- Test: `tests/test_retrieve.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_retrieve.py` (fake embedder + fake reranker — offline, no model downloads):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retrieve.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.retrieve'`.

- [ ] **Step 3: Write `rag/retrieve.py`**

```python
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


def _default_reranker():
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

    reranker = reranker or _default_reranker()
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_n]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retrieve.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/retrieve.py tests/test_retrieve.py
git commit -m "feat: hybrid retrieval with reciprocal rank fusion and reranking"
```

---

### Task 6: Grounded generation

**Files:**
- Create: `rag/generate.py`
- Test: `tests/test_generate.py`

- [ ] **Step 1: Write the failing test**

`tests/test_generate.py` (tests prompt construction only — never loads the 3B model):

```python
from rag.generate import SYSTEM_PROMPT, build_prompt
from rag.ingest import Chunk


def test_build_prompt_includes_context_and_question():
    chunks = [
        Chunk(id="1", source="a.md", heading="Intro", text="FastAPI is a web framework."),
    ]
    messages = build_prompt("What is FastAPI?", chunks)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "FastAPI is a web framework." in messages[1]["content"]
    assert "What is FastAPI?" in messages[1]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.generate'`.

- [ ] **Step 3: Write `rag/generate.py`**

```python
from __future__ import annotations

from rag.ingest import Chunk

GENERATOR_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = (
    'Answer the question using ONLY the provided context. If the answer is '
    'not in the context, reply exactly: "Not found in the docs."'
)


def build_prompt(question: str, context_chunks: list[Chunk]) -> list[dict]:
    context = "\n\n".join(f"[{c.source} — {c.heading}]\n{c.text}" for c in context_chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def load_generator():
    """Load the text-generation pipeline. Downloads ~6GB on first call —
    never invoked from automated tests, only from app.py at Space startup."""
    from transformers import pipeline

    return pipeline(
        "text-generation",
        model=GENERATOR_MODEL_NAME,
        device_map="cpu",
        max_new_tokens=300,
    )


def answer_question(generator, question: str, context_chunks: list[Chunk]) -> str:
    messages = build_prompt(question, context_chunks)
    output = generator(messages)
    return output[0]["generated_text"][-1]["content"].strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_generate.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/generate.py tests/test_generate.py
git commit -m "feat: grounded prompt construction and LLM generation"
```

---

### Task 7: Streamlit app

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write `app.py`**

```python
from pathlib import Path

import streamlit as st

from rag.generate import answer_question, load_generator
from rag.ingest import load_docs
from rag.index import build_index
from rag.retrieve import retrieve

DATA_DIR = Path(__file__).parent / "data" / "fastapi-docs"


@st.cache_resource(show_spinner="Building the document index...")
def get_index():
    chunks = load_docs(DATA_DIR)
    return build_index(chunks)


@st.cache_resource(show_spinner="Loading the language model (first run only)...")
def get_generator():
    return load_generator()


st.title("RAG Hybrid Search — FastAPI Docs Q&A")
st.caption(
    "Hybrid (dense + BM25) retrieval, cross-encoder reranking, and a grounded "
    "local LLM answer over a bundled set of FastAPI documentation pages. "
    "Answers only from the docs shown below the answer — the first question "
    "may take 15-30s on CPU."
)

question = st.text_input("Ask a question about FastAPI:")

if question:
    index = get_index()
    generator = get_generator()
    with st.spinner("Searching the docs and generating an answer..."):
        chunks = retrieve(index, question)
        answer = answer_question(generator, question, chunks)

    st.markdown("### Answer")
    st.write(answer)

    with st.expander("Sources"):
        for chunk in chunks:
            st.markdown(f"**{chunk.source} — {chunk.heading}**")
            st.text(chunk.text[:500])
```

- [ ] **Step 2: Manual smoke run**

Run: `streamlit run app.py`
Expected: browser opens at `localhost:8501`; type "How do I declare a path parameter in FastAPI?"; after the model loads (first run downloads Qwen2.5-3B-Instruct, ~6GB — this step alone can take several minutes on a slow connection) an answer appears citing `path-params.md` under Sources.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit UI wiring retrieval and generation"
```

---

### Task 8: End-to-end smoke test

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write the test**

This is the one test that uses the real (small) embedding + reranker models against the real bundled corpus — it downloads bge-small (~130MB) and the cross-encoder (~90MB) on first run, but never the 6GB generator, since it checks retrieval/citation only (matches the spec's definition of done: "a known question returns an answer citing the right file" — the citation set is fully determined by retrieval, generation only phrases it).

`tests/test_smoke.py`:

```python
from pathlib import Path

from rag.index import build_index
from rag.ingest import load_docs
from rag.retrieve import retrieve

DATA_DIR = Path(__file__).parent.parent / "data" / "fastapi-docs"


def test_retrieval_cites_the_right_source_file():
    chunks = load_docs(DATA_DIR)
    index = build_index(chunks)
    results = retrieve(
        index, "How do I declare a path parameter in FastAPI?", top_n=3
    )
    sources = {c.source for c in results}
    assert "path-params.md" in sources
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS (the first run downloads the two small models — verified at ~130s on one connection, budget a couple minutes rather than seconds; fast on subsequent runs once cached).

- [ ] **Step 3: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass (ingest, index, retrieve, generate, smoke).

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: end-to-end retrieval smoke test against the bundled corpus"
```

---

### Task 9: README with HF Space config

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

The YAML front matter is required by Hugging Face Spaces — it's how the Space knows which SDK and entry file to run.

```markdown
---
title: RAG Hybrid Search Demo
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
license: mit
---

# RAG Hybrid Search Demo

Q&A over a bundled set of FastAPI documentation pages. Answers only from
those docs, and cites the exact passage used.

## How it works

```
startup: load docs -> heading-aware chunks -> embed (bge-small-en-v1.5)
         -> Chroma (dense index) + BM25 (rank_bm25)

query:   dense top-20 + BM25 top-20
         -> reciprocal rank fusion
         -> cross-encoder rerank (ms-marco-MiniLM-L-6-v2) -> top 5
         -> Qwen2.5-3B-Instruct answers from those 5 chunks only
         -> UI shows the answer + the source file/heading/snippet per chunk
```

## Stack

Streamlit · sentence-transformers (bge-small-en-v1.5, ms-marco-MiniLM-L-6-v2)
· ChromaDB (in-memory) · rank_bm25 · transformers (Qwen2.5-3B-Instruct)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

First run downloads the embedding model (~130MB), the reranker (~90MB), and
the generator (~6GB) — expect several minutes before the first answer.

## Sample corpus

`data/fastapi-docs/` — 12 pages vendored from the FastAPI project's own
documentation (MIT license). See `data/fastapi-docs/SOURCE.md`.

## Tests

```bash
pytest -v
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with architecture overview and HF Space config"
```

---

### Task 10: Deploy to GitHub and Hugging Face

**Files:** none (repo-level operations)

- [ ] **Step 1: Create and push the GitHub repo**

Run:
```bash
gh repo create menko09/rag-hybrid-search-demo --public --source=. --remote=origin --push
```
Expected: repo created at `github.com/menko09/rag-hybrid-search-demo`, current branch pushed.

- [ ] **Step 2: Create the HF Space**

Run:
```bash
python -c "
from huggingface_hub import create_repo
create_repo(
    repo_id='Jaypare13/rag-hybrid-search-demo',
    repo_type='space',
    space_sdk='streamlit',
    exist_ok=True,
)
print('Space created')
"
```
Expected: prints `Space created`; repo now exists at `huggingface.co/spaces/Jaypare13/rag-hybrid-search-demo` (empty, showing HF's default placeholder page).

- [ ] **Step 3: Push code to the Space**

Run:
```bash
git remote add space https://huggingface.co/spaces/Jaypare13/rag-hybrid-search-demo
git push space master:main
```
Expected: push succeeds (auth via the `hf auth login` token from earlier in this session, cached by `huggingface_hub`'s git credential helper).

- [ ] **Step 4: Verify the Space builds**

Run:
```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
print(api.space_info('Jaypare13/rag-hybrid-search-demo').runtime)
"
```
Expected: `stage` eventually reads `RUNNING` (it will show `BUILDING` right after the push — building installs `requirements.txt` and downloads the models on the Space's own machine, which can take several minutes for the 3B model; re-run this check after a few minutes if it's still building).
