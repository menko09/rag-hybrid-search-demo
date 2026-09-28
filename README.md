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
