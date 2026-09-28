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
         -> Qwen2.5-0.5B-Instruct answers from those 5 chunks only
         -> UI shows the answer + the source file/heading/snippet per chunk
```

## Stack

Streamlit · sentence-transformers (bge-small-en-v1.5, ms-marco-MiniLM-L-6-v2)
· ChromaDB (in-memory) · rank_bm25 · transformers (Qwen2.5-0.5B-Instruct)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

First run downloads the embedding model (~130MB), the reranker (~90MB), and
the generator (~1GB) — a minute or two on a normal connection.

## Deploy

Hosted on [Streamlit Community Cloud](https://share.streamlit.io) (free,
public repos only): connect this GitHub repo, set the main file to `app.py`,
deploy. No extra config needed — `requirements.txt` covers it.

The generator is deliberately a small model (0.5B params, ~1GB RAM) so the
whole app fits Streamlit Community Cloud's free-tier memory budget. A larger
model (e.g. Qwen2.5-3B-Instruct, ~6-12GB RAM) gives noticeably better answers
but needs a host with more headroom — Hugging Face Spaces' CPU-basic tier
(16GB RAM) works well for that, but hosting a Docker/Gradio Space there
currently requires an [HF PRO](https://huggingface.co/pro) subscription
($9/mo); static (no-Python) Spaces are the only free SDK on HF right now.

## Sample corpus

`data/fastapi-docs/` — 12 pages vendored from the FastAPI project's own
documentation (MIT license). See `data/fastapi-docs/SOURCE.md`.

## Tests

```bash
pytest -v
```
