# RAG Hybrid Search — HF Space demo design

Source: "AI Portfolio Projects — Scope (Free Stack)" doc, Project 1 (RAG Pipeline
with Hybrid Search). This is a scoped-down demo of that project for a public
Hugging Face Space, not the full 5-7 day spec.

## Purpose

Portfolio demo: a Q&A app that answers questions only from a bundled set of
docs and shows which passage backs each answer. Showcases hybrid
(dense + BM25) retrieval, reranking, and grounded generation.

## Scope decisions

- **Lean demo, not full spec.** Dropped from the original project scope:
  the 30-question Ragas eval harness, automated citation-verification
  checking, and side-by-side comparison of two chunking strategies. These are
  dev-process/eval artifacts, not demo-visible features. Kept: ingestion,
  hybrid retrieval, reranking, grounded answers, citation *display*.
- **No Ollama.** The source doc's stack uses Ollama for local LLM/embeddings.
  HF Spaces' free CPU tier can't run an Ollama daemon well. Swapped for
  in-process `transformers`/`sentence-transformers` — no daemon, standard
  for HF Spaces.
- **Single Streamlit process, no FastAPI.** The source doc's architecture is
  a FastAPI backend + Streamlit frontend calling it over HTTP. For a single
  Space container, an in-process function call is simpler than an HTTP hop to
  a sidecar service. The `/ingest` + `/ask` API surface is a "when this
  becomes the full 3-project system" concern, not this Space.
- **Bundled sample corpus, no upload UI.** ~12 core pages of FastAPI's own
  documentation (MIT-licensed), vendored into the repo. Ingested once at
  startup, cached in memory. No live document upload.
- **Single chunking strategy.** Heading-aware chunking only.
- **Model:** Qwen2.5-3B-Instruct via `transformers` (ungated, CPU-workable).

## Architecture

One Streamlit app (`app.py`), single container, HF Spaces Streamlit SDK.

```
startup (st.cache_resource, runs once per Space instance):
  load bundled docs (data/fastapi-docs/*.md)
    -> heading-aware chunker -> chunks
    -> bge-small-en-v1.5 embeddings -> Chroma in-memory collection
    -> rank_bm25 index over the same chunks

query (per user question):
  dense_hits  = chroma.query(question, k=20)
  bm25_hits   = bm25.get_top_n(question, k=20)
  fused       = reciprocal_rank_fusion(dense_hits, bm25_hits)
  reranked    = cross_encoder("ms-marco-MiniLM-L-6-v2").rerank(fused)[:5]
  answer      = qwen2.5-3b-instruct(prompt(question, reranked))
  UI renders: answer text + expandable "Sources" section
              (file name, chunk heading, snippet) per cited chunk
```

## Components

- `data/fastapi-docs/` — vendored markdown source files (bundled corpus).
- `rag/ingest.py` — load + chunk the bundled docs.
- `rag/index.py` — build/hold the Chroma collection + BM25 index, embedding
  model load.
- `rag/retrieve.py` — dense + BM25 retrieval, reciprocal rank fusion,
  cross-encoder reranking.
- `rag/generate.py` — prompt construction, Qwen2.5-3B-Instruct load + call,
  grounding/"not found" behavior.
- `app.py` — Streamlit UI: question box, spinner, answer, sources panel.
  Wires the above via `st.cache_resource` (index + models load once).
- `tests/test_smoke.py` — one pytest: index builds, a known question about
  FastAPI path parameters returns an answer citing the right source file.

## Data flow

User types a question in Streamlit → `app.py` calls `retrieve.py` (which
reads the cached index from `index.py`) → top-5 reranked chunks go to
`generate.py` → model output + the chunks used are returned together →
`app.py` renders the answer and a sources expander listing each chunk's file
and snippet.

## Error handling

- Model load failure at startup → Streamlit shows the exception directly
  (fail loud, not a blank page).
- No relevant chunks / low-relevance retrieval → the grounding prompt makes
  the model say "not found in the docs" — no special-case retrieval code
  needed, the LLM prompt carries this.
- CPU inference is slow (~5-10 tok/s expected) → UI shows a spinner and a
  one-line note that the first answer may take 15-30s.

## Testing

One smoke test (`tests/test_smoke.py`): build the index from the bundled
corpus, ask a known question ("How do I declare a path parameter in
FastAPI?"), assert the answer's cited source file is the expected doc page.
Not the source doc's full 30-question eval harness — that's a full-spec
concern, out of scope for this Space.

## Deployment

- New local repo (this one), pushed to two remotes:
  - GitHub: `menko09/rag-hybrid-search-demo` (public)
  - HF Space: `huggingface.co/spaces/Jaypare13/rag-hybrid-search-demo`,
    Streamlit SDK, CPU basic (free tier)
- HF auth: `hf auth login` (done, user `Jaypare13`), used for
  `huggingface_hub.create_repo(repo_type="space", space_sdk="streamlit")` +
  git push to the Space's git remote.
- `requirements.txt`: streamlit, transformers, sentence-transformers,
  chromadb, rank_bm25, torch (cpu).
