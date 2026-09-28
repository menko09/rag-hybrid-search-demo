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

Superseded at deploy time — see addendum below. Original plan: GitHub +
a Hugging Face Space (Streamlit SDK, CPU basic free tier).

- `requirements.txt`: streamlit, transformers, sentence-transformers,
  chromadb, rank_bm25, torch (cpu).

## Addendum (deploy-time revision, 2026-09-28)

Two things discovered only at actual deploy time changed the plan above:

1. **HF Spaces no longer accepts `sdk: streamlit`** — only `gradio`,
   `docker`, or `static`. Streamlit apps now need `sdk: docker` with a
   Dockerfile.
2. **Hosting a Docker or Gradio Space on this HF account's free CPU-basic
   tier returned `402 Payment Required`** — needs an HF PRO subscription
   ($9/mo). Only static (no-Python) Spaces are free on this account.

Given user's account isn't paying for HF PRO, chose to deploy free instead:
- **Generator model swapped**: Qwen2.5-3B-Instruct (~6-12GB RAM) →
  Qwen2.5-0.5B-Instruct (~1GB RAM, loaded with `torch_dtype="auto"` to use
  the checkpoint's native bf16 rather than upcasting to fp32). Answer
  quality is noticeably weaker at this size, but it fits a free host's RAM.
- **Host swapped**: HF Spaces → Streamlit Community Cloud (genuinely free,
  native Streamlit support, no Dockerfile needed — removed the Dockerfile
  and `.dockerignore` added for the HF docker-SDK attempt).
- GitHub repo (`menko09/rag-hybrid-search-demo`) is unaffected — still the
  canonical source, Streamlit Community Cloud deploys straight from it.
- If HF PRO is ever added, the code still works there too — just needs a
  Dockerfile again (the original `docker run streamlit on :7860` approach)
  and reverting `GENERATOR_MODEL_NAME` to the 3B model for better answers.

## Addendum 2 (domain pivot, post-launch)

After the first live deploy, swapped the demo's subject matter from FastAPI
docs to an internal-company-handbook Q&A assistant — a closer match to the
actual portfolio pitch (automation consulting for businesses) than
developer docs, and lower reputational risk than the alternative domains
considered (legal/healthcare demos read worse when a small model
occasionally answers from general knowledge instead of strict citation,
which is a real, observed failure mode of the 0.5B model).

- Corpus: `data/fastapi-docs/` → `data/handbook-docs/` — 10 pages vendored
  from GitLab's public Team Handbook (MIT license): communication norms,
  company values, total rewards. One fetched page (`top-misused-terms.md`)
  was dropped — its actual list content is injected by a Hugo shortcode at
  GitLab's build time, so the vendored copy had no real static content.
- `rag/ingest.py` gained YAML-frontmatter stripping (`_strip_frontmatter`)
  in `load_docs` — every handbook page has a `---\ntitle: ...\n---` header
  that the FastAPI corpus never had; left unstripped it produced junk
  heading-less chunks. `chunk_markdown` itself is unchanged.
  `tests/test_smoke.py`'s question/expected-source pair updated to match
  (`"What is the Power of the Pause?"` → `power-of-the-pause.md`).
  Verified for real: full suite passes (13 tests) and a real end-to-end
  generation run (actual models, actual corpus) was checked by hand.
