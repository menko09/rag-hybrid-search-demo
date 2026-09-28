from pathlib import Path

import streamlit as st

from rag.generate import answer_question, load_generator
from rag.ingest import load_docs
from rag.index import build_index
from rag.retrieve import load_reranker, retrieve

DATA_DIR = Path(__file__).parent / "data" / "fastapi-docs"


@st.cache_resource(show_spinner="Building the document index...")
def get_index():
    chunks = load_docs(DATA_DIR)
    return build_index(chunks)


@st.cache_resource(show_spinner="Loading the language model (first run only)...")
def get_generator():
    return load_generator()


@st.cache_resource(show_spinner="Loading the reranker...")
def get_reranker():
    return load_reranker()


st.title("RAG Hybrid Search — FastAPI Docs Q&A")
st.caption(
    "Hybrid (dense + BM25) retrieval, cross-encoder reranking, and a grounded "
    "local LLM answer over a bundled set of FastAPI documentation pages. "
    "Answers only from the docs shown below the answer — the first question "
    "may take 15-30s on CPU."
)

question = st.text_input("Ask a question about FastAPI:")

if question and question.strip():
    index = get_index()
    generator = get_generator()
    reranker = get_reranker()
    with st.spinner("Searching the docs and generating an answer..."):
        chunks = retrieve(index, question, reranker=reranker)
        answer = answer_question(generator, question, chunks)

    st.markdown("### Answer")
    st.write(answer)

    with st.expander("Sources"):
        for chunk in chunks:
            st.markdown(f"**{chunk.source} — {chunk.heading}**")
            snippet = chunk.text[:500]
            st.text(snippet + "…" if len(chunk.text) > 500 else snippet)
