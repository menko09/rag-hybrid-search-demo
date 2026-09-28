from __future__ import annotations

import uuid
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

    embedder = embedder if embedder is not None else _default_embedder()
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name=f"docs-{uuid.uuid4().hex}")

    embeddings = embedder.encode([c.text for c in chunks]).tolist()
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
    )

    bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    return Index(chunks=chunks, collection=collection, bm25=bm25, embedder=embedder)
