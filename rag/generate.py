from __future__ import annotations

from rag.ingest import Chunk

GENERATOR_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = (
    'Answer the question using ONLY the provided context. Each context block '
    'is labeled [source — heading]; mention the source when you use it, but '
    'do not copy the bracket label itself into your answer. If the answer is '
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
        do_sample=False,
    )


def answer_question(generator, question: str, context_chunks: list[Chunk]) -> str:
    messages = build_prompt(question, context_chunks)
    output = generator(messages)
    return output[0]["generated_text"][-1]["content"].strip()
