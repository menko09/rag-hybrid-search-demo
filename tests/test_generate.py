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
