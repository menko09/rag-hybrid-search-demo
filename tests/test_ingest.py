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
    assert [c.heading for c in chunks] == ["Filled"]


def test_load_docs_reads_all_markdown_files(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nBody A.\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n\nBody B.\n", encoding="utf-8")
    chunks = load_docs(tmp_path)
    assert {c.source for c in chunks} == {"a.md", "b.md"}


def test_chunk_markdown_strips_mkdocs_attr_list_from_heading():
    text = "## Body — Fields { #body-fields }\n\nSome text.\n"
    chunks = chunk_markdown(text, source="doc.md")
    assert chunks[0].heading == "Body — Fields"
    assert "{" not in chunks[0].heading and "}" not in chunks[0].heading


def test_chunk_markdown_keeps_content_before_first_heading():
    text = "Intro paragraph before any heading.\n\n# Title\n\nBody text.\n"
    chunks = chunk_markdown(text, source="doc.md")
    assert chunks[0].heading == ""
    assert "Intro paragraph before any heading." in chunks[0].text
    assert [c.heading for c in chunks] == ["", "Title"]


def test_load_docs_strips_yaml_frontmatter(tmp_path):
    (tmp_path / "a.md").write_text(
        '---\ntitle: "A"\ndescription: "desc"\n---\n\n## Heading\n\nBody A.\n',
        encoding="utf-8",
    )
    chunks = load_docs(tmp_path)
    assert [c.heading for c in chunks] == ["Heading"]
    assert "title:" not in chunks[0].text
    assert "---" not in chunks[0].text
