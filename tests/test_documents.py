from pathlib import Path

import pytest

from rag_pipeline.documents import chunk_documents, chunk_text, load_documents


def test_load_documents_reads_markdown_in_stable_order(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("Deuxième", encoding="utf-8")
    (tmp_path / "a.md").write_text("Été", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("Ignoré", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert documents == [
        {"source": "a.md", "text": "Été"},
        {"source": "b.md", "text": "Deuxième"},
    ]


def test_load_documents_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Répertoire de documents introuvable"):
        load_documents(tmp_path / "missing")


def test_load_documents_rejects_directory_without_markdown(tmp_path: Path) -> None:
    (tmp_path / "document.txt").write_text("Texte", encoding="utf-8")

    with pytest.raises(ValueError, match="Aucun fichier Markdown"):
        load_documents(tmp_path)


@pytest.mark.parametrize(
    ("text", "chunk_size", "overlap", "expected"),
    [
        ("", 4, 1, []),
        ("abcd", 4, 0, ["abcd"]),
        ("abcdefghij", 4, 1, ["abcd", "defg", "ghij", "j"]),
        ("abcde", 4, 0, ["abcd", "e"]),
    ],
)
def test_chunk_text(
    text: str, chunk_size: int, overlap: int, expected: list[str]
) -> None:
    assert chunk_text(text, chunk_size=chunk_size, overlap=overlap) == expected


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)],
)
def test_chunk_text_rejects_invalid_parameters(
    chunk_size: int, overlap: int
) -> None:
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=chunk_size, overlap=overlap)


def test_chunk_documents_preserves_source_and_resets_ids() -> None:
    documents = [
        {"source": "first.md", "text": "abcdef"},
        {"source": "second.md", "text": "xyz"},
    ]

    chunks = chunk_documents(documents, chunk_size=4, overlap=1)

    assert chunks == [
        {"source": "first.md", "chunk_id": 0, "text": "abcd"},
        {"source": "first.md", "chunk_id": 1, "text": "def"},
        {"source": "second.md", "chunk_id": 0, "text": "xyz"},
    ]
