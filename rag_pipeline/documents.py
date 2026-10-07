"""Chargement et découpage des documents sources."""

from pathlib import Path
from typing import TypedDict


class Document(TypedDict):
    source: str
    text: str


class Chunk(Document):
    chunk_id: int


def load_documents(documents_dir: Path) -> list[Document]:
    """Charge tous les fichiers Markdown d'un répertoire, dans un ordre stable."""
    documents_dir = Path(documents_dir)
    if not documents_dir.is_dir():
        raise FileNotFoundError(f"Répertoire de documents introuvable : {documents_dir}")

    documents = [
        Document(source=path.name, text=path.read_text(encoding="utf-8"))
        for path in sorted(documents_dir.glob("*.md"))
    ]
    if not documents:
        raise ValueError(f"Aucun fichier Markdown trouvé dans {documents_dir}")
    return documents


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Découpe un texte en segments de taille fixe avec chevauchement."""
    if chunk_size <= 0:
        raise ValueError("chunk_size doit être strictement positif")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap doit être compris entre 0 et chunk_size - 1")

    step = chunk_size - overlap
    return [text[start : start + chunk_size] for start in range(0, len(text), step)]


def chunk_documents(
    documents: list[Document], chunk_size: int = 500, overlap: int = 100
) -> list[Chunk]:
    """Découpe les documents et conserve leur provenance."""
    return [
        Chunk(source=document["source"], chunk_id=chunk_id, text=text)
        for document in documents
        for chunk_id, text in enumerate(
            chunk_text(document["text"], chunk_size=chunk_size, overlap=overlap)
        )
    ]
