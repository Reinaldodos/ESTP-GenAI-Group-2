"""Création de l'index vectoriel et recherche des passages pertinents."""

from typing import TypedDict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .documents import Chunk


class SearchResult(Chunk):
    score: float


class VectorRetriever:
    """Index FAISS associé à un modèle d'embeddings et à ses passages."""

    def __init__(self, chunks: list[Chunk], model_name: str) -> None:
        if not chunks:
            raise ValueError("Impossible de créer un index sans passage")

        self.chunks = chunks
        self.embedding_model = SentenceTransformer(model_name)
        embeddings = self.embedding_model.encode(
            [chunk["text"] for chunk in chunks], normalize_embeddings=True
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, question: str, k: int = 3) -> list[SearchResult]:
        """Retourne les k passages les plus proches de la question."""
        if not question.strip():
            raise ValueError("La question ne peut pas être vide")
        if k <= 0:
            raise ValueError("k doit être strictement positif")

        query_embedding = self.embedding_model.encode(
            [question], normalize_embeddings=True
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        result_count = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, result_count)

        return [
            SearchResult(
                **self.chunks[int(index_number)],
                score=float(score),
            )
            for score, index_number in zip(scores[0], indices[0])
        ]
