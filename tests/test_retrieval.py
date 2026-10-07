from collections.abc import Sequence

import numpy as np
import pytest

import rag_pipeline.retrieval as retrieval_module
from rag_pipeline.retrieval import VectorRetriever


CHUNKS = [
    {"source": "first.md", "chunk_id": 0, "text": "First passage"},
    {"source": "second.md", "chunk_id": 0, "text": "Second passage"},
]


class FakeEmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls: list[tuple[list[str], bool]] = []

    def encode(
        self, texts: Sequence[str], *, normalize_embeddings: bool
    ) -> list[list[float]]:
        values = list(texts)
        self.calls.append((values, normalize_embeddings))
        if values == ["Question"]:
            return [[0.5, 0.5]]
        return [[1.0, 0.0], [0.0, 1.0]]


class FakeIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.ntotal = 0
        self.added: np.ndarray | None = None
        self.search_calls: list[tuple[np.ndarray, int]] = []

    def add(self, embeddings: np.ndarray) -> None:
        self.added = embeddings
        self.ntotal = len(embeddings)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        self.search_calls.append((query, k))
        return (
            np.array([[0.9, 0.25]], dtype=np.float32)[:, :k],
            np.array([[1, 0]], dtype=np.int64)[:, :k],
        )


@pytest.fixture
def fake_boundaries(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    created: dict[str, object] = {}

    def create_model(model_name: str) -> FakeEmbeddingModel:
        model = FakeEmbeddingModel(model_name)
        created["model"] = model
        return model

    def create_index(dimension: int) -> FakeIndex:
        index = FakeIndex(dimension)
        created["index"] = index
        return index

    monkeypatch.setattr(retrieval_module, "SentenceTransformer", create_model)
    monkeypatch.setattr(retrieval_module.faiss, "IndexFlatIP", create_index)
    return created


def test_retriever_rejects_empty_chunks() -> None:
    with pytest.raises(ValueError, match="sans passage"):
        VectorRetriever([], "model")


def test_retriever_builds_normalized_float32_index(
    fake_boundaries: dict[str, object],
) -> None:
    retriever = VectorRetriever(CHUNKS, "embedding-model")
    model = fake_boundaries["model"]
    index = fake_boundaries["index"]

    assert isinstance(model, FakeEmbeddingModel)
    assert isinstance(index, FakeIndex)
    assert retriever.embedding_model is model
    assert retriever.index is index
    assert model.model_name == "embedding-model"
    assert model.calls == [(["First passage", "Second passage"], True)]
    assert index.dimension == 2
    assert index.added is not None
    assert index.added.dtype == np.float32
    np.testing.assert_array_equal(index.added, [[1.0, 0.0], [0.0, 1.0]])


@pytest.mark.parametrize("question", ["", " ", "\n\t"])
def test_search_rejects_blank_question(
    fake_boundaries: dict[str, object], question: str
) -> None:
    retriever = VectorRetriever(CHUNKS, "model")

    with pytest.raises(ValueError, match="question ne peut pas être vide"):
        retriever.search(question)


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_non_positive_k(
    fake_boundaries: dict[str, object], k: int
) -> None:
    retriever = VectorRetriever(CHUNKS, "model")

    with pytest.raises(ValueError, match="k doit être strictement positif"):
        retriever.search("Question", k=k)


def test_search_caps_k_and_maps_scores_to_chunks(
    fake_boundaries: dict[str, object],
) -> None:
    retriever = VectorRetriever(CHUNKS, "model")

    results = retriever.search("Question", k=99)
    model = fake_boundaries["model"]
    index = fake_boundaries["index"]

    assert isinstance(model, FakeEmbeddingModel)
    assert isinstance(index, FakeIndex)
    assert model.calls[-1] == (["Question"], True)
    assert index.search_calls[0][1] == 2
    assert index.search_calls[0][0].dtype == np.float32
    assert results == [
        {
            "source": "second.md",
            "chunk_id": 0,
            "text": "Second passage",
            "score": pytest.approx(0.9),
        },
        {
            "source": "first.md",
            "chunk_id": 0,
            "text": "First passage",
            "score": pytest.approx(0.25),
        },
    ]
