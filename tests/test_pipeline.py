from pathlib import Path

import pytest

import rag_pipeline.pipeline as pipeline_module
from rag_pipeline.config import RAGConfig
from rag_pipeline.pipeline import RAGPipeline, RAGResponse


def make_config() -> RAGConfig:
    return RAGConfig(
        documents_dir=Path("documents"),
        embedding_model="embedding-model",
        llm_base_url="http://localhost:1234/v1",
        llm_api_key="key",
        llm_model="llm-model",
        llm_temperature=0.2,
        chunk_size=400,
        chunk_overlap=50,
        retrieval_k=4,
    )


def test_pipeline_initialization_orchestrates_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config()
    documents = [{"source": "doc.md", "text": "Document"}]
    chunks = [{"source": "doc.md", "chunk_id": 0, "text": "Chunk"}]
    client = object()
    calls: dict[str, object] = {}

    def fake_load_documents(path: Path) -> object:
        calls["documents_dir"] = path
        return documents

    def fake_chunk_documents(
        loaded_documents: object, *, chunk_size: int, overlap: int
    ) -> object:
        calls["chunk"] = (loaded_documents, chunk_size, overlap)
        return chunks

    class FakeRetriever:
        def __init__(self, received_chunks: object, model_name: str) -> None:
            calls["retriever"] = (received_chunks, model_name)
            self.chunks = received_chunks

    def fake_create_client(base_url: str, api_key: str) -> object:
        calls["client"] = (base_url, api_key)
        return client

    monkeypatch.setattr(pipeline_module, "load_documents", fake_load_documents)
    monkeypatch.setattr(pipeline_module, "chunk_documents", fake_chunk_documents)
    monkeypatch.setattr(pipeline_module, "VectorRetriever", FakeRetriever)
    monkeypatch.setattr(pipeline_module, "create_client", fake_create_client)

    pipeline = RAGPipeline(config)

    assert pipeline.config is config
    assert pipeline.client is client
    assert pipeline.chunk_count == 1
    assert calls == {
        "documents_dir": Path("documents"),
        "chunk": (documents, 400, 50),
        "retriever": (chunks, "embedding-model"),
        "client": ("http://localhost:1234/v1", "key"),
    }


@pytest.mark.parametrize(("requested_k", "expected_k"), [(None, 4), (2, 2), (0, 0)])
def test_retrieve_uses_config_default_or_explicit_k(
    requested_k: int | None, expected_k: int
) -> None:
    calls: list[tuple[str, int]] = []
    expected = [{"source": "doc.md", "chunk_id": 0, "text": "Text", "score": 1.0}]

    class FakeRetriever:
        chunks = [{"text": "Text"}]

        def search(self, question: str, k: int) -> object:
            calls.append((question, k))
            return expected

    pipeline = object.__new__(RAGPipeline)
    pipeline.config = make_config()
    pipeline.retriever = FakeRetriever()

    assert pipeline.retrieve("Question", requested_k) is expected
    assert calls == [("Question", expected_k)]


def test_ask_returns_generated_answer_and_retrieved_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config()
    sources = [{"source": "doc.md", "chunk_id": 0, "text": "Text", "score": 0.8}]
    client = object()
    calls: dict[str, object] = {}

    class FakeRetriever:
        chunks = [{"text": "Text"}]

        def search(self, question: str, k: int) -> object:
            calls["search"] = (question, k)
            return sources

    def fake_generate_answer(
        received_client: object,
        model: str,
        temperature: float,
        question: str,
        results: object,
    ) -> str:
        calls["generation"] = (
            received_client,
            model,
            temperature,
            question,
            results,
        )
        return "Generated answer"

    monkeypatch.setattr(pipeline_module, "generate_answer", fake_generate_answer)
    pipeline = object.__new__(RAGPipeline)
    pipeline.config = config
    pipeline.retriever = FakeRetriever()
    pipeline.client = client

    response = pipeline.ask("Question", k=2)

    assert response == RAGResponse(answer="Generated answer", sources=sources)
    assert calls == {
        "search": ("Question", 2),
        "generation": (
            client,
            "llm-model",
            0.2,
            "Question",
            sources,
        ),
    }
