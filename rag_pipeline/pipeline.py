"""Façade orchestrant toutes les étapes du pipeline RAG."""

from dataclasses import dataclass

from .config import RAGConfig
from .documents import chunk_documents, load_documents
from .llm import create_client, generate_answer
from .retrieval import SearchResult, VectorRetriever


@dataclass(frozen=True, slots=True)
class RAGResponse:
    answer: str
    sources: list[SearchResult]


class RAGPipeline:
    """Pipeline prêt à répondre après chargement et indexation des documents."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.documents = load_documents(self.config.documents_dir)
        chunks = chunk_documents(
            self.documents,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        self.retriever = VectorRetriever(chunks, self.config.embedding_model)
        self.client = create_client(self.config.llm_base_url, self.config.llm_api_key)

    @property
    def chunk_count(self) -> int:
        return len(self.retriever.chunks)

    @property
    def document_sources(self) -> tuple[str, ...]:
        """Retourne les noms des documents indexés, dans leur ordre de chargement."""
        return tuple(document["source"] for document in self.documents)

    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        result_count = self.config.retrieval_k if k is None else k
        return self.retriever.search(question, result_count)

    def ask(
        self,
        question: str,
        k: int | None = None,
        temperature: float | None = None,
    ) -> RAGResponse:
        generation_temperature = (
            self.config.llm_temperature if temperature is None else float(temperature)
        )
        if not 0 <= generation_temperature <= 2:
            raise ValueError("temperature doit être comprise entre 0 et 2")
        results = self.retrieve(question, k)
        answer = generate_answer(
            self.client,
            self.config.llm_model,
            generation_temperature,
            question,
            results,
        )
        return RAGResponse(answer=answer, sources=results)
