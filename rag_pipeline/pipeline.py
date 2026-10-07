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

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        documents = load_documents(self.config.documents_dir)
        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        self.retriever = VectorRetriever(chunks, self.config.embedding_model)
        self.client = create_client(self.config.llm_base_url, self.config.llm_api_key)

    @property
    def chunk_count(self) -> int:
        return len(self.retriever.chunks)

    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        result_count = self.config.retrieval_k if k is None else k
        return self.retriever.search(question, result_count)

    def ask(self, question: str, k: int | None = None) -> RAGResponse:
        results = self.retrieve(question, k)
        answer = generate_answer(
            self.client, self.config.llm_model, question, results
        )
        return RAGResponse(answer=answer, sources=results)
