"""Configuration centrale du pipeline RAG."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """Paramètres nécessaires à l'indexation et à la génération."""

    documents_dir: Path = Path("data")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_base_url: str = "http://127.0.0.1:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_model: str = "mistral-7b"
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_k: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "documents_dir", Path(self.documents_dir))
        if self.chunk_size <= 0:
            raise ValueError("chunk_size doit être strictement positif")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "chunk_overlap doit être compris entre 0 et chunk_size - 1"
            )
        if self.retrieval_k <= 0:
            raise ValueError("retrieval_k doit être strictement positif")
