"""Configuration centrale du pipeline RAG."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """Paramètres nécessaires à l'indexation et à la génération."""

    documents_dir: Path
    embedding_model: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_temperature: float
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RAGConfig":
        """Charge la configuration depuis un fichier YAML en texte brut."""
        config_path = Path(path)
        try:
            raw_config: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Fichier de configuration introuvable : {config_path}"
            ) from None
        except yaml.YAMLError as error:
            raise ValueError(f"YAML invalide dans {config_path} : {error}") from error

        if not isinstance(raw_config, dict):
            raise ValueError(
                f"La racine de {config_path} doit être une table de paramètres YAML"
            )

        expected_keys = set(cls.__dataclass_fields__)
        unknown_keys = set(raw_config) - expected_keys
        if unknown_keys:
            names = ", ".join(sorted(map(str, unknown_keys)))
            raise ValueError(f"Paramètre(s) inconnu(s) dans {config_path} : {names}")

        missing_keys = expected_keys - set(raw_config)
        if missing_keys:
            names = ", ".join(sorted(missing_keys))
            raise ValueError(f"Paramètre(s) manquant(s) dans {config_path} : {names}")

        if "documents_dir" in raw_config:
            documents_dir = Path(raw_config["documents_dir"])
            if not documents_dir.is_absolute():
                documents_dir = config_path.parent / documents_dir
            raw_config["documents_dir"] = documents_dir

        return cls(**raw_config)

    def __post_init__(self) -> None:
        object.__setattr__(self, "documents_dir", Path(self.documents_dir))
        object.__setattr__(self, "llm_temperature", float(self.llm_temperature))
        if not 0 <= self.llm_temperature <= 2:
            raise ValueError("llm_temperature doit être compris entre 0 et 2")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size doit être strictement positif")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "chunk_overlap doit être compris entre 0 et chunk_size - 1"
            )
        if self.retrieval_k <= 0:
            raise ValueError("retrieval_k doit être strictement positif")
