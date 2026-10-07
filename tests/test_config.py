from pathlib import Path

import pytest
import yaml

from rag_pipeline.config import RAGConfig


def valid_config(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "documents_dir": "documents",
        "embedding_model": "embedding-model",
        "llm_base_url": "http://localhost:1234/v1",
        "llm_api_key": "test-key",
        "llm_model": "llm-model",
        "llm_temperature": 0,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "retrieval_k": 3,
    }
    values.update(overrides)
    return values


def write_config(path: Path, values: object) -> None:
    path.write_text(yaml.safe_dump(values), encoding="utf-8")


def test_from_yaml_loads_values_and_resolves_relative_documents_path(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path, valid_config())

    config = RAGConfig.from_yaml(config_path)

    assert config.documents_dir == tmp_path / "documents"
    assert config.embedding_model == "embedding-model"
    assert config.llm_base_url == "http://localhost:1234/v1"
    assert config.llm_api_key == "test-key"
    assert config.llm_model == "llm-model"
    assert config.llm_temperature == 0.0
    assert config.chunk_size == 500
    assert config.chunk_overlap == 100
    assert config.retrieval_k == 3


def test_from_yaml_preserves_absolute_documents_path(tmp_path: Path) -> None:
    documents_dir = tmp_path / "absolute-documents"
    config_path = tmp_path / "config.yaml"
    write_config(config_path, valid_config(documents_dir=str(documents_dir)))

    assert RAGConfig.from_yaml(config_path).documents_dir == documents_dir


def test_from_yaml_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Fichier de configuration introuvable"):
        RAGConfig.from_yaml(tmp_path / "missing.yaml")


def test_from_yaml_rejects_invalid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("value: [unclosed", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML invalide"):
        RAGConfig.from_yaml(config_path)


@pytest.mark.parametrize("root", [None, [], "plain text"])
def test_from_yaml_rejects_non_mapping_root(tmp_path: Path, root: object) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path, root)

    with pytest.raises(ValueError, match="doit être une table"):
        RAGConfig.from_yaml(config_path)


def test_from_yaml_rejects_unknown_parameter(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path, valid_config(unexpected=True))

    with pytest.raises(ValueError, match=r"Paramètre\(s\) inconnu\(s\).+unexpected"):
        RAGConfig.from_yaml(config_path)


def test_from_yaml_rejects_missing_parameter(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    values = valid_config()
    del values["llm_model"]
    write_config(config_path, values)

    with pytest.raises(ValueError, match=r"Paramètre\(s\) manquant\(s\).+llm_model"):
        RAGConfig.from_yaml(config_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"llm_temperature": -0.1}, "llm_temperature"),
        ({"llm_temperature": 2.1}, "llm_temperature"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"chunk_overlap": -1}, "chunk_overlap"),
        ({"chunk_overlap": 500}, "chunk_overlap"),
        ({"retrieval_k": 0}, "retrieval_k"),
    ],
)
def test_from_yaml_rejects_invalid_numeric_parameters(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path, valid_config(**overrides))

    with pytest.raises(ValueError, match=message):
        RAGConfig.from_yaml(config_path)


def test_project_configuration_is_valid() -> None:
    project_root = Path(__file__).parents[1]

    config = RAGConfig.from_yaml(project_root / "config.yaml")

    assert config.documents_dir.is_dir()
