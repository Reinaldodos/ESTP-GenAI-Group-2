from pathlib import Path
from types import SimpleNamespace

import pytest

import run_rag


class FakeConfigType:
    loaded_paths: list[Path] = []

    @classmethod
    def from_yaml(cls, path: Path) -> object:
        cls.loaded_paths.append(path)
        return object()


def prepare_cli(
    monkeypatch: pytest.MonkeyPatch, pipeline: object, answers: list[str]
) -> None:
    FakeConfigType.loaded_paths = []
    values = iter(answers)
    monkeypatch.setattr(run_rag, "RAGConfig", FakeConfigType)
    monkeypatch.setattr(run_rag, "RAGPipeline", lambda config: pipeline)
    monkeypatch.setattr("builtins.input", lambda prompt: next(values))


def test_main_loads_project_config_and_exits_immediately(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline = SimpleNamespace(chunk_count=2)
    prepare_cli(monkeypatch, pipeline, ["exit"])

    run_rag.main()

    assert FakeConfigType.loaded_paths == [run_rag.CONFIG_PATH]
    assert "Pipeline prêt (2 passages indexés)." in capsys.readouterr().out


def test_main_ignores_blank_questions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakePipeline:
        chunk_count = 1

        def ask(self, question: str) -> object:
            pytest.fail(f"ask ne doit pas être appelé pour {question!r}")

    prepare_cli(monkeypatch, FakePipeline(), ["   ", "exit"])

    run_rag.main()

    assert "--- Réponse ---" not in capsys.readouterr().out


def test_main_prints_sources_scores_and_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked: list[str] = []

    class FakePipeline:
        chunk_count = 1

        def ask(self, question: str) -> object:
            asked.append(question)
            return SimpleNamespace(
                sources=[
                    {
                        "source": "policy.md",
                        "chunk_id": 0,
                        "text": "Policy text",
                        "score": 0.8765,
                    }
                ],
                answer="Allowed",
            )

    prepare_cli(monkeypatch, FakePipeline(), ["My question", "exit"])

    run_rag.main()

    output = capsys.readouterr().out
    assert asked == ["My question"]
    assert "policy.md (score=0.876)" in output
    assert "Policy text" in output
    assert "--- Réponse ---" in output
    assert "Allowed" in output
