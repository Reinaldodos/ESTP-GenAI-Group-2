from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import rag_pipeline
from rag_pipeline.config import RAGConfig


APP_PATH = Path(__file__).parents[1] / "app.py"


class FakePipeline:
    instances: list["FakePipeline"] = []
    fail_requests = False

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.chunk_count = 12
        self.document_sources = ("policy.md", "hotels.md")
        self.ask_calls: list[tuple[str, int | None, float | None]] = []
        self.instances.append(self)

    def ask(
        self,
        question: str,
        k: int | None = None,
        temperature: float | None = None,
    ) -> object:
        self.ask_calls.append((question, k, temperature))
        if self.fail_requests:
            raise ConnectionError("LM Studio unavailable")
        return SimpleNamespace(
            answer="Travel is allowed.",
            sources=[
                {
                    "source": "policy.md",
                    "chunk_id": 2,
                    "text": "Relevant policy text",
                    "score": 0.875,
                }
            ],
        )


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    st.cache_resource.clear()
    FakePipeline.instances = []
    FakePipeline.fail_requests = False
    monkeypatch.setattr(rag_pipeline, "RAGPipeline", FakePipeline)


def start_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
    assert not app.exception
    return app


def test_app_starts_and_displays_pipeline_metadata() -> None:
    app = start_app()

    assert len(FakePipeline.instances) == 1
    assert any("Mistral" in caption.value or "local LLM" in caption.value for caption in app.caption)
    assert any("Documents:** 2" in markdown.value for markdown in app.markdown)
    assert any("Chunks:** 12" in markdown.value for markdown in app.markdown)
    assert app.info[0].value == "Ask a question to search the travel documents."


def test_cache_rebuilds_only_when_chunk_settings_change() -> None:
    app = start_app()

    app.slider[2].set_value(5).run()
    app.slider[3].set_value(0.8).run()
    assert len(FakePipeline.instances) == 1

    app.slider[0].set_value(600).run()
    assert len(FakePipeline.instances) == 2
    assert FakePipeline.instances[-1].config.chunk_size == 600

    app.slider[1].set_value(150).run()
    assert len(FakePipeline.instances) == 3
    assert FakePipeline.instances[-1].config.chunk_overlap == 150


def test_question_uses_selected_generation_settings_and_displays_sources() -> None:
    app = start_app()
    app.slider[2].set_value(5).run()
    app.slider[3].set_value(0.7).run()
    app.text_input[0].set_value("Can I book this hotel?").run()

    pipeline = FakePipeline.instances[-1]
    assert pipeline.ask_calls == [("Can I book this hotel?", 5, 0.7)]
    assert any("Travel is allowed." in markdown.value for markdown in app.markdown)
    assert app.expander[0].label == "1. policy.md — similarity 0.875"


def test_unavailable_llm_displays_configured_model() -> None:
    app = start_app()
    FakePipeline.fail_requests = True

    app.text_input[0].set_value("Can I travel?").run()

    assert "mistral-7b" in app.error[0].value
    assert "LM Studio" in app.error[0].value
    assert app.exception[0].message == "LM Studio unavailable"
