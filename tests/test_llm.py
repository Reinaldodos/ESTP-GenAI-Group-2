from types import SimpleNamespace

import pytest

import rag_pipeline.llm as llm_module
from rag_pipeline.llm import FALLBACK_ANSWER, build_prompt, create_client, generate_answer


RESULTS = [
    {"source": "first.md", "chunk_id": 0, "text": "First", "score": 0.9},
    {"source": "second.md", "chunk_id": 1, "text": "Second", "score": 0.8},
]


def test_create_client_forwards_connection_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}
    sentinel = object()

    def fake_openai(*, base_url: str, api_key: str) -> object:
        captured.update(base_url=base_url, api_key=api_key)
        return sentinel

    monkeypatch.setattr(llm_module, "OpenAI", fake_openai)

    client = create_client("http://localhost/v1", "key")

    assert client is sentinel
    assert captured == {"base_url": "http://localhost/v1", "api_key": "key"}


def test_build_prompt_contains_rules_question_and_ordered_context() -> None:
    prompt = build_prompt("What is allowed?", RESULTS)

    assert FALLBACK_ANSWER in prompt
    assert "Do not invent information." in prompt
    assert "Question:\nWhat is allowed?" in prompt
    assert "[Source: first.md]\nFirst" in prompt
    assert "[Source: second.md]\nSecond" in prompt
    assert prompt.index("first.md") < prompt.index("second.md")


def test_build_prompt_accepts_empty_results() -> None:
    prompt = build_prompt("Unknown?", [])

    assert "Question:\nUnknown?" in prompt
    assert FALLBACK_ANSWER in prompt


@pytest.mark.parametrize(("content", "expected"), [("Answer", "Answer"), (None, "")])
def test_generate_answer_sends_expected_request(
    content: str | None, expected: str
) -> None:
    calls: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            calls.append(kwargs)
            message = SimpleNamespace(content=content)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )

    answer = generate_answer(client, "model", 0.25, "Question?", RESULTS)

    assert answer == expected
    assert calls == [
        {
            "model": "model",
            "messages": [
                {"role": "user", "content": build_prompt("Question?", RESULTS)}
            ],
            "temperature": 0.25,
        }
    ]
