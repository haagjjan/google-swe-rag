"""Tests for context formatting and grounded answer generation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from swe_google_rag import generation
from swe_google_rag.config import Settings
from swe_google_rag.schemas import DocumentChunk, RetrievedChunk


def _settings() -> Settings:
    return Settings(
        google_api_key="test-google-key",
        generation_model="generation-test",
        embedding_model="embedding-test",
        embedding_dimension=2,
        pdf_storage_path=Path("/pdfs"),
        vector_store_path=Path("/vectors"),
        chunk_size_tokens=512,
        chunk_overlap_tokens=64,
        top_k=3,
    )


def _result() -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id="chapter-p2-c0001-id",
            text="Tests protect behavior during change.",
            source_filename="chapter.pdf",
            page_number=2,
            section="Testing",
        ),
        similarity_score=0.9,
    )


def test_context_contains_text_and_provenance() -> None:
    context = generation.format_retrieved_context([_result()])

    assert "[Source 1]" in context
    assert "File: chapter.pdf" in context
    assert "Page: 2" in context
    assert "Section: Testing" in context
    assert "Tests protect behavior during change." in context


def test_empty_context_returns_insufficient_information_without_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_client(**_: object) -> None:
        raise AssertionError("Google API should not be called")

    monkeypatch.setattr(generation.genai, "Client", fail_client)

    answer = generation.generate_grounded_answer(
        "What is missing?",
        "",
        _settings(),
    )

    assert answer == generation._INSUFFICIENT_INFORMATION_MESSAGE


def test_generation_uses_grounding_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeModels:
        def generate_content(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(text="Grounded answer. [Source 1]")

    monkeypatch.setattr(
        generation.genai,
        "Client",
        lambda **_: SimpleNamespace(models=FakeModels()),
    )

    answer = generation.generate_grounded_answer(
        "Why test?",
        generation.format_retrieved_context([_result()]),
        _settings(),
    )

    assert answer == "Grounded answer. [Source 1]"
    assert "Why test?" in calls[0]["contents"]
    config = calls[0]["config"]
    assert "using only the supplied source excerpts" in (config.system_instruction)


def test_generation_failure_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModels:
        def generate_content(self, **_: object) -> None:
            raise ValueError("provider rejected request")

    monkeypatch.setattr(
        generation.genai,
        "Client",
        lambda **_: SimpleNamespace(models=FailingModels()),
    )

    with pytest.raises(RuntimeError, match="generation-test"):
        generation.generate_grounded_answer(
            "Why test?",
            "Some context",
            _settings(),
        )
