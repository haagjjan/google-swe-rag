"""Integration test for the local query-time RAG orchestration."""

from pathlib import Path

import pytest

from swe_google_rag import rag
from swe_google_rag.config import Settings
from swe_google_rag.schemas import DocumentChunk
from swe_google_rag.vector_store import save_index


def _settings(vector_path: Path) -> Settings:
    return Settings(
        google_api_key="test-google-key",
        generation_model="generation-test",
        embedding_model="embedding-test",
        embedding_dimension=2,
        pdf_storage_path=Path("/pdfs"),
        vector_store_path=vector_path,
        chunk_size_tokens=512,
        chunk_overlap_tokens=64,
        top_k=1,
    )


def test_answer_question_retrieves_and_generates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        DocumentChunk(
            chunk_id="testing",
            text="Testing supports safe changes.",
            source_filename="chapter-1.pdf",
            page_number=3,
        ),
        DocumentChunk(
            chunk_id="teamwork",
            text="Teams need clear communication.",
            source_filename="chapter-2.pdf",
            page_number=4,
        ),
    ]
    save_index(
        tmp_path,
        chunks,
        [[1.0, 0.0], [0.0, 1.0]],
        "embedding-test",
        2,
    )
    monkeypatch.setattr(rag, "embed_question", lambda **_: [1.0, 0.0])
    captured: dict[str, str] = {}

    def fake_generation(
        question: str,
        context: str,
        settings: Settings,
    ) -> str:
        captured["question"] = question
        captured["context"] = context
        return "Testing supports safe changes. [Source 1]"

    monkeypatch.setattr(rag, "generate_grounded_answer", fake_generation)

    answer = rag.answer_question(
        "Why do we test?",
        _settings(tmp_path),
    )

    assert answer.text.endswith("[Source 1]")
    assert answer.sources[0].chunk.chunk_id == "testing"
    assert captured["question"] == "Why do we test?"
    assert "Testing supports safe changes." in captured["context"]


def test_stored_model_must_match_configuration(tmp_path: Path) -> None:
    chunk = DocumentChunk(
        chunk_id="one",
        text="text",
        source_filename="chapter.pdf",
        page_number=1,
    )
    save_index(tmp_path, [chunk], [[1.0, 0.0]], "other-model", 2)

    with pytest.raises(ValueError, match="does not match"):
        rag.retrieve_for_question("question", _settings(tmp_path))
