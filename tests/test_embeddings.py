"""Tests for Google embedding request orchestration."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from swe_google_rag import embeddings
from swe_google_rag.config import Settings
from swe_google_rag.schemas import DocumentChunk


def _settings(dimension: int | None = 2) -> Settings:
    return Settings(
        google_api_key="test-google-key",
        generation_model="generation-test",
        embedding_model="embedding-test",
        embedding_dimension=dimension,
        pdf_storage_path=Path("/pdfs"),
        vector_store_path=Path("/vectors"),
        chunk_size_tokens=512,
        chunk_overlap_tokens=64,
        top_k=3,
    )


def _chunk(identifier: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        text=text,
        source_filename="chapter.pdf",
        page_number=1,
    )


class _FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def embed_content(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        contents = kwargs["contents"]
        assert isinstance(contents, list)
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[float(len(text)), 1.0]) for text in contents
            ]
        )


def test_document_embeddings_are_batched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = _FakeModels()
    client = SimpleNamespace(models=models)
    monkeypatch.setattr(
        embeddings.genai,
        "Client",
        lambda **_: client,
    )
    monkeypatch.setattr(embeddings, "_EMBEDDING_BATCH_SIZE", 2)
    chunks = [
        _chunk("one", "alpha"),
        _chunk("two", "bravo"),
        _chunk("three", "charlie"),
    ]

    vectors = embeddings.embed_document_chunks(chunks, _settings())

    assert vectors == [[5.0, 1.0], [5.0, 1.0], [7.0, 1.0]]
    assert [len(call["contents"]) for call in models.calls] == [2, 1]
    assert models.calls[0]["model"] == "embedding-test"


def test_question_embedding_returns_one_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = _FakeModels()
    monkeypatch.setattr(
        embeddings.genai,
        "Client",
        lambda **_: SimpleNamespace(models=models),
    )

    vector = embeddings.embed_question("  question  ", _settings())

    assert vector == [8.0, 1.0]
    assert models.calls[0]["contents"] == ["question"]


def test_empty_question_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        embeddings.embed_question("   ", _settings())


def test_api_failure_is_wrapped_without_exposing_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModels:
        def embed_content(self, **_: object) -> None:
            raise ValueError("provider rejected request")

    monkeypatch.setattr(
        embeddings.genai,
        "Client",
        lambda **_: SimpleNamespace(models=FailingModels()),
    )

    with pytest.raises(RuntimeError, match="embedding-test") as error:
        embeddings.embed_question("question", _settings())

    assert "test-google-key" not in str(error.value)


def test_unexpected_embedding_dimension_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WrongDimensionModels:
        def embed_content(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 2.0, 3.0])])

    monkeypatch.setattr(
        embeddings.genai,
        "Client",
        lambda **_: SimpleNamespace(models=WrongDimensionModels()),
    )

    with pytest.raises(RuntimeError, match="Expected embedding dimension"):
        embeddings.embed_question("question", _settings())
