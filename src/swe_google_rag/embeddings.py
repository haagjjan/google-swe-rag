"""Create document and query embeddings with the Google Gen AI SDK."""

from collections.abc import Sequence

from google import genai
from google.genai import types

from .config import Settings
from .schemas import DocumentChunk

_EMBEDDING_BATCH_SIZE = 100


def embed_document_chunks(
    chunks: Sequence[DocumentChunk],
    settings: Settings,
) -> list[list[float]]:
    """Return one embedding vector for each document chunk."""
    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]

    vectors = _embed_texts(
        texts=texts,
        task_type="RETRIEVAL_DOCUMENT",
        settings=settings,
    )

    if len(vectors) != len(chunks):
        raise RuntimeError(
            "The embedding API returned a different number of vectors "
            "than the number of document chunks."
        )

    return vectors


def embed_question(question: str, settings: Settings) -> list[float]:
    """Embed one complete short question for retrieval."""
    question = question.strip()

    if not question:
        raise ValueError("Question must not be empty.")

    vectors = _embed_texts(
        texts=[question],
        task_type="RETRIEVAL_QUERY",
        settings=settings,
    )

    if len(vectors) != 1:
        raise RuntimeError("The embedding API did not return exactly one query vector.")

    return vectors[0]


def _embed_texts(
    texts: Sequence[str],
    task_type: str,
    settings: Settings,
) -> list[list[float]]:
    """Request embeddings and validate their dimensions."""
    normalized_texts = [text.strip() for text in texts]
    if not normalized_texts:
        return []
    if any(not text for text in normalized_texts):
        raise ValueError("Embedding inputs must not be empty.")

    client = genai.Client(api_key=settings.google_api_key)

    vectors: list[list[float]] = []
    config_kwargs: dict[str, str | int] = {"task_type": task_type}
    if settings.embedding_dimension is not None:
        config_kwargs["output_dimensionality"] = settings.embedding_dimension

    for start in range(0, len(normalized_texts), _EMBEDDING_BATCH_SIZE):
        batch = normalized_texts[start : start + _EMBEDDING_BATCH_SIZE]

        try:
            response = client.models.embed_content(
                model=settings.embedding_model,
                contents=batch,
                config=types.EmbedContentConfig(**config_kwargs),
            )
        except Exception as exc:
            raise RuntimeError(
                f"The embedding request failed for model {settings.embedding_model!r}."
            ) from exc

        if not response.embeddings:
            raise RuntimeError("The embedding API returned no embeddings.")

        for embedding in response.embeddings:
            if embedding.values is None:
                raise RuntimeError("The embedding API returned an empty vector.")

            vectors.append(list(embedding.values))

    if len(vectors) != len(normalized_texts):
        raise RuntimeError(
            "The embedding API returned a different number of vectors "
            "than the number of inputs."
        )

    _validate_embedding_dimensions(
        vectors=vectors,
        expected_dimension=settings.embedding_dimension,
    )

    return vectors


def _validate_embedding_dimensions(
    vectors: Sequence[Sequence[float]],
    expected_dimension: int | None,
) -> None:
    """Ensure all returned vectors have one consistent dimension."""
    if not vectors:
        raise ValueError("No vectors were provided for validation.")

    actual_dimension = len(vectors[0])

    if actual_dimension == 0:
        raise RuntimeError("The embedding API returned a zero-length vector.")

    for vector in vectors:
        if len(vector) != actual_dimension:
            raise RuntimeError(
                "The embedding API returned inconsistent vector dimensions."
            )

    if expected_dimension is not None and actual_dimension != expected_dimension:
        raise RuntimeError(
            f"Expected embedding dimension {expected_dimension}, "
            f"but received {actual_dimension}."
        )
