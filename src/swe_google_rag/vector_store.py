"""Persist chunk vectors locally and perform cosine-similarity retrieval."""

import json
import os
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import numpy as np

from .schemas import DocumentChunk, RetrievedChunk

_INDEX_FILENAME = "index.npz"
_INDEX_FORMAT_VERSION = 1


def save_index(
    storage_path: Path,
    chunks: Sequence[DocumentChunk],
    embeddings: Sequence[Sequence[float]],
    embedding_model: str,
    embedding_dimension: int,
) -> None:
    """Persist vectors, chunk metadata, and embedding configuration atomically."""
    embedding_model = embedding_model.strip()
    if not embedding_model:
        raise ValueError("embedding_model must not be empty.")

    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be greater than zero.")

    matrix = _create_embedding_matrix(embeddings)

    if len(chunks) != matrix.shape[0]:
        raise ValueError(
            f"Received {len(chunks)} chunks but {matrix.shape[0]} embeddings."
        )

    if matrix.shape[1] != embedding_dimension:
        raise ValueError(
            f"Expected dimension {embedding_dimension}, "
            f"but embeddings have dimension {matrix.shape[1]}."
        )

    _validate_unique_chunk_ids(chunks)

    payload = {
        "format_version": _INDEX_FORMAT_VERSION,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_filename": chunk.source_filename,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "metadata": dict(chunk.metadata),
            }
            for chunk in chunks
        ],
    }

    metadata_bytes = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")

    storage_path = storage_path.expanduser().resolve()
    storage_path.mkdir(parents=True, exist_ok=True)

    index_path = storage_path / _INDEX_FILENAME
    temporary_path = storage_path / (f".{_INDEX_FILENAME}.{uuid4().hex}.tmp")

    try:
        with temporary_path.open("wb") as file:
            np.savez_compressed(
                file,
                embeddings=matrix,
                metadata=np.frombuffer(metadata_bytes, dtype=np.uint8),
            )
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, index_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_index(
    storage_path: Path,
) -> tuple[list[DocumentChunk], list[list[float]], str, int]:
    """Load chunks, vectors, model ID, and vector dimension from local storage."""
    index_path = storage_path.expanduser().resolve() / _INDEX_FILENAME

    if not index_path.exists():
        raise FileNotFoundError(
            f"Vector index does not exist: {index_path}. Run 'index' first."
        )

    try:
        with np.load(index_path, allow_pickle=False) as index:
            if "embeddings" not in index or "metadata" not in index:
                raise ValueError("Index is missing required data.")

            matrix = np.asarray(index["embeddings"], dtype=np.float32)
            metadata_bytes = np.asarray(
                index["metadata"],
                dtype=np.uint8,
            ).tobytes()

        payload = json.loads(metadata_bytes.decode("utf-8"))
    except (
        OSError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(f"Could not load vector index: {index_path}") from exc

    if not isinstance(payload, dict):
        raise TypeError(f"Vector index metadata is invalid: {index_path}")

    if payload.get("format_version") != _INDEX_FORMAT_VERSION:
        raise ValueError("The stored vector index uses an unsupported format version.")

    try:
        embedding_model = payload["embedding_model"]
        embedding_dimension = payload["embedding_dimension"]
        saved_chunk_count = payload["chunk_count"]
        raw_chunks = payload["chunks"]

        if not isinstance(embedding_model, str):
            raise TypeError("Invalid embedding model metadata.")
        if not embedding_model.strip():
            raise ValueError("Invalid embedding model metadata.")
        if not isinstance(embedding_dimension, int) or isinstance(
            embedding_dimension,
            bool,
        ):
            raise TypeError("Invalid embedding dimension metadata.")
        if embedding_dimension <= 0:
            raise ValueError("Invalid embedding dimension metadata.")
        if not isinstance(saved_chunk_count, int) or isinstance(
            saved_chunk_count,
            bool,
        ):
            raise TypeError("Invalid chunk count metadata.")
        if not isinstance(raw_chunks, list):
            raise TypeError("Invalid chunk metadata.")

        chunks = [_deserialize_chunk(item) for item in raw_chunks]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Vector index metadata is invalid: {index_path}") from exc

    if saved_chunk_count != len(chunks):
        raise ValueError("Stored chunk count does not match the saved chunk metadata.")

    if matrix.ndim != 2:
        raise ValueError("Stored embeddings are not a two-dimensional matrix.")

    if matrix.shape != (len(chunks), embedding_dimension):
        raise ValueError("Stored embeddings do not match the saved chunk metadata.")

    if not np.isfinite(matrix).all():
        raise ValueError("Stored embeddings contain invalid numeric values.")

    _validate_unique_chunk_ids(chunks)

    return (
        chunks,
        matrix.tolist(),
        embedding_model,
        embedding_dimension,
    )


def search_index(
    chunks: Sequence[DocumentChunk],
    embeddings: Sequence[Sequence[float]],
    query_vector: Sequence[float],
    top_k: int,
) -> list[RetrievedChunk]:
    """Return the highest-similarity chunks for one query vector."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not chunks:
        return []

    matrix = _create_embedding_matrix(embeddings)

    if len(chunks) != matrix.shape[0]:
        raise ValueError(
            "The number of chunks does not match the number of embeddings."
        )

    query = np.asarray(query_vector, dtype=np.float32)

    if query.ndim != 1:
        raise ValueError("query_vector must be one-dimensional.")

    if query.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Query dimension {query.shape[0]} does not match "
            f"index dimension {matrix.shape[1]}."
        )

    if not np.isfinite(query).all():
        raise ValueError("query_vector contains invalid numeric values.")

    document_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)

    if query_norm == 0:
        raise ValueError("query_vector must not be a zero vector.")

    if np.any(document_norms == 0):
        raise ValueError("The index contains a zero-length embedding vector.")

    similarity_scores = (matrix @ query) / (document_norms * query_norm)
    similarity_scores = np.clip(similarity_scores, -1.0, 1.0)

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: (-float(similarity_scores[index]), index),
    )
    selected_indices = ranked_indices[: min(top_k, len(chunks))]

    return [
        RetrievedChunk(
            chunk=chunks[index],
            similarity_score=float(similarity_scores[index]),
        )
        for index in selected_indices
    ]


def _deserialize_chunk(item: object) -> DocumentChunk:
    """Validate and deserialize one chunk metadata object."""
    if not isinstance(item, dict):
        raise TypeError("Chunk metadata entry is not an object.")

    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("Chunk metadata field is not an object.")

    chunk_id = item["chunk_id"]
    text = item["text"]
    source_filename = item["source_filename"]
    page_number = item["page_number"]
    section = item.get("section")

    if not isinstance(chunk_id, str) or not chunk_id:
        raise TypeError("Chunk ID is invalid.")
    if not isinstance(text, str):
        raise TypeError("Chunk text is invalid.")
    if not isinstance(source_filename, str) or not source_filename:
        raise TypeError("Chunk source filename is invalid.")
    if page_number is not None and (
        not isinstance(page_number, int) or isinstance(page_number, bool)
    ):
        raise TypeError("Chunk page number is invalid.")
    if section is not None and not isinstance(section, str):
        raise TypeError("Chunk section is invalid.")

    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        source_filename=source_filename,
        page_number=page_number,
        section=section,
        metadata=metadata,
    )


def _create_embedding_matrix(
    embeddings: Sequence[Sequence[float]],
) -> np.ndarray:
    """Convert embeddings into a validated two-dimensional NumPy matrix."""
    if not embeddings:
        raise ValueError("At least one embedding is required.")

    try:
        matrix = np.asarray(embeddings, dtype=np.float32)
    except ValueError as exc:
        raise ValueError("Embeddings must all have the same dimension.") from exc

    if matrix.ndim != 2:
        raise ValueError("Embeddings must form a two-dimensional matrix.")

    if matrix.shape[1] == 0:
        raise ValueError("Embedding vectors must not be empty.")

    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain invalid numeric values.")

    return matrix


def _validate_unique_chunk_ids(
    chunks: Sequence[DocumentChunk],
) -> None:
    """Ensure every stored chunk has a unique identifier."""
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Chunk IDs must be unique.")
