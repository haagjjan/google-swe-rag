"""Tests for vector persistence and cosine retrieval."""

from pathlib import Path

import pytest

from swe_google_rag.schemas import DocumentChunk
from swe_google_rag.vector_store import load_index, save_index, search_index


def _chunk(identifier: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        text=f"text for {identifier}",
        source_filename="chapter.pdf",
        page_number=1,
        metadata={"chunk_index": 0},
    )


def test_search_ranks_by_cosine_similarity() -> None:
    chunks = [_chunk("x"), _chunk("y"), _chunk("diagonal")]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    results = search_index(chunks, embeddings, [1.0, 0.0], top_k=2)

    assert [result.chunk.chunk_id for result in results] == ["x", "diagonal"]
    assert results[0].similarity_score == pytest.approx(1.0)
    assert results[1].similarity_score == pytest.approx(2**-0.5)


def test_search_ties_keep_index_order_and_caps_top_k() -> None:
    chunks = [_chunk("first"), _chunk("second")]

    results = search_index(
        chunks,
        [[1.0, 0.0], [1.0, 0.0]],
        [1.0, 0.0],
        top_k=10,
    )

    assert [result.chunk.chunk_id for result in results] == [
        "first",
        "second",
    ]


def test_index_round_trip_preserves_vectors_and_metadata(
    tmp_path: Path,
) -> None:
    chunks = [_chunk("first"), _chunk("second")]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    save_index(
        storage_path=tmp_path,
        chunks=chunks,
        embeddings=embeddings,
        embedding_model="embedding-test",
        embedding_dimension=2,
    )
    loaded_chunks, loaded_vectors, model, dimension = load_index(tmp_path)

    assert loaded_chunks == chunks
    assert loaded_vectors == embeddings
    assert model == "embedding-test"
    assert dimension == 2


def test_missing_index_explains_that_indexing_is_required(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Run 'index' first"):
        load_index(tmp_path)


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ([1.0], "does not match"),
        ([0.0, 0.0], "zero vector"),
        ([float("nan"), 0.0], "invalid numeric"),
    ],
)
def test_invalid_queries_fail(query: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        search_index([_chunk("one")], [[1.0, 0.0]], query, top_k=1)


def test_duplicate_chunk_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        save_index(
            tmp_path,
            [_chunk("same"), _chunk("same")],
            [[1.0, 0.0], [0.0, 1.0]],
            "embedding-test",
            2,
        )
