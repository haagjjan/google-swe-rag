"""Tests for deterministic chunking and provenance preservation."""

from pathlib import Path

import pytest

from swe_google_rag.chunking import chunk_pages
from swe_google_rag.schemas import ExtractedPage


def _page(text: str, page_number: int = 1) -> ExtractedPage:
    return ExtractedPage(
        source_path=Path("/documents/chapter.pdf"),
        page_number=page_number,
        text=text,
        section="Testing",
    )


def test_chunks_use_overlap_and_keep_original_text() -> None:
    chunks = chunk_pages(
        [_page("one two three four five six")],
        chunk_size_tokens=4,
        chunk_overlap_tokens=2,
    )

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "three four five six",
    ]
    assert [chunk.metadata["token_count"] for chunk in chunks] == [4, 4]
    assert chunks[1].metadata["token_start"] == 2


def test_punctuation_counts_as_lexical_tokens() -> None:
    chunks = chunk_pages(
        [_page("Hello, world!")],
        chunk_size_tokens=3,
        chunk_overlap_tokens=1,
    )

    assert [chunk.text for chunk in chunks] == ["Hello, world", "world!"]


def test_chunk_ids_are_stable_and_provenance_is_preserved() -> None:
    page = _page("Stable chunk content.", page_number=7)

    first = chunk_pages([page], 20, 2)
    second = chunk_pages([page], 20, 2)

    assert first == second
    assert first[0].source_filename == "chapter.pdf"
    assert first[0].page_number == 7
    assert first[0].section == "Testing"
    assert first[0].metadata["source_path"] == "/documents/chapter.pdf"


def test_empty_pages_are_ignored() -> None:
    assert chunk_pages([_page(" \n\t ")], 10, 2) == []


@pytest.mark.parametrize(
    ("size", "overlap", "message"),
    [
        (0, 0, "greater than zero"),
        (10, -1, "must not be negative"),
        (10, 10, "must be smaller"),
        (10, 11, "must be smaller"),
    ],
)
def test_invalid_chunk_settings_fail(
    size: int,
    overlap: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        chunk_pages([_page("text")], size, overlap)
