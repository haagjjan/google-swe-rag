"""Split extracted PDF text into deterministic overlapping chunks."""

import re
from collections.abc import Sequence
from hashlib import sha256

from .schemas import DocumentChunk, ExtractedPage

# Google does not publish a local tokenizer for gemini-embedding-001. This
# lightweight lexical tokenizer keeps chunking deterministic and offline while
# remaining conservative relative to the embedding model's input limit.
_TOKEN_PATTERN = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", re.UNICODE)


def chunk_pages(
    pages: Sequence[ExtractedPage],
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[DocumentChunk]:
    """Create deterministic lexical-token chunks from extracted PDF pages.

    Each page is chunked independently so every chunk retains an exact page
    number. Empty pages are ignored. The text of each chunk is sliced from the
    original page text, preserving punctuation and whitespace inside the span.
    """
    _validate_chunk_settings(
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )

    chunks: list[DocumentChunk] = []

    for page in pages:
        token_spans = [match.span() for match in _TOKEN_PATTERN.finditer(page.text)]
        if not token_spans:
            continue

        start = 0
        chunk_index = 0

        while start < len(token_spans):
            end = min(start + chunk_size_tokens, len(token_spans))
            character_start = token_spans[start][0]
            character_end = token_spans[end - 1][1]
            chunk_text = page.text[character_start:character_end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=_create_chunk_id(
                            page=page,
                            chunk_index=chunk_index,
                            chunk_text=chunk_text,
                        ),
                        text=chunk_text,
                        source_filename=page.source_path.name,
                        page_number=page.page_number,
                        section=page.section,
                        metadata={
                            "source_path": page.source_path.as_posix(),
                            "chunk_index": chunk_index,
                            "token_start": start,
                            "token_end": end,
                            "token_count": end - start,
                        },
                    )
                )

            if end == len(token_spans):
                break

            start = end - chunk_overlap_tokens
            chunk_index += 1

    return chunks


def _validate_chunk_settings(
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> None:
    """Validate chunk size and overlap values."""
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be greater than zero.")

    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens must not be negative.")

    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens.")


def _create_chunk_id(
    page: ExtractedPage,
    chunk_index: int,
    chunk_text: str,
) -> str:
    """Create a deterministic identifier for one document chunk."""
    identity = (
        f"{page.source_path.as_posix()}\n"
        f"{page.page_number}\n"
        f"{chunk_index}\n"
        f"{chunk_text}"
    )

    digest = sha256(identity.encode("utf-8")).hexdigest()[:16]
    page_label = page.page_number if page.page_number is not None else "unknown"

    return f"{page.source_path.stem}-p{page_label}-c{chunk_index:04d}-{digest}"
