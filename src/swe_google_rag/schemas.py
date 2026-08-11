"""Define the shared page, chunk, retrieval, and answer data structures."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

MetadataValue: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text and provenance extracted from one source PDF page."""

    source_path: Path
    page_number: int | None
    text: str
    section: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A lexical-token-bounded text chunk with source metadata."""

    chunk_id: str
    text: str
    source_filename: str
    page_number: int | None
    section: str | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A document chunk returned by similarity search."""

    chunk: DocumentChunk
    similarity_score: float


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """A grounded answer and the chunks used as its sources."""

    text: str
    sources: tuple[RetrievedChunk, ...]


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    """Summary of a successfully persisted document index."""

    pdf_count: int
    page_count: int
    chunk_count: int
    embedding_dimension: int
    index_path: Path
