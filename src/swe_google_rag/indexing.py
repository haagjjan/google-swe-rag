"""Orchestrate PDF discovery, extraction, chunking, embedding, and persistence."""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .chunking import chunk_pages
from .config import Settings
from .documents import discover_pdf_files, extract_pdf_pages
from .embeddings import embed_document_chunks, embedding_request_count
from .schemas import ExtractedPage, IndexBuildResult
from .vector_store import save_index


def build_document_index(settings: Settings) -> IndexBuildResult:
    """Build and persist a local vector index from configured PDF inputs."""
    started_at = perf_counter()
    pdf_files = discover_pdf_files(settings.pdf_storage_path)

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {settings.pdf_storage_path}")

    pages: list[ExtractedPage] = []

    for pdf_file in pdf_files:
        extracted_pages = extract_pdf_pages(pdf_file)
        pages.extend(extracted_pages)

    chunks = chunk_pages(
        pages=pages,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )

    if not chunks:
        raise ValueError(
            "No text chunks were created. The PDFs may contain no extractable text."
        )

    embeddings = embed_document_chunks(
        chunks=chunks,
        settings=settings,
    )

    if not embeddings:
        raise RuntimeError("No embeddings were returned.")

    actual_embedding_dimension = len(embeddings[0])

    save_index(
        storage_path=settings.vector_store_path,
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=settings.embedding_model,
        embedding_dimension=actual_embedding_dimension,
    )

    total_latency_ms = (perf_counter() - started_at) * 1_000
    index_path = settings.vector_store_path.resolve() / "index.npz"
    manifest_path = settings.vector_store_path.resolve() / "index-build.json"
    model_calls = embedding_request_count(len(chunks))
    _write_build_manifest(
        path=manifest_path,
        settings=settings,
        pdf_files=pdf_files,
        pdf_count=len(pdf_files),
        page_count=len(pages),
        chunk_count=len(chunks),
        embedding_dimension=actual_embedding_dimension,
        embedding_model_calls=model_calls,
        total_latency_ms=total_latency_ms,
    )

    return IndexBuildResult(
        pdf_count=len(pdf_files),
        page_count=len(pages),
        chunk_count=len(chunks),
        embedding_dimension=actual_embedding_dimension,
        index_path=index_path,
        manifest_path=manifest_path,
        total_latency_ms=total_latency_ms,
        embedding_model_calls=model_calls,
    )


def _write_build_manifest(
    *,
    path: Path,
    settings: Settings,
    pdf_files: list[Path],
    pdf_count: int,
    page_count: int,
    chunk_count: int,
    embedding_dimension: int,
    embedding_model_calls: int,
    total_latency_ms: float,
) -> None:
    source_evidence = [
        {
            "filename": source.name,
            "sha256": _sha256_file(source),
            "size_bytes": source.stat().st_size,
        }
        for source in pdf_files
    ]
    stable_configuration = {
        "embedding_model": settings.embedding_model,
        "embedding_dimension": embedding_dimension,
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "sources": source_evidence,
    }
    configuration_hash = hashlib.sha256(
        json.dumps(
            stable_configuration, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "configuration_hash": configuration_hash,
        "configuration": stable_configuration,
        "pdf_count": pdf_count,
        "page_count": page_count,
        "chunk_count": chunk_count,
        "embedding_model_calls": embedding_model_calls,
        "total_latency_ms": round(total_latency_ms, 3),
    }
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
