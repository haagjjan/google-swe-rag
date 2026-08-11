"""Orchestrate PDF discovery, extraction, chunking, embedding, and persistence."""

from .chunking import chunk_pages
from .config import Settings
from .documents import discover_pdf_files, extract_pdf_pages
from .embeddings import embed_document_chunks
from .schemas import ExtractedPage, IndexBuildResult
from .vector_store import save_index


def build_document_index(settings: Settings) -> IndexBuildResult:
    """Build and persist a local vector index from configured PDF inputs."""
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

    return IndexBuildResult(
        pdf_count=len(pdf_files),
        page_count=len(pages),
        chunk_count=len(chunks),
        embedding_dimension=actual_embedding_dimension,
        index_path=settings.vector_store_path.resolve() / "index.npz",
    )
