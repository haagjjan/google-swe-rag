"""Discover PDF inputs and extract page-aware text from each document."""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .schemas import ExtractedPage


def discover_pdf_files(storage_path: Path) -> list[Path]:
    """Return recursively discovered PDF files in deterministic order."""
    storage_path = storage_path.expanduser().resolve()

    if not storage_path.exists():
        raise FileNotFoundError(f"PDF storage directory does not exist: {storage_path}")

    if not storage_path.is_dir():
        raise NotADirectoryError(f"PDF storage path is not a directory: {storage_path}")

    pdf_files = [
        path
        for path in storage_path.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".pdf"
    ]

    return sorted(
        pdf_files,
        key=lambda path: path.relative_to(storage_path).as_posix().casefold(),
    )


def extract_pdf_pages(pdf_path: Path) -> list[ExtractedPage]:
    """Extract text and page provenance from every page of one PDF.

    Empty pages are retained with empty text. OCR and section inference are
    deliberately outside this small pipeline.
    """
    pdf_path = pdf_path.expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"PDF path is not a file: {pdf_path}")

    if pdf_path.suffix.casefold() != ".pdf":
        raise ValueError(f"File does not have a .pdf extension: {pdf_path}")

    extracted_pages: list[ExtractedPage] = []

    try:
        with pdf_path.open("rb") as pdf_file:
            reader = PdfReader(pdf_file, strict=False)

            if reader.is_encrypted:
                raise ValueError(f"Encrypted PDFs are not supported: {pdf_path.name}")

            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    raw_text = page.extract_text()
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to extract text from "
                        f"{pdf_path.name}, page {page_number}."
                    ) from exc

                extracted_pages.append(
                    ExtractedPage(
                        source_path=pdf_path,
                        page_number=page_number,
                        text=_normalize_extracted_text(raw_text),
                    )
                )
    except PdfReadError as exc:
        raise PdfReadError(f"Could not read PDF file: {pdf_path}") from exc

    return extracted_pages


def _normalize_extracted_text(text: str | None) -> str:
    """Apply minimal normalization without destroying page structure."""
    if text is None:
        return ""

    return text.replace("\r\n", "\n").replace("\r", "\n").strip()
