"""Tests for PDF discovery and page extraction."""

from pathlib import Path

import pytest
from pypdf import PdfWriter

from swe_google_rag.documents import discover_pdf_files, extract_pdf_pages


def test_discovery_is_recursive_case_insensitive_and_sorted(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "b.PDF").touch()
    (nested / "A.pdf").touch()
    (tmp_path / "notes.txt").touch()

    paths = discover_pdf_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
        "b.PDF",
        "nested/A.pdf",
    ]


def test_extract_blank_pdf_preserves_page_provenance(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with pdf_path.open("wb") as pdf_file:
        writer.write(pdf_file)

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 1
    assert pages[0].source_path == pdf_path.resolve()
    assert pages[0].page_number == 1
    assert pages[0].text == ""


def test_missing_pdf_storage_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_pdf_files(tmp_path / "missing")


def test_non_pdf_input_is_rejected(tmp_path: Path) -> None:
    text_path = tmp_path / "document.txt"
    text_path.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.pdf"):
        extract_pdf_pages(text_path)
