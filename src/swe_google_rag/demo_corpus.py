"""Generate deterministic, text-extractable PDFs from the public Markdown corpus."""

import re
import textwrap
from pathlib import Path

_PAGE_BREAK = "<!-- PAGE BREAK -->"


def build_demo_corpus(source_path: Path, output_path: Path) -> list[Path]:
    """Generate one deterministic PDF per Markdown source document."""
    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not source_path.is_dir():
        raise NotADirectoryError(f"Demo corpus source does not exist: {source_path}")

    markdown_files = sorted(source_path.glob("*.md"), key=lambda path: path.name)
    if not markdown_files:
        raise FileNotFoundError(f"No Markdown corpus files found in: {source_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for markdown_path in markdown_files:
        pages = _markdown_pages(markdown_path.read_text(encoding="utf-8"))
        pdf_path = output_path / f"{markdown_path.stem}.pdf"
        pdf_path.write_bytes(_build_pdf(pages))
        generated.append(pdf_path)
    return generated


def default_corpus_paths() -> tuple[Path, Path]:
    """Return repository-relative source and generated corpus locations."""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "eval/corpus/source", project_root / "data/demo_pdfs"


def _markdown_pages(markdown: str) -> list[list[str]]:
    pages: list[list[str]] = []
    for raw_page in markdown.split(_PAGE_BREAK):
        lines: list[str] = []
        for raw_line in raw_page.strip().splitlines():
            line = re.sub(r"^#{1,6}\s+", "", raw_line.strip())
            line = re.sub(r"^[-*]\s+", "- ", line)
            line = re.sub(r"[`*_]", "", line)
            if not line:
                lines.append("")
                continue
            lines.extend(textwrap.wrap(line, width=82, break_long_words=False))
        if not lines:
            lines = [" "]
        pages.append(lines[:48])
    return pages


def _build_pdf(pages: list[list[str]]) -> bytes:
    objects: dict[int, bytes] = {}
    page_object_ids = [4 + index * 2 for index in range(len(pages))]
    kids = " ".join(f"{object_id} 0 R" for object_id in page_object_ids)

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>".encode()
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for index, lines in enumerate(pages):
        page_object_id = 4 + index * 2
        content_object_id = page_object_id + 1
        commands = ["BT", "/F1 10 Tf", "72 748 Td", "14 TL"]
        for line in lines:
            escaped = (
                line.encode("ascii", errors="replace")
                .decode("ascii")
                .replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
            )
            commands.extend((f"({escaped}) Tj", "T*"))
        commands.append("ET")
        stream = ("\n".join(commands) + "\n").encode("ascii")
        objects[page_object_id] = (
            "<< /Type /Page /Parent 2 0 R "
            "/MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_object_id} 0 R >>"
        ).encode("ascii")
        objects[content_object_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"endstream"
        )

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id in range(1, max(objects) + 1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode("ascii"))
        output.extend(objects[object_id])
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)
