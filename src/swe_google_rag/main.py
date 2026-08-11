"""Command-line entry point for indexing PDFs and asking RAG questions."""

import argparse
from pathlib import Path

from .config import load_settings
from .indexing import build_document_index
from .rag import answer_question


def main() -> None:
    """Parse CLI arguments and run the selected indexing or query operation."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        _run_command(args)
    # The CLI is the user-facing boundary; provider and file errors should be
    # concise here while library callers still receive typed exceptions.
    except Exception as exc:  # noqa: BLE001
        parser.exit(status=1, message=f"Error: {exc}\n")


def _run_command(args: argparse.Namespace) -> None:
    """Run one parsed command."""
    settings = load_settings(env_file=args.env_file)

    if args.command == "index":
        result = build_document_index(settings)
        print(
            f"Indexed {result.pdf_count} PDF(s), "
            f"{result.page_count} page(s), and "
            f"{result.chunk_count} chunk(s)."
        )
        print(
            f"Embedding dimension: {result.embedding_dimension}. "
            f"Index: {result.index_path}"
        )
        return

    if args.command == "ask":
        question = " ".join(args.question)

        answer = answer_question(
            question=question,
            settings=settings,
        )

        print("\nAnswer:\n")
        print(answer.text)

        print("\nRetrieved sources:\n")

        for source_number, result in enumerate(answer.sources, start=1):
            chunk = result.chunk

            page = chunk.page_number if chunk.page_number is not None else "unknown"

            print(
                f"[Source {source_number}] "
                f"{chunk.source_filename}, "
                f"page {page}, "
                f"score={result.similarity_score:.4f}"
            )

        return

    raise RuntimeError(f"Unsupported command: {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="RAG over Software Engineering at Google PDFs."
    )

    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional path to the environment file.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "index",
        help="Build the vector index from configured PDF files.",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question using the stored vector index.",
    )

    ask_parser.add_argument(
        "question",
        nargs="+",
        help="Question to answer from the indexed documents.",
    )

    return parser


if __name__ == "__main__":
    main()
