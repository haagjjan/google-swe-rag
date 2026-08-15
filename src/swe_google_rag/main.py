"""Command-line entry point for indexing PDFs and asking RAG questions."""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from .config import load_settings
from .demo_corpus import build_demo_corpus, default_corpus_paths
from .evaluation import (
    experiment_config_from_mapping,
    load_experiment_config,
    run_evaluation,
)
from .indexing import build_document_index
from .judge import run_blinded_judge
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
    if args.command == "demo-corpus":
        source_path, output_path = default_corpus_paths()
        generated = build_demo_corpus(
            source_path=args.source or source_path,
            output_path=args.output or output_path,
        )
        print(f"Generated {len(generated)} deterministic demo PDF(s):")
        for path in generated:
            print(path)
        return

    settings = load_settings(env_file=args.env_file)

    if args.command == "index":
        build_result = build_document_index(settings)
        print(
            f"Indexed {build_result.pdf_count} PDF(s), "
            f"{build_result.page_count} page(s), and "
            f"{build_result.chunk_count} chunk(s)."
        )
        print(
            f"Embedding dimension: {build_result.embedding_dimension}. "
            f"Index: {build_result.index_path}"
        )
        print(
            f"Indexing latency: {build_result.total_latency_ms:.1f} ms. "
            f"Embedding calls: {build_result.embedding_model_calls}."
        )
        print(f"Build manifest: {build_result.manifest_path}")
        return

    if args.command == "evaluate":
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        config = load_experiment_config(args.config)
        run_directory = run_evaluation(
            settings=settings,
            dataset_path=args.dataset,
            config=config,
            output_path=args.output,
        )
        print(f"Evaluation complete: {run_directory}")
        print(f"Report: {run_directory / 'REPORT.md'}")
        return

    if args.command == "compare":
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        config = experiment_config_from_mapping(
            {
                "name": "cli-model-comparison",
                "version": "1.0.0",
                "retrieval_methods": [args.retrieval_method],
                "generation_models": args.models,
                "generation_modes": ["direct", "rag", "oracle"],
                "generation_retrieval_method": args.retrieval_method,
                "top_k": args.top_k or settings.top_k,
                "rrf_k": args.rrf_k,
                "max_cases": args.max_cases,
                "pricing_usd_per_million_tokens": {},
            }
        )
        run_directory = run_evaluation(
            settings=settings,
            dataset_path=args.dataset,
            config=config,
            output_path=args.output,
        )
        print(f"Comparison complete: {run_directory / 'REPORT.md'}")
        return

    if args.command == "judge":
        judged_path, audit_path = run_blinded_judge(
            settings=settings,
            report_path=args.report,
            judge_model=args.judge_model,
            output_path=args.output,
            audit_size=args.audit_size,
        )
        print(f"Blinded judgments: {judged_path}")
        print(f"Manual audit sample: {audit_path}")
        return

    if args.command == "serve":
        import uvicorn

        from .web import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)
        return

    if args.command == "demo":
        import uvicorn

        from .web import create_app

        source_path, pdf_path = default_corpus_paths()
        build_demo_corpus(source_path, pdf_path)
        demo_settings = replace(
            settings,
            pdf_storage_path=pdf_path,
            vector_store_path=pdf_path.parent / "demo_vector_store",
        )
        index_path = demo_settings.vector_store_path / "index.npz"
        if args.rebuild or not index_path.is_file():
            build_document_index(demo_settings)
        uvicorn.run(
            create_app(settings_loader=lambda: demo_settings),
            host=args.host,
            port=args.port,
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

        for source_number, retrieved_source in enumerate(answer.sources, start=1):
            chunk = retrieved_source.chunk

            page = chunk.page_number if chunk.page_number is not None else "unknown"

            print(
                f"[Source {source_number}] "
                f"{chunk.source_filename}, "
                f"page {page}, "
                f"score={retrieved_source.similarity_score:.4f}"
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

    corpus_parser = subparsers.add_parser(
        "demo-corpus",
        help="Regenerate the redistributable demo PDFs from Markdown.",
    )
    corpus_parser.add_argument("--source", type=Path, default=None)
    corpus_parser.add_argument("--output", type=Path, default=None)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run a versioned retrieval and generation evaluation.",
    )
    evaluate_parser.add_argument("--dataset", type=Path, required=True)
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare direct, retrieved, and oracle context across models.",
    )
    compare_parser.add_argument("--dataset", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Google generation model IDs to compare.",
    )
    compare_parser.add_argument(
        "--retrieval-method",
        choices=("dense", "bm25", "hybrid"),
        default="hybrid",
    )
    compare_parser.add_argument("--top-k", type=int, default=None)
    compare_parser.add_argument("--rrf-k", type=int, default=60)
    compare_parser.add_argument("--max-cases", type=int, default=None)

    judge_parser = subparsers.add_parser(
        "judge",
        help="Supplement deterministic metrics with a blinded model judge.",
    )
    judge_parser.add_argument("--report", type=Path, required=True)
    judge_parser.add_argument("--judge-model", required=True)
    judge_parser.add_argument("--output", type=Path, required=True)
    judge_parser.add_argument("--audit-size", type=int, default=12)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the local FastAPI comparison demo.",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    demo_parser = subparsers.add_parser(
        "demo",
        help="Build the public demo index if needed and run the local UI.",
    )
    demo_parser.add_argument("--host", default="127.0.0.1")
    demo_parser.add_argument("--port", type=int, default=8000)
    demo_parser.add_argument("--rebuild", action="store_true")

    return parser


if __name__ == "__main__":
    main()
