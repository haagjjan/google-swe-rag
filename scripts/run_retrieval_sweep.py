"""Run one-factor-at-a-time retrieval sweeps against the public benchmark."""

import argparse
from dataclasses import replace
from pathlib import Path

from swe_google_rag.config import load_settings
from swe_google_rag.evaluation import run_evaluation
from swe_google_rag.indexing import build_document_index
from swe_google_rag.schemas import ExperimentConfig

_PROFILES = (
    ("baseline", 400, 60, 768, 4),
    ("chunk-256", 256, 60, 768, 4),
    ("chunk-640", 640, 60, 768, 4),
    ("overlap-0", 400, 0, 768, 4),
    ("overlap-120", 400, 120, 768, 4),
    ("dimension-384", 400, 60, 384, 4),
    ("dimension-1536", 400, 60, 1536, 4),
    ("top-k-1", 400, 60, 768, 1),
    ("top-k-5", 400, 60, 768, 5),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and evaluate controlled retrieval configurations."
    )
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument(
        "--dataset", type=Path, default=Path("eval/datasets/public-v1.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("eval/results/retrieval-sweep")
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=[profile[0] for profile in _PROFILES],
    )
    args = parser.parse_args()
    selected = {profile[0]: profile for profile in _PROFILES}
    unknown = sorted(set(args.profiles) - set(selected))
    if unknown:
        parser.error(f"Unknown profiles: {', '.join(unknown)}")

    base_settings = load_settings(args.env_file)
    output_root = args.output.expanduser().resolve()
    for profile_name in args.profiles:
        _, chunk_size, overlap, dimension, top_k = selected[profile_name]
        profile_root = output_root / profile_name
        settings = replace(
            base_settings,
            vector_store_path=profile_root / "index",
            chunk_size_tokens=chunk_size,
            chunk_overlap_tokens=overlap,
            embedding_dimension=dimension,
            top_k=top_k,
        )
        print(f"Building {profile_name}...")
        build_document_index(settings)
        run_evaluation(
            settings=settings,
            dataset_path=args.dataset,
            config=ExperimentConfig(
                name=f"retrieval-sweep-{profile_name}",
                version="1.0.0",
                retrieval_methods=("dense", "bm25", "hybrid"),
                generation_models=(),
                generation_modes=(),
                generation_retrieval_method="hybrid",
                top_k=top_k,
            ),
            output_path=profile_root / "runs",
        )


if __name__ == "__main__":
    main()
