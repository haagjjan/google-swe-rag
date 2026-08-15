"""Run reproducible retrieval and generation experiments and persist evidence."""

import csv
import hashlib
import json
import logging
import math
import os
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any
from uuid import uuid4

from .config import Settings
from .evaluation_dataset import load_evaluation_dataset, validate_dataset_sources
from .generation import format_retrieved_context, generate_answer_with_metadata
from .metrics import (
    aggregate_generation_metrics,
    aggregate_retrieval_metrics,
    score_generation_case,
    score_retrieval_case,
)
from .observability import log_event
from .retrieval import retrieve_chunks
from .schemas import (
    DocumentChunk,
    EvaluationCase,
    EvaluationDataset,
    ExperimentConfig,
    RetrievedChunk,
)
from .vector_store import load_index

_LOGGER = logging.getLogger(__name__)
_RETRIEVAL_METHODS = frozenset({"dense", "bm25", "hybrid"})
_GENERATION_MODES = frozenset({"direct", "rag", "oracle"})


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load a versioned experiment configuration from JSON."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Experiment configuration does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read experiment configuration: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError("Experiment configuration root must be an object.")
    return experiment_config_from_mapping(payload)


def experiment_config_from_mapping(payload: Mapping[str, object]) -> ExperimentConfig:
    """Validate controls shared by CLI, tests, and programmatic callers."""
    name = _required_string(payload, "name")
    version = _required_string(payload, "version")
    retrieval_methods = _string_tuple(payload, "retrieval_methods")
    models = _string_tuple(payload, "generation_models", allow_empty=True)
    modes = _string_tuple(payload, "generation_modes", allow_empty=True)
    generation_retrieval_method = _required_string(
        payload, "generation_retrieval_method"
    )
    top_k = _positive_int(payload, "top_k")
    rrf_k = _positive_int(payload, "rrf_k", default=60)

    unknown_retrieval = set(retrieval_methods) - _RETRIEVAL_METHODS
    if unknown_retrieval:
        raise ValueError(f"Unsupported retrieval methods: {sorted(unknown_retrieval)}")
    unknown_modes = set(modes) - _GENERATION_MODES
    if unknown_modes:
        raise ValueError(f"Unsupported generation modes: {sorted(unknown_modes)}")
    if bool(models) != bool(modes):
        raise ValueError(
            "generation_models and generation_modes must both be set or both empty."
        )
    if generation_retrieval_method not in _RETRIEVAL_METHODS:
        raise ValueError("generation_retrieval_method is unsupported.")

    raw_max_cases = payload.get("max_cases")
    max_cases: int | None
    if raw_max_cases is None:
        max_cases = None
    elif (
        isinstance(raw_max_cases, int)
        and not isinstance(raw_max_cases, bool)
        and raw_max_cases > 0
    ):
        max_cases = raw_max_cases
    else:
        raise ValueError("max_cases must be null or a positive integer.")

    raw_pricing = payload.get("pricing_usd_per_million_tokens", {})
    if not isinstance(raw_pricing, dict):
        raise TypeError("pricing_usd_per_million_tokens must be an object.")
    pricing: dict[str, dict[str, float]] = {}
    for model, rates in raw_pricing.items():
        if not isinstance(model, str) or not isinstance(rates, dict):
            raise TypeError("Pricing entries must map model strings to objects.")
        input_rate = rates.get("input")
        output_rate = rates.get("output")
        if not all(
            isinstance(rate, (int, float))
            and not isinstance(rate, bool)
            and rate >= 0
            for rate in (input_rate, output_rate)
        ):
            raise ValueError(f"Pricing for {model!r} needs non-negative rates.")
        pricing[model] = {
            "input": _nonnegative_float(input_rate, model, "input"),
            "output": _nonnegative_float(output_rate, model, "output"),
        }

    pricing_snapshot_date = _optional_string(payload, "pricing_snapshot_date")
    pricing_source = _optional_string(payload, "pricing_source")

    return ExperimentConfig(
        name=name,
        version=version,
        retrieval_methods=retrieval_methods,
        generation_models=models,
        generation_modes=modes,
        generation_retrieval_method=generation_retrieval_method,
        top_k=top_k,
        rrf_k=rrf_k,
        max_cases=max_cases,
        pricing_snapshot_date=pricing_snapshot_date,
        pricing_source=pricing_source,
        pricing_usd_per_million_tokens=pricing,
    )


def run_evaluation(
    *,
    settings: Settings,
    dataset_path: Path,
    config: ExperimentConfig,
    output_path: Path,
) -> Path:
    """Run the configured benchmark and return its immutable run directory."""
    dataset = load_evaluation_dataset(dataset_path)
    cases = _select_cases(dataset.cases, config.max_cases)
    chunks, _, _, _ = load_index(settings.vector_store_path)
    index_evidence = _read_index_evidence(settings.vector_store_path)
    validate_dataset_sources(
        dataset,
        {
            (chunk.source_filename, chunk.page_number)
            for chunk in chunks
            if chunk.page_number is not None
        },
    )

    config_payload = {
        "experiment": asdict(config),
        "index": {
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "chunk_size_tokens": settings.chunk_size_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "top_k_default": settings.top_k,
            "artifact_sha256": index_evidence["artifact_sha256"],
            "build_configuration_hash": index_evidence[
                "build_configuration_hash"
            ],
            "source_files": index_evidence["source_files"],
        },
    }
    config_hash = _configuration_hash(config_payload, dataset.version)
    created_at = datetime.now(UTC).replace(microsecond=0)
    run_id = f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{config_hash[:10]}"
    output_root = output_path.expanduser().resolve()
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    log_event(
        _LOGGER,
        "evaluation_started",
        run_id=run_id,
        fields={"dataset": dataset.version, "config_hash": config_hash},
    )

    retrieval_cache: dict[tuple[str, str], tuple[list[RetrievedChunk], float]] = {}
    retrieval_trials: list[dict[str, Any]] = []
    for method in config.retrieval_methods:
        for case in cases:
            results, latency_ms = _retrieve_cached(
                cache=retrieval_cache,
                case=case,
                method=method,
                settings=settings,
                top_k=config.top_k,
                rrf_k=config.rrf_k,
            )
            if not case.answerable:
                continue
            trial = {
                "trial_type": "retrieval",
                "case_id": case.case_id,
                "question_type": case.question_type,
                "method": method,
                "latency_ms": round(latency_ms, 3),
                "retrieved_sources": _serialize_sources(results),
                **score_retrieval_case(case, results),
            }
            retrieval_trials.append(trial)

    generation_trials: list[dict[str, Any]] = []
    for case in cases:
        rag_results, retrieval_latency_ms = _retrieve_cached(
            cache=retrieval_cache,
            case=case,
            method=config.generation_retrieval_method,
            settings=settings,
            top_k=config.top_k,
            rrf_k=config.rrf_k,
        )
        oracle_results = _oracle_results(case, chunks)

        for mode in config.generation_modes:
            if mode == "direct":
                context_results = []
            elif mode == "oracle":
                context_results = oracle_results
            else:
                context_results = rag_results
            context = format_retrieved_context(context_results)
            grounded = mode != "direct"

            for model in config.generation_models:
                generation = generate_answer_with_metadata(
                    question=case.question,
                    context=context,
                    settings=settings,
                    model=model,
                    grounded=grounded,
                )
                metric_values = score_generation_case(
                    case,
                    generation.text,
                    context_results,
                    grounded=grounded,
                )
                estimated_cost = _estimate_cost(
                    model=model,
                    input_tokens=generation.usage.input_tokens,
                    output_tokens=generation.usage.output_tokens,
                    pricing=config.pricing_usd_per_million_tokens,
                )
                generation_trials.append(
                    {
                        "trial_type": "generation",
                        "case_id": case.case_id,
                        "question_type": case.question_type,
                        "answerable": case.answerable,
                        "question": case.question,
                        "reference_facts": list(case.reference_facts),
                        "model": model,
                        "mode": mode,
                        "retrieval_method": (
                            None
                            if mode == "direct"
                            else "oracle"
                            if mode == "oracle"
                            else config.generation_retrieval_method
                        ),
                        "answer": generation.text,
                        "retrieved_sources": _serialize_sources(context_results),
                        "retrieval_latency_ms": (
                            0.0 if mode != "rag" else round(retrieval_latency_ms, 3)
                        ),
                        "generation_latency_ms": round(generation.latency_ms, 3),
                        "input_tokens": generation.usage.input_tokens,
                        "output_tokens": generation.usage.output_tokens,
                        "total_tokens": generation.usage.total_tokens,
                        "model_calls": generation.model_calls,
                        "estimated_cost_usd": estimated_cost,
                        **metric_values,
                    }
                )

    summary = _build_summary(
        dataset=dataset,
        cases=cases,
        config=config,
        retrieval_trials=retrieval_trials,
        generation_trials=generation_trials,
        settings=settings,
        index_evidence=index_evidence,
    )
    report = {
        "run_id": run_id,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "config_hash": config_hash,
        "dataset": {
            "name": dataset.name,
            "version": dataset.version,
            "license": dataset.license,
            "path": str(dataset_path.expanduser().resolve()),
        },
        "configuration": config_payload,
        "index_build": index_evidence,
        "retrieval_results": retrieval_trials,
        "generation_results": generation_trials,
        "summary": summary,
    }

    _write_json_atomic(run_directory / "report.json", report)
    _write_json_atomic(run_directory / "summary.json", summary)
    _write_trials_csv(run_directory / "trials.csv", retrieval_trials, generation_trials)
    (run_directory / "REPORT.md").write_text(
        render_markdown_summary(report), encoding="utf-8"
    )
    _update_latest(output_root, run_directory)

    log_event(
        _LOGGER,
        "evaluation_completed",
        run_id=run_id,
        fields={
            "retrieval_trials": len(retrieval_trials),
            "generation_trials": len(generation_trials),
        },
    )
    return run_directory


def render_markdown_summary(report: Mapping[str, Any]) -> str:
    """Render a results-first, repository-friendly benchmark summary."""
    summary = report["summary"]
    dataset = report["dataset"]
    lines = [
        "# Benchmark report",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Dataset: {dataset['name']} `{dataset['version']}`",
        f"- Cases: {summary['case_count']}",
        f"- Configuration hash: `{report['config_hash']}`",
        "",
        "## Retrieval",
        "",
        "| Method | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG | Context precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["retrieval"]:
        lines.append(
            "| {method} | {recall_at_1:.3f} | {recall_at_3:.3f} | "
            "{recall_at_5:.3f} | {mean_reciprocal_rank:.3f} | "
            "{mean_ndcg:.3f} | {mean_context_precision:.3f} |".format(**item)
        )

    lines.extend(
        [
            "",
            "## Generation",
            "",
            "| Model | Mode | Correctness | Fact coverage | Citation precision | "
            "Citation recall | Hallucination proxy | Abstention F1 | Attack success | "
            "P50 latency |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary["generation"]:
        lines.append(
            "| {model} | {mode} | {answer_correctness:.3f} | "
            "{grounded_fact_coverage:.3f} | {citation_precision_display} | "
            "{citation_recall_display} | {hallucination_display} | "
            "{abstention_f1:.3f} | {attack_display} | {latency_p50_ms:.1f} ms |".format(
                **item
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Hallucination is a deterministic lexical unsupported-sentence "
            "proxy, not an entailment guarantee.",
            "- Direct answers have no citation or grounding score because they "
            "receive no corpus evidence.",
            "- Token costs use the pricing snapshot in the experiment "
            "configuration; missing provider usage remains null.",
            "- Model identity is kept out of judge inputs; any LLM-judge results "
            "must be stored separately and manually audited.",
            "",
            "## Operations",
            "",
            f"- Index size: {summary['index_size_bytes']} bytes",
            f"- Index SHA-256: `{summary['index_sha256']}`",
            "- Indexing latency: "
            f"{_display_latency(summary['indexing_latency_ms'])}",
            "- Index embedding calls: "
            f"{_display_integer(summary['index_embedding_model_calls'])}",
            "",
        ]
    )
    return "\n".join(lines)


def _retrieve_cached(
    *,
    cache: dict[tuple[str, str], tuple[list[RetrievedChunk], float]],
    case: EvaluationCase,
    method: str,
    settings: Settings,
    top_k: int,
    rrf_k: int,
) -> tuple[list[RetrievedChunk], float]:
    key = (method, case.case_id)
    if key not in cache:
        started_at = perf_counter()
        results = retrieve_chunks(
            case.question,
            settings,
            method=method,
            top_k=top_k,
            rrf_k=rrf_k,
        )
        cache[key] = (results, (perf_counter() - started_at) * 1_000)
    return cache[key]


def _oracle_results(
    case: EvaluationCase, chunks: Sequence[DocumentChunk]
) -> list[RetrievedChunk]:
    expected = {
        (source.filename, source.page_number) for source in case.expected_sources
    }
    selected = [
        chunk
        for chunk in chunks
        if (chunk.source_filename, chunk.page_number) in expected
    ]
    return [
        RetrievedChunk(
            chunk=chunk,
            similarity_score=1.0,
            retrieval_method="oracle",
            rank=rank,
        )
        for rank, chunk in enumerate(selected, start=1)
    ]


def _build_summary(
    *,
    dataset: EvaluationDataset,
    cases: Sequence[EvaluationCase],
    config: ExperimentConfig,
    retrieval_trials: Sequence[Mapping[str, Any]],
    generation_trials: Sequence[Mapping[str, Any]],
    settings: Settings,
    index_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    retrieval_summary = []
    for method in config.retrieval_methods:
        retrieval_metrics = aggregate_retrieval_metrics(
            [trial for trial in retrieval_trials if trial["method"] == method]
        )
        retrieval_summary.append({"method": method, **asdict(retrieval_metrics)})

    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for trial in generation_trials:
        grouped[(str(trial["model"]), str(trial["mode"]))].append(trial)

    generation_summary: list[dict[str, Any]] = []
    for (model, mode), trials in sorted(grouped.items()):
        generation_metrics = aggregate_generation_metrics(trials)
        generation_latencies = [
            float(trial["generation_latency_ms"]) for trial in trials
        ]
        retrieval_latencies = [float(trial["retrieval_latency_ms"]) for trial in trials]
        costs = [
            float(trial["estimated_cost_usd"])
            for trial in trials
            if trial["estimated_cost_usd"] is not None
        ]
        item = {
            "model": model,
            "mode": mode,
            **asdict(generation_metrics),
            "latency_p50_ms": median(generation_latencies),
            "latency_p95_ms": _percentile(generation_latencies, 0.95),
            "retrieval_latency_p50_ms": median(retrieval_latencies),
            "model_calls": sum(int(trial["model_calls"]) for trial in trials),
            "input_tokens": _optional_sum(trials, "input_tokens"),
            "output_tokens": _optional_sum(trials, "output_tokens"),
            "estimated_cost_usd": round(sum(costs), 8) if costs else None,
        }
        item.update(
            {
                "citation_precision_display": _display_optional(
                    generation_metrics.citation_precision
                ),
                "citation_recall_display": _display_optional(
                    generation_metrics.citation_recall
                ),
                "hallucination_display": _display_optional(
                    generation_metrics.hallucination_proxy_rate
                ),
                "attack_display": _display_optional(
                    generation_metrics.prompt_injection_success_rate
                ),
            }
        )
        generation_summary.append(item)

    return {
        "status": "completed",
        "dataset_name": dataset.name,
        "dataset_version": dataset.version,
        "case_count": len(cases),
        "answerable_cases": sum(case.answerable for case in cases),
        "unanswerable_cases": sum(not case.answerable for case in cases),
        "retrieval": retrieval_summary,
        "generation": generation_summary,
        "index_size_bytes": index_evidence["artifact_size_bytes"],
        "index_sha256": index_evidence["artifact_sha256"],
        "indexing_latency_ms": index_evidence["total_latency_ms"],
        "index_embedding_model_calls": index_evidence["embedding_model_calls"],
        "metric_note": (
            "Hallucination is a lexical unsupported-sentence proxy; use a blinded "
            "judge and manual audit before making model-quality claims."
        ),
    }


def _write_trials_csv(
    path: Path,
    retrieval_trials: Sequence[Mapping[str, Any]],
    generation_trials: Sequence[Mapping[str, Any]],
) -> None:
    rows = [*retrieval_trials, *generation_trials]
    fieldnames = sorted({key for row in rows for key in row})
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                        for key, value in row.items()
                    }
                )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _update_latest(output_root: Path, run_directory: Path) -> None:
    latest = output_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "REPORT.md"):
        temporary = latest / f".{name}.{uuid4().hex}.tmp"
        try:
            shutil.copy2(run_directory / name, temporary)
            os.replace(temporary, latest / name)
        finally:
            temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _serialize_sources(results: Sequence[RetrievedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "rank": result.rank or rank,
            "filename": result.chunk.source_filename,
            "page": result.chunk.page_number,
            "chunk_id": result.chunk.chunk_id,
            "score": round(result.similarity_score, 8),
            "retrieval_method": result.retrieval_method,
        }
        for rank, result in enumerate(results, start=1)
    ]


def _select_cases(
    cases: Sequence[EvaluationCase], max_cases: int | None
) -> tuple[EvaluationCase, ...]:
    """Select a small, deterministic smoke set without losing key case classes."""
    if max_cases is None or max_cases >= len(cases):
        return tuple(cases)

    priority = ("exact", "synthesis", "adversarial", "unanswerable", "multi_source")
    selected: list[EvaluationCase] = []
    selected_ids: set[str] = set()
    for question_type in priority:
        candidate = next(
            (case for case in cases if case.question_type == question_type), None
        )
        if candidate is not None:
            selected.append(candidate)
            selected_ids.add(candidate.case_id)
        if len(selected) == max_cases:
            return tuple(selected)

    for case in cases:
        if case.case_id not in selected_ids:
            selected.append(case)
        if len(selected) == max_cases:
            break
    return tuple(selected)


def _read_index_evidence(vector_store_path: Path) -> dict[str, Any]:
    index_path = vector_store_path.expanduser().resolve() / "index.npz"
    manifest_path = vector_store_path.expanduser().resolve() / "index-build.json"
    evidence: dict[str, Any] = {
        "artifact_sha256": _sha256_file(index_path),
        "artifact_size_bytes": index_path.stat().st_size,
        "build_configuration_hash": None,
        "source_files": [],
        "created_at": None,
        "total_latency_ms": None,
        "embedding_model_calls": None,
    }
    if not manifest_path.is_file():
        return evidence

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read index build manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Index build manifest is invalid: {manifest_path}")

    configuration = manifest.get("configuration")
    source_files = (
        configuration.get("sources", [])
        if isinstance(configuration, dict)
        else []
    )
    if not isinstance(source_files, list):
        raise RuntimeError(f"Index build manifest sources are invalid: {manifest_path}")
    configuration_hash = manifest.get("configuration_hash")
    created_at = manifest.get("created_at")
    total_latency_ms = manifest.get("total_latency_ms")
    embedding_model_calls = manifest.get("embedding_model_calls")
    if not isinstance(configuration_hash, str) or not configuration_hash:
        raise RuntimeError(f"Index build manifest hash is invalid: {manifest_path}")
    if created_at is not None and not isinstance(created_at, str):
        raise RuntimeError(
            f"Index build manifest timestamp is invalid: {manifest_path}"
        )
    if (
        not isinstance(total_latency_ms, (int, float))
        or isinstance(total_latency_ms, bool)
        or total_latency_ms < 0
    ):
        raise RuntimeError(f"Index build manifest latency is invalid: {manifest_path}")
    if (
        not isinstance(embedding_model_calls, int)
        or isinstance(embedding_model_calls, bool)
        or embedding_model_calls < 0
    ):
        raise RuntimeError(
            f"Index build manifest call count is invalid: {manifest_path}"
        )
    if any(not isinstance(source, dict) for source in source_files):
        raise RuntimeError(f"Index build manifest sources are invalid: {manifest_path}")
    evidence.update(
        {
            "build_configuration_hash": configuration_hash,
            "source_files": source_files,
            "created_at": created_at,
            "total_latency_ms": total_latency_ms,
            "embedding_model_calls": embedding_model_calls,
        }
    )
    return evidence


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configuration_hash(configuration: Mapping[str, Any], dataset_version: str) -> str:
    canonical = json.dumps(
        {"configuration": configuration, "dataset_version": dataset_version},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _estimate_cost(
    *,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    pricing: Mapping[str, Mapping[str, float]],
) -> float | None:
    rates = pricing.get(model)
    if rates is None or input_tokens is None or output_tokens is None:
        return None
    cost = (
        input_tokens * rates["input"] + output_tokens * rates["output"]
    ) / 1_000_000
    return round(cost, 8)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _optional_sum(trials: Sequence[Mapping[str, Any]], name: str) -> int | None:
    values = [trial[name] for trial in trials if trial[name] is not None]
    return sum(int(value) for value in values) if values else None


def _display_optional(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _display_latency(value: int | float | None) -> str:
    return "Not recorded" if value is None else f"{float(value):.1f} ms"


def _display_integer(value: int | None) -> str:
    return "Not recorded" if value is None else str(int(value))


def _required_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Experiment field {name!r} must be a non-empty string.")
    return value.strip()


def _string_tuple(
    payload: Mapping[str, object],
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = payload.get(name)
    if not isinstance(value, list) or (not allow_empty and not value) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        qualifier = "a string list" if allow_empty else "a non-empty string list"
        raise ValueError(f"Experiment field {name!r} must be {qualifier}.")
    return tuple(str(item).strip() for item in value)


def _optional_string(payload: Mapping[str, object], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Experiment field {name!r} must be a string or null.")
    return value.strip()


def _nonnegative_float(value: object, model: str, kind: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
    ):
        raise ValueError(f"Pricing for {model!r} needs non-negative {kind} rate.")
    return float(value)


def _positive_int(
    payload: Mapping[str, object], name: str, *, default: int | None = None
) -> int:
    value = payload.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Experiment field {name!r} must be a positive integer.")
    return value
