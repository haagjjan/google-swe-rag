"""Define the shared page, chunk, retrieval, and answer data structures."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

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
    retrieval_method: str = "dense"
    rank: int | None = None


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
    manifest_path: Path
    total_latency_ms: float
    embedding_model_calls: int


@dataclass(frozen=True, slots=True)
class GenerationUsage:
    """Provider-reported token usage for one generation request."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Generated text plus operational measurements."""

    text: str
    model: str
    mode: str
    latency_ms: float
    model_calls: int
    usage: GenerationUsage = field(default_factory=GenerationUsage)


@dataclass(frozen=True, slots=True)
class EvaluationSource:
    """One labelled document location expected to support an answer."""

    filename: str
    page_number: int


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """A single versioned benchmark question and its reference evidence."""

    case_id: str
    question: str
    answerable: bool
    question_type: str
    expected_sources: tuple[EvaluationSource, ...]
    reference_facts: tuple[str, ...]
    tags: tuple[str, ...] = ()
    attack_success_markers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """Validated evaluation dataset with explicit provenance and license."""

    name: str
    version: str
    license: str
    corpus_path: str
    cases: tuple[EvaluationCase, ...]


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Aggregate retrieval metrics for one retrieval configuration."""

    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mean_reciprocal_rank: float
    mean_ndcg: float
    mean_context_precision: float
    evaluated_cases: int


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Aggregate generation, abstention, citation, and safety metrics."""

    answer_correctness: float
    grounded_fact_coverage: float
    answer_relevance: float
    citation_precision: float | None
    citation_recall: float | None
    hallucination_proxy_rate: float | None
    abstention_precision: float
    abstention_recall: float
    abstention_f1: float
    prompt_injection_success_rate: float | None
    evaluated_cases: int


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Versioned controls for a reproducible benchmark run."""

    name: str
    version: str
    retrieval_methods: tuple[str, ...]
    generation_models: tuple[str, ...]
    generation_modes: tuple[str, ...]
    generation_retrieval_method: str
    top_k: int
    rrf_k: int = 60
    max_cases: int | None = None
    pricing_snapshot_date: str | None = None
    pricing_source: str | None = None
    pricing_usd_per_million_tokens: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Complete serialized benchmark output."""

    run_id: str
    created_at: str
    config_hash: str
    dataset_name: str
    dataset_version: str
    configuration: Mapping[str, Any]
    retrieval_results: tuple[Mapping[str, Any], ...]
    generation_results: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
