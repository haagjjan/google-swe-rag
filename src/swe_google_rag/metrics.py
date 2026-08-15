"""Deterministic RAG evaluation metrics with explicit proxy boundaries."""

import math
import re
from collections.abc import Mapping, Sequence

from .schemas import (
    EvaluationCase,
    GenerationMetrics,
    RetrievalMetrics,
    RetrievedChunk,
)

_CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)]", re.IGNORECASE)
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", re.IGNORECASE)
_ABSTENTION_PHRASES = (
    "do not contain enough information",
    "don't contain enough information",
    "cannot answer",
    "can't answer",
    "do not know",
    "don't know",
    "insufficient information",
    "not enough information",
)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
    }
)


def score_retrieval_case(
    case: EvaluationCase,
    results: Sequence[RetrievedChunk],
) -> dict[str, float]:
    """Score one answerable case using binary document-page relevance."""
    if not case.answerable:
        raise ValueError("Retrieval relevance is undefined for unanswerable cases.")

    expected = {
        (source.filename, source.page_number) for source in case.expected_sources
    }
    relevance = [
        (result.chunk.source_filename, result.chunk.page_number) in expected
        for result in results
    ]
    seen_expected: set[tuple[str, int | None]] = set()
    novelty_relevance = []
    for result, relevant in zip(results, relevance, strict=True):
        source = (result.chunk.source_filename, result.chunk.page_number)
        is_new_relevant = relevant and source not in seen_expected
        novelty_relevance.append(is_new_relevant)
        if is_new_relevant:
            seen_expected.add(source)

    def recall_at(k: int) -> float:
        found = {
            (result.chunk.source_filename, result.chunk.page_number)
            for result in results[:k]
            if (result.chunk.source_filename, result.chunk.page_number) in expected
        }
        return len(found) / len(expected)

    first_relevant = next(
        (rank for rank, relevant in enumerate(relevance, start=1) if relevant), None
    )
    reciprocal_rank = 0.0 if first_relevant is None else 1.0 / first_relevant

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, relevant in enumerate(novelty_relevance, start=1)
        if relevant
    )
    ideal_count = min(len(expected), len(results))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    return {
        "recall_at_1": recall_at(1),
        "recall_at_3": recall_at(3),
        "recall_at_5": recall_at(5),
        "reciprocal_rank": reciprocal_rank,
        "ndcg": ndcg,
        "context_precision": sum(relevance) / len(results) if results else 0.0,
    }


def aggregate_retrieval_metrics(
    trial_metrics: Sequence[Mapping[str, float]],
) -> RetrievalMetrics:
    """Average per-question retrieval metrics without hiding sample count."""
    if not trial_metrics:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    def average(name: str) -> float:
        return sum(trial[name] for trial in trial_metrics) / len(trial_metrics)

    return RetrievalMetrics(
        recall_at_1=average("recall_at_1"),
        recall_at_3=average("recall_at_3"),
        recall_at_5=average("recall_at_5"),
        mean_reciprocal_rank=average("reciprocal_rank"),
        mean_ndcg=average("ndcg"),
        mean_context_precision=average("context_precision"),
        evaluated_cases=len(trial_metrics),
    )


def score_generation_case(
    case: EvaluationCase,
    answer: str,
    context_results: Sequence[RetrievedChunk],
    *,
    grounded: bool,
) -> dict[str, float | bool | None]:
    """Score one answer with transparent lexical proxies and exact labels."""
    abstained = is_abstention(answer)
    fact_coverage = _fact_coverage(answer, case.reference_facts)

    if case.answerable:
        correctness = fact_coverage
        relevance = float(not abstained and fact_coverage > 0)
    else:
        correctness = float(abstained)
        relevance = float(abstained)

    citation_precision: float | None = None
    citation_recall: float | None = None
    hallucination_proxy: float | None = None
    if grounded:
        citation_precision, citation_recall = _citation_metrics(
            case, answer, context_results
        )
        hallucination_proxy = _hallucination_proxy(answer, context_results, abstained)

    attack_success: bool | None = None
    if case.question_type == "adversarial":
        folded_answer = answer.casefold()
        attack_success = any(
            marker.casefold() in folded_answer
            for marker in case.attack_success_markers
        )

    return {
        "answer_correctness": correctness,
        "grounded_fact_coverage": fact_coverage,
        "answer_relevance": relevance,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "hallucination_proxy_rate": hallucination_proxy,
        "abstained": abstained,
        "attack_success": attack_success,
    }


def aggregate_generation_metrics(
    trials: Sequence[Mapping[str, object]],
) -> GenerationMetrics:
    """Aggregate generation metrics, including answerability classification."""
    if not trials:
        return GenerationMetrics(
            0.0, 0.0, 0.0, None, None, None, 0.0, 0.0, 0.0, None, 0
        )

    def average(name: str) -> float:
        return sum(_numeric_value(trial[name], name) for trial in trials) / len(
            trials
        )

    def optional_average(name: str) -> float | None:
        values = [
            _numeric_value(trial[name], name)
            for trial in trials
            if trial.get(name) is not None
        ]
        return sum(values) / len(values) if values else None

    true_positive = sum(
        int(bool(trial["abstained"]) and not bool(trial["answerable"]))
        for trial in trials
    )
    false_positive = sum(
        int(bool(trial["abstained"]) and bool(trial["answerable"]))
        for trial in trials
    )
    false_negative = sum(
        int(not bool(trial["abstained"]) and not bool(trial["answerable"]))
        for trial in trials
    )
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)

    attack_values = [
        bool(trial["attack_success"])
        for trial in trials
        if trial.get("attack_success") is not None
    ]
    attack_rate = (
        sum(attack_values) / len(attack_values) if attack_values else None
    )

    return GenerationMetrics(
        answer_correctness=average("answer_correctness"),
        grounded_fact_coverage=average("grounded_fact_coverage"),
        answer_relevance=average("answer_relevance"),
        citation_precision=optional_average("citation_precision"),
        citation_recall=optional_average("citation_recall"),
        hallucination_proxy_rate=optional_average("hallucination_proxy_rate"),
        abstention_precision=precision,
        abstention_recall=recall,
        abstention_f1=f1,
        prompt_injection_success_rate=attack_rate,
        evaluated_cases=len(trials),
    )


def is_abstention(answer: str) -> bool:
    folded = answer.casefold()
    return any(phrase in folded for phrase in _ABSTENTION_PHRASES)


def _fact_coverage(answer: str, facts: Sequence[str]) -> float:
    if not facts:
        return 0.0
    answer_tokens = set(_content_tokens(answer))
    covered = 0
    for fact in facts:
        fact_tokens = set(_content_tokens(fact))
        overlap = _safe_ratio(len(answer_tokens & fact_tokens), len(fact_tokens))
        covered += overlap >= 0.7
    return covered / len(facts)


def _citation_metrics(
    case: EvaluationCase,
    answer: str,
    context_results: Sequence[RetrievedChunk],
) -> tuple[float, float]:
    citation_numbers = {
        int(match.group(1)) for match in _CITATION_PATTERN.finditer(answer)
    }
    cited_sources = {
        (
            context_results[number - 1].chunk.source_filename,
            context_results[number - 1].chunk.page_number,
        )
        for number in citation_numbers
        if 1 <= number <= len(context_results)
    }
    expected = {
        (source.filename, source.page_number) for source in case.expected_sources
    }
    valid = cited_sources & expected

    if case.answerable:
        precision = _safe_ratio(len(valid), len(citation_numbers))
        recall = _safe_ratio(len(valid), len(expected))
        return precision, recall

    return (1.0 if not citation_numbers else 0.0), 1.0


def _hallucination_proxy(
    answer: str,
    context_results: Sequence[RetrievedChunk],
    abstained: bool,
) -> float:
    """Estimate unsupported sentences by lexical context overlap.

    This is intentionally named a proxy: paraphrases can be misclassified and
    fluent agreement is not proof of entailment. An optional blinded LLM judge
    and manual audit should supplement it in a published live benchmark.
    """
    if abstained:
        return 0.0

    context_token_sets = [
        set(_content_tokens(result.chunk.text)) for result in context_results
    ]
    claims = []
    for sentence in _SENTENCE_PATTERN.split(answer):
        tokens = set(_content_tokens(_CITATION_PATTERN.sub("", sentence)))
        if len(tokens) >= 4:
            claims.append(tokens)

    if not claims:
        return 0.0
    if not context_token_sets:
        return 1.0

    unsupported = 0
    for claim in claims:
        best_overlap = max(
            _safe_ratio(len(claim & context), len(claim))
            for context in context_token_sets
        )
        unsupported += best_overlap < 0.55
    return unsupported / len(claims)


def _content_tokens(text: str) -> list[str]:
    candidates = (
        match.group(0).casefold() for match in _WORD_PATTERN.finditer(text)
    )
    return [
        token
        for token in candidates
        if token not in _STOP_WORDS
    ]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _numeric_value(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Metric {name!r} must be numeric.")
    return float(value)
