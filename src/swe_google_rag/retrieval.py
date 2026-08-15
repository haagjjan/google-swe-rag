"""Dense, lexical BM25, and reciprocal-rank-fusion retrieval strategies."""

import math
import re
from collections import Counter
from collections.abc import Sequence

from .config import Settings
from .embeddings import embed_question
from .schemas import DocumentChunk, RetrievedChunk
from .vector_store import load_index, search_index

_WORD_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)
_SUPPORTED_METHODS = frozenset({"dense", "bm25", "hybrid"})


def retrieve_chunks(
    question: str,
    settings: Settings,
    *,
    method: str = "dense",
    top_k: int | None = None,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Retrieve chunks with one explicitly selected, inspectable strategy."""
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")
    if len(question) > settings.max_question_chars:
        raise ValueError(
            f"Question must not exceed {settings.max_question_chars} characters."
        )
    if method not in _SUPPORTED_METHODS:
        choices = ", ".join(sorted(_SUPPORTED_METHODS))
        raise ValueError(f"Unsupported retrieval method {method!r}; choose {choices}.")

    selected_top_k = settings.top_k if top_k is None else top_k
    if selected_top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero.")

    chunks, embeddings, stored_model, stored_dimension = load_index(
        settings.vector_store_path
    )

    if method == "bm25":
        return search_bm25(chunks, question, selected_top_k)

    _validate_embedding_compatibility(
        settings=settings,
        stored_model=stored_model,
        stored_dimension=stored_dimension,
    )
    query_vector = embed_question(question=question, settings=settings)
    if len(query_vector) != stored_dimension:
        raise ValueError(
            f"Query embedding dimension {len(query_vector)} does not match "
            f"stored index dimension {stored_dimension}."
        )

    dense = search_index(
        chunks=chunks,
        embeddings=embeddings,
        query_vector=query_vector,
        top_k=len(chunks) if method == "hybrid" else selected_top_k,
    )
    if method == "dense":
        return dense

    lexical = search_bm25(chunks, question, len(chunks))
    return reciprocal_rank_fusion(
        rankings=(dense, lexical),
        top_k=selected_top_k,
        rrf_k=rrf_k,
    )


def search_bm25(
    chunks: Sequence[DocumentChunk],
    question: str,
    top_k: int,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[RetrievedChunk]:
    """Rank chunks with a small dependency-free Okapi BM25 implementation."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if not chunks:
        return []

    query_terms = _tokenize(question)
    documents = [_tokenize(chunk.text) for chunk in chunks]
    lengths = [len(document) for document in documents]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    document_frequency = Counter(
        term for document in documents for term in set(document)
    )

    scores: list[float] = []
    for document, length in zip(documents, lengths, strict=True):
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            containing = document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (len(documents) - containing + 0.5) / (containing + 0.5)
            )
            normalizer = frequency + k1 * (
                1 - b + b * length / average_length
            ) if average_length else frequency
            score += inverse_document_frequency * frequency * (k1 + 1) / normalizer
        scores.append(score)

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: (-scores[index], index),
    )[: min(top_k, len(chunks))]
    return [
        RetrievedChunk(
            chunk=chunks[index],
            similarity_score=scores[index],
            retrieval_method="bm25",
            rank=rank,
        )
        for rank, index in enumerate(ranked_indices, start=1)
    ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse result lists by stable chunk ID using reciprocal ranks."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero.")

    scores: dict[str, float] = {}
    chunks_by_id: dict[str, DocumentChunk] = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            chunk_id = result.chunk.chunk_id
            if chunk_id not in first_seen:
                first_seen[chunk_id] = seen_counter
                seen_counter += 1
            chunks_by_id[chunk_id] = result.chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rrf_k + rank)

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], first_seen[chunk_id]),
    )[:top_k]
    return [
        RetrievedChunk(
            chunk=chunks_by_id[chunk_id],
            similarity_score=scores[chunk_id],
            retrieval_method="hybrid",
            rank=rank,
        )
        for rank, chunk_id in enumerate(ranked_ids, start=1)
    ]


def _validate_embedding_compatibility(
    *,
    settings: Settings,
    stored_model: str,
    stored_dimension: int,
) -> None:
    if stored_model != settings.embedding_model:
        raise ValueError(
            "The configured embedding model does not match the model "
            "used to build the stored index."
        )
    if (
        settings.embedding_dimension is not None
        and settings.embedding_dimension != stored_dimension
    ):
        raise ValueError(
            "The configured embedding dimension does not match the "
            "dimension used to build the stored index."
        )


def _tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_PATTERN.finditer(text)]
