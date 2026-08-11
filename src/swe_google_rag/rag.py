"""Orchestrate query embedding, retrieval, context building, and generation."""

from .config import Settings
from .embeddings import embed_question
from .generation import format_retrieved_context, generate_grounded_answer
from .schemas import RagAnswer, RetrievedChunk
from .vector_store import load_index, search_index


def retrieve_for_question(
    question: str,
    settings: Settings,
) -> list[RetrievedChunk]:
    """Embed one question and retrieve the configured number of chunks."""
    question = question.strip()

    if not question:
        raise ValueError("Question must not be empty.")

    (
        chunks,
        embeddings,
        stored_embedding_model,
        stored_embedding_dimension,
    ) = load_index(settings.vector_store_path)

    if stored_embedding_model != settings.embedding_model:
        raise ValueError(
            "The configured embedding model does not match the model "
            "used to build the stored index."
        )

    if (
        settings.embedding_dimension is not None
        and settings.embedding_dimension != stored_embedding_dimension
    ):
        raise ValueError(
            "The configured embedding dimension does not match the "
            "dimension used to build the stored index."
        )

    query_vector = embed_question(
        question=question,
        settings=settings,
    )

    if len(query_vector) != stored_embedding_dimension:
        raise ValueError(
            f"Query embedding dimension {len(query_vector)} does not match "
            f"stored index dimension {stored_embedding_dimension}."
        )

    return search_index(
        chunks=chunks,
        embeddings=embeddings,
        query_vector=query_vector,
        top_k=settings.top_k,
    )


def answer_question(question: str, settings: Settings) -> RagAnswer:
    """Run the complete query-time RAG pipeline for one question."""
    retrieved_chunks = retrieve_for_question(
        question=question,
        settings=settings,
    )

    context = format_retrieved_context(retrieved_chunks)

    answer_text = generate_grounded_answer(
        question=question,
        context=context,
        settings=settings,
    )

    return RagAnswer(
        text=answer_text,
        sources=tuple(retrieved_chunks),
    )
