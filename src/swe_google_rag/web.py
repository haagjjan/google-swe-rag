"""FastAPI application for an inspectable direct-versus-RAG portfolio demo."""

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings, load_settings
from .generation import format_retrieved_context, generate_answer_with_metadata
from .retrieval import retrieve_chunks
from .schemas import GenerationResult, RetrievedChunk

ALLOWED_DEMO_MODELS = ("gemma-4-26b-a4b-it", "gemini-2.5-flash")


class RetrievalOptions(BaseModel):
    """Bounded retrieval controls exposed by the local demo."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["dense", "bm25", "hybrid"] = "hybrid"
    top_k: int = Field(default=4, ge=1, le=10)
    rrf_k: int = Field(default=60, ge=1, le=1_000)


class AskRequest(BaseModel):
    """Contract for one direct or RAG answer."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2_000)
    mode: Literal["direct", "rag"] = "rag"
    generation_model: str = ALLOWED_DEMO_MODELS[0]
    retrieval: RetrievalOptions = Field(default_factory=RetrievalOptions)


class CompareRequest(BaseModel):
    """Contract for matched direct/RAG responses over one retrieved context."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2_000)
    models: list[str] = Field(
        default_factory=lambda: list(ALLOWED_DEMO_MODELS),
        min_length=1,
        max_length=len(ALLOWED_DEMO_MODELS),
    )
    retrieval: RetrievalOptions = Field(default_factory=RetrievalOptions)


SettingsLoader = Callable[[], Settings]
Retriever = Callable[..., list[RetrievedChunk]]
Generator = Callable[..., GenerationResult]


def create_app(
    *,
    settings_loader: SettingsLoader = load_settings,
    retriever: Retriever = retrieve_chunks,
    generator: Generator = generate_answer_with_metadata,
) -> FastAPI:
    """Create the application with injectable boundaries for offline tests."""
    app = FastAPI(
        title="Google SWE RAG Portfolio Demo",
        version="2.0.0",
        description=(
            "Compare hosted Gemma and Gemini with and without retrieved evidence."
        ),
    )
    static_path = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_path), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_path / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        try:
            settings = settings_loader()
            index_path = settings.vector_store_path / "index.npz"
            return {
                "status": "ready" if index_path.is_file() else "index_required",
                "configured": True,
                "index_ready": index_path.is_file(),
                "embedding_model": settings.embedding_model,
                "embedding_dimension": settings.embedding_dimension,
                "generation_models": list(ALLOWED_DEMO_MODELS),
                "provider": "Google Gemini API (hosted)",
            }
        except Exception:  # noqa: BLE001
            return {
                "status": "configuration_required",
                "configured": False,
                "index_ready": False,
                "generation_models": list(ALLOWED_DEMO_MODELS),
                "provider": "Google Gemini API (hosted)",
            }

    @app.get("/api/evaluations/latest")
    def latest_evaluation() -> dict[str, Any]:
        project_summary = Path.cwd() / "eval/results/latest/summary.json"
        packaged_summary = static_path / "benchmark-summary.json"
        summary_path = (
            project_summary if project_summary.is_file() else packaged_summary
        )
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=503, detail="No validated benchmark summary is available."
            ) from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=500, detail="Benchmark summary is invalid.")
        return payload

    @app.post("/api/ask")
    def ask(request: AskRequest) -> dict[str, Any]:
        _validate_models([request.generation_model])
        settings = _load_settings_or_503(settings_loader)
        try:
            return _run_answer(
                request=request,
                settings=settings,
                retriever=retriever,
                generator=generator,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/compare")
    def compare(request: CompareRequest) -> dict[str, Any]:
        _validate_models(request.models)
        if len(set(request.models)) != len(request.models):
            raise HTTPException(status_code=422, detail="Model IDs must be unique.")
        settings = _load_settings_or_503(settings_loader)
        started_at = perf_counter()
        try:
            retrieved = retriever(
                request.question,
                settings,
                method=request.retrieval.method,
                top_k=request.retrieval.top_k,
                rrf_k=request.retrieval.rrf_k,
            )
            context = format_retrieved_context(retrieved)
            responses = []
            for model in request.models:
                for mode in ("direct", "rag"):
                    generated = generator(
                        question=request.question,
                        context=context if mode == "rag" else "",
                        settings=settings,
                        model=model,
                        grounded=mode == "rag",
                    )
                    responses.append(
                        _serialize_answer(
                            generated=generated,
                            mode=mode,
                            sources=retrieved if mode == "rag" else [],
                            retrieval=request.retrieval,
                            retrieval_latency_ms=None,
                        )
                    )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return {
            "question": request.question,
            "responses": responses,
            "shared_retrieved_sources": _serialize_sources(retrieved),
            "total_latency_ms": round((perf_counter() - started_at) * 1_000, 3),
            "provider_disclosure": (
                "Gemma is open-weight and Gemini is proprietary; both are hosted "
                "through Google's Gemini API in this demo."
            ),
        }

    return app


def _run_answer(
    *,
    request: AskRequest,
    settings: Settings,
    retriever: Retriever,
    generator: Generator,
) -> dict[str, Any]:
    retrieved: list[RetrievedChunk] = []
    retrieval_latency_ms = 0.0
    if request.mode == "rag":
        started_at = perf_counter()
        retrieved = retriever(
            request.question,
            settings,
            method=request.retrieval.method,
            top_k=request.retrieval.top_k,
            rrf_k=request.retrieval.rrf_k,
        )
        retrieval_latency_ms = (perf_counter() - started_at) * 1_000

    generated = generator(
        question=request.question,
        context=format_retrieved_context(retrieved),
        settings=settings,
        model=request.generation_model,
        grounded=request.mode == "rag",
    )
    return _serialize_answer(
        generated=generated,
        mode=request.mode,
        sources=retrieved,
        retrieval=request.retrieval,
        retrieval_latency_ms=retrieval_latency_ms,
    )


def _serialize_answer(
    *,
    generated: GenerationResult,
    mode: str,
    sources: list[RetrievedChunk],
    retrieval: RetrievalOptions,
    retrieval_latency_ms: float | None,
) -> dict[str, Any]:
    return {
        "answer": generated.text,
        "mode": mode,
        "generation_model": generated.model,
        "generation_latency_ms": round(generated.latency_ms, 3),
        "retrieval_latency_ms": (
            None if retrieval_latency_ms is None else round(retrieval_latency_ms, 3)
        ),
        "total_latency_ms": round(
            generated.latency_ms + (retrieval_latency_ms or 0.0), 3
        ),
        "usage": {
            "input_tokens": generated.usage.input_tokens,
            "output_tokens": generated.usage.output_tokens,
            "total_tokens": generated.usage.total_tokens,
            "model_calls": generated.model_calls,
        },
        "retrieval": retrieval.model_dump(),
        "sources": _serialize_sources(sources),
        "provider": "Google Gemini API (hosted)",
    }


def _serialize_sources(results: list[RetrievedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "rank": result.rank or rank,
            "filename": result.chunk.source_filename,
            "page": result.chunk.page_number,
            "chunk_id": result.chunk.chunk_id,
            "score": round(result.similarity_score, 6),
            "retrieval_method": result.retrieval_method,
            "excerpt": result.chunk.text[:1_200],
        }
        for rank, result in enumerate(results, start=1)
    ]


def _validate_models(models: list[str]) -> None:
    unsupported = sorted(set(models) - set(ALLOWED_DEMO_MODELS))
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported demo model(s): {', '.join(unsupported)}",
        )


def _load_settings_or_503(settings_loader: SettingsLoader) -> Settings:
    try:
        return settings_loader()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Configure the local environment before using model endpoints.",
        ) from exc
