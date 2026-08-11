"""Load and validate environment-based configuration for the RAG project."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration shared by indexing and query-time pipelines."""

    google_api_key: str
    generation_model: str
    embedding_model: str
    embedding_dimension: int | None
    pdf_storage_path: Path
    vector_store_path: Path
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    top_k: int


def load_settings(env_file: Path | None = None) -> Settings:
    """Load validated settings from the environment and optional env file."""
    project_root = Path(__file__).resolve().parents[2]
    env_path = _resolve_env_path(env_file, project_root)

    if env_path.exists():
        if not env_path.is_file():
            raise ValueError(f"Environment path is not a file: {env_path}")
        load_dotenv(env_path, override=False)
    elif env_file is not None:
        raise FileNotFoundError(f"Environment file does not exist: {env_path}")

    google_api_key = _required_env("GOOGLE_API_KEY")
    generation_model = _required_env("GENERATION_MODEL")
    embedding_model = _required_env("EMBEDDING_MODEL")
    embedding_dimension = _optional_positive_int("EMBEDDING_DIMENSION")

    pdf_storage_path = _resolve_path(
        _required_env("PDF_STORAGE_PATH"),
        project_root,
    )
    vector_store_path = _resolve_path(
        _required_env("VECTOR_STORE_PATH"),
        project_root,
    )

    chunk_size_tokens = _positive_int("CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens = _non_negative_int("CHUNK_OVERLAP_TOKENS")
    top_k = _positive_int("TOP_K")

    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS.")

    return Settings(
        google_api_key=google_api_key,
        generation_model=generation_model,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        pdf_storage_path=pdf_storage_path,
        vector_store_path=vector_store_path,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        top_k=top_k,
    )


def _resolve_env_path(env_file: Path | None, project_root: Path) -> Path:
    """Resolve the default or explicitly configured environment file."""
    if env_file is None:
        return project_root / ".env"

    path = env_file.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _required_env(name: str) -> str:
    """Return a required non-empty environment variable."""
    value = os.getenv(name)

    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")

    return value.strip()


def _positive_int(name: str) -> int:
    """Read a positive integer environment variable."""
    value = _parse_int(name)

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return value


def _non_negative_int(name: str) -> int:
    """Read a non-negative integer environment variable."""
    value = _parse_int(name)

    if value < 0:
        raise ValueError(f"{name} must not be negative.")

    return value


def _optional_positive_int(name: str) -> int | None:
    """Read an optional positive integer environment variable."""
    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return None

    value = _parse_int(name)

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return value


def _parse_int(name: str) -> int:
    """Read an integer while keeping validation errors user-facing."""
    raw_value = _required_env(name)
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _resolve_path(raw_path: str, project_root: Path) -> Path:
    """Resolve an absolute or project-relative configured path."""
    path = Path(raw_path).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()
