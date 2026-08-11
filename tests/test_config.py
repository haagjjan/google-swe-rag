"""Tests for environment configuration loading and validation."""

from pathlib import Path

import pytest

from swe_google_rag.config import load_settings

_SETTING_NAMES = (
    "GOOGLE_API_KEY",
    "GENERATION_MODEL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "PDF_STORAGE_PATH",
    "VECTOR_STORE_PATH",
    "CHUNK_SIZE_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "TOP_K",
)


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent the developer's shell or .env from affecting these tests."""
    for name in _SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)


def _write_env(
    path: Path,
    *,
    api_key: str = "test-google-key",
    chunk_size: str = "512",
    overlap: str = "64",
    top_k: str = "3",
    dimension: str = "768",
) -> None:
    path.write_text(
        "\n".join(
            [
                f"GOOGLE_API_KEY={api_key}",
                "GENERATION_MODEL=gemma-test",
                "EMBEDDING_MODEL=embedding-test",
                f"EMBEDDING_DIMENSION={dimension}",
                f"PDF_STORAGE_PATH={path.parent / 'pdfs'}",
                f"VECTOR_STORE_PATH={path.parent / 'vectors'}",
                f"CHUNK_SIZE_TOKENS={chunk_size}",
                f"CHUNK_OVERLAP_TOKENS={overlap}",
                f"TOP_K={top_k}",
            ]
        ),
        encoding="utf-8",
    )


def test_load_settings_reads_and_resolves_values(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path)

    settings = load_settings(env_path)

    assert settings.google_api_key == "test-google-key"
    assert settings.generation_model == "gemma-test"
    assert settings.embedding_model == "embedding-test"
    assert settings.embedding_dimension == 768
    assert settings.pdf_storage_path == (tmp_path / "pdfs").resolve()
    assert settings.vector_store_path == (tmp_path / "vectors").resolve()
    assert settings.chunk_size_tokens == 512
    assert settings.chunk_overlap_tokens == 64
    assert settings.top_k == 3


def test_existing_environment_takes_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path, api_key="file-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "shell-key")

    settings = load_settings(env_path)

    assert settings.google_api_key == "shell-key"


def test_embedding_dimension_can_be_omitted(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path, dimension="")

    settings = load_settings(env_path)

    assert settings.embedding_dimension is None


def test_missing_key_error_never_exposes_other_values(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path, api_key="")

    with pytest.raises(ValueError, match="GOOGLE_API_KEY") as error:
        load_settings(env_path)

    assert "test-google-key" not in str(error.value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("chunk_size", "abc", "CHUNK_SIZE_TOKENS must be an integer"),
        ("chunk_size", "0", "CHUNK_SIZE_TOKENS must be greater than zero"),
        ("overlap", "-1", "CHUNK_OVERLAP_TOKENS must not be negative"),
        ("top_k", "0", "TOP_K must be greater than zero"),
        ("dimension", "-2", "EMBEDDING_DIMENSION must be greater than zero"),
    ],
)
def test_invalid_numeric_settings_are_clear(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    env_path = tmp_path / ".env"
    arguments = {field: value}
    _write_env(env_path, **arguments)

    with pytest.raises(ValueError, match=message):
        load_settings(env_path)


def test_overlap_must_be_smaller_than_chunk_size(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path, chunk_size="10", overlap="10")

    with pytest.raises(ValueError, match="must be smaller"):
        load_settings(env_path)


def test_explicit_missing_env_file_fails_clearly(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.env"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_settings(missing_path)
