"""Load and validate versioned JSON evaluation datasets."""

import json
from pathlib import Path

from .schemas import EvaluationCase, EvaluationDataset, EvaluationSource

_QUESTION_TYPES = frozenset(
    {"exact", "synthesis", "multi_source", "unanswerable", "adversarial"}
)


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    """Return a validated dataset and reject ambiguous benchmark labels."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read evaluation dataset: {path}") from exc

    if not isinstance(payload, dict):
        raise TypeError("Evaluation dataset root must be an object.")

    name = _required_string(payload, "name")
    version = _required_string(payload, "version")
    license_name = _required_string(payload, "license")
    corpus_path = _required_string(payload, "corpus_path")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Evaluation dataset must contain a non-empty cases list.")

    cases = tuple(_parse_case(item) for item in raw_cases)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation case IDs must be unique.")

    answerable_count = sum(case.answerable for case in cases)
    if answerable_count == 0 or answerable_count == len(cases):
        raise ValueError(
            "Evaluation dataset must include answerable and unanswerable cases."
        )

    return EvaluationDataset(
        name=name,
        version=version,
        license=license_name,
        corpus_path=corpus_path,
        cases=cases,
    )


def validate_dataset_sources(
    dataset: EvaluationDataset,
    available_sources: set[tuple[str, int]],
) -> None:
    """Ensure every labelled source resolves to a generated corpus page."""
    missing = sorted(
        {
            (source.filename, source.page_number)
            for case in dataset.cases
            for source in case.expected_sources
            if (source.filename, source.page_number) not in available_sources
        }
    )
    if missing:
        formatted = ", ".join(f"{name}:p{page}" for name, page in missing)
        raise ValueError(f"Dataset references missing corpus pages: {formatted}")


def _parse_case(value: object) -> EvaluationCase:
    if not isinstance(value, dict):
        raise TypeError("Every evaluation case must be an object.")

    case_id = _required_string(value, "id")
    question = _required_string(value, "question")
    answerable = value.get("answerable")
    if not isinstance(answerable, bool):
        raise TypeError(f"Case {case_id!r} answerable must be a boolean.")

    question_type = _required_string(value, "question_type")
    if question_type not in _QUESTION_TYPES:
        choices = ", ".join(sorted(_QUESTION_TYPES))
        raise ValueError(
            f"Case {case_id!r} has unsupported question_type; choose {choices}."
        )

    raw_sources = value.get("expected_sources", [])
    if not isinstance(raw_sources, list):
        raise TypeError(f"Case {case_id!r} expected_sources must be a list.")
    expected_sources = tuple(_parse_source(item, case_id) for item in raw_sources)

    reference_facts = _string_tuple(value, "reference_facts", case_id)
    tags = _string_tuple(value, "tags", case_id)
    attack_markers = _string_tuple(value, "attack_success_markers", case_id)

    if answerable and (not expected_sources or not reference_facts):
        raise ValueError(
            f"Answerable case {case_id!r} needs expected sources and reference facts."
        )
    if not answerable and (expected_sources or reference_facts):
        raise ValueError(
            f"Unanswerable case {case_id!r} must not contain reference evidence."
        )
    if question_type == "unanswerable" and answerable:
        raise ValueError(f"Case {case_id!r} cannot be answerable and unanswerable.")
    if question_type == "adversarial" and not attack_markers:
        raise ValueError(
            f"Adversarial case {case_id!r} needs attack_success_markers."
        )

    return EvaluationCase(
        case_id=case_id,
        question=question,
        answerable=answerable,
        question_type=question_type,
        expected_sources=expected_sources,
        reference_facts=reference_facts,
        tags=tags,
        attack_success_markers=attack_markers,
    )


def _parse_source(value: object, case_id: str) -> EvaluationSource:
    if not isinstance(value, dict):
        raise TypeError(f"Case {case_id!r} source labels must be objects.")
    filename = _required_string(value, "filename")
    page_number = value.get("page")
    if (
        not isinstance(page_number, int)
        or isinstance(page_number, bool)
        or page_number <= 0
    ):
        raise ValueError(f"Case {case_id!r} source pages must be positive integers.")
    return EvaluationSource(filename=filename, page_number=page_number)


def _required_string(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Evaluation field {name!r} must be a non-empty string.")
    return value.strip()


def _string_tuple(
    payload: dict[str, object], name: str, case_id: str
) -> tuple[str, ...]:
    value = payload.get(name, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise TypeError(f"Case {case_id!r} field {name!r} must be a string list.")
    return tuple(item.strip() for item in value)
