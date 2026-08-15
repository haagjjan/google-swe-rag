"""Optional blinded LLM judging with a deterministic manual-audit sample."""

import csv
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from google import genai
from google.genai import types

from .config import Settings


def run_blinded_judge(
    *,
    settings: Settings,
    report_path: Path,
    judge_model: str,
    output_path: Path,
    audit_size: int = 12,
) -> tuple[Path, Path]:
    """Judge saved answers without exposing candidate model identity to the judge."""
    if audit_size <= 0:
        raise ValueError("audit_size must be greater than zero.")
    report = _load_report(report_path)
    trials = report.get("generation_results")
    if not isinstance(trials, list) or not trials:
        raise ValueError("The report contains no generation trials to judge.")

    output_path = output_path.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=settings.google_api_key)
    judgments: list[dict[str, Any]] = []

    for position, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise TypeError("Generation trials must be objects.")
        blind_id = _blind_id(str(report.get("run_id")), position, trial)
        prompt = _judge_prompt(blind_id, trial)
        started_at = perf_counter()
        try:
            response = client.models.generate_content(
                model=judge_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "correctness": {"type": "number"},
                            "relevance": {"type": "number"},
                            "unsupported_claim_rate": {"type": "number"},
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "correctness",
                            "relevance",
                            "unsupported_claim_rate",
                            "rationale",
                        ],
                    },
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Blinded judging failed for judge model {judge_model!r}."
            ) from exc

        try:
            parsed = json.loads(response.text or "")
        except json.JSONDecodeError as exc:
            raise RuntimeError("Judge returned invalid JSON.") from exc
        _validate_judgment(parsed)
        usage = getattr(response, "usage_metadata", None)
        judgments.append(
            {
                "blind_id": blind_id,
                "case_id": trial.get("case_id"),
                "candidate_model": trial.get("model"),
                "candidate_mode": trial.get("mode"),
                "judge_model": judge_model,
                "latency_ms": round((perf_counter() - started_at) * 1_000, 3),
                "input_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
                **parsed,
            }
        )

    judged_path = output_path / "judge-results.json"
    _write_json_atomic(
        judged_path,
        {
            "source_run_id": report.get("run_id"),
            "judge_model": judge_model,
            "blinding": (
                "Candidate model and mode were excluded from judge prompts and "
                "reattached only after each judgment."
            ),
            "judgments": judgments,
        },
    )
    audit_path = output_path / "manual-audit.csv"
    _write_audit_sample(
        path=audit_path,
        trials=trials,
        judgments=judgments,
        audit_size=min(audit_size, len(trials)),
        run_id=str(report.get("run_id")),
    )
    return judged_path, audit_path


def _judge_prompt(blind_id: str, trial: Mapping[str, Any]) -> str:
    payload = {
        "blind_id": blind_id,
        "question": trial.get("question"),
        "answerable": trial.get("answerable"),
        "reference_facts": trial.get("reference_facts", []),
        "candidate_answer": trial.get("answer"),
    }
    return (
        "Evaluate the anonymous candidate answer only against the supplied "
        "question, answerability label, and reference facts. Return values from "
        "0 to 1 for correctness, relevance, and unsupported_claim_rate. An "
        "appropriate refusal is correct when answerable is false. Do not infer "
        "or guess the candidate model. Keep rationale under 50 words.\n\n"
        + json.dumps(payload, sort_keys=True, ensure_ascii=False)
    )


def _validate_judgment(value: object) -> None:
    if not isinstance(value, dict):
        raise TypeError("Judge result must be an object.")
    for name in ("correctness", "relevance", "unsupported_claim_rate"):
        score = value.get(name)
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not 0 <= float(score) <= 1
        ):
            raise ValueError(f"Judge score {name!r} must be between zero and one.")
    if not isinstance(value.get("rationale"), str):
        raise TypeError("Judge rationale must be a string.")


def _write_audit_sample(
    *,
    path: Path,
    trials: list[object],
    judgments: list[dict[str, Any]],
    audit_size: int,
    run_id: str,
) -> None:
    ranked = sorted(
        range(len(trials)),
        key=lambda index: hashlib.sha256(
            f"{run_id}:{index}".encode()
        ).hexdigest(),
    )[:audit_size]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "blind_id",
                "case_id",
                "candidate_model",
                "candidate_mode",
                "judge_correctness",
                "judge_relevance",
                "judge_unsupported_claim_rate",
                "human_correctness",
                "human_relevance",
                "human_unsupported_claim_rate",
                "reviewer_notes",
            ],
        )
        writer.writeheader()
        for index in ranked:
            judgment = judgments[index]
            writer.writerow(
                {
                    "blind_id": judgment["blind_id"],
                    "case_id": judgment["case_id"],
                    "candidate_model": judgment["candidate_model"],
                    "candidate_mode": judgment["candidate_mode"],
                    "judge_correctness": judgment["correctness"],
                    "judge_relevance": judgment["relevance"],
                    "judge_unsupported_claim_rate": judgment[
                        "unsupported_claim_rate"
                    ],
                    "human_correctness": "",
                    "human_relevance": "",
                    "human_unsupported_claim_rate": "",
                    "reviewer_notes": "",
                }
            )


def _blind_id(run_id: str, position: int, trial: Mapping[str, Any]) -> str:
    identity = f"{run_id}:{position}:{trial.get('case_id')}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _load_report(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read evaluation report: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError("Evaluation report root must be an object.")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
