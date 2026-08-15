"""Small structured-logging helpers for benchmark and demo operations."""

import json
import logging
from collections.abc import Mapping


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    run_id: str | None = None,
    fields: Mapping[str, object] | None = None,
) -> None:
    """Emit one stable JSON object without credentials or document text."""
    payload: dict[str, object] = {"event": event}
    if run_id:
        payload["run_id"] = run_id
    if fields:
        payload.update(fields)
    logger.info(json.dumps(payload, sort_keys=True, ensure_ascii=True))
