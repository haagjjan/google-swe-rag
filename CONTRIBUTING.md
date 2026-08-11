# Contributing

This repository is primarily a learning and portfolio project, but focused bug
reports and improvements are welcome.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the offline suite before submitting a change:

```bash
python -m pytest -q
```

## Change guidelines

- Keep the RAG stages explicit and independently testable.
- Add or update tests for behavior changes.
- Do not commit API keys, `.env` files, source PDFs, extracted document text,
  or generated vector indices.
- Avoid adding a framework or hosted service without documenting the trade-off
  in `docs/DESIGN_DECISIONS.md`.
- Preserve clear operator-facing errors at the CLI boundary.
- Keep provider calls mocked in the default test suite.

## Reporting problems

Include the Python version, command, sanitized error, and minimal reproduction.
Never paste an API key or confidential document content into an issue.
