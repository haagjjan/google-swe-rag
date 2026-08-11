# Testing and evaluation

The repository separates deterministic software verification from live model
behavior. This distinction matters: a passing unit suite does not prove that a
RAG system retrieves the best evidence or always produces a supported answer.

## Offline test suite

Run:

```bash
python -m pytest -q
```

The suite contains 42 tests and does not require an API key. Google client calls
are replaced with controlled fakes.

| Area | What is covered |
| --- | --- |
| Configuration | Environment loading, precedence, path resolution, numeric validation, secret-safe errors |
| Documents | Recursive and case-insensitive discovery, stable ordering, PDF validation, page provenance |
| Chunking | Size and overlap, stable IDs, punctuation, empty pages, provenance |
| Embeddings | Batching, document/query inputs, empty questions, dimensions, provider failures |
| Vector store | Round trips, metadata, matrix validation, duplicate IDs, cosine ranking, ties, invalid vectors |
| Generation | Evidence formatting, no-context behavior, grounding instruction, provider failures |
| RAG orchestration | Stored-model compatibility and query-to-answer integration |

## CI

GitHub Actions installs the package and runs the offline suite on Python 3.11
and 3.13. The workflow does not receive a Google API key and cannot spend model
quota.

## Completed live smoke test

The implementation was manually verified with two locally held chapter PDFs:

```text
2 PDFs discovered
37 pages extracted
59 chunks created
59 unique chunk IDs
768-dimensional vectors stored
```

One relevant question retrieved the expected pages and produced a cited answer.
One unrelated question produced the configured insufficient-evidence response.
The source PDFs, extracted chunks, and vector index are excluded from Git.

## What this evidence supports

The current evidence supports the claims that:

- the implemented stages connect end to end;
- configuration and data-contract failures are handled explicitly;
- persistence round-trips without losing chunk provenance;
- cosine retrieval ranks controlled synthetic vectors correctly;
- model requests are constructed with the intended task types and grounding
  instructions;
- the complete workflow has succeeded against the real provider.

## What is not yet proven

The current suite does not establish:

- retrieval recall across a representative question set;
- citation precision or answer faithfulness at scale;
- the best chunk size, overlap, or `TOP_K` value;
- robustness across complex layouts, tables, or scanned documents;
- latency and cost distributions;
- concurrency or large-corpus performance.

## Next evaluation step

A stronger evaluation would create a versioned dataset with approximately
50-100 questions, expected source pages, answerability labels, and reference
facts. It could then report:

- retrieval hit rate at k;
- mean reciprocal rank;
- answerable versus unanswerable classification;
- citation precision;
- grounded-fact coverage;
- latency and estimated API cost.

Those metrics should be recorded separately for retrieval and generation so a
fluent answer cannot hide poor evidence selection.
