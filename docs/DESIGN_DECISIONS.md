# Design decisions

This record explains why the project looks the way it does. The aim is not to
claim that each choice is globally optimal, but to make scope and trade-offs
explicit.

## 1. Implement the RAG primitives directly

**Status:** Accepted

The project uses the Google Gen AI SDK, NumPy, pypdf, and standard Python code
instead of LangChain or another orchestration framework.

**Why:** The learning objective was to understand the boundaries between
document extraction, chunking, embedding, retrieval, prompt construction, and
generation. Hiding those boundaries behind a framework would have made the
first implementation faster but less educational.

**Trade-off:** More application code and fewer ready-made integrations.

## 2. Support one model provider

**Status:** Accepted

Google AI is the only provider. The application does not introduce a provider
interface merely to make the code appear flexible.

**Why:** There is no current requirement to switch providers, and an unused
abstraction would add complexity without proving portability.

**Trade-off:** Changing provider requires implementation work.

## 3. Use deterministic lexical chunking

**Status:** Accepted

Chunk limits use a local regular-expression tokenizer that recognizes words,
word-internal hyphens/apostrophes, and punctuation.

**Why:** The selected embedding API does not expose a matching local tokenizer.
Using an unrelated tokenizer would suggest false precision; calling a remote
token-count endpoint repeatedly would make chunking network-dependent.

**Trade-off:** `CHUNK_SIZE_TOKENS` represents lexical tokens, not provider
billing tokens. Chunk configuration must remain conservative.

## 4. Preserve page boundaries

**Status:** Accepted

Chunks never span extracted PDF pages.

**Why:** Physical page provenance makes the returned evidence easier to inspect
and cite.

**Trade-off:** Short page endings can produce small chunks, and content spanning
two pages is not combined automatically.

## 5. Store vectors in a compressed NumPy file

**Status:** Accepted for the current scale

Vectors and metadata are persisted in one versioned `index.npz` file.

**Why:** The reference corpus is small, local, and single-user. NumPy keeps the
storage format and exact retrieval implementation understandable.

**Trade-off:** Full-memory loading, linear search, no concurrent writes, no
metadata query engine, and no access-control layer.

## 6. Write the index atomically

**Status:** Accepted

The application writes a temporary file, flushes it, and replaces the active
index only after the new file is complete.

**Why:** A failed or interrupted write should not leave a half-written active
index.

**Trade-off:** A rebuild temporarily requires enough disk space for both files.

## 7. Use exact cosine similarity as the baseline

**Status:** Accepted for the current scale

Retrieval computes exact cosine similarity over all vectors.

**Why:** Exact search is deterministic, easy to test, and sufficient for tens or
hundreds of chunks.

**Trade-off:** It will not scale to large corpora and does not address lexical
matching, metadata filtering, or reranking.

## 8. Perform complete index rebuilds

**Status:** Accepted for the current scale

Any source or chunking change requires rebuilding the index.

**Why:** Full rebuilds avoid document identity, deletion, stale-chunk, and
partial-update complexity while the corpus is small.

**Trade-off:** Re-embedding unchanged documents consumes time and API quota.

## 9. Keep generation deterministic through rules, not sampling settings

**Status:** Accepted

The generation request relies on explicit grounding instructions and a fixed
insufficient-evidence response. It does not force provider sampling parameters
that may be unsupported across model families.

**Trade-off:** The provider can still produce variable wording, and prompting
does not guarantee citation correctness.

## 10. Keep tests offline by default

**Status:** Accepted

Tests replace the Google client at the application boundary.

**Why:** Unit and orchestration tests should be fast, deterministic, secret-free,
and usable in pull requests from forks.

**Trade-off:** Provider availability and live model behavior require separate,
intentional smoke tests.

## 11. Do not distribute reference documents or derived chunks

**Status:** Accepted

The repository ignores source PDFs, generated vector data, and extracted text.

**Why:** The code license does not grant redistribution rights for third-party
documents. It also prevents local absolute paths and potentially sensitive
content from entering Git history.

**Trade-off:** A new user must supply an authorized PDF before running the full
workflow.
