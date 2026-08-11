# RAG pipeline implementation log

Date: 30 July 2026

## My goal

I wanted the initial scaffold to become a complete, understandable RAG
pipeline. Not a collection of mostly filled-in files that only looked
finished. My target workflow was:

```text
PDFs
  → page-aware extraction
  → overlapping chunks
  → Google document embeddings
  → local vector index
  → Google query embedding
  → cosine retrieval
  → grounded Google answer with sources
```

I deliberately kept the original learning-oriented architecture. I did not
introduce LangChain, a hosted vector database, or an abstraction over multiple
model providers. Google AI is the only model provider in this project.

## What I found at the start

Most modules expressed the intended idea, but they were not yet connected into
a runnable or verified application.

### Configuration defects

- `config.py` imported `detenv.load_detenv`, which does not exist in the
  declared dependencies.
- The implementation then called `load_dotenv`, which had never been
  imported.
- Invalid integer values leaked raw Python conversion errors instead of clear
  configuration messages.
- An explicitly supplied missing environment file was silently ignored.
- The example environment file did not contain usable model defaults.

### PDF extraction defects

- The output list was declared as `extract_pdf_pages` but the function tried
  to append to and return `extracted_pages`.
- The loop variable was `pagenumber`, while the constructor used
  `page_number`.
- `ExtractedPage` requires `source_path`, but the extractor passed the
  nonexistent `source_filename` argument.
- Several validation messages described files as folders or contained
  misleading wording.

The first real PDF therefore crashed immediately with a `NameError`.

### Chunking defects

- The module imported Hugging Face `transformers`, but that dependency was
  absent from `requirements.txt`.
- `chunk_pages` used a `tokenizer` variable that was not in its function
  signature.
- `indexing.py` passed a `tokenizer` argument that `chunk_pages` did not
  accept.
- `indexing.py` expected `settings.tokenizer_model`, which did not exist in
  `Settings`.

This meant the declared dependency set could never import the chunking layer.

### Embedding defects

- `embed_question` requested the vector but never returned it.
- Provider errors were not consistently translated into useful
  pipeline-level errors.
- There was no request batching for a larger set of chunks.
- Empty inputs and mismatched response counts were not fully guarded.

### Vector-store defects

- `load_index` was defined twice in succession, leaving a dead first
  definition and making the file difficult to reason about.
- Stored metadata was trusted without structural validation.
- Corrupt or mismatched count, shape, model, and dimension information did
  not all fail at the clearest boundary.

### Orchestration and CLI defects

- Indexing depended on the broken tokenizer contract.
- The CLI could not import because configuration import failed first.
- A successful index build did not report document, page, chunk, dimension,
  or output-path details.
- Runtime failures produced developer-facing tracebacks instead of concise
  operator errors.
- There was no installable console command or package metadata.

### Verification and documentation gaps

- The three test files only contained TODO comments, so `pytest` collected
  zero tests.
- The README still said the repository was only a scaffold.
- There was no complete setup-to-answer instruction manual.
- There was no record of implementation decisions or known limitations.

## How I completed the repository

### 1. I made configuration deterministic and secret-safe

I switched to the declared `python-dotenv` package and made `.env` resolution
project-relative by default. An explicit `--env-file` is resolved and checked
before use. Shell environment values take precedence over file values, which
also makes CI and temporary overrides possible.

All required strings and numeric settings now have focused validation. Error
messages name the setting but never include the Google API key.

I populated `.env.example` with verified defaults:

```dotenv
GENERATION_MODEL=gemma-4-26b-a4b-it
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSION=768
```

The working credential remained in an ignored local `.env` file and was never
printed or committed.

### 2. I repaired PDF discovery and extraction

PDF discovery now:

- validates that the configured path exists and is a directory;
- discovers `.pdf` files recursively and case-insensitively;
- sorts by relative path for deterministic indexing.

Extraction now:

- validates the input path and extension;
- rejects encrypted PDFs clearly;
- preserves one-based physical page numbers;
- constructs the actual `ExtractedPage` schema with a resolved source path;
- retains empty pages while allowing chunking to ignore them;
- wraps page-specific extraction failures with filename and page context.

### 3. I replaced the unusable tokenizer dependency

Google does not publish a local tokenizer for
`gemini-embedding-001`. Adding an unrelated tokenizer would make token counts
look precise when they were not, and using Google token-count API calls during
every chunking decision would make indexing slow and network-dependent.

I therefore implemented a deterministic local lexical tokenizer with Python
regular expressions. It recognizes words, word-internal hyphens/apostrophes,
and punctuation. Chunks are slices of the original text, so their punctuation
and internal whitespace are preserved.

The existing `CHUNK_SIZE_TOKENS` and `CHUNK_OVERLAP_TOKENS` names remain for
configuration compatibility, but the documentation explicitly states that
they are lexical tokens, not provider billing tokens. A 512-token chunk is
conservative relative to the embedding input limit.

Every chunk retains:

- a stable hash-based chunk ID;
- source filename and source path;
- page number and optional section;
- chunk index;
- token start, end, and count.

### 4. I completed Google embedding integration

Document chunks use the `RETRIEVAL_DOCUMENT` task type and questions use
`RETRIEVAL_QUERY`. Both paths share the same model and optional output
dimension.

I fixed the missing query-vector return and added:

- whitespace and empty-input validation;
- batches of at most 100 inputs;
- response-count validation;
- consistent dimension validation;
- errors that identify the selected model without disclosing credentials.

### 5. I hardened local persistence and retrieval

The vector store still uses one compressed NumPy file because that is
appropriate for this learning-scale repository.

I cleaned the duplicate function definition and retained atomic persistence:
the new index is written to a temporary file, flushed, and then replaces the
old index.

On load, I now verify:

- index format version;
- required arrays;
- JSON metadata shape;
- embedding model and dimension;
- chunk count;
- vector-matrix shape;
- finite numeric values;
- unique chunk IDs.

Cosine retrieval validates the query shape, dimension, finite values, and
nonzero norms. Equal scores keep original index order, which makes results
deterministic.

### 6. I completed grounded generation

The generation prompt requires the model to:

- use only supplied excerpts;
- cite `[Source N]` labels;
- ignore instructions found inside source text;
- return one exact insufficient-information sentence when evidence is
  inadequate.

I removed the explicit `temperature` option. Google deprecated sampling
parameters for newer model families, and the grounding rules already control
the behavior needed here.

### 7. I made the project installable and operable

I added `pyproject.toml` with:

- Python 3.11+ metadata;
- runtime and development dependencies;
- `src/` package discovery;
- pytest configuration;
- the `swe-google-rag` console command.

The `index` command now reports the PDF, page, and chunk counts, embedding
dimension, and saved path. The `ask` command prints the answer plus each
retrieved source, physical page, and similarity score. Expected runtime
failures are shown as concise `Error:` messages.

### 8. I replaced placeholders with real tests

I created 42 offline tests covering:

- environment loading, precedence, path resolution, and invalid values;
- lexical chunk bounds, overlap, stable IDs, and provenance;
- recursive PDF discovery and extraction;
- embedding batching, query returns, dimensions, and provider failures;
- index round trips, malformed inputs, ranking, ties, and vector errors;
- context formatting, no-context behavior, grounded prompts, and generation
  failures;
- query-time retrieval and generation orchestration.

The tests mock Google API boundaries, so routine test runs do not use quota.

## My final verification

I verified the code in a fresh `.venv` and installed it as an editable package.

Local verification:

```text
42 passed
2 PDFs discovered
37 pages extracted
37 non-empty pages
59 chunks created
59 unique chunk IDs
512 maximum lexical tokens per chunk
```

Live Google verification:

```text
gemini-embedding-001 → 768-dimensional vector returned
gemma-4-26b-a4b-it → non-empty grounded response returned
full index → index.npz written successfully
```

For the relevant question, “What are the main differences between programming
and software engineering?”, the pipeline retrieved Chapter 1 pages 1, 2, and
6 and answered with citations. The top similarity score was `0.7447`.

For the unrelated question, “What is the recipe for baking a sourdough
loaf?”, the pipeline responded:

> The retrieved documents do not contain enough information to answer this
> question.

This verifies both the normal grounded-answer path and the refusal path.

## What remains deliberately limited

These are bounded follow-up improvements, not blockers for the current
pipeline:

- Scanned/image-only PDFs require a future OCR stage.
- Lexical token counts approximate model tokens.
- Adding or changing a PDF requires a full `index` rebuild.
- Retrieval has no similarity threshold, hybrid keyword search, or reranker.
- Section names are not inferred automatically.
- The local NumPy index is not intended for concurrent or very large
  production workloads.
- There is no conversation history or graphical interface.

The pipeline now works completely for its stated scope: extractable PDFs,
local dense retrieval, and grounded single-question answers through Google AI.
