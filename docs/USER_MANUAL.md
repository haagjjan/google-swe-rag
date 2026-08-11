# User manual

This guide takes a new user from a clean checkout to a complete
PDF-indexing and question-answering workflow.

## 1. Prerequisites

You need:

- macOS, Linux, or another environment with a shell;
- Python 3.11 or newer;
- a Google AI Studio API key with access to the selected generation and
  embedding models;
- at least one text-based PDF.

The repository defaults are:

- `gemma-4-26b-a4b-it` for grounded answers;
- `gemini-embedding-001` for document and question embeddings.

Model availability can differ by Google account, project, region, and current
API policy. If a default is unavailable, replace it with a compatible model ID
available to your key.

## 2. Open the project

From the cloned repository's parent directory:

```bash
cd google-swe-rag
```

All following commands assume this is the current directory.

## 3. Create and activate a virtual environment

Standard Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Or with `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

The editable installation creates the `swe-google-rag` command and includes
pytest for local verification.

Whenever a new terminal is opened, return to this directory and reactivate:

```bash
source .venv/bin/activate
```

## 4. Configure Google AI

Create the ignored local environment file:

```bash
cp .env.example .env
```

Open `.env` in an editor and set only the credential value:

```dotenv
GOOGLE_API_KEY=your_personal_google_ai_studio_key
```

Do not put quotes around the key unless the key itself requires them. Never
commit `.env` or paste its contents into issue reports.

The copied file includes the configuration used for the verified reference run:

```dotenv
GENERATION_MODEL=gemma-4-26b-a4b-it
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSION=768
PDF_STORAGE_PATH=data/pdfs
VECTOR_STORE_PATH=data/vector_store
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_TOKENS=64
TOP_K=3
```

### Configuration reference

| Variable | Required | Meaning |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Yes | Google AI Studio credential used by the SDK |
| `GENERATION_MODEL` | Yes | Google model used for the grounded answer |
| `EMBEDDING_MODEL` | Yes | Google model used for both documents and questions |
| `EMBEDDING_DIMENSION` | No | Requested vector size; blank uses the model default |
| `PDF_STORAGE_PATH` | Yes | Directory scanned recursively for PDFs |
| `VECTOR_STORE_PATH` | Yes | Directory containing `index.npz` |
| `CHUNK_SIZE_TOKENS` | Yes | Maximum lexical tokens in a chunk |
| `CHUNK_OVERLAP_TOKENS` | Yes | Lexical tokens shared by adjacent page chunks |
| `TOP_K` | Yes | Maximum chunks supplied to answer generation |

Relative data paths are resolved from the installed package's project root,
not from the terminal's current directory or a custom environment file's
directory.

`CHUNK_OVERLAP_TOKENS` must be smaller than `CHUNK_SIZE_TOKENS`. All numeric
values must be whole numbers; size, dimension, and `TOP_K` must be positive.

## 5. Add source PDFs

Place one or more PDFs in:

```text
data/pdfs/
```

Subdirectories are allowed. Discovery is recursive and `.pdf` matching is
case-insensitive.

The repository does not contain source PDFs. Input documents are intentionally
ignored by Git and must be supplied locally by each user.

The extractor supports PDFs with an existing text layer. Scanned pages that
contain only images need OCR before they can be indexed. Encrypted PDFs are
rejected.

## 6. Run the offline tests

Before using API quota:

```bash
python -m pytest -q
```

Expected result:

```text
42 passed
```

A deprecation warning originating inside a dependency may appear on newer
Python versions; it does not indicate a pipeline failure.

## 7. Build the vector index

Run:

```bash
swe-google-rag index
```

For the two locally held chapters used in the reference run, the verified
output was:

```text
Indexed 2 PDF(s), 37 page(s), and 59 chunk(s).
Embedding dimension: 768. Index: .../data/vector_store/index.npz
```

Indexing makes Google embedding requests. The implementation submits at most
100 chunks in one batch and validates that the provider returns one
same-dimensional vector per chunk.

The resulting file is:

```text
data/vector_store/index.npz
```

It contains:

- the float vector matrix;
- chunk text;
- source filename and path;
- physical PDF page number;
- optional section;
- chunk ID and chunk positions;
- embedding model and dimension;
- index format version.

The write is atomic: a fully written temporary file replaces the previous
index only after the new data is flushed.

## 8. Ask a question

Quote the complete question:

```bash
swe-google-rag ask "What are the main differences between programming and software engineering?"
```

The command prints:

1. the grounded answer with `[Source N]` citations;
2. the retrieved filename and physical PDF page for each source;
3. the cosine similarity score for each source.

Example source output:

```text
[Source 1] SWE-at-Google-Ch1.pdf, page 1, score=0.7447
```

The score ranks retrieved chunks for this question. It is not a probability
or a guarantee that the answer is supported. The generation prompt separately
checks whether the retrieved excerpts contain enough evidence.

For an unrelated question, the expected answer is:

```text
The retrieved documents do not contain enough information to answer this question.
```

Sources are still displayed in that case so the retrieval behavior remains
inspectable.

## 9. Repeat the normal workflow

For questions over the same unchanged PDFs:

```bash
swe-google-rag ask "Your next question"
```

The documents are not embedded again. Only the new question is embedded, then
the saved vectors are searched.

After adding, replacing, or removing any PDF, rebuild:

```bash
swe-google-rag index
```

The current implementation performs a complete rebuild and atomically
replaces the old index.

After changing `EMBEDDING_MODEL` or `EMBEDDING_DIMENSION`, rebuild the index.
If the stored and configured values differ, question answering stops with a
clear compatibility error instead of comparing incompatible vectors.

Changing only `GENERATION_MODEL`, `TOP_K`, or the API key does not require a
rebuild.

Changing chunk size or overlap should be followed by a rebuild because those
settings determine the stored evidence units.

## 10. Use a different environment file

Pass `--env-file` before the subcommand:

```bash
swe-google-rag --env-file .env.experiment index
swe-google-rag --env-file .env.experiment ask "What does the chapter say about time?"
```

An explicitly supplied file must exist. Values already exported in the shell
take precedence over values in that file.

## 11. Run without the console command

When the virtual environment is active but the editable package has not been
installed, use:

```bash
PYTHONPATH=src python -m swe_google_rag.main index
PYTHONPATH=src python -m swe_google_rag.main ask "Your question"
```

Installing with `python -m pip install -e ".[dev]"` is the recommended path.

## 12. Troubleshooting

### `Missing required environment variable: GOOGLE_API_KEY`

Confirm that `.env` exists in this project and contains a non-empty
`GOOGLE_API_KEY`. Do not print the key while diagnosing.

### `Environment file does not exist`

The path passed to `--env-file` is wrong. Correct the path or omit the option
to use the project `.env`.

### `No PDF files found`

Add a `.pdf` file below `data/pdfs/`, or correct `PDF_STORAGE_PATH`.

### `Encrypted PDFs are not supported`

Create a decrypted local copy that you are authorized to use, place it in the
PDF folder, and rebuild.

### `No text chunks were created`

The PDFs probably contain scanned images without a text layer. Run OCR outside
this pipeline, replace the files, and rebuild.

### `Vector index does not exist ... Run 'index' first`

Run:

```bash
swe-google-rag index
```

### `configured embedding model does not match`

The `.env` embedding configuration changed after indexing. Restore the old
model/dimension or rebuild the index with the new values.

### `embedding request failed` or `grounded-answer generation request failed`

Check:

- that the Google key is valid;
- that the configured model is available to that key;
- Google API quota and rate limits;
- network connectivity;
- the exact model IDs.

The verified defaults follow the official
[Google embedding API](https://ai.google.dev/api/embeddings) and
[Gemma Gemini API guide](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api).

### The answer cites an irrelevant chunk

Try:

- asking a more specific question;
- increasing `TOP_K` moderately and asking again;
- reducing chunk size and rebuilding;
- confirming that the needed passage was extracted from the PDF.

Very large `TOP_K` values add weak context and can make answers worse.

### The answer uses `page unknown` or `section unknown`

Physical page numbers should be present for extracted PDFs. Sections are
currently not inferred, so `section unknown` is expected.

## 13. Security and data handling

- `.env`, PDFs, and `data/vector_store/*` are ignored by Git.
- Chunk text is sent to Google's embedding API while indexing.
- The retrieved chunk text and the user's question are sent to Google's
  generation API while asking.
- Vectors and source metadata are stored locally in `index.npz`.
- The project does not send requests to OpenAI.
- Deleting the local index is safe; it can be rebuilt from the PDFs.
