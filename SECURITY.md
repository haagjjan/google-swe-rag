# Security and privacy

## Reporting a vulnerability

Please report security concerns through GitHub rather than posting secrets or
confidential source material in a public issue. Include a minimal sanitized
reproduction and the affected version or commit.

If a credential is ever exposed, revoke it with the provider immediately. Git
history cleanup does not make an exposed key safe to reuse.

## Data-flow warning

This is a local application, but it is not an offline application:

- PDF chunks are sent to Google's embedding API during indexing.
- The question and retrieved excerpts are sent to Google's generation API.
- The vector index and document metadata remain on the local filesystem.

Do not process confidential, regulated, or personal data until you have
reviewed the provider agreement, retention settings, access controls, and your
legal basis for sending that data to an external service.

## Repository hygiene

The repository ignores:

- `.env` and environment variants;
- `data/pdfs/*`;
- `data/vector_store/*`;
- virtual environments, caches, and build metadata.

The ignore rules are a last line of defense, not a substitute for reviewing
staged changes before every commit.
