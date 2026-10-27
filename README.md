# IARAG Chunking Service

FastAPI service that converts Markdown documents or plain text into token-counted chunks for downstream retrieval and indexing.

The service uses [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) by default for both semantic embeddings and token counting. Markdown can be split semantically within heading-delimited sections or deterministically by title and natural text boundaries.

## Features

- Semantic Markdown chunking with configurable token limits and rolling-window size.
- Deterministic Markdown chunking that preserves title boundaries and enforces a maximum token count.
- Semantic chunking for short plain text.
- BGE-M3 CLS-token pooling with normalized embeddings.
- English/Spanish language detection for Markdown inputs.
- JSON structured logs and a health endpoint.
- CPU fallback when CUDA is unavailable; the provided Compose deployment is configured for an NVIDIA GPU.

## API

The application listens on port `8084` in the container. Docker Compose publishes it only on the host loopback interface at `http://127.0.0.1:8085`.

Interactive OpenAPI documentation is available at `/docs` while the service is running.

### Health check

```http
GET /health
```

```json
{"status": "healthy"}
```

### Semantic Markdown chunking

```http
POST /services/chunking/doc
Content-Type: multipart/form-data
```

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `content_parsed` | string | yes | — | Markdown content to split. |
| `doc_id` | string | no | `null` | Identifier included in logs; it is not included in the response. |
| `min_split_tokens` | integer | no | `150` | Preferred minimum size of a semantic chunk. Must be at least 1 and smaller than `max_split_tokens`. |
| `max_split_tokens` | integer | no | `300` | Maximum semantic split size. Must be at least 1. |
| `split_tokens_tolerance` | integer | no | `10` | Token tolerance used by the rolling-window splitter. |
| `window_size` | integer | no | `5` | Number of units considered by the semantic rolling window. |

```bash
curl -X POST http://127.0.0.1:8085/services/chunking/doc \
  -F 'doc_id=example-001' \
  -F 'min_split_tokens=150' \
  -F 'max_split_tokens=300' \
  -F 'split_tokens_tolerance=10' \
  -F 'window_size=5' \
  --form-string $'content_parsed=# Introduction\nThis is the first section.\n\n# Results\nThese are the results.'
```

The input is first divided at Markdown headings. A section that already fits within `max_split_tokens` remains intact; a larger section is split semantically. An undersized final split is merged with or rebalanced against its predecessor when possible.

### Short-text semantic chunking

```http
POST /services/chunking/short_text
Content-Type: multipart/form-data
```

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `short_text` | string | yes | — | Plain text to split semantically. |
| `doc_id` | string | no | `null` | Identifier included in logs. |

This endpoint uses fixed splitter settings: 100 minimum tokens, 200 maximum tokens, a tolerance of 10, and a window size of 2.

```bash
curl -X POST http://127.0.0.1:8085/services/chunking/short_text \
  -F 'doc_id=note-001' \
  --form-string 'short_text=Text to divide into semantically coherent chunks.'
```

### Markdown chunking by title

```http
POST /services/chunking/doc_by_titles
Content-Type: multipart/form-data
```

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `content_parsed` | string | yes | — | Markdown content to split. |
| `doc_id` | string | no | `null` | Identifier included in logs. |
| `max_split_tokens` | integer | no | `800` | Hard maximum number of tokens per returned chunk. |

```bash
curl -X POST http://127.0.0.1:8085/services/chunking/doc_by_titles \
  -F 'doc_id=example-001' \
  -F 'max_split_tokens=800' \
  --form-string $'content_parsed=# Introduction\nIntroductory text.\n\n# Methods\nMethod details.'
```

This mode does not calculate semantic similarity. It splits each heading-delimited section at progressively finer natural boundaries—paragraphs, lines, sentences, then words—and falls back to token-bounded text slices when necessary.

### Response format

All chunking endpoints return the same structure:

```json
{
  "chunk_list": ["Introductory text.", "Method details."],
  "chunk_title": ["Introduction", "Methods"],
  "token_count": [5, 4],
  "chunk_size": 2,
  "chunk_language": "en",
  "output_path": ""
}
```

- `chunk_list`, `chunk_title`, and `token_count` are aligned by index for the two Markdown endpoints.
- `chunk_language` is `es` when the detected language is Spanish and `en` otherwise.
- The short-text endpoint currently returns an empty `chunk_title` list and an empty `chunk_language` value.
- Token counts use the tokenizer belonging to the configured model, without special tokens.
- Invalid form data or parameter constraints produce HTTP `422`; processing failures produce HTTP `503`.

## Run with Docker Compose

Requirements:

- Docker with the Compose plugin.
- NVIDIA Container Toolkit and a CUDA-capable GPU.
- An existing external Docker network named `n8n-net`.
- The host cache directory `/srv/iarag/services/vectorizer/cache`.

Create the shared network once if it does not already exist:

```bash
docker network create n8n-net
```

Build and start the service:

```bash
cd /srv/iarag/services/chunking
docker compose up -d --build
docker compose logs -f chunker
```

Verify it from the host:

```bash
curl http://127.0.0.1:8085/health
```

Other containers attached to `n8n-net` can call the service at `http://chunker:8084`. The published port is intentionally bound to `127.0.0.1`; it is not directly exposed on external host interfaces.

Stop the service with:

```bash
docker compose down
```

## Run locally

Python 3.12 is the container runtime and is the recommended local version.

```bash
cd /srv/iarag/services/chunking
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn main_API:app --host 0.0.0.0 --port 8084
```

On first startup, Hugging Face downloads the configured model unless it is already cached. The model is loaded while the application module is imported, so readiness may take longer than usual on the first run. When CUDA is not available, the code automatically selects CPU execution.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_NAME` | `BAAI/bge-m3` | Hugging Face model used by the encoder and tokenizer. Keep both on the same model so split limits and reported token counts remain consistent. |
| `HF_HOME` | Hugging Face default | Model/cache location. Compose sets this to `/data/hf` and mounts the vectorizer cache there. |

The Compose file also mounts `./tmp_shared/output_ocr` (via its absolute host path) at `/tmp/output_ocr`. The current HTTP endpoints return an empty `output_path` and do not write chunk results to that directory.

## Chunking behavior

Markdown headings are detected by a lightweight line parser:

- Any trimmed line beginning with `#` is treated as a heading; heading depth is not retained.
- Heading markers are removed and the clean heading is returned separately in `chunk_title`.
- Consecutive headings before body text are combined into one title.
- Content before the first heading becomes an introductory section with a single-space title (`" "`).
- A heading without body content may not produce a chunk.

Semantic requests to `/doc` and `/short_text` are serialized with an application-level lock so the shared model is used by one such request at a time. `/doc_by_titles` is non-semantic and does not use that lock.

## Tests

Run the unit tests from the service directory:

```bash
python -m unittest discover -s tests -v
```

The current tests cover the default model, BGE tokenizer-based token counting, and English/Spanish language detection.

## Project layout

```text
.
├── main_API.py             # FastAPI routes, validation, locking, and logging
├── main_chunking.py        # Encoder and chunking strategies
├── utils.py                # Markdown parsing, language detection, and token helpers
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container image
└── docker-compose.yml      # GPU deployment and network configuration
```
