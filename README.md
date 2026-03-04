# claude-code-proxy

A modular proxy server that translates [Anthropic API](https://docs.anthropic.com/en/api) requests into calls to any LLM provider (OpenAI, Google Gemini, or native Anthropic) via [LiteLLM](https://github.com/BerriAI/litellm).

Point any tool that speaks the Anthropic Messages API (e.g. [Claude Code](https://docs.anthropic.com/en/docs/claude-code)) at this proxy and transparently route requests to your preferred backend.

## Features

- **Anthropic-compatible API** — drop-in `/v1/messages` and `/v1/messages/count_tokens` endpoints
- **Multi-provider** — OpenAI, Google Gemini (incl. Vertex AI), and native Anthropic backends
- **Extensible** — add new providers by subclassing `AbstractProvider` and registering them
- **Model mapping** — `haiku` and `sonnet` model names are automatically resolved to your configured models
- **Streaming** — full SSE streaming support with Anthropic-format events
- **Tool use** — Anthropic tool schemas are converted to OpenAI function-calling format, with provider-specific schema cleaning

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install & Run

```bash
# Clone
git clone https://github.com/sleepysoong/claude-code-proxy.git
cd claude-code-proxy

# Copy and edit environment variables
cp .env.example .env

# Install dependencies
uv sync

# Start the server
uv run python server.py
```

The server starts on `http://0.0.0.0:8082` by default.

### Docker

```bash
docker build -t claude-code-proxy .
docker run -p 8082:8082 --env-file .env claude-code-proxy
```

## Configuration

All configuration is via environment variables (or `.env` file). See [`.env.example`](.env.example) for the full list.

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | If using OpenAI | OpenAI API key |
| `GEMINI_API_KEY` | If using Gemini | Google AI Studio API key |
| `ANTHROPIC_API_KEY` | If using Anthropic | Anthropic API key |
| `PREFERRED_PROVIDER` | No | `openai` (default), `google`, or `anthropic` |
| `BIG_MODEL` | No | Model to map `sonnet` to (default: `gpt-4.1`) |
| `SMALL_MODEL` | No | Model to map `haiku` to (default: `gpt-4.1-mini`) |
| `OPENAI_BASE_URL` | No | Custom OpenAI-compatible base URL |
| `USE_VERTEX_AUTH` | No | Set `true` to use Vertex AI ADC instead of API key |
| `VERTEX_PROJECT` | If Vertex | GCP project ID |
| `VERTEX_LOCATION` | If Vertex | GCP region |

## Architecture

```
app/
├── main.py              # FastAPI app factory
├── config.py            # Environment variable loading
├── logging.py           # Logging setup & pretty request printer
├── models/
│   ├── request.py       # Pydantic models (MessagesRequest, etc.)
│   └── response.py      # Response models (MessagesResponse, Usage)
├── providers/
│   ├── base.py          # AbstractProvider ABC
│   ├── openai.py        # OpenAI provider (message flattening)
│   ├── gemini.py        # Gemini provider (schema cleaning, Vertex AI)
│   ├── anthropic.py     # Anthropic passthrough provider
│   └── registry.py      # ProviderRegistry factory + registration
├── converters/
│   ├── request.py       # Anthropic → LiteLLM request conversion
│   ├── response.py      # LiteLLM → Anthropic response conversion
│   └── streaming.py     # SSE streaming event generator
└── routers/
    └── messages.py      # /v1/messages endpoints
server.py                # Slim entry point (uvicorn)
```

### Adding a New Provider

1. Create `app/providers/your_provider.py`
2. Subclass `AbstractProvider` and implement all abstract methods
3. Register the instance in `app/providers/registry.py`

```python
from app.providers.base import AbstractProvider

class MistralProvider(AbstractProvider):
    def get_model_prefix(self) -> str:
        return "mistral"

    def get_supported_models(self) -> list[str]:
        return ["mistral-large-latest", "mistral-small-latest"]

    def configure_request(self, litellm_request):
        litellm_request["api_key"] = os.environ.get("MISTRAL_API_KEY")
        return litellm_request

    def preprocess_messages(self, messages):
        return messages  # or apply provider-specific transforms
```

## Usage with Claude Code

```bash
# Set the API base URL to point at the proxy
export ANTHROPIC_BASE_URL=http://localhost:8082
export ANTHROPIC_API_KEY=anything  # the proxy uses its own keys

claude
```

## License

MIT
