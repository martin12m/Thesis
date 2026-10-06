# Context-Aware E-Commerce Chatbot Backend

A self-hosted Python backend that answers customer questions about a product catalog, by text or voice. It assembles the right context for every turn (session, catalog, documents, business rules) and compares three ways of answering.

## What it does

- **One pipeline for text and voice.** `POST /chat` takes JSON. `POST /chat/voice` takes an audio file, transcribes it, then runs the exact same pipeline. Optional speech output.
- **One context mechanism.** Session history, catalog data, policy documents and business-rule tools are all reduced to a common shape and filtered per turn by a single `ContextManager`.
- **Three interchangeable reasoning variants:**

| Variant | How it answers |
|---|---|
| `retrieval` | Hybrid search (BM25 + local embeddings), then the model answers from the retrieved text |
| `tool_first` | Rule-based routing to one catalog/policy tool; the model only verbalizes its output |
| `hybrid` | Runs every matching tool and retrieval, then answers from both |

- **Swappable models.** Run a local GGUF model or any OpenAI-compatible endpoint by changing environment variables, no code changes.
- **Built-in evaluation.** A scenario runner scores every variant on every model: deterministic metrics, an LLM judge, p50/p95 latency and reliability.

## How a request flows

```
POST /chat  ──────────────┐
POST /chat/voice ─ STT ───┤
                          ▼
                  run_chat_pipeline
                          │
                  variant router (rules)
                          │
        ┌─────────────────┼─────────────────┐
    retrieval         tool_first          hybrid
        └─────────────────┼─────────────────┘
                          ▼
                    context manager
                          │
                       LLM backend
                          │
                   answer + trace  ─ optional TTS
```

## Layout

```
api/            FastAPI app and routes
orchestration/  pipeline, variant router, context manager, structured routing
variants/       retrieval, tool_first, hybrid
retrieval/      SQLite FTS5 (BM25) + ONNX embeddings
tools/          catalog and policy tools
data/           data sources, loaders, fixtures
llm/            model backends (llama.cpp, OpenAI-compatible)
integrations/   optional external product-reviews API client
speech/         STT (faster-whisper) and TTS (Piper)
evaluation/     scoring, LLM judge, scenario runner, scenarios
scripts/        evaluation CLI, database seeding
results/        raw evaluation output (CSV, JSON, logs, audio)
tests/          pytest suite
```

## Setup

Requires Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in the <...> values
```

Seed the catalog and build the search index:

```bash
python -m scripts.seed_data --db data/catalog.db \
  --products data/fixtures/products.json --policies data/fixtures/policies.json
python -c "from retrieval.indexer import rebuild_index; rebuild_index('data/catalog.db')"
```

Always build the index from a database seeded by `scripts.seed_data`: it expects the full schema, including the brand, rating and review columns. Re-seed any older database first.

Voice needs a Piper voice model:

```bash
python -m piper.download_voices en_US-lessac-medium --download-dir models/piper_voices
```

## Run

```bash
uvicorn api.app:app --reload
```

```bash
curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is your return policy?"}'

curl -X POST localhost:8000/chat/voice -F "audio=@question.wav"
```

Every response includes a trace: chosen variant, routing rule, tools executed, evidence used and active topic, plus transcription and synthesis latency on the voice path.

Example response (trimmed):

```json
{
  "message": "Running shoes can be returned within 30 days of delivery if they are in resellable condition.",
  "tool_calls": [{"tool_name": "get_return_policy_info", "arguments": {}, "status": "completed"}],
  "metadata": {
    "status": "ok",
    "trace": {
      "variant": "tool_first",
      "active_topic": "policy",
      "session_summary": "What is your return policy?",
      "routing_decision": {
        "variant": "tool_first",
        "rule": "query requests structured facts that map to tools",
        "combined": false
      }
    }
  }
}
```


## Models

Set the backend in `.env`:

- **Hosted / OpenAI-compatible:** `LLM_BACKEND=openai_compatible`, plus `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.
- **Local:** `LLM_BACKEND=llama_cpp` and `MODEL_PATH` pointing to a `.gguf` file. Download one from [Hugging Face](https://huggingface.co/models?library=gguf&sort=trending) into `models/`; model files are git-ignored and not part of the repo.
- **Reasoning models:** use `LLM_EXTRA_PARAMS` to suppress hidden reasoning, otherwise it can eat the whole token budget and return empty answers.

Compare several models with named profiles (`LLM_PROFILES` and `LLM_PROFILE_<NAME>_<FIELD>`, see `.env.example`).

## Evaluate

```bash
python scripts/run_evaluation.py \
  --scenarios evaluation/scenarios/context_categories_suite.json \
  --output results/my_run.csv \
  --model qwen,gptoss20b \
  --judge cross \
  --simulate 20
```

- `--model`: comma-separated profile names (`all` for everything configured).
- `--judge cross`: each model is judged by a different one. `self` and `none` also work.
- `--simulate N`: adds N LLM-generated scenarios. They have no ground truth, so only the judge scores them meaningfully.

Writes the results CSV plus latency-percentile and success-rate summaries.

## Results

`results/` holds the raw output of the evaluation runs.

| Files | Contents |
|---|---|
| `results_qwen_gptoss120b_sim150*` | qwen and gpt-oss-120b, 158 scenarios x 3 variants |
| `results_ministral8b_14b_sim150*` | ministral-8b and ministral-14b, same 158 scenarios |
| `results_groq_pair*`, `results_mistral_trio*` | gpt-oss-20b and the Ministral models on the 28-scenario set |
| `results_llama32*`, `results_phi4mini*` | the two local GGUF models on the 28-scenario set |
| `*_ratelimited_run*` | a rerun distorted by rate-limit backoff, kept for reference |
| `*_simulated_scenarios.json`, `combined_scenarios_28.json` | the scenario sets used |
| `voice_demo_v2/` | speech-to-text and text-to-speech run: results JSON and audio |

Each run has three files: the row-level results, `_latency_summary.csv` (p50/p95) and `_success_rate.csv`.

## Test

```bash
pytest
```

## Scope

A research backend, not a production service: no authentication, rate limiting or frontend. Routing is rule-based by design, so unusual phrasing can miss a rule. Session state is carried inside the pipeline and is not yet returned to HTTP clients.
