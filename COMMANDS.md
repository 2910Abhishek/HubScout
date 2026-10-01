# HubScout commands

Run everything from the project folder:

```bash
cd ~/Documents/Projects/HubScout
```

## 1. One-time setup

```bash
docker context use default   # use Docker CE (not Docker Desktop); redo if Docker Desktop resets it
uv sync                      # install Python dependencies into .venv
make env                     # create .env (your API keys) and .env.infra (local passwords)
# then open .env and paste your API keys
uv run python scripts/check_env.py   # shows which keys are set (never prints values)
```

## 2. Start the project

```bash
make up        # start Docker services: Postgres :5433, Redis :6379, SearXNG :8888
make studio    # start the HubScout agent server on :2024 (keep this terminal open)
```

Then open LangGraph Studio in your browser and choose the **hubscout** graph:
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

Each run saves its README in the `outputs/` folder.

## 3. Stop the project

```bash
# In the terminal running `make studio`: press Ctrl+C to stop the agent server
make down      # stop the Docker services (your data is kept)
```

## 4. Check status

```bash
make ps                                  # Docker services and their health
curl -s localhost:2024/ok                # agent server alive? prints {"ok":true}
curl -s localhost:11434/api/version      # Ollama (local fallback LLM) alive?
make prereqs                             # check all system prerequisites
```

## 5. Optional: tracing with Langfuse (uses ~3 GB RAM)

```bash
make up-obs    # start core services + Langfuse UI on http://localhost:3000
make down-obs  # stop everything, including Langfuse
```

## 6. Quality checks (what CI runs)

```bash
make test      # unit tests (offline, no API calls)
make check     # lint + type check + tests + security audit
```

## 7. Troubleshooting

```bash
make logs                                  # follow Docker service logs (Ctrl+C to exit)
make logs SERVICE=redis                    # logs for one service
sudo systemctl restart ollama              # restart the local LLM if it stops answering
docker context use default                 # "Cannot connect to the Docker daemon" fix
```

> Free-tier note: OpenRouter allows 50 LLM calls per day (resets 05:30 IST). After that,
> HubScout automatically uses the local Ollama model, so runs work but are slower.
