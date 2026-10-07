---
name: verify-external-id
description: Verify that a model ID, dataset ID, OpenRouter model ID, PyPI package or Docker image tag really exists BEFORE it appears in code, config, fixtures or docs. Use every time you are about to write an external identifier.
---

# Verify an external identifier

Invented IDs are the #1 failure HubScout exists to prevent — don't commit one ourselves.

| Kind | How to verify |
|---|---|
| HF model / dataset | Hugging Face MCP (model/dataset details) or `https://huggingface.co/api/models/<id>` / `/api/datasets/<id>` → 200 |
| OpenRouter model | `curl -s https://openrouter.ai/api/v1/models \| jq '.data[] \| select(.id=="<id>")'` (also check `supported_parameters` for `tools` / `structured_outputs`) |
| PyPI package | `curl -s https://pypi.org/pypi/<name>/json \| jq .info.version` → exists; note the latest version |
| Ollama model | the tag exists on ollama.com/library/<name>/tags, then `ollama pull` succeeds |
| Docker image tag | `docker manifest inspect <image>:<tag>` succeeds |
| Library API | check-library-docs skill |

Record the verification date in a comment or the commit body when it matters (e.g. model IDs in `.env.example`). If it can't be verified, don't use it; ask the user.
