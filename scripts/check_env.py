"""Report which .env keys are set and whether settings validate, WITHOUT printing any values.

Used instead of reading .env: it shows SET/empty per key and only the field names and
error types of validation failures (pydantic error messages can echo input values).
"""

from __future__ import annotations

import sys

from dotenv import dotenv_values
from pydantic import ValidationError

from app.config import DEFAULT_ENV_FILE, load_settings

REQUIRED_FOR_PHASE_1 = [
    "OPENROUTER_API_KEY",
    "ARTIFICIAL_ANALYSIS_API_KEY",
    "LANGSMITH_API_KEY",
    "LLM_MODEL_STRONG",
    "LLM_MODEL_CHEAP",
    "LLM_MODEL_JUDGE",
    "OLLAMA_MODEL",
    "DATABASE_URL",
]
RECOMMENDED = ["HF_TOKEN"]


def main() -> int:
    if not DEFAULT_ENV_FILE.exists():
        print(f"{DEFAULT_ENV_FILE} not found (create it with scripts/init_env.sh)")
        return 1

    values = dotenv_values(DEFAULT_ENV_FILE)
    missing = [k for k in REQUIRED_FOR_PHASE_1 if not values.get(k)]
    print(
        f"{DEFAULT_ENV_FILE.name}: {sum(1 for v in values.values() if v)} of {len(values)} keys set"
    )
    for key in REQUIRED_FOR_PHASE_1 + RECOMMENDED:
        state = "SET" if values.get(key) else "empty"
        tag = "required" if key in REQUIRED_FOR_PHASE_1 else "recommended"
        print(f"  {state:5}  {key}  ({tag})")

    try:
        load_settings()
    except ValidationError as exc:
        print("settings: INVALID")
        for err in exc.errors(include_input=False, include_url=False):
            print(f"  {'.'.join(map(str, err['loc']))}: {err['type']}")
        return 1
    print("settings: valid")
    if missing:
        print(f"missing required: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
