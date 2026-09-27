"""Report which env keys are set and whether settings validate, WITHOUT printing any values.

Used instead of reading the env files: it shows SET/empty per key and only the field names
and error types of validation failures (pydantic error messages can echo input values).
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import dotenv_values
from pydantic import ValidationError

from app.config import PROJECT_ROOT, load_settings

ENV = PROJECT_ROOT / ".env"
INFRA = PROJECT_ROOT / ".env.infra"
REQUIRED = ["OPENROUTER_API_KEY", "ARTIFICIAL_ANALYSIS_API_KEY", "LANGSMITH_API_KEY"]


def _report(path: Path, required: list[str]) -> list[str]:
    if not path.exists():
        print(f"{path.name}: MISSING (run scripts/init_env.sh)")
        return [f"{path.name} file"]
    values = dotenv_values(path)
    print(f"{path.name}: {sum(1 for v in values.values() if v)} of {len(values)} keys set")
    for key in values:
        tag = " (required)" if key in required else ""
        print(f"  {'SET' if values[key] else 'empty':5}  {key}{tag}")
    return [key for key in required if not values.get(key)]


def main() -> int:
    missing = _report(ENV, REQUIRED) + _report(INFRA, [])
    try:
        settings = load_settings()
        settings.infra.database_url  # noqa: B018 - raises if the Postgres password is missing
    except ValidationError as exc:
        print("settings: INVALID")
        for err in exc.errors(include_input=False, include_url=False):
            print(f"  {'.'.join(map(str, err['loc']))}: {err['type']}")
        return 1
    except RuntimeError as exc:
        print(f"settings: {exc}")
        return 1
    print("settings: valid")
    if missing:
        print(f"missing: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
