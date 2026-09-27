"""`.env.example` must document every setting and contain no values."""

import re
from pathlib import Path

from pydantic_settings import BaseSettings

from app.config import PROJECT_ROOT, Settings

ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")


def _example_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in Path(ENV_EXAMPLE).read_text().splitlines():
        match = ASSIGNMENT.match(line)
        if match:
            entries[match.group(1)] = match.group(2)
    return entries


def _settings_env_names() -> set[str]:
    names: set[str] = set()
    for field in Settings.model_fields.values():
        group = field.annotation
        assert isinstance(group, type)
        assert issubclass(group, BaseSettings)
        prefix = group.model_config.get("env_prefix", "")
        names.update(f"{prefix}{name}".upper() for name in group.model_fields)
    return names


def test_every_setting_is_documented() -> None:
    missing = _settings_env_names() - _example_entries().keys()
    assert not missing, f"add to .env.example: {sorted(missing)}"


def test_example_has_no_values() -> None:
    filled = {key for key, value in _example_entries().items() if value.strip()}
    assert not filled, f".env.example must not contain values: {sorted(filled)}"


def test_example_has_no_duplicate_keys() -> None:
    keys = [
        m.group(1) for line in ENV_EXAMPLE.read_text().splitlines() if (m := ASSIGNMENT.match(line))
    ]
    assert len(keys) == len(set(keys))
