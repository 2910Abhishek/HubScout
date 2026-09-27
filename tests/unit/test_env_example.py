"""`.env.example` lists only API keys, every key is a real setting, and it has no values."""

import re

from pydantic import SecretStr
from pydantic_settings import BaseSettings

from app.config import PROJECT_ROOT, Settings

ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")


def _example_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text().splitlines():
        if match := ASSIGNMENT.match(line):
            assert match.group(1) not in entries, f"duplicate key {match.group(1)}"
            entries[match.group(1)] = match.group(2)
    return entries


def _secret_env_names() -> set[str]:
    names: set[str] = set()
    for field in Settings.model_fields.values():
        group = field.annotation
        assert isinstance(group, type)
        assert issubclass(group, BaseSettings)
        prefix = group.model_config.get("env_prefix", "")
        for name, info in group.model_fields.items():
            if info.annotation in (SecretStr, SecretStr | None):
                names.add(f"{prefix}{name}".upper())
    return names


def test_every_key_is_a_secret_setting() -> None:
    unknown = _example_entries().keys() - _secret_env_names()
    assert not unknown, f"not secret settings (put defaults in app/config.py): {sorted(unknown)}"


def test_required_api_keys_are_listed() -> None:
    required = {"OPENROUTER_API_KEY", "ARTIFICIAL_ANALYSIS_API_KEY", "LANGSMITH_API_KEY"}
    assert required <= _example_entries().keys()


def test_example_has_no_values() -> None:
    filled = {key for key, value in _example_entries().items() if value.strip()}
    assert not filled, f".env.example must not contain values: {sorted(filled)}"
