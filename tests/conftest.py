"""Shared test setup: keep every test hermetic from the developer's real `.env`."""

from collections.abc import Iterator

import pytest

from app import config


@pytest.fixture(autouse=True)
def _isolate_env_file(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Point the default env file at a path that never exists and reset cached settings."""
    missing = tmp_path_factory.mktemp("no-env") / ".env"
    original = config.DEFAULT_ENV_FILE
    config.DEFAULT_ENV_FILE = missing
    config.get_settings.cache_clear()
    yield
    config.DEFAULT_ENV_FILE = original
    config.get_settings.cache_clear()
