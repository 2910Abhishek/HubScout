"""Shared test setup: keep every test hermetic from the developer's real env files."""

from collections.abc import Iterator

import pytest

from app import config


@pytest.fixture(autouse=True)
def _isolate_env_files(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Point the default env files at paths that never exist and reset cached settings."""
    missing = tmp_path_factory.mktemp("no-env")
    original = config.DEFAULT_ENV_FILES
    config.DEFAULT_ENV_FILES = (missing / ".env.infra", missing / ".env")
    config.get_settings.cache_clear()
    yield
    config.DEFAULT_ENV_FILES = original
    config.get_settings.cache_clear()
