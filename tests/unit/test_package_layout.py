"""The package skeleton from the project card imports cleanly."""

import importlib

import pytest

PACKAGES = [
    "app",
    "app.schemas",
    "app.graph",
    "app.graph.nodes",
    "app.tools",
    "app.tools.registries",
    "app.tools.checks",
    "app.retrieval",
    "app.memory",
    "app.guardrails",
    "app.api",
    "app.report",
    "mcp_servers.arxiv_mcp",
    "mcp_servers.ml_insights_mcp",
    "verifier",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} should document its purpose"
