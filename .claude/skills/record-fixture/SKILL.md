---
name: record-fixture
description: Capture a real external API response once (locally, via an integration run) into tests/fixtures/ with secrets scrubbed, so unit tests stay offline yet realistic. Use when a unit test needs Hub, OpenRouter, Artificial Analysis, PyPI, arXiv, GitHub or SearXNG data.
---

# Record a fixture

1. Write or run the matching `@pytest.mark.integration` test (or a small script under `scripts/`) that calls the real API once.
2. Save the raw response as JSON to `tests/fixtures/<source>/<descriptive_name>.json`.
3. **Scrub**: remove or replace Authorization headers, API keys, tokens, emails and user IDs. Trim huge arrays to what the test needs (keep realistic structure).
4. Add `"_recorded_at": "<ISO date>"` and `"_source": "<URL without secrets>"` at the top level.
5. Grep the fixture for secret patterns before committing (`sk-or-`, `hf_`, `Bearer`).
6. Unit tests load it via a shared helper / `respx` route. They never hit the network.
7. Commit with `test: record <source> fixture for <case>`.
