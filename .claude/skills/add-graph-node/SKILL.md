---
name: add-graph-node
description: Step-by-step recipe for adding a node to a HubScout LangGraph graph - schema, node with injected LLM, fake-LLM unit test, graph registration, langgraph.json registration, and a Studio load check. Use whenever creating or substantially changing a graph node.
---

# Add a graph node

1. **Docs**: run check-library-docs for any LangGraph API you will use (interrupt, Send, Command, reducers).
2. **Schema**: define the node's output as a Pydantic v2 model in `app/schemas/` (strict fields, validators). LLM output that feeds code MUST be parsed into it.
3. **State**: add only the fields the node writes to the graph state (`app/graph/state.py`), with reducers where lists accumulate.
4. **Node**: in `app/graph/nodes/<name>.py`, write a factory `make_<name>_node(llm: Runnable) -> Callable[[State], dict]`. The LLM is **injected**, never created inside the node. Return a partial state update. No hardcoded model names or URLs.
5. **Unit test**: `tests/unit/graph/test_<name>.py` using `GenericFakeChatModel` / a fake returning the structured object. Cover the happy path, invalid LLM output (validation error path), and each branch.
6. **Register in graph**: add the node and edges in the graph builder; production wiring uses `get_llm(tier)`.
7. **langgraph.json**: make sure the graph is registered (`"name": "./app/graph/<file>.py:graph"`).
8. **Studio check**: `uv run langgraph dev --no-browser` in the background → `/assistants/search` lists the graph → run one thread via the API → stop the server.
9. Run definition-of-done, then git-commit (`feat: add <name> node`).
