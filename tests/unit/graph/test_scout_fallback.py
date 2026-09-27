"""The scout degrades gracefully when the report step fails."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph.nodes.scout import repo_ids_in_tool_results


def test_repo_ids_extracted_from_tool_output_in_order_without_urls_or_duplicates() -> None:
    messages = [
        HumanMessage("find openai/not-from-a-tool"),
        AIMessage(content=""),
        ToolMessage(
            "1. openai/whisper-small (3M)\n2. vasista22/whisper-hindi-small.\n"
            "see https://huggingface.co/openai/whisper-small\n3. OpenAI/whisper-small",
            tool_call_id="c1",
        ),
    ]

    ids = [c.repo_id for c in repo_ids_in_tool_results(messages, limit=5)]

    assert ids == ["openai/whisper-small", "vasista22/whisper-hindi-small"]


def test_limit_is_respected() -> None:
    messages = [ToolMessage("org1/one org2/two org3/three", tool_call_id="c1")]

    assert len(repo_ids_in_tool_results(messages, limit=2)) == 2
