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


async def test_grounded_search_tries_languages_then_task_only() -> None:
    from langchain_core.tools import tool

    from app.graph.nodes.scout import grounded_search

    calls: list[list[str]] = []

    @tool
    def hub_repo_search(filters: list[str], sort: str, limit: int, repo_types: list[str]) -> str:
        """Search."""
        calls.append(filters)
        return "No repositories found" if len(filters) > 1 else "### openai/whisper-small"

    messages = await grounded_search({"hub_repo_search": hub_repo_search}, "asr", ["hi"])

    assert calls == [["asr", "hi"], ["asr"]]
    assert [c.repo_id for c in repo_ids_in_tool_results(messages, 5)] == ["openai/whisper-small"]


async def test_scout_survives_a_failing_llm_and_uses_grounded_search() -> None:
    from langchain_core.runnables import RunnableLambda
    from langchain_core.tools import tool

    from app.config import load_settings
    from app.graph.nodes.scout import make_scout_node
    from app.schemas import Constraints, ResearchPlan
    from tests.unit.graph.fakes import FakeHub, make_deps

    limits: list[int] = []

    @tool
    def hub_repo_search(
        filters: list[str],
        sort: str = "downloads",
        limit: int = 20,
        repo_types: list[str] | None = None,
    ) -> str:
        """Search."""
        limits.append(limit)
        return "### openai/whisper-small\n### vasista22/whisper-hindi-small"

    def broken_llm(tier: str, **kw: object) -> RunnableLambda:  # type: ignore[type-arg]
        def fail(_: object) -> object:
            raise TimeoutError("local model too slow")

        return RunnableLambda(fail)

    deps = make_deps(broken_llm, FakeHub({}), [hub_repo_search], load_settings(env_files=None))  # type: ignore[arg-type]
    state = {
        "constraints": Constraints(
            task_family="speech",
            task_description="ASR",
            deployment_mode="open_weight",
            commercial_use=True,
            languages=["hi"],
            gpu_vram_gb=16,
        ),
        "plan": ResearchPlan(
            hf_task="automatic-speech-recognition", search_queries=["x"], steps=["y"]
        ),
    }

    out = await make_scout_node(deps)(state, {})

    assert [c.repo_id for c in out["candidates"]] == [
        "openai/whisper-small",
        "vasista22/whisper-hindi-small",
    ]
    assert limits == [8]
    assert len(out["errors"]) == 2


def test_dataset_tags_and_url_fragments_are_not_candidates() -> None:
    messages = [
        ToolMessage(
            "### moorlee/qwen3-asr-0.6b-hinglish\n**Tags:** dataset:shrutisingh/HiACC, "
            "base_model:Qwen/Qwen3-ASR, see hf.co/moorlee and https://huggingface.co/org9/x",
            tool_call_id="c1",
        )
    ]

    ids = [c.repo_id for c in repo_ids_in_tool_results(messages, limit=5)]

    assert ids == ["moorlee/qwen3-asr-0.6b-hinglish"]
