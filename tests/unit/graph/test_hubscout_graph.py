"""End-to-end runs of the HubScout graph with fakes: interrupts, parallel scouts, verification,
metadata linking, method selection, gaps and the rendered README."""

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph.build import build_graph
from app.schemas import (
    ConstraintDraft,
    KitNarrative,
    MethodSelection,
    ResearchPlan,
    ScoutCandidate,
    ScoutReport,
    StarterKit,
)
from app.tools.registries.arxiv import Paper
from app.tools.search import WebResult
from tests.unit.graph.fakes import (
    FakeArxiv,
    FakeHub,
    FakeLinks,
    FakeWeb,
    ScriptedLlm,
    dataset_facts,
    facts,
    make_deps,
    make_search_tool,
)

REQUEST = "Speech-to-text for Hindi-English customer calls."
PLAN = ResearchPlan(
    hf_task="automatic-speech-recognition",
    search_queries=["whisper hindi"],
    dataset_queries=["hindi speech"],
    method_queries=["code switching asr"],
    steps=["search the Hub", "check constraints"],
)
FULL_DRAFT = ConstraintDraft(
    task_family="speech",
    task_description="Transcribe Hindi-English calls",
    commercial_use=True,
    languages=["hi", "en"],
    gpu_vram_gb=16,
)
REPORT = ScoutReport(
    candidates=[
        ScoutCandidate(repo_id="openai/whisper-small", why="multilingual ASR"),
        ScoutCandidate(repo_id="fake/huge-model", why="big"),
        ScoutCandidate(repo_id="nobody/does-not-exist", why="made up"),
        ScoutCandidate(repo_id="OpenAI/whisper-small", why="duplicate, different case"),
    ]
)
WHISPER_PAPER = Paper("2212.04356", "Robust Speech Recognition via Weak Supervision", "…", "2022")
SEARCH_PAPER = Paper("2401.00001", "Code-Switching ASR for Hinglish", "A method paper.", "2024")
HUB = FakeHub(
    models={
        "openai/whisper-small": facts(
            "openai/whisper-small", tags=["dataset:org/hinglish-speech", "arxiv:2212.04356"]
        ),
        "fake/huge-model": facts("fake/huge-model", params=70_000_000_000, licence="llama3.1"),
    },
    datasets={
        "org/hinglish-speech": dataset_facts("org/hinglish-speech", downloads=100),
        "org/hindi-asr": dataset_facts("org/hindi-asr", downloads=90_000),
        "org/nc-data": dataset_facts("org/nc-data", licence="cc-by-nc-4.0"),
    },
)
WEB = FakeWeb(
    [
        WebResult("Fine-tuning Whisper for Hindi", "https://example.com/whisper-hindi", "…", "t"),
        WebResult("Moved page", "https://example.com/dead", "…", "t"),
        WebResult("Reference code", "https://github.com/org/hinglish-asr", "…", "t"),
    ]
)


def thread() -> dict[str, Any]:
    return {"configurable": {"thread_id": "t1"}}


def interrupt_value(result: dict[str, Any]) -> dict[str, Any]:
    return result["__interrupt__"][0].value  # type: ignore[no-any-return]


async def test_full_run_builds_a_verified_connected_starter_kit() -> None:
    partial = FULL_DRAFT.model_copy(update={"commercial_use": None, "gpu_vram_gb": None})
    searches: list[dict[str, Any]] = []
    llm = ScriptedLlm(
        structured={
            ConstraintDraft: [partial, FULL_DRAFT],
            ResearchPlan: [PLAN],
            ScoutReport: [REPORT],
            MethodSelection: [MethodSelection(picks=[0, 1], reasons=["method paper", "guide"])],
            KitNarrative: [KitNarrative(tldr="Use whisper-small.", fit_story="They connect.")],
        },
        tool_replies=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "hub_repo_search", "args": {"query": "whisper hindi"}, "id": "c1"}
                ],
            ),
            AIMessage(content="found enough"),
        ],
    )
    tool = make_search_tool(
        searches, models="### openai/whisper-small", datasets="### org/hindi-asr\n### org/nc-data"
    )
    arxiv = FakeArxiv(papers=[WHISPER_PAPER], search_results=[SEARCH_PAPER])
    deps = make_deps(
        llm, HUB, [tool], arxiv=arxiv, web=WEB, links=FakeLinks({"https://example.com/dead"})
    )
    graph = build_graph(deps, checkpointer=InMemorySaver())

    # 1) Only the missing questions are asked; there is no deployment-mode question any more.
    result = await graph.ainvoke({"request": REQUEST}, thread())
    questions = interrupt_value(result)["questions"]
    assert len(questions) == 2
    assert "commercial" in questions[0]
    assert "hardware" in questions[1]

    # 2) Answer -> plan approval shows all three query lists.
    result = await graph.ainvoke(Command(resume="commercial, one 16 GB GPU"), thread())
    plan = interrupt_value(result)["plan"]
    assert plan["dataset_queries"] == ["hindi speech"]
    assert plan["method_queries"] == ["code switching asr"]

    # 3) Approve -> three scouts in parallel -> verify -> write.
    result = await graph.ainvoke(Command(resume="yes"), thread())
    kit: StarterKit = result["starter_kit"]

    # Models: only the verified one, duplicates and fakes removed.
    assert [m.repo_id for m in kit.models] == ["openai/whisper-small"]
    assert len(result["model_candidates"]) == 3

    # Datasets: the one the model was trained on comes from Hub metadata and ranks first.
    assert [d.repo_id for d in kit.datasets] == ["org/hinglish-speech", "org/hindi-asr"]
    assert kit.datasets[0].used_by == ["openai/whisper-small"]
    assert kit.datasets[0].sample_rows

    # Methods: the linked paper first, then the LLM's picks among verified results only.
    assert [m.url for m in kit.methods] == [
        "https://arxiv.org/abs/2212.04356",
        "https://arxiv.org/abs/2401.00001",
        "https://example.com/whisper-hindi",
    ]
    assert kit.methods[0].source == "hub-metadata"
    assert kit.methods[0].describes == ["openai/whisper-small"]
    assert kit.methods[1].why == "method paper"

    rejected = {r.repo_id: r for r in kit.rejected}
    assert rejected["nobody/does-not-exist"].stage == "existence"
    assert any("licence" in r for r in rejected["fake/huge-model"].reasons)
    assert rejected["org/nc-data"].kind == "dataset"
    assert rejected["https://example.com/dead"].reasons == ["HTTP 404"]

    assert len(kit.links) <= 15
    assert kit.tldr == "Use whisper-small."
    assert kit.quick_start is not None
    assert "trust_remote_code=False" in kit.quick_start
    assert "org/hinglish-speech" in kit.quick_start
    assert result["readme_path"] is None  # tests never write files

    readme = result["readme"]
    for section in ("## TL;DR", "## 1. Models", "## 2. Datasets", "## 3. Methods", "Sample rows"):
        assert section in readme
    assert "used to train openai/whisper-small" in readme
    assert "## What we ruled out (4)" in readme

    dataset_searches = [s for s in searches if s["repo_types"] == ["dataset"]]
    assert dataset_searches[0]["filters"] == ["task_categories:automatic-speech-recognition"]
    assert arxiv.searches == ["code switching asr"]
    assert llm.calls == [
        "strong:ConstraintDraft",
        "strong:ConstraintDraft",
        "strong:ResearchPlan",
        "cheap:tools",
        "cheap:tools",
        "cheap:ScoutReport",
        "cheap:MethodSelection",
        "strong:KitNarrative",
    ]


async def test_nothing_found_gives_an_honest_empty_kit_after_a_plan_revision() -> None:
    revised = PLAN.model_copy(update={"search_queries": ["indic asr"]})
    llm = ScriptedLlm(
        structured={
            ConstraintDraft: [FULL_DRAFT],
            ResearchPlan: [PLAN, revised],
            ScoutReport: [ScoutReport(candidates=[])],
        }
    )
    graph = build_graph(
        make_deps(llm, FakeHub(), [make_search_tool([])]), checkpointer=InMemorySaver()
    )

    result = await graph.ainvoke({"request": REQUEST}, thread())
    assert interrupt_value(result)["type"] == "plan_approval"
    result = await graph.ainvoke(Command(resume="also search Indic models"), thread())
    assert interrupt_value(result)["plan"]["search_queries"] == ["indic asr"]
    result = await graph.ainvoke(Command(resume={"approved": True}), thread())

    kit: StarterKit = result["starter_kit"]
    assert kit.links == []
    assert len(kit.gaps) == 3
    assert all("passed verification" in gap for gap in kit.gaps)
    assert "_No model passed verification" in result["readme"]
    # No narrative LLM call when there is nothing to summarise.
    assert "strong:KitNarrative" not in llm.calls


async def test_empty_resume_re_asks_without_using_a_round() -> None:
    partial = FULL_DRAFT.model_copy(update={"commercial_use": None})
    llm = ScriptedLlm(structured={ConstraintDraft: [partial, FULL_DRAFT], ResearchPlan: [PLAN]})
    graph = build_graph(make_deps(llm, HUB, []), checkpointer=InMemorySaver())

    await graph.ainvoke({"request": REQUEST}, thread())
    result = await graph.ainvoke(Command(resume=""), thread())  # Studio's default ""

    value = interrupt_value(result)
    assert value["type"] == "clarification"
    assert "empty" in value["error"]
    assert llm.calls == ["strong:ConstraintDraft"]  # no extra LLM call for the empty answer

    result = await graph.ainvoke(Command(resume="yes, commercial"), thread())
    assert interrupt_value(result)["type"] == "plan_approval"
    state = (await graph.aget_state(thread())).values
    assert state["clarify_rounds"] == 1
    assert state["clarification_log"][0].endswith("A: yes, commercial")


async def test_empty_plan_review_re_asks() -> None:
    llm = ScriptedLlm(structured={ConstraintDraft: [FULL_DRAFT], ResearchPlan: [PLAN]})
    graph = build_graph(make_deps(llm, HUB, []), checkpointer=InMemorySaver())

    await graph.ainvoke({"request": REQUEST}, thread())
    result = await graph.ainvoke(Command(resume="  "), thread())

    assert interrupt_value(result)["type"] == "plan_approval"
    assert "empty" in interrupt_value(result)["error"]
    assert llm.calls.count("strong:ResearchPlan") == 1


async def test_clarifier_stops_asking_after_max_rounds() -> None:
    llm = ScriptedLlm(structured={ConstraintDraft: [ConstraintDraft()] * 3, ResearchPlan: [PLAN]})
    graph = build_graph(make_deps(llm, HUB, []), checkpointer=InMemorySaver())

    result = await graph.ainvoke({"request": "something"}, thread())
    assert len(interrupt_value(result)["questions"]) == 3  # commercial, task, hardware
    for _ in range(2):
        result = await graph.ainvoke(Command(resume="no idea"), thread())

    assert interrupt_value(result)["type"] == "plan_approval"
    state = (await graph.aget_state(thread())).values
    assert state["constraints"].cpu_only is True
    assert len(state["assumptions"]) == 3
