"""End-to-end run of the HubScout graph with fakes: interrupts, routing, checks, blueprint."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph.build import build_graph
from app.schemas import (
    AggregatorNarrative,
    Blueprint,
    ConstraintDraft,
    ResearchPlan,
    ScoutCandidate,
    ScoutReport,
)
from tests.unit.graph.fakes import FakeHub, ScriptedLlm, facts, make_deps, make_search_tool

REQUEST = "Speech-to-text for Hindi-English customer calls."
PLAN = ResearchPlan(
    hf_task="automatic-speech-recognition",
    search_queries=["whisper hindi"],
    steps=["search the Hub", "check constraints"],
)
FULL_DRAFT = ConstraintDraft(
    task_family="speech",
    task_description="Transcribe Hindi-English calls",
    deployment_mode="open_weight",
    commercial_use=True,
    languages=["hi", "en"],
    gpu_vram_gb=16,
)
NARRATIVE = AggregatorNarrative(
    recommendation_summary="Use whisper-small.", pick_reasons=["fits"], risks=["accents"]
)
REPORT = ScoutReport(
    candidates=[
        ScoutCandidate(repo_id="openai/whisper-small", why="multilingual ASR"),
        ScoutCandidate(repo_id="fake/huge-model", why="big"),
        ScoutCandidate(repo_id="nobody/does-not-exist", why="made up"),
        ScoutCandidate(repo_id="OpenAI/whisper-small", why="duplicate, different case"),
    ]
)
HUB = FakeHub(
    {
        "openai/whisper-small": facts("openai/whisper-small"),
        "fake/huge-model": facts("fake/huge-model", params=70_000_000_000, licence="llama3.1"),
    }
)


def thread() -> dict[str, Any]:
    return {"configurable": {"thread_id": "t1"}}


def interrupt_value(result: dict[str, Any]) -> dict[str, Any]:
    return result["__interrupt__"][0].value  # type: ignore[no-any-return]


async def test_full_run_with_clarification_and_plan_approval() -> None:
    partial = FULL_DRAFT.model_copy(update={"deployment_mode": None, "gpu_vram_gb": None})
    search_calls: list[dict[str, Any]] = []
    llm = ScriptedLlm(
        structured={
            ConstraintDraft: [partial, FULL_DRAFT],
            ResearchPlan: [PLAN],
            ScoutReport: [REPORT],
            AggregatorNarrative: [NARRATIVE],
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
    graph = build_graph(
        make_deps(llm, HUB, [make_search_tool(search_calls)]), checkpointer=InMemorySaver()
    )

    # 1) Clarifier pauses with only the missing questions.
    result = await graph.ainvoke({"request": REQUEST}, thread())
    questions = interrupt_value(result)["questions"]
    assert len(questions) == 2
    assert "hosted API" in questions[0]
    assert "hardware" in questions[1]

    # 2) Answer -> clarifier settles -> planner -> pauses for plan approval.
    result = await graph.ainvoke(Command(resume="Self-hosted, one 16 GB GPU"), thread())
    assert interrupt_value(result)["type"] == "plan_approval"
    assert interrupt_value(result)["plan"]["hf_task"] == "automatic-speech-recognition"

    # 3) Approve -> scout (MCP tool) -> checker -> aggregator.
    result = await graph.ainvoke(Command(resume="yes"), thread())
    blueprint: Blueprint = result["blueprint"]

    assert search_calls == [{"query": "whisper hindi", "filters": None}]
    assert blueprint.open_weight_pick is not None
    assert blueprint.open_weight_pick.repo_id == "openai/whisper-small"
    assert blueprint.open_weight_pick.precision == "bf16"
    rejected = {r.repo_id: r for r in blueprint.rejected}
    assert rejected["nobody/does-not-exist"].stage == "existence"
    assert any("licence" in reason for reason in rejected["fake/huge-model"].reasons)
    assert len(result["candidates"]) == 3  # duplicate removed
    assert blueprint.pipeline_mermaid is not None
    assert "whisper-small" in blueprint.pipeline_mermaid
    assert llm.calls == [
        "strong:ConstraintDraft",
        "strong:ConstraintDraft",
        "strong:ResearchPlan",
        "cheap:tools",
        "cheap:tools",
        "cheap:ScoutReport",
        "strong:AggregatorNarrative",
    ]


async def test_plan_feedback_triggers_one_revision() -> None:
    revised = PLAN.model_copy(update={"search_queries": ["indic asr"]})
    llm = ScriptedLlm(
        structured={
            ConstraintDraft: [FULL_DRAFT],
            ResearchPlan: [PLAN, revised],
            ScoutReport: [ScoutReport(candidates=[])],
        },
    )
    graph = build_graph(make_deps(llm, HUB, []), checkpointer=InMemorySaver())

    result = await graph.ainvoke({"request": REQUEST}, thread())
    assert interrupt_value(result)["type"] == "plan_approval"
    result = await graph.ainvoke(Command(resume="also search Indic models"), thread())
    assert interrupt_value(result)["plan"]["search_queries"] == ["indic asr"]
    result = await graph.ainvoke(Command(resume={"approved": True}), thread())

    blueprint: Blueprint = result["blueprint"]
    assert blueprint.open_weight_pick is None
    assert any("No open-weight candidate" in n for n in blueprint.notes)


async def test_api_mode_skips_scouting_and_explains() -> None:
    llm = ScriptedLlm(
        structured={ConstraintDraft: [FULL_DRAFT.model_copy(update={"deployment_mode": "api"})]}
    )
    graph = build_graph(make_deps(llm, HUB, []), checkpointer=InMemorySaver())

    result = await graph.ainvoke({"request": REQUEST}, thread())

    blueprint: Blueprint = result["blueprint"]
    assert blueprint.open_weight_pick is None
    assert any("hosted-API path" in n for n in blueprint.notes)
    assert llm.calls == ["strong:ConstraintDraft"]


@pytest.mark.parametrize("rounds", [2])
async def test_clarifier_stops_asking_after_max_rounds(rounds: int) -> None:
    empty = ConstraintDraft()
    llm = ScriptedLlm(
        structured={
            ConstraintDraft: [empty] * (rounds + 1),
            ResearchPlan: [PLAN],
        }
    )
    graph = build_graph(make_deps(llm, HUB, []), checkpointer=InMemorySaver())

    await graph.ainvoke({"request": "something"}, thread())
    for _ in range(rounds):
        result = await graph.ainvoke(Command(resume="no idea"), thread())

    assert interrupt_value(result)["type"] == "plan_approval"
    state = (await graph.aget_state(thread())).values
    assert state["constraints"].cpu_only is True
    assert len(state["assumptions"]) == 4
