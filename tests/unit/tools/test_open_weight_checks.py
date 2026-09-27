"""Code-only constraint checks for open-weight candidates."""

import pytest

from app.config import load_settings
from app.schemas import Constraints, OpenWeightCandidate, RejectedCandidate
from app.tools.checks.open_weight import check_candidate, estimate_vram_gb, hardware_fit
from tests.unit.graph.fakes import facts

POLICY = load_settings(env_files=None).policy
TASK = "automatic-speech-recognition"


def constraints(**kw: object) -> Constraints:
    base: dict[str, object] = {
        "task_family": "speech",
        "task_description": "ASR",
        "deployment_mode": "open_weight",
        "commercial_use": True,
        "gpu_vram_gb": 16,
    }
    return Constraints.model_validate({**base, **kw})


def reasons(result: object) -> list[str]:
    assert isinstance(result, RejectedCandidate)
    return result.reasons


def test_vram_estimate_uses_bytes_per_param_and_overhead() -> None:
    assert estimate_vram_gb(7_000_000_000, "bf16", 1.2) == 16.8
    assert estimate_vram_gb(7_000_000_000, "int4", 1.2) == 4.2


def test_hardware_fit_quantises_only_when_needed() -> None:
    assert hardware_fit(1_000_000_000, constraints(), POLICY).precision == "bf16"
    assert hardware_fit(8_000_000_000, constraints(), POLICY).precision == "int8"
    assert hardware_fit(20_000_000_000, constraints(), POLICY).precision == "int4"
    assert hardware_fit(40_000_000_000, constraints(), POLICY).fits is False


def test_cpu_only_uses_ram_budget() -> None:
    fit = hardware_fit(1_000_000_000, constraints(cpu_only=True, gpu_vram_gb=None), POLICY)
    assert fit.fits
    assert fit.precision == "fp32"
    assert "CPU" in fit.note


def test_good_candidate_is_approved_with_code_computed_numbers() -> None:
    result = check_candidate(
        facts("openai/whisper-small"), constraints(languages=["hi"]), TASK, POLICY, "why"
    )

    assert isinstance(result, OpenWeightCandidate)
    assert result.precision == "bf16"
    assert result.est_vram_gb == 0.58
    assert 0 < result.fit_score <= 1
    assert result.sources[0].url == "https://huggingface.co/openai/whisper-small"


def test_missing_repo_rejected_at_existence_stage() -> None:
    result = check_candidate(facts("x/y", exists=False), constraints(), TASK, POLICY, "why")

    assert isinstance(result, RejectedCandidate)
    assert result.stage == "existence"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"licence": "cc-by-nc-4.0"}, "licence 'cc-by-nc-4.0' not in allowed"),
        ({"licence": None}, "licence missing"),
        ({"gated": True}, "gated"),
        ({"tags": ["custom_code"]}, "trust_remote_code"),
        ({"pipeline_tag": "text-generation"}, "task mismatch"),
        ({"params": 60_000_000_000}, "too large"),
        ({"params": None}, "parameter count unknown"),
        ({"languages": ["en"]}, "does not list required languages ['hi']"),
    ],
)
def test_each_rule_rejects_with_a_readable_reason(
    overrides: dict[str, object], expected: str
) -> None:
    result = check_candidate(
        facts("a/b", **overrides), constraints(languages=["hi"]), TASK, POLICY, "why"
    )

    assert any(expected in r for r in reasons(result))


def test_non_commercial_accepts_any_known_licence() -> None:
    result = check_candidate(
        facts("a/b", licence="cc-by-nc-4.0"), constraints(commercial_use=False), TASK, POLICY, "why"
    )
    assert isinstance(result, OpenWeightCandidate)


def test_unlisted_languages_are_a_note_not_a_rejection() -> None:
    result = check_candidate(
        facts("a/b", languages=[]), constraints(languages=["hi"]), TASK, POLICY, "why"
    )

    assert isinstance(result, OpenWeightCandidate)
    assert any("not stated" in n for n in result.notes)
