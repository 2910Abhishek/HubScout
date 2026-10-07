"""Small helpers: model-name cleanup and the de-duplicating reducer."""

from app.graph.state import as_text, unique_merge
from app.graph.usage import undouble


def test_undouble_fixes_streamed_concatenation() -> None:
    assert undouble("org/model:freeorg/model:free") == "org/model:free"
    assert undouble("aaaa") == "a"
    assert undouble("qwen3:4b-instruct") == "qwen3:4b-instruct"


def test_unique_merge_keeps_order_and_drops_duplicates() -> None:
    assert unique_merge(["a", "b"], ["b", "c", "a"]) == ["a", "b", "c"]
    assert unique_merge(None, ["x"]) == ["x"]


def test_as_text_accepts_strings_dicts_and_others() -> None:
    assert as_text("  hi ") == "hi"
    assert as_text({"answer": "yes"}) == "yes"
    assert as_text(True) == "True"


def test_infer_task_family_keywords() -> None:
    from app.graph.nodes.clarifier import infer_task_family

    assert infer_task_family("speech-to-text for Hindi calls") == "speech"
    assert infer_task_family("Transcribe customer call recordings") == "speech"
    assert infer_task_family("detect defects in product photos") == "vision"
    assert infer_task_family("Classify support tickets into 12 categories") == "text"
    assert infer_task_family("compare both") is None
