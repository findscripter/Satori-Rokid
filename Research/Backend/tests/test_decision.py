"""Engineering baseline: only warranted assistance, with interruption controls."""

from copy import deepcopy

import pytest

from satori_repro.decision import decide
from satori_repro.state import initial_state


TASK = {"task_id": "tea", "title": "泡茶", "steps": [
    {"step_id": "water", "instruction": "倒水", "completion_criteria": "有水"},
    {"step_id": "tea", "instruction": "放茶", "completion_criteria": "有茶"},
]}


def state(**changes):
    result = initial_state(TASK)
    result.update(task_confirmed=True, current_step_id="water", confidence=0.9,
                  uncertain=False, evidence_frame_ids=["f1"])
    result.update(changes)
    return result


def test_unconfirmed_goal_is_a_clarification_even_with_model_help():
    result = decide(state(task_confirmed=False, needs_help=True, help_text="倒水"), timestamp_ms=0)
    assert result["kind"] == "clarify"
    assert "泡茶" in result["message"]


def test_clear_progress_without_help_requirement_stays_silent():
    assert decide(state(), timestamp_ms=0)["kind"] == "observe"


@pytest.mark.parametrize("changes", [{"uncertain": True}, {"confidence": 0.59}])
def test_uncertainty_is_not_confident_guidance(changes):
    result = decide(state(needs_help=True, help_text="倒水", **changes), timestamp_ms=0)
    assert result["kind"] == "clarify"


def test_help_request_produces_grounded_guide_without_mutation():
    data = state(needs_help=True, help_text="请把水倒入杯子。")
    snapshot = deepcopy(data)
    result = decide(data, timestamp_ms=100)
    assert result["kind"] == "guide"
    assert result["message"] == "请把水倒入杯子。"
    assert result["evidence_frame_ids"] == ["f1"]
    assert result["timestamp_ms"] == 100
    assert data == snapshot


def test_model_only_completion_cannot_finish_task():
    result = decide(state(observed_completed_step_ids=["water", "tea"]), timestamp_ms=0)
    assert result["kind"] != "complete"


def test_all_user_confirmations_allow_completion_even_during_cooldown():
    previous = decide(state(needs_help=True, help_text="倒水"), timestamp_ms=0)
    result = decide(state(confirmed_completed_step_ids=["water", "tea"]), previous,
                    timestamp_ms=100)
    assert result["kind"] == "complete"


def test_duplicate_guidance_is_suppressed_even_after_cooldown():
    data = state(needs_help=True, help_text="倒水")
    previous = decide(data, timestamp_ms=0)
    result = decide(data, previous, timestamp_ms=20000)
    assert result["kind"] == "observe"
    assert result["message"] == ""
    assert "repeat" in result["reason"]


def test_changed_guidance_is_cooled_down_then_allowed():
    previous = decide(state(needs_help=True, help_text="倒水"), timestamp_ms=0)
    changed = state(needs_help=True, help_text="放茶")
    assert decide(changed, previous, timestamp_ms=14999)["kind"] == "observe"
    assert decide(changed, previous, timestamp_ms=15000)["kind"] == "guide"


def test_repeated_completion_is_suppressed():
    data = state(confirmed_completed_step_ids=["water", "tea"])
    previous = decide(data, timestamp_ms=0)
    assert decide(data, previous, timestamp_ms=20000)["kind"] == "observe"


def test_missing_evidence_or_help_text_cannot_generate_guidance():
    for changes in ({"evidence_frame_ids": []}, {"help_text": ""}):
        data = state(needs_help=True, help_text="倒水")
        data.update(changes)
        assert decide(data, timestamp_ms=0)["kind"] != "guide"


@pytest.mark.parametrize("kwargs", [
    {"timestamp_ms": -1}, {"timestamp_ms": True},
    {"timestamp_ms": 0, "cooldown_ms": -1},
])
def test_invalid_time_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        decide(state(), **kwargs)


def test_future_previous_decision_rejected():
    previous = decide(state(needs_help=True, help_text="倒水"), timestamp_ms=100)
    with pytest.raises(ValueError):
        decide(state(), previous, timestamp_ms=99)
