"""User confirmations are causal, authoritative, and never silently erased."""

from copy import deepcopy

import pytest

from satori_repro.state import initial_state, update_state


TASK = {"task_id": "tea", "title": "泡茶", "steps": [
    {"step_id": "water", "instruction": "倒水", "completion_criteria": "有水"},
    {"step_id": "tea", "instruction": "放茶", "completion_criteria": "有茶"},
]}


def obs(**changes):
    result = {"scene": "桌面", "objects": [{"name": "杯子", "state": "有水"}],
              "action": "倒水", "current_step_id": "water", "completed_step_ids": [],
              "confidence": 0.9, "uncertain": False, "evidence_frame_ids": ["f1"],
              "needs_help": False, "help_text": ""}
    result.update(changes)
    return result


def update(previous, observation=None, events=None, timestamp=100):
    return update_state(previous, observation or obs(), events or [], task=TASK,
                        timestamp_ms=timestamp, frame_ids=["f1"])


def test_initial_state_has_no_inferred_or_confirmed_progress():
    state = initial_state(TASK)
    assert state["task_id"] == "tea"
    assert state["task_confirmed"] is False
    assert state["current_step_id"] is None
    assert state["confirmed_completed_step_ids"] == []
    assert state["observed_completed_step_ids"] == []
    assert state["uncertain"] is True


def test_model_completed_steps_remain_observations_and_inputs_are_immutable():
    before = initial_state(TASK)
    observation = obs(completed_step_ids=["water"], current_step_id="tea")
    snapshots = deepcopy((before, observation))
    after = update(before, observation)
    assert after["observed_completed_step_ids"] == ["water"]
    assert after["confirmed_completed_step_ids"] == []
    assert after["current_step_id"] == "tea"
    assert (before, observation) == snapshots
    after["objects"][0]["name"] = "changed"
    assert observation["objects"][0]["name"] == "杯子"


def test_confirmed_completion_wins_over_model_and_survives_uncertainty():
    before = initial_state(TASK)
    after = update(before, events=[
        {"type": "confirm_task", "task_id": "tea", "timestamp_ms": 0},
        {"type": "confirm_step", "step_id": "water", "timestamp_ms": 90},
    ])
    assert after["task_confirmed"] is True
    assert after["confirmed_completed_step_ids"] == ["water"]
    assert after["current_step_id"] == "tea"
    uncertain = update(after, obs(confidence=0.2, uncertain=True), timestamp=200)
    assert uncertain["confirmed_completed_step_ids"] == ["water"]
    assert uncertain["current_step_id"] == "tea"
    assert uncertain["uncertain"] is True


def test_same_time_reopen_wins_after_confirm_and_retracts_completion():
    events = [
        {"timestamp_ms": 100, "type": "confirm_step", "step_id": "water"},
        {"timestamp_ms": 100, "type": "reopen_step", "step_id": "water"},
    ]
    after = update(initial_state(TASK), obs(current_step_id="tea", completed_step_ids=["water"]), events)
    assert after["confirmed_completed_step_ids"] == []
    assert after["current_step_id"] == "water"
    assert "water" not in after["observed_completed_step_ids"]


def test_confirm_after_reopen_at_same_time_is_authoritative():
    events = [
        {"timestamp_ms": 100, "type": "reopen_step", "step_id": "water"},
        {"timestamp_ms": 100, "type": "confirm_step", "step_id": "water"},
    ]
    after = update(initial_state(TASK), events=events)
    assert after["confirmed_completed_step_ids"] == ["water"]
    assert after["current_step_id"] == "tea"


def test_low_confidence_does_not_invent_new_progress():
    after = update(initial_state(TASK), obs(confidence=0.59, completed_step_ids=["water"]))
    assert after["current_step_id"] is None
    assert after["observed_completed_step_ids"] == []
    assert after["uncertain"] is True


def test_all_confirmed_steps_have_no_current_step():
    after = update(initial_state(TASK), events=[
        {"timestamp_ms": 1, "type": "confirm_step", "step_id": "water"},
        {"timestamp_ms": 2, "type": "confirm_step", "step_id": "tea"},
    ])
    assert after["current_step_id"] is None
    assert after["confirmed_completed_step_ids"] == ["water", "tea"]


@pytest.mark.parametrize("events", [
    [{"timestamp_ms": 101, "type": "confirm_step", "step_id": "water"}],
    [{"timestamp_ms": 1, "type": "confirm_step", "step_id": "missing"}],
    [{"timestamp_ms": 1, "type": "confirm_task", "task_id": "other"}],
    [{"timestamp_ms": 1, "type": "erase_task"}],
])
def test_invalid_or_future_events_cannot_change_state(events):
    before = initial_state(TASK)
    snapshot = deepcopy(before)
    with pytest.raises(ValueError):
        update(before, events=events)
    assert before == snapshot


def test_future_evidence_and_time_reversal_are_rejected():
    with pytest.raises(ValueError):
        update(initial_state(TASK), obs(evidence_frame_ids=["future"]))
    before = update(initial_state(TASK))
    with pytest.raises(ValueError):
        update(before, timestamp=99)

def test_confirming_observed_step_discards_its_old_guidance():
    observation = obs(needs_help=True, help_text="把水倒进杯子。")
    after = update(initial_state(TASK), observation, [
        {"timestamp_ms": 0, "type": "confirm_task", "task_id": "tea"},
        {"timestamp_ms": 100, "type": "confirm_step", "step_id": "water"},
    ])
    assert after["current_step_id"] == "tea"
    assert after["needs_help"] is False
    assert after["help_text"] == ""
    from satori_repro.decision import decide
    assert decide(after, timestamp_ms=100)["kind"] == "observe"


def test_reopening_a_different_step_discards_guidance_for_the_previous_step():
    before = update(initial_state(TASK), events=[
        {"timestamp_ms": 0, "type": "confirm_task", "task_id": "tea"},
        {"timestamp_ms": 100, "type": "confirm_step", "step_id": "water"},
    ])
    after = update(before, obs(current_step_id="tea", needs_help=True, help_text="放茶叶。"),
                   [{"timestamp_ms": 200, "type": "reopen_step", "step_id": "water"}],
                   timestamp=200)
    assert after["current_step_id"] == "water"
    assert after["needs_help"] is False
    assert after["help_text"] == ""


def test_later_observation_cannot_guide_an_already_confirmed_step():
    before = update(initial_state(TASK), events=[
        {"timestamp_ms": 100, "type": "confirm_step", "step_id": "water"},
    ])
    after = update(before, obs(needs_help=True, help_text="把水倒进杯子。"), timestamp=200)
    assert after["current_step_id"] == "tea"
    assert after["needs_help"] is False
    assert after["help_text"] == ""


def test_discarding_stale_guidance_preserves_final_completion():
    after = update(initial_state(TASK), obs(needs_help=True, help_text="继续倒水。"), [
        {"timestamp_ms": 0, "type": "confirm_task", "task_id": "tea"},
        {"timestamp_ms": 50, "type": "confirm_step", "step_id": "water"},
        {"timestamp_ms": 100, "type": "confirm_step", "step_id": "tea"},
    ])
    assert after["current_step_id"] is None
    assert after["needs_help"] is False
    assert after["help_text"] == ""
    from satori_repro.decision import decide
    assert decide(after, timestamp_ms=100)["kind"] == "complete"


def test_new_observation_for_pending_step_can_offer_fresh_guidance():
    before = update(initial_state(TASK), obs(needs_help=True, help_text="倒水。"), [
        {"timestamp_ms": 0, "type": "confirm_task", "task_id": "tea"},
        {"timestamp_ms": 100, "type": "confirm_step", "step_id": "water"},
    ])
    after = update(before, obs(current_step_id="tea", needs_help=True, help_text="放茶叶。"),
                   timestamp=200)
    assert after["current_step_id"] == "tea"
    assert after["needs_help"] is True
    assert after["help_text"] == "放茶叶。"
    from satori_repro.decision import decide
    assert decide(after, timestamp_ms=200)["kind"] == "guide"
