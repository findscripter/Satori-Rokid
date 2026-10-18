from copy import deepcopy

import pytest

from satori_repro.satori_method import SatoriMethodSession, build_stage_prompt


def catalog():
    return [
        {"task_id": name, "title": name, "adaptation": "handwritten test task",
         "steps": [{"step_id": f"{name}{i}", "instruction": f"Do {name}{i}",
                    "checkpoints": [{"checkpoint_id": f"{name}{i}c", "question": "Is the result visible?"}],
                    "required_objects": ["cup"]} for i in (1, 2)]}
        for name in ("a", "b")
    ]


def frames(timestamp):
    return [{"frame_id": f"f{timestamp}", "timestamp_ms": timestamp}]


def event(timestamp, event_type, **kwargs):
    return {"timestamp_ms": timestamp, "type": event_type, **kwargs}


class Provider:
    def __init__(self):
        self.calls = []
        self.checkpoint_status = "IN_PROGRESS"
        self.goal = "a"
        self.bad_evidence = False
        self.forecast_available = True
        self.forecast_override = None
        self.scene = "desk"
        self.objects = [{"name": "cup", "state": "visible"}]
        self.uncertain = False
        self.belief_confidence = 0.9
        self.overall_status = None
        self.action = "handling cup"

    def complete_json(self, stage, input_frames, payload):
        self.calls.append((stage, deepcopy(input_frames), deepcopy(payload)))
        evidence = ["future"] if self.bad_evidence else [input_frames[-1]["frame_id"]]
        if stage == "belief":
            return {"scene": self.scene, "objects": deepcopy(self.objects),
                    "action": self.action, "uncertain": self.uncertain, "confidence": self.belief_confidence,
                    "evidence_frame_ids": evidence}
        if stage == "desire":
            return {"task_id": self.goal, "confidence": 0.9, "evidence_frame_ids": evidence}
        if stage == "forecast":
            target = self.forecast_override or payload["candidate_steps"][0]["step_id"]
            return {"target_step_id": target, "required_objects_available": self.forecast_available,
                    "confidence": 0.9, "evidence_frame_ids": evidence}
        if stage == "guidance":
            return {"step_id": payload["target_step"]["step_id"], "text": "Place the cup carefully.",
                    "modalities": ["text"], "evidence_frame_ids": evidence}
        step = payload["active_step"]
        return {"step_id": step["step_id"],
                "checkpoints": [{"checkpoint_id": c["checkpoint_id"], "status": self.checkpoint_status}
                                for c in step["checkpoints"]],
                "overall_status": self.overall_status or ("COMPLETED" if self.checkpoint_status == "COMPLETED" else "PROCESSING"),
                "evidence_frame_ids": evidence}


def bootstrap(provider=None):
    provider = provider or Provider()
    session = SatoriMethodSession(catalog(), provider)
    session.process_frame(frames(0), 0, [])
    session.process_frame(frames(1), 1, [event(1, "confirm_task", task_id="a")])
    return session, provider


def test_goal_confirmation_and_step_acceptance_do_not_complete_steps():
    provider = Provider()
    session = SatoriMethodSession(catalog(), provider)
    first = session.process_frame(frames(0), 0, [])
    assert first["confirmed_task_id"] is None
    assert first["pending_confirmation"] == {"kind": "confirm_task", "task_id": "a"}
    second = session.process_frame(frames(1), 1, [event(1, "confirm_task", task_id="a")])
    assert second["active_step_id"] is None
    assert second["pending_confirmation"] == {"kind": "accept_step", "step_id": "a1"}
    assert second["guidance"] is None
    accepted = session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    assert accepted["active_step_id"] == "a1"
    assert accepted["confirmed_completed_step_ids"] == []
    assert accepted["guidance"]["step_id"] == "a1"


def test_next_guidance_is_cached_before_completion_and_reused_after_confirmation():
    session, provider = bootstrap()
    active = session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    assert active["cached_forecast"]["target_step_id"] == "a2"
    assert active["cached_forecast"]["generated_at_ms"] == 2
    assert active["pending_confirmation"] is None
    assert active["confirmed_completed_step_ids"] == []
    provider.checkpoint_status = "COMPLETED"
    done = session.process_frame(frames(3), 3, [])
    assert done["pending_confirmation"] == {"kind": "confirm_step", "step_id": "a1"}
    assert done["confirmed_completed_step_ids"] == []
    before = [stage for stage, _, _ in provider.calls].count("guidance")
    transitioned = session.process_frame(frames(4), 4, [event(4, "confirm_step", step_id="a1")])
    assert transitioned["pending_confirmation"] == {"kind": "accept_step", "step_id": "a2"}
    assert [stage for stage, _, _ in provider.calls].count("guidance") == before
    assert transitioned["cached_forecast"]["generated_at_ms"] == 2
    assert transitioned["confirmed_completed_step_ids"] == ["a1"]


def test_accepting_cached_next_step_is_distinct_from_confirming_it_complete():
    session, provider = bootstrap()
    session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    result = session.process_frame(frames(3), 3, [event(3, "confirm_step", step_id="a1"), event(3, "accept_step", step_id="a2")])
    assert result["active_step_id"] == "a2"
    assert result["confirmed_completed_step_ids"] == ["a1"]
    final = session.process_frame(frames(4), 4, [event(4, "confirm_step", step_id="a2")])
    assert final["phase"] == "complete"
    assert final["confirmed_completed_step_ids"] == ["a1", "a2"]


def test_switching_task_invalidates_cached_guidance_and_confirmations():
    session, provider = bootstrap()
    session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    switched = session.process_frame(frames(3), 3, [event(3, "switch_task", task_id="b")])
    assert switched["confirmed_task_id"] is None
    assert switched["cached_forecast"] is None
    assert switched["guidance"] is None
    assert switched["confirmed_completed_step_ids"] == []
    assert switched["pending_confirmation"] == {"kind": "confirm_task", "task_id": "b"}
    next_result = session.process_frame(frames(4), 4, [event(4, "confirm_task", task_id="b")])
    assert next_result["cached_forecast"]["task_id"] == "b"
    assert next_result["cached_forecast"]["target_step_id"] == "b1"


def test_reopen_clears_cached_future_and_requires_accepting_reopened_step():
    session, _ = bootstrap()
    session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    session.process_frame(frames(3), 3, [event(3, "confirm_step", step_id="a1")])
    reopened = session.process_frame(frames(4), 4, [event(4, "reopen_step", step_id="a1")])
    assert reopened["confirmed_completed_step_ids"] == []
    assert reopened["pending_confirmation"] == {"kind": "accept_step", "step_id": "a1"}
    assert reopened["cached_forecast"]["generated_at_ms"] == 4
    assert reopened["cached_forecast"]["target_step_id"] == "a1"


@pytest.mark.parametrize("which", ["future_frame", "future_event", "old_event", "duplicate_frame", "changed_frame_timestamp"])
def test_causal_input_validation_runs_before_any_model_call(which):
    session, provider = bootstrap()
    input_frames, events = frames(2), []
    if which == "future_frame":
        input_frames = frames(3)
    elif which == "future_event":
        events = [event(3, "accept_step", step_id="a1")]
    elif which == "old_event":
        events = [event(1, "accept_step", step_id="a1")]
    elif which == "duplicate_frame":
        input_frames *= 2
    else:
        input_frames = [{"frame_id": "f1", "timestamp_ms": 2}]
    before = len(provider.calls)
    with pytest.raises(ValueError):
        session.process_frame(input_frames, 2, events)
    assert len(provider.calls) == before


def test_future_model_evidence_is_rejected():
    provider = Provider()
    provider.bad_evidence = True
    session = SatoriMethodSession(catalog(), provider)
    with pytest.raises(ValueError, match="evidence"):
        session.process_frame(frames(0), 0, [])


def test_wrong_step_cannot_be_accepted_or_confirmed():
    session, _ = bootstrap()
    with pytest.raises(ValueError):
        session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a2")])
    with pytest.raises(ValueError):
        session.process_frame(frames(2), 2, [event(2, "confirm_step", step_id="a1")])


def test_missing_required_objects_prevents_guidance_release():
    provider = Provider()
    provider.forecast_available = False
    session, provider = bootstrap(provider)
    assert not any(stage == "guidance" for stage, _, _ in provider.calls)
    record = session.process_frame(frames(2), 2, [])
    assert record["guidance"] is None and record["cached_forecast"] is None


def test_unknown_forecast_step_is_rejected():
    provider = Provider()
    session = SatoriMethodSession(catalog(), provider)
    session.process_frame(frames(0), 0, [])
    provider.forecast_override = "not-in-task"
    with pytest.raises(ValueError):
        session.process_frame(frames(1), 1, [event(1, "confirm_task", task_id="a")])


def test_history_is_past_only_and_returned_records_cannot_mutate_state():
    session, provider = bootstrap()
    result = session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    result["confirmed_completed_step_ids"].append("a2")
    result["belief"]["scene"] = "mutated"
    next_record = session.process_frame(frames(3), 3, [])
    assert next_record["confirmed_completed_step_ids"] == []
    for _, input_frames, payload in provider.calls:
        now = input_frames[-1]["timestamp_ms"]
        assert all(h["timestamp_ms"] < now for h in payload["history"])
    assert next_record["metadata"]["method"] == "satori-structured-reimplementation-v1"
    assert "text" in next_record["metadata"]["modality_scope"]
    assert next_record["metadata"]["task_adaptation"] == "handwritten test task"


@pytest.mark.parametrize("stage", ["belief", "desire", "forecast", "checkpoints", "guidance"])
def test_prompts_describe_exact_json_contract_and_evidence(stage):
    prompt = build_stage_prompt(stage, {})
    assert "JSON" in prompt and "evidence_frame_ids" in prompt
    assert "instruction" in prompt.lower()


def test_changed_scene_cannot_release_guidance_accepted_from_old_context():
    session, provider = bootstrap()
    provider.scene = "hallway"
    provider.objects = []
    provider.forecast_available = False
    result = session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    assert result["active_step_id"] is None
    assert result["guidance"] is None
    assert result["cached_forecast"] is None
    assert result["pending_confirmation"] is None
    assert result["phase"] == "awaiting_feasible_forecast"
    assert result["event_outcomes"][0]["status"] == "deferred"


def test_changed_required_object_state_refreshes_cached_future_guidance():
    session, provider = bootstrap()
    session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    provider.objects = [{"name": "cup", "state": "broken"}]
    provider.forecast_available = False
    result = session.process_frame(frames(3), 3, [])
    assert result["active_step_id"] == "a1"
    assert result["cached_forecast"] is None
    assert any(call["stage"] == "forecast" for call in result["stage_calls"])


@pytest.mark.parametrize("uncertain,confidence", [(True, 0.9), (False, 0.3), (False, 0.0)])
def test_uncertain_acceptance_waits_without_throwing_away_valid_prefetch(uncertain, confidence):
    session, provider = bootstrap()
    provider.uncertain, provider.belief_confidence = uncertain, confidence
    result = session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    assert result["active_step_id"] is None and result["guidance"] is None
    assert result["cached_forecast"]["generated_at_ms"] == 1
    assert result["event_outcomes"][0]["status"] == "deferred"
    provider.uncertain, provider.belief_confidence = False, 0.9
    accepted = session.process_frame(frames(3), 3, [event(3, "accept_step", step_id="a1")])
    assert accepted["active_step_id"] == "a1"
    assert accepted["guidance"]["evidence_frame_ids"] == ["f1"]


def test_same_frame_confirm_and_accept_preserves_human_fact_but_defers_stale_next_guidance():
    session, provider = bootstrap()
    session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    provider.scene, provider.objects, provider.forecast_available = "hallway", [], False
    result = session.process_frame(frames(3), 3, [event(3, "confirm_step", step_id="a1"),
                                                event(3, "accept_step", step_id="a2")])
    assert result["confirmed_completed_step_ids"] == ["a1"]
    assert result["active_step_id"] is None and result["guidance"] is None
    assert result["pending_confirmation"] is None


def test_normal_action_change_keeps_prefetch_and_guidance_stable():
    session, provider = bootstrap()
    active = session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    provider.action = "putting cup down"
    provider.objects.reverse()
    result = session.process_frame(frames(3), 3, [])
    assert result["cached_forecast"] == active["cached_forecast"]
    assert [call["stage"] for call in result["stage_calls"]] == ["belief", "checkpoints"]


def test_clear_noncompletion_retracts_model_completion_without_reopening_human_steps():
    session, provider = bootstrap()
    session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    provider.checkpoint_status = "COMPLETED"
    session.process_frame(frames(3), 3, [])
    provider.checkpoint_status = "IN_PROGRESS"
    result = session.process_frame(frames(4), 4, [])
    assert result["checkpoint_states"]["a1c"]["status"] == "IN_PROGRESS"
    assert result["pending_confirmation"] is None
    assert result["phase"] == "active_step"
    assert result["confirmed_completed_step_ids"] == []
    session.process_frame(frames(5), 5, [event(5, "confirm_step", step_id="a1"),
                                        event(5, "accept_step", step_id="a2")])
    result = session.process_frame(frames(6), 6, [])
    assert result["confirmed_completed_step_ids"] == ["a1"]


@pytest.mark.parametrize("status,overall,uncertain", [("UNKNOWN", "UNKNOWN", False),
                                                       ("NOT_STARTED", "UNRELATED", False),
                                                       ("IN_PROGRESS", "PROCESSING", True)])
def test_occlusion_or_uncertainty_does_not_retract_observed_completion(status, overall, uncertain):
    session, provider = bootstrap()
    session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    provider.checkpoint_status = "COMPLETED"
    session.process_frame(frames(3), 3, [])
    provider.checkpoint_status, provider.overall_status, provider.uncertain = status, overall, uncertain
    result = session.process_frame(frames(4), 4, [])
    assert result["checkpoint_states"]["a1c"]["status"] == "COMPLETED"
    assert result["pending_confirmation"] == {"kind": "confirm_step", "step_id": "a1"}


def test_user_rejects_completion_until_fresh_progress_then_new_completion():
    session, provider = bootstrap()
    session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    provider.checkpoint_status = "COMPLETED"
    session.process_frame(frames(3), 3, [])
    rejected = session.process_frame(frames(4), 4, [event(4, "reject_completion", step_id="a1")])
    assert rejected["active_step_id"] == "a1"
    assert rejected["confirmed_completed_step_ids"] == []
    assert rejected["pending_confirmation"] is None
    assert rejected["completion_rejected"] is True
    still_rejected = session.process_frame(frames(5), 5, [])
    assert still_rejected["pending_confirmation"] is None
    provider.checkpoint_status = "IN_PROGRESS"
    progress = session.process_frame(frames(6), 6, [])
    assert progress["completion_rejected"] is False
    provider.checkpoint_status = "COMPLETED"
    finished = session.process_frame(frames(7), 7, [])
    assert finished["pending_confirmation"] == {"kind": "confirm_step", "step_id": "a1"}


def test_reject_completion_requires_the_current_pending_active_step():
    session, provider = bootstrap()
    before = len(provider.calls)
    with pytest.raises(ValueError):
        session.process_frame(frames(2), 2, [event(2, "reject_completion", step_id="a1")])
    assert len(provider.calls) == before


def test_manual_completion_override_remains_allowed_during_uncertainty():
    session, provider = bootstrap()
    session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    provider.uncertain = True
    result = session.process_frame(frames(3), 3, [event(3, "confirm_step", step_id="a1")])
    assert result["confirmed_completed_step_ids"] == ["a1"]
    assert result["active_step_id"] is None


def test_irrelevant_added_object_does_not_invalidate_matching_prerequisites():
    session, provider = bootstrap()
    active = session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    provider.scene = "  DESK  "
    provider.objects = [{"name": "phone", "state": "on desk"}, {"name": " CUP ", "state": "VISIBLE"}]
    result = session.process_frame(frames(3), 3, [])
    assert result["cached_forecast"] == active["cached_forecast"]
    assert not any(call["stage"] in {"forecast", "guidance"} for call in result["stage_calls"])


def test_translated_object_names_are_not_treated_as_proven_missing_prerequisites():
    provider = Provider()
    provider.scene = "桌面"
    provider.objects = [{"name": "杯子", "state": "可见"}]
    session, provider = bootstrap(provider)
    result = session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    assert result["active_step_id"] == "a1"
    provider.objects = [{"name": "盘子", "state": "可见"}]
    provider.forecast_available = False
    changed = session.process_frame(frames(3), 3, [])
    assert changed["cached_forecast"] is None


def test_additional_same_named_prerequisite_is_not_lost_from_cache_context():
    session, provider = bootstrap()
    session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    provider.objects = [{"name": "cup", "state": "broken"}, {"name": "cup", "state": "visible"}]
    provider.forecast_available = False
    result = session.process_frame(frames(3), 3, [])
    assert result["cached_forecast"] is None


def test_deferred_acceptance_is_not_automatically_executed_on_later_frames():
    session, provider = bootstrap()
    provider.uncertain = True
    session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    provider.uncertain = False
    result = session.process_frame(frames(3), 3, [])
    assert result["active_step_id"] is None and result["guidance"] is None
    assert result["pending_confirmation"] == {"kind": "accept_step", "step_id": "a1"}


def test_partial_reliable_counterevidence_retracts_only_that_checkpoint():
    class MultiCheckpointProvider(Provider):
        def complete_json(self, stage, input_frames, payload):
            response = super().complete_json(stage, input_frames, payload)
            if stage == "checkpoints" and input_frames[-1]["timestamp_ms"] == 4:
                response["checkpoints"][0]["status"] = "IN_PROGRESS"
                response["checkpoints"][1]["status"] = "UNKNOWN"
                response["overall_status"] = "PROCESSING"
            return response

    tasks, provider = catalog(), MultiCheckpointProvider()
    tasks[0]["steps"][0]["checkpoints"].append({"checkpoint_id": "a1d", "question": "Is the second result visible?"})
    session = SatoriMethodSession(tasks, provider)
    session.process_frame(frames(0), 0, [])
    session.process_frame(frames(1), 1, [event(1, "confirm_task", task_id="a")])
    session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    provider.checkpoint_status = "COMPLETED"
    session.process_frame(frames(3), 3, [])
    result = session.process_frame(frames(4), 4, [])
    assert result["checkpoint_states"]["a1c"]["status"] == "IN_PROGRESS"
    assert result["checkpoint_states"]["a1d"]["status"] == "COMPLETED"
    assert result["pending_confirmation"] is None


def test_reopen_and_switch_clear_rejected_completion_gate():
    session, provider = bootstrap()
    session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    provider.checkpoint_status = "COMPLETED"
    session.process_frame(frames(3), 3, [])
    session.process_frame(frames(4), 4, [event(4, "reject_completion", step_id="a1")])
    reopened = session.process_frame(frames(5), 5, [event(5, "reopen_step", step_id="a1")])
    assert reopened["completion_rejected"] is False
    switched = session.process_frame(frames(6), 6, [event(6, "switch_task", task_id="b")])
    assert switched["completion_rejected"] is False


@pytest.mark.parametrize("invalid_uncertain", [0, "", None])
def test_falsey_nonboolean_uncertainty_cannot_release_guidance(invalid_uncertain):
    session, provider = bootstrap()
    provider.uncertain = invalid_uncertain
    with pytest.raises(ValueError, match="uncertainty"):
        session.process_frame(frames(2), 2, [event(2, "accept_step", step_id="a1")])
    # Rejected observations do not consume the event timestamp or commit a step.
    provider.uncertain = False
    retry = session.process_frame(frames(2), 2, [])
    assert retry["active_step_id"] is None and retry["guidance"] is None
