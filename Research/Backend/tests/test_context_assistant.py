from copy import deepcopy

import pytest

from satori_repro.context_assistant import ContextAssistant, build_context_prompt


FRAMES = [{"frame_id": "v0", "timestamp_ms": 1000}, {"frame_id": "v1", "timestamp_ms": 2000}]


def answer():
    return {"answer": "你似乎在厨房，正在擦拭台面。", "environment": {"place_type": "厨房", "visible_objects": ["台面", "布"]},
            "activity": {"action": "擦拭台面", "phase": "in_progress"}, "uncertainties": ["无法从当前视角确认整个清洁任务是否完成"],
            "evidence_frame_ids": ["v1"], "sensor_evidence_ids": []}


def sample(sample_id="i0", timestamp=1500, *, mount="head", source="recorded"):
    return {"sample_id": sample_id, "timestamp_ms": timestamp,
            "accelerometer_m_s2": [0, 0, 9.81], "gyroscope_rad_s": [0, 0, 0.5], "mount": mount, "source": source}


class Provider:
    def __init__(self, response=None):
        self.response = answer() if response is None else response
        self.calls = []

    def complete_json(self, stage, frames, payload):
        self.calls.append((stage, deepcopy(frames), deepcopy(payload)))
        return self.response


def test_visual_question_is_independent_of_task_catalog_and_marks_missing_sensors():
    provider = Provider()
    assistant = ContextAssistant(provider)
    history = [{"timestamp_ms": 1000, "question": "我在哪？", "answer": "可能在厨房。"}]
    result = assistant.ask(FRAMES, "我身处什么环境，正在做什么？", history=history)
    stage, frames, payload = provider.calls[0]
    assert stage == "context_question" and frames == FRAMES
    assert payload["question"] == "我身处什么环境，正在做什么？"
    assert payload["history"] == history
    assert set(payload) == {"question", "history", "sensor_context"}
    assert payload["sensor_context"]["status"] == "missing"
    assert result == answer() and result["sensor_evidence_ids"] == []
    assert assistant.last_context["missing_sensors"] == ["imu"]
    assert assistant.last_context["available_modalities"] == ["vision"]


def test_real_imu_is_summarized_with_units_mount_provenance_and_ids():
    response = answer()
    response.update(answer="画面显示你在擦拭台面；头戴IMU测得头部转动，不能据此判断手的动作。", sensor_evidence_ids=["i0", "i1"])
    samples = [sample(), sample("i1", 1800), sample("w0", 1800, mount="wrist", source="live")]
    samples[1]["gyroscope_rad_s"] = [0, 0, 1.5]
    provider = Provider(response)
    assistant = ContextAssistant(provider)
    original = deepcopy(samples)
    result = assistant.ask(FRAMES, "我在做什么？", sensor_samples=samples)
    sensors = provider.calls[0][2]["sensor_context"]
    assert samples == original
    assert sensors["status"] == "provided" and sensors["sample_count"] == 3
    assert sensors["sample_ids"] == ["i0", "i1", "w0"]
    assert sensors["sources"] == ["live", "recorded"] and sensors["mounts"] == ["head", "wrist"]
    head = next(stream for stream in sensors["streams"] if stream["mount"] == "head")
    assert head["gyroscope_rad_s"]["magnitude_mean"] == 1.0
    assert head["accelerometer_m_s2"]["magnitude_mean"] == pytest.approx(9.81)
    assert "gravity" in sensors["limitations"][0].lower()
    assert result["sensor_evidence_ids"] == ["i0", "i1"]
    assert assistant.last_context["missing_sensors"] == []


@pytest.mark.parametrize("invalid", ["future", "nan", "inf", "bool", "dimension", "wrong_mount", "synthetic_source", "synthetic_extra", "duplicate", "out_of_order"])
def test_invalid_sensor_data_is_rejected_before_provider_call(invalid):
    samples = [sample()]
    if invalid == "future":
        samples[0]["timestamp_ms"] = 2001
    elif invalid in {"nan", "inf"}:
        samples[0]["accelerometer_m_s2"][0] = float(invalid)
    elif invalid == "bool":
        samples[0]["gyroscope_rad_s"][0] = True
    elif invalid == "dimension":
        samples[0]["gyroscope_rad_s"] = [1, 2]
    elif invalid == "wrong_mount":
        samples[0]["mount"] = "glasses-or-hand"
    elif invalid == "synthetic_source":
        samples[0]["source"] = "synthetic"
    elif invalid == "synthetic_extra":
        samples[0]["synthetic"] = True
    elif invalid == "duplicate":
        samples.append(sample())
    elif invalid == "out_of_order":
        samples.append(sample("earlier", 1000))
    provider = Provider()
    with pytest.raises(ValueError):
        ContextAssistant(provider).ask(FRAMES, "What am I doing?", sensor_samples=samples)
    assert not provider.calls


@pytest.mark.parametrize("invalid", ["empty_frames", "duplicate_frames", "unordered_frames", "bad_time", "empty_question", "history_future", "history_missing_time", "history_task_catalog"])
def test_invalid_visual_or_history_context_fails_before_model(invalid):
    frames, question, history = deepcopy(FRAMES), "What am I doing?", []
    if invalid == "empty_frames":
        frames = []
    elif invalid == "duplicate_frames":
        frames[1]["frame_id"] = "v0"
    elif invalid == "unordered_frames":
        frames.reverse()
    elif invalid == "bad_time":
        frames[1]["timestamp_ms"] = True
    elif invalid == "empty_question":
        question = " "
    else:
        history = [{"timestamp_ms": 1000, "question": "Where?", "answer": "Kitchen"}]
        if invalid == "history_future":
            history[0]["timestamp_ms"] = 2001
        elif invalid == "history_missing_time":
            del history[0]["timestamp_ms"]
        else:
            history[0]["task_catalog"] = {"answer": "cleaning"}
    provider = Provider()
    with pytest.raises(ValueError):
        ContextAssistant(provider).ask(frames, question, history=history)
    assert not provider.calls


@pytest.mark.parametrize("invalid", ["extra_field", "missing_field", "future_evidence", "empty_evidence", "false_sensor_ids", "imu_claim", "phase", "object_shape", "uncertainty_type"])
def test_invalid_model_response_is_not_returned_as_grounded_answer(invalid):
    response = answer()
    if invalid == "extra_field":
        response["task_id"] = "cleaning"
    elif invalid == "missing_field":
        del response["activity"]
    elif invalid == "future_evidence":
        response["evidence_frame_ids"] = ["v2"]
    elif invalid == "empty_evidence":
        response["evidence_frame_ids"] = []
    elif invalid == "false_sensor_ids":
        response["sensor_evidence_ids"] = ["imaginary-imu"]
    elif invalid == "imu_claim":
        response["answer"] = "根据IMU数据，你正在走动。"
    elif invalid == "phase":
        response["activity"]["phase"] = "task_complete_confirmed"
    elif invalid == "object_shape":
        response["environment"]["visible_objects"] = [{"name": "cup"}]
    else:
        response["uncertainties"] = "none"
    assistant = ContextAssistant(Provider(response))
    with pytest.raises(ValueError):
        assistant.ask(FRAMES, "What am I doing?")
    assert assistant.last_context["response_status"] == "failed"


def test_declared_sensor_usage_requires_sensor_evidence_even_when_data_was_supplied():
    response = answer()
    response["answer"] = "The IMU shows head rotation."
    with pytest.raises(ValueError):
        ContextAssistant(Provider(response)).ask(FRAMES, "What is happening?", sensor_samples=[sample()])


def test_missing_imu_disclosure_is_allowed_but_not_a_following_sensor_claim():
    response = answer()
    response["uncertainties"].append("未提供IMU数据。")
    assert ContextAssistant(Provider(response)).ask(FRAMES, "What is happening?")["sensor_evidence_ids"] == []
    response["answer"] = "未提供IMU数据。陀螺仪显示你正在转头。"
    with pytest.raises(ValueError):
        ContextAssistant(Provider(response)).ask(FRAMES, "What is happening?")


def test_inputs_response_and_audit_metadata_have_no_mutable_aliases():
    provider = Provider()
    assistant = ContextAssistant(provider)
    frames, samples = deepcopy(FRAMES), [sample()]
    original = deepcopy((frames, samples, provider.response))
    result = assistant.ask(frames, "What am I doing?", sensor_samples=samples)
    result["environment"]["visible_objects"].append("invented")
    metadata = assistant.last_context
    metadata["sensor_context"]["sample_ids"].append("invented")
    assert (frames, samples, provider.response) == original
    assert assistant.last_context["sensor_context"]["sample_ids"] == ["i0"]


def test_empty_sensor_list_is_explicitly_missing():
    assistant = ContextAssistant(Provider())
    assistant.ask(FRAMES, "What am I doing?", sensor_samples=[])
    assert assistant.last_context["sensor_context"]["status"] == "missing"
    assert assistant.last_context["sensor_context"]["reason"] == "empty"


def test_prompt_limits_head_motion_to_head_and_does_not_infer_identity_or_absolute_location():
    prompt = build_context_prompt()
    assert "JSON" in prompt and "sensor_evidence_ids" in prompt
    assert "head" in prompt.lower() and "hand" in prompt.lower()
    assert "identity" in prompt.lower() and "absolute" in prompt.lower()
    assert "task catalog" in prompt.lower()
