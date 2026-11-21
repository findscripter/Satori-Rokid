import json

from analyze_replay_metrics import analyze


def test_clip_uses_request_and_audio_ids_and_excludes_cancelled_turn(tmp_path):
    ns = 1_000_000_000
    (tmp_path / "metadata").mkdir()
    (tmp_path / "session").mkdir()
    (tmp_path / "metadata/clip.json").write_text(json.dumps({
        "source_session": "run", "boot": "boot",
        "start_elapsed_ns_inclusive": ns, "end_elapsed_ns_exclusive": 21 * ns,
    }))
    identity = {"session_id": "run", "boot_id": "boot"}

    def event(at, name, **data):
        return dict(identity, elapsed_ns=at * ns, name=name, data=data)

    frame = dict(identity, ts_ns=3 * ns, frame_id="current")
    (tmp_path / "session/frames.jsonl").write_text(json.dumps(frame))
    events = [
        event(1, "dialogue_request", request_id="a"),
        event(2, "dialogue_discarded", request_id="a"),
        event(3, "dialogue_request", request_id="b"),
        event(4, "dialogue_visual_snapshot", request_id="b", frame_ids=["current"]),
        event(4, "web_search_plan_request", parent_request_id="b"),
        event(6, "web_search_plan", parent_request_id="b"),
        event(7, "dialogue_route", request_id="b", frame_count=1),
        event(10, "dialogue_response", request_id="b", answer={"kind": "clarify", "text": "Which?"}),
        event(11, "speech_requested", id="other", text="Device status"),
        event(12, "audio_started", id="other"),
        event(13, "speech_requested", id="answer", text="Which?"),
        event(14, "audio_started", id="answer"),
        event(16, "audio_done", id="answer"),
        event(19, "dialogue_request", request_id="c"),
        event(20, "dialogue_response", request_id="c", answer={"kind": "answer", "text": "Which?"}),
        event(21, "dialogue_request", request_id="outside"),
    ]
    (tmp_path / "session/events.jsonl").write_text("\n".join(map(json.dumps, events)))
    report = analyze(tmp_path)
    assert report["requests"] == 3
    assert report["outcomes"] == {"discarded": 1, "answered": 2}
    assert report["playback_completed"] == report["answered_with_frame"] == 1
    assert report["turns"][1]["speech_duration_s"] == 2
    assert report["turns"][2]["request_to_audio_s"] is None
    metrics = report["seconds_for_answered_turns"]
    assert metrics["request_to_audio_s"]["median"] == 11
    assert metrics["newest_frame_to_audio_s"]["median"] == 11
    assert metrics["request_to_text_s"]["median"] == 4
