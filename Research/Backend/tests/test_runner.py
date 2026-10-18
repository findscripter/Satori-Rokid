import json
from pathlib import Path

import pytest

from satori_repro.runner import run_session


def inputs(tmp_path):
    task = {"task_id": "display", "title": "连接显示器", "steps": [
        {"step_id": "cable", "instruction": "连接视频线", "completion_criteria": "两端已连接"},
        {"step_id": "power", "instruction": "打开显示器", "completion_criteria": "显示画面"}]}
    session = {"session_id": "causal", "frames": [{"frame_id": f"f{i}", "timestamp_ms": i * 1000} for i in range(3)],
               "events": [{"timestamp_ms": 0, "type": "confirm_task", "task_id": "display"},
                          {"timestamp_ms": 10000, "type": "confirm_step", "step_id": "cable"}]}
    paths = tmp_path / "task.json", tmp_path / "session.json"
    for path, data in zip(paths, (task, session)):
        path.write_text(json.dumps(data), encoding="utf-8")
    return paths


class Provider:
    name, synthetic = "test-replay", True

    def __init__(self, fail_on=None, bad_evidence=False):
        self.calls, self.fail_on, self.bad_evidence = [], fail_on, bad_evidence

    def infer(self, frames, *, task, history, state):
        self.calls.append({"frames": frames, "history": history, "state": state})
        if len(self.calls) == self.fail_on:
            raise RuntimeError("secret-that-must-not-be-recorded")
        return {"scene": "桌面", "objects": [], "action": "检查视频线", "current_step_id": "cable",
                "completed_step_ids": [], "confidence": 0.9, "uncertain": False,
                "evidence_frame_ids": ["future" if self.bad_evidence else frames[-1]["frame_id"]],
                "needs_help": False, "help_text": ""}


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_causal_windows_histories_and_future_events(tmp_path):
    task, session = inputs(tmp_path)
    provider, out = Provider(), tmp_path / "out"
    result = run_session(session, task, provider, out, window_size=2, history_limit=1)
    assert result["processed_frames"] == 3
    assert [[f["frame_id"] for f in c["frames"]] for c in provider.calls] == [["f0"], ["f0", "f1"], ["f1", "f2"]]
    assert [len(c["history"]) for c in provider.calls] == [0, 1, 1]
    assert all(r["state"]["confirmed_completed_step_ids"] == [] for r in rows(out / "timeline.jsonl"))
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "completed" and run["unconsumed_events"] == 1
    assert len(run["inputs"]["session_sha256"]) == 64
    assert json.loads((out / "unity-report.json").read_text(encoding="utf-8"))["synthetic"] is True


@pytest.mark.parametrize("mode", ["single", "window"])
def test_baselines_do_not_receive_state_or_history(tmp_path, mode):
    task, session = inputs(tmp_path)
    provider = Provider()
    run_session(session, task, provider, tmp_path / "out", context_mode=mode)
    assert all(c["state"] is None and c["history"] == [] for c in provider.calls)
    if mode == "single":
        assert all(len(c["frames"]) == 1 for c in provider.calls)


def test_partial_failure_is_marked_without_secret_leak(tmp_path):
    task, session = inputs(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="inference"):
        run_session(session, task, Provider(fail_on=2), out)
    assert len(rows(out / "timeline.jsonl")) == 1
    text = (out / "run.json").read_text()
    assert json.loads(text)["status"] == "failed" and "secret-that" not in text


def test_invalid_evidence_is_rejected(tmp_path):
    task, session = inputs(tmp_path)
    with pytest.raises(RuntimeError, match="validation"):
        run_session(session, task, Provider(bad_evidence=True), tmp_path / "out")


def test_existing_output_not_overwritten(tmp_path):
    task, session = inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep").write_text("keep")
    with pytest.raises(FileExistsError):
        run_session(session, task, Provider(), out)
    assert (out / "keep").read_text() == "keep"


@pytest.mark.parametrize("kwargs", [{"context_mode": "typo"}, {"window_size": 0}, {"history_limit": -1}, {"cooldown_ms": -1}])
def test_bad_config_does_not_create_run(tmp_path, kwargs):
    task, session = inputs(tmp_path)
    with pytest.raises(ValueError):
        run_session(session, task, Provider(), tmp_path / "out", **kwargs)
    assert not (tmp_path / "out").exists()
