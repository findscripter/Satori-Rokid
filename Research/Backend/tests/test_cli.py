import json
import subprocess
import sys

from satori_repro.cli import main


def call(*args):
    return subprocess.run([sys.executable, "-m", "satori_repro", *map(str, args)], capture_output=True, text=True, encoding="utf-8")


def test_demo_without_key_and_windows_utf8_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = tmp_path / "demo"
    result = call("demo", "--output", out)
    assert result.returncode == 0, result.stderr
    report = json.loads((out / "unity-report.json").read_text(encoding="utf-8"))
    assert len(report["records"]) == 9 and report["synthetic"] is True
    assert report["records"][-1]["decision_kind"] == "complete"
    before = (out / "timeline.jsonl").read_bytes()
    repeated = call("demo", "--output", out)
    assert repeated.returncode == 2 and repeated.stderr.startswith("error:")
    assert before == (out / "timeline.jsonl").read_bytes()


def test_direct_demo_evaluation_and_output_protection(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo", "--output", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["processed_frames"] == 9
    from importlib.resources import files
    labels = files("satori_repro").joinpath("data", "demo", "labels.json")
    score = tmp_path / "score.json"
    assert main(["evaluate", "--timeline", str(out / "timeline.jsonl"), "--labels", str(labels), "--output", str(score)]) == 0
    assert json.loads(score.read_text())["data_kind"] == "synthetic"
    assert main(["evaluate", "--timeline", str(out / "timeline.jsonl"), "--labels", str(labels), "--output", str(score)]) == 2


def test_infer_requires_model():
    result = call("infer", "--session", "missing", "--task", "missing", "--output", "unused")
    assert result.returncode == 2 and "--model" in result.stderr


def test_ask_context_requires_explicit_model_and_has_no_task_argument():
    from satori_repro.cli import make_parser
    parser = make_parser()
    args = parser.parse_args(["ask-context", "--session", "s.json", "--question", "我在哪？",
                              "--output", "out", "--model", "chosen-model"])
    assert args.window == 8 and args.at_ms is None and args.api_key_env == "SATORI_API_KEY"
    assert not hasattr(args, "task")
    result = call("ask-context", "--session", "s.json", "--question", "我在哪？", "--output", "out")
    assert result.returncode == 2 and "--model" in result.stderr


def test_ask_context_dispatches_one_question_without_reading_key(tmp_path, monkeypatch, capsys):
    from satori_repro import cli
    calls = []
    class FakeStage:
        def __init__(self, **kwargs):
            calls.append(("provider", kwargs))
    def fake_run(session, question, provider, output, **kwargs):
        calls.append(("run", str(session), question, str(output), kwargs))
        return {"status": "completed", "answer": "室内"}
    monkeypatch.setattr(cli, "SatoriStageProvider", FakeStage)
    monkeypatch.setattr(cli, "run_context_question", fake_run)
    assert cli.main(["ask-context", "--session", "s.json", "--question", "我在哪？",
                     "--output", str(tmp_path/"out"), "--model", "chosen-model",
                     "--window", "3", "--at-ms", "2000", "--sensors", "imu.json"]) == 0
    assert len(calls) == 2
    assert calls[0][1]["api_key_env"] == "SATORI_API_KEY"
    assert calls[1][-1] == {"window_size":3,"at_ms":2000,"sensors_path":__import__("pathlib").Path("imu.json")}
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_method_evaluation_requires_explicit_run_metadata():
    result = call("evaluate-method", "--timeline", "method.jsonl", "--labels", "labels.json")
    assert result.returncode == 2
    assert "--run-metadata" in result.stderr


def test_method_evaluation_dispatch_and_output_protection(tmp_path, monkeypatch, capsys):
    from pathlib import Path
    from satori_repro import cli

    calls = []

    def fake_evaluate_method(timeline, labels, run_metadata):
        calls.append((timeline, labels, run_metadata))
        return {"data_kind": "model", "state_active_step_accuracy": 0.5,
                "raw_checkpoint_status_accuracy": None}

    monkeypatch.setattr(cli, "evaluate_method", fake_evaluate_method, raising=False)
    score = tmp_path / "method-score.json"
    args = ["evaluate-method", "--timeline", "method.jsonl", "--labels", "labels.json",
            "--run-metadata", "run.json", "--output", str(score)]
    assert cli.main(args) == 0
    assert calls == [(Path("method.jsonl"), Path("labels.json"), Path("run.json"))]
    assert json.loads(capsys.readouterr().out)["state_active_step_accuracy"] == 0.5
    before = score.read_bytes()
    assert cli.main(args) == 2
    assert score.read_bytes() == before
    assert len(calls) == 1
    assert "already exists" in capsys.readouterr().err


def test_method_evaluation_error_does_not_write_a_score(tmp_path, monkeypatch, capsys):
    from satori_repro import cli

    def fail(*args):
        raise ValueError("No matching independently labeled method frames")

    monkeypatch.setattr(cli, "evaluate_method", fail, raising=False)
    score = tmp_path / "missing-score.json"
    assert cli.main(["evaluate-method", "--timeline", "method.jsonl", "--labels", "labels.json",
                     "--run-metadata", "run.json", "--output", str(score)]) == 2
    assert not score.exists()
    assert "independently labeled" in capsys.readouterr().err


def test_method_evaluation_cli_scores_a_synthetic_fixture(tmp_path):
    timeline = tmp_path / "method.jsonl"
    labels = tmp_path / "labels.json"
    metadata = tmp_path / "run.json"
    state = {"frame_id": "fixture-frame", "active_step_id": "fixture-step", "phase": "active_step",
             "confirmed_completed_step_ids": []}
    timeline.write_text(json.dumps({**state, "timestamp_ms": 0, "pending_confirmation": None,
                                     "guidance": None, "stage_calls": [], "events": []}) + "\n", encoding="utf-8")
    labels.write_text(json.dumps({"labels": [state]}), encoding="utf-8")
    metadata.write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    result = call("evaluate-method", "--timeline", timeline, "--labels", labels, "--run-metadata", metadata)
    assert result.returncode == 0, result.stderr
    score = json.loads(result.stdout)
    assert score["data_kind"] == "synthetic"
    assert score["state_active_step_accuracy"] == 1
    assert score["raw_checkpoint_status_accuracy"] is None
