"""Read-only native-output audit supplement; never interprets answers for scoring."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent / "Research/Backend"))
import satori_paper_benchmark as benchmark


def audit(run_dir):
    run_dir = Path(run_dir)
    base = benchmark.audit_run(run_dir)
    manifest, _ = benchmark._read(run_dir / "manifest.json")
    taxonomy, _ = benchmark._read(run_dir / "taxonomy.json")
    samples = {row["sample_id"]: row for row in manifest["samples"]}
    errors, finishes, answers = Counter(), Counter(), Counter()
    silence = generated = 0
    latencies = []
    slots = [i for i in range(4) for _ in range(4)]
    for line in (run_dir / "api-calls.jsonl").read_text().splitlines():
        row = json.loads(line)
        errors[row.get("error_code", "success")] += 1
        finishes[row.get("finish_reason", "no_generation_result")] += 1
        latencies.append(row["latency_ms"])
        prep = row.get("preprocessing")
        if prep is None:
            continue
        sample = samples[row["sample_id"]]
        stamps = [frame["timestamp_ms"] for frame in sample["frames"]]
        visible = benchmark.visible_payload(sample, taxonomy)
        question = benchmark._json_text(visible) + "\n" + "\n".join(
            f"Frame ID: {frame['frame_id']}; timestamp_ms: {frame['timestamp_ms']}" for frame in visible["frames"])
        assert prep["unique_input_images"] == 4
        assert prep["model_frame_slots"] == 16
        assert prep["slot_to_unique_image"] == slots
        assert prep["unique_timestamps_ms"] == stamps
        assert prep["slot_timestamps_ms"] == [stamps[i] for i in slots]
        assert prep["clip_time_range_seconds"] == [stamps[0] / 1000, stamps[-1] / 1000]
        assert prep["user_text"] == question
        assert prep["reaction_called_once_per_user_query"] is True
        assert prep["native_streaming_trigger_exercised"] is False
        assert prep["checkpoint_weight_changes"] is False
        assert prep["model_revision"] == "7c5517eaaa4dc3788602af40e077063792c75f6c"
        assert row["usage"]["completion_tokens"] == len(row["generated_token_ids"])
        assert isinstance(row["raw_native_text"], str)
        if prep["keep_silent_emitted"]:
            assert "<keep_silent>" in row["raw_native_text"]
            assert row["raw_json_text"] == ""
            assert row["error_code"] == "NativeSilenceError"
            silence += 1
        elif row["finish_reason"] == "stop":
            answers[row["raw_json_text"]] += 1
        generated += 1
    return {"passed": True, "base_audit": base, "generated_outputs": generated,
            "native_silence_outputs": silence, "errors": dict(errors), "finish_reasons": dict(finishes),
            "unique_raw_answers": len(answers), "most_common_raw_answers": answers.most_common(10),
            "total_generation_seconds": sum(latencies) / 1000}


if __name__ == "__main__":
    result = audit(sys.argv[1])
    benchmark._write(Path(sys.argv[1]) / "native-audit.json", result)
    print(json.dumps(result), flush=True)
