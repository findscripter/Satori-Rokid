"""Recompute descriptive metrics from a Satori clip export (Python 3.9+, no dependencies).

    python Tools/analyze_replay_metrics.py path/to/extracted-clip --output metrics.json

Without --output, print JSON to stdout. Times are device-observed seconds. Medians
use answered turns with both required events. A clarification counts as an answer.
This measures delivery and timing; evaluation of visible facts requires separate labels.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median


def event_time(event):
    return None if event is None else event["elapsed_ns"]


def seconds(start, end):
    if start is None or end is None or end < start:
        return None
    return (end - start) / 1_000_000_000


def analyze(clip_dir):
    meta = json.loads((clip_dir / "metadata/clip.json").read_text(encoding="utf-8-sig"))
    start, end = meta["start_elapsed_ns_inclusive"], meta["end_elapsed_ns_exclusive"]
    if end <= start:
        raise ValueError("Invalid clip time window")

    def read_stream(name, clock):
        path = clip_dir / "session" / (name + ".jsonl")
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["session_id"] != meta["source_session"] or row["boot_id"] != meta["boot"]:
                    raise ValueError(f"{path.name}: session or boot does not match the clip")
                if start <= row[clock] < end:
                    yield row

    frames = {row["frame_id"]: row["ts_ns"] for row in read_stream("frames", "ts_ns")}
    events = sorted(read_stream("events", "elapsed_ns"), key=event_time)
    turns = {}
    requests = []
    speech = defaultdict(list)
    audio = defaultdict(dict)
    for event in events:
        name, data = event["name"], event["data"]
        if name == "speech_requested":
            speech[data["text"]].append(event)
        elif name in ("audio_started", "audio_done"):
            audio[data["id"]][name] = event["elapsed_ns"]
        elif name.startswith("dialogue_") or name in ("web_search_plan_request", "web_search_plan"):
            rid = data.get("request_id") or data.get("parent_request_id")
            if rid:
                turn = turns.setdefault(rid, {})
                if name == "dialogue_request":
                    if name in turn:
                        raise ValueError(f"Duplicate dialogue request: {rid}")
                    requests.append(rid)
                turn[name] = event

    results = []
    for i, rid in enumerate(requests):
        turn = turns[rid]
        request = event_time(turn["dialogue_request"])
        response = turn.get("dialogue_response")
        reply = event_time(response)
        outcome = next((tag for name, tag in [("dialogue_response", "answered"),
                        ("dialogue_discarded", "discarded"), ("dialogue_error", "error")]
                        if name in turn), "unfinished_in_clip")
        item = {"request_id": rid, "outcome": outcome, "video_s": seconds(start, request)}
        results.append(item)
        if response is None:
            terminal = turn.get("dialogue_discarded") or turn.get("dialogue_error")
            item["request_to_terminal_s"] = seconds(request, event_time(terminal))
            continue

        snapshot = turn.get("dialogue_visual_snapshot", {}).get("data", {})
        route = turn.get("dialogue_route", {}).get("data", {})
        ids = snapshot.get("frame_ids", [])
        image_count = route.get("frame_count")
        sampled = None
        if ids and len(ids) == image_count and all(fid in frames for fid in ids):
            sampled = max(frames[fid] for fid in ids)
        answer = response["data"]["answer"]
        # Speech submission has no request_id. Match exact text within this turn,
        # then match playback by its own ID; leave ambiguous submissions unassigned.
        next_request = event_time(turns[requests[i + 1]]["dialogue_request"]) if i + 1 < len(requests) else end
        matches = [e for e in speech[answer["text"]] if reply <= event_time(e) < next_request]
        speech_id = matches[0]["data"]["id"] if len(matches) == 1 else None
        playback = audio.get(speech_id, {})
        began, done = playback.get("audio_started"), playback.get("audio_done")
        item.update(answer_kind=answer["kind"], answer_frame_count=image_count, speech_id=speech_id,
                    request_to_snapshot_s=seconds(request, event_time(turn.get("dialogue_visual_snapshot"))),
                    search_decision_s=seconds(event_time(turn.get("web_search_plan_request")),
                                              event_time(turn.get("web_search_plan"))),
                    final_answer_stage_s=seconds(event_time(turn.get("dialogue_route")), reply),
                    request_to_text_s=seconds(request, reply), request_to_audio_s=seconds(request, began),
                    text_to_audio_s=seconds(reply, began), speech_duration_s=seconds(began, done),
                    newest_frame_to_audio_s=seconds(sampled, began))

    answered = [row for row in results if row["outcome"] == "answered"]
    metrics = {}
    for field in ("request_to_snapshot_s", "search_decision_s", "final_answer_stage_s", "request_to_text_s",
                  "request_to_audio_s", "text_to_audio_s", "speech_duration_s", "newest_frame_to_audio_s"):
        values = [row[field] for row in answered if row[field] is not None]
        metrics[field] = {"n": len(values)}
        if values:
            metrics[field].update(median=median(values), mean=mean(values), min=min(values), max=max(values))
    return {"requests": len(results), "outcomes": dict(Counter(row["outcome"] for row in results)),
            "reply_fraction": len(answered) / len(results) if results else None,
            "answered_with_frame": sum((r["answer_frame_count"] or 0) > 0 for r in answered),
            "playback_completed": sum(r["speech_duration_s"] is not None for r in answered),
            "seconds_for_answered_turns": metrics, "turns": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(analyze(args.clip_dir), ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
