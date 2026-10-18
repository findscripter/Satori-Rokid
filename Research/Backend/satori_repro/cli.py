import argparse
from importlib.resources import files
import json
from pathlib import Path
import sys

from .evaluate import evaluate, evaluate_method
from .context_runner import run_context_question
from .stage_provider import SatoriStageProvider
from .providers import OpenAICompatibleProvider, ReplayProvider
from .runner import run_session, write_json
from .video import extract_video


def make_parser():
    parser = argparse.ArgumentParser(description="Offline task-state replay for Satori reproduction")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Synthetic regression; no network")
    demo.add_argument("--output", type=Path, required=True)
    for name in ("replay", "infer"):
        item = sub.add_parser(name)
        item.add_argument("--session", type=Path, required=True)
        item.add_argument("--task", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--mode", choices=["single", "window", "stateful"], default="stateful")
        item.add_argument("--window", type=int, default=4)
        item.add_argument("--history-limit", type=int, default=8)
        item.add_argument("--cooldown-ms", type=int, default=15000)
        if name == "replay":
            item.add_argument("--responses", type=Path, required=True)
        else:
            item.add_argument("--model", required=True)
            item.add_argument("--base-url", default="https://api.openai.com/v1")
            item.add_argument("--api-key-env", default="OPENAI_API_KEY")
            item.add_argument("--timeout", type=float, default=60)
    context = sub.add_parser("ask-context", help="Ask once about environment and action; no task catalog")
    context.add_argument("--session", type=Path, required=True)
    context.add_argument("--question", required=True)
    context.add_argument("--output", type=Path, required=True)
    context.add_argument("--model", required=True)
    context.add_argument("--base-url", default="https://api.openai.com/v1")
    context.add_argument("--api-key-env", default="SATORI_API_KEY")
    context.add_argument("--timeout", type=float, default=90)
    context.add_argument("--window", type=int, default=8)
    context.add_argument("--at-ms", type=int)
    context.add_argument("--sensors", type=Path)
    extract = sub.add_parser("extract-video")
    extract.add_argument("--video", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--fps", type=float, default=1)
    extract.add_argument("--max-frames", type=int, default=600)
    score = sub.add_parser("evaluate")
    score.add_argument("--timeline", type=Path, required=True)
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--output", type=Path)
    method_score = sub.add_parser("evaluate-method", help="Score staged Satori logs against independent method labels")
    method_score.add_argument("--timeline", type=Path, required=True)
    method_score.add_argument("--labels", type=Path, required=True)
    method_score.add_argument("--run-metadata", type=Path, required=True)
    method_score.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = make_parser().parse_args(argv)
    try:
        if args.command == "demo":
            data = Path(str(files("satori_repro").joinpath("data", "demo")))
            result = run_session(data / "session.json", data / "task.json", ReplayProvider(data / "responses.json"), args.output)
            write_json(args.output / "evaluation.json", evaluate(args.output / "timeline.jsonl", data / "labels.json"))
        elif args.command in {"replay", "infer"}:
            provider = ReplayProvider(args.responses) if args.command == "replay" else OpenAICompatibleProvider(
                model=args.model, base_url=args.base_url, api_key_env=args.api_key_env, timeout=args.timeout)
            result = run_session(args.session, args.task, provider, args.output, context_mode=args.mode,
                                 window_size=args.window, history_limit=args.history_limit, cooldown_ms=args.cooldown_ms)
        elif args.command == "ask-context":
            provider = SatoriStageProvider(model=args.model, base_url=args.base_url,
                                           api_key_env=args.api_key_env, timeout=args.timeout)
            result = run_context_question(args.session, args.question, provider, args.output,
                                          window_size=args.window, at_ms=args.at_ms, sensors_path=args.sensors)
        elif args.command == "extract-video":
            result = {"session_path": str(extract_video(args.video, args.output, fps=args.fps, max_frames=args.max_frames))}
        else:
            if args.output and args.output.exists():
                raise FileExistsError("Evaluation output already exists")
            result = (evaluate_method(args.timeline, args.labels, args.run_metadata)
                      if args.command == "evaluate-method" else evaluate(args.timeline, args.labels))
            if args.output:
                write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
