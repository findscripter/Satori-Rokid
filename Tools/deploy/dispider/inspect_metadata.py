"""CPU-only inspection of the actual pinned tokenizer/control tokens."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = AutoTokenizer.from_pretrained(args.model, use_fast=False, local_files_only=True)
decision = AutoTokenizer.from_pretrained(args.model / "perception_decision", local_files_only=True)
output = {"source": "actual pinned local tokenizer metadata, no model weights loaded",
          "reaction": {"vocabulary_length": len(root), "eos_token_id": root.eos_token_id,
                       "pad_token_id": root.pad_token_id,
                       "keep_silent_ids": root.encode("<keep_silent>")},
          "decision": {"vocabulary_length": len(decision),
                       "eos_token_id": decision.eos_token_id, "pad_token_id": decision.pad_token_id,
                       "has_answer_ids": decision.encode("<has_answer>"),
                       "to_do_ids": decision.encode("<to_do>")},
          "reaction_config": json.loads((args.model / "config.json").read_text()),
          "decision_config": json.loads((args.model / "perception_decision/config.json").read_text())}
args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps({"reaction": output["reaction"], "decision": output["decision"]}, indent=2))
