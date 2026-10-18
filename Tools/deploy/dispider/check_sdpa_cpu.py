"""Check causal masks, padding, and incremental KV on tiny random CPU Qwen2."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["DISPIDER_ATTN_IMPLEMENTATION"] = "sdpa"
    sys.path.insert(0, str(args.runtime))
    import torch
    import transformers
    from transformers import Qwen2Config
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as Original
    from dispider.model.decision_backbone.adapter import Qwen2Model as Adapted

    torch.manual_seed(6381)
    torch.set_num_threads(2)
    parameters = dict(vocab_size=97, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128,
                      attention_dropout=0.0, pad_token_id=0)
    config = Qwen2Config(**parameters)
    config._attn_implementation = "eager"
    eager = Original(config).eval().cpu()
    sdpa = Adapted(Qwen2Config(**parameters)).eval().cpu()
    sdpa.load_state_dict(eager.state_dict(), strict=True)
    ids = torch.tensor([[3, 7, 11, 13, 17, 19, 23, 29]])
    checks = []

    def compare(name, actual, expected):
        delta = float((actual - expected).abs().max())
        passed = torch.allclose(actual, expected, atol=3e-5, rtol=3e-5)
        checks.append({"name": name, "max_absolute_difference": delta, "passed": bool(passed)})
        if not passed:
            raise AssertionError(f"{name}: {delta}")

    with torch.inference_mode():
        baseline = eager(ids, use_cache=False).last_hidden_state
        actual = sdpa(ids, use_cache=False).last_hidden_state
        compare("eager_vs_sdpa_unpadded", actual, baseline)
        changed = ids.clone()
        changed[:, 4:] = torch.tensor([[31, 37, 41, 43]])
        causal = sdpa(changed, use_cache=False).last_hidden_state
        compare("future_tokens_cannot_change_prefix", causal[:, :4], actual[:, :4])
        for side in ("right", "left"):
            pad = torch.zeros((1, 3), dtype=torch.long)
            padded = torch.cat((ids, pad), 1) if side == "right" else torch.cat((pad, ids), 1)
            mask = padded.ne(0).long()
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            lhs = sdpa(padded, attention_mask=mask, position_ids=positions,
                       use_cache=False).last_hidden_state
            rhs = eager(padded, attention_mask=mask, position_ids=positions,
                        use_cache=False).last_hidden_state
            compare(f"eager_vs_sdpa_{side}_padding", lhs[mask.bool()], rhs[mask.bool()])
            compare(f"{side}_padding_preserves_live_tokens", lhs[mask.bool()], actual[0])
        cache = DynamicCache()
        prefix = sdpa(ids[:, :4], past_key_values=cache, use_cache=True,
                      cache_position=torch.arange(4)).last_hidden_state
        suffix = sdpa(ids[:, 4:], attention_mask=torch.ones_like(ids), past_key_values=cache,
                      use_cache=True, cache_position=torch.arange(4, 8)).last_hidden_state
        compare("incremental_kv_equals_full_pass", torch.cat((prefix, suffix), 1), actual)
        rejected = False
        try:
            sdpa(ids[:, :1], past_key_values=cache, use_cache=True,
                 cache_position=torch.tensor([0]))
        except ValueError:
            rejected = True
        checks.append({"name": "noncontiguous_cache_positions_rejected", "passed": rejected})
        assert rejected
    output = {
        "status": "passed", "device": "cpu", "python": platform.python_version(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "checks": checks, "weights": "tiny randomly initialized decoder; no checkpoint weights loaded",
        "limitation": "Validates mathematical causal/padding/cache behavior only; original flash-attn kernel and full-model equivalence untested.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
