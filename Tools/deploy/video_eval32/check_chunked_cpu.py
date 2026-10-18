"""Compare the shim against dense masked native SDPA, including block isolation."""

import argparse
import json
from pathlib import Path

from chunked_sdpa import chunked_sdpa_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as functional
    torch.manual_seed(20261017)
    records = []
    for dtype in (torch.float32, torch.bfloat16):
        for lengths in ([7], [5, 7], [4, 4, 4, 4, 4, 4, 4, 4], [3, 7, 2]):
            n, heads, dimension = sum(lengths), 3, 8
            q, k, v = [torch.randn(n, heads, dimension, dtype=dtype) for _ in range(3)]
            bounds = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
            mask = torch.zeros(1, n, n, dtype=torch.bool)
            for a, b in zip(bounds.tolist(), bounds.tolist()[1:]):
                mask[:, a:b, a:b] = True
            dense = functional.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), mask, dropout_p=0.0).transpose(0, 1).reshape(n, -1)
            actual = chunked_sdpa_attention(q, k, v, bounds, bounds)
            difference = (dense.float()-actual.float()).abs().max().item()
            tolerance = 1e-5 if dtype == torch.float32 else 0.02
            assert torch.allclose(dense, actual, atol=tolerance, rtol=tolerance)
            # Changing a different native temporal group cannot affect group 0.
            if len(lengths) > 1:
                changed = v.clone()
                changed[lengths[0]:] += 100
                perturbed = chunked_sdpa_attention(q, k, changed, bounds, bounds)
                assert torch.equal(actual[:lengths[0]], perturbed[:lengths[0]])
            records.append({"dtype": str(dtype), "native_lengths": lengths, "max_absolute_error": difference,
                            "tolerance": tolerance, "block_isolation": True})
    assert not torch.cuda.is_initialized()
    value = {"passed": True, "cpu_only": True, "model_loaded": False, "cuda_initialized": False,
             "cases": records, "scope": "Mathematical block semantics and CPU numerics; GPU kernel rounding is not proven bitwise identical"}
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value))


if __name__ == "__main__":
    main()
