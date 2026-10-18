"""Native32 backend with a separately recorded, block-equivalent vision shim."""

import importlib.util
from pathlib import Path
import sys

from chunked_sdpa import chunked_sdpa_attention

SPEC = importlib.util.spec_from_file_location("frozen_native32_dense_backend", Path(__file__).with_name("backend32.py"))
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)


class Backend(base.Backend):
    def __init__(self, model_path):
        super().__init__(model_path)
        module = sys.modules[self.model.__class__.__module__]
        functions = module.VL_VISION_ATTENTION_FUNCTIONS
        if functions["sdpa"].__name__ != "sdpa_attention":
            raise ValueError("unexpected_existing_vision_attention_override")
        self._original_vision_sdpa = functions["sdpa"]
        functions["sdpa"] = chunked_sdpa_attention

    def generate(self, **kwargs):
        value = super().generate(**kwargs)
        value["preprocessing"]["vision_attention_execution"] = "per-native-cu_seqlens-block SDPA"
        value["preprocessing"]["global_dense_mask_avoided"] = True
        value["preprocessing"]["native_visibility_boundaries_preserved"] = True
        value["preprocessing"]["floating_point_kernel_equivalence_scope"] = "same block attention function; numerical rounding/kernel selection may differ"
        return value
