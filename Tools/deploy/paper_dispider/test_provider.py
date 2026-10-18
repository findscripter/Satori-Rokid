"""CPU contract tests using recorded fake generations, never performance claims."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import satori_paper_benchmark as benchmark

_SPEC = importlib.util.spec_from_file_location("paper_dispider_provider", Path(__file__).with_name("provider.py"))
provider = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(provider)


class FakeBackend:
    def __init__(self, text='{"answer":"g0","evidence_frame_ids":["v000"]}',
                 silence=False, finish="stop", failure=None):
        self.text, self.silence, self.finish, self.failure = text, silence, finish, failure
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        return {"text": self.text, "raw_native_text": "<keep_silent><|im_end|>" if self.silence else self.text,
                "finish_reason": self.finish, "prompt_tokens": 100, "completion_tokens": 3,
                "generated_token_ids": [1, 2, 3], "model_revision": provider.MODEL_REVISION,
                "preprocessing": {"keep_silent_emitted": self.silence}}


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sample = {"sample_id": "s000", "task_kind": "goal_recognition", "question": "What task?",
                       "cutoff_ms": 3000, "candidate_group": "goals", "frames": []}
        for index in range(4):
            path = self.root / f"frame-{index}.png"
            Image.new("RGB", (8, 8), (index * 40, index, 1)).save(path)
            self.sample["frames"].append({"frame_id": f"v{index:03d}", "timestamp_ms": index * 1000,
                                          "image_path": str(path), "sha256": benchmark._sha(path.read_bytes())})
        self.manifest = {"schema_version": 1, "benchmark_id": "fixed", "samples": [self.sample]}
        self.taxonomy = {"schema_version": 1, "goals": [{"id": "g0", "label": "make tea"}], "actions": []}

    def tearDown(self):
        self.temp.cleanup()

    def test_four_sources_expand_only_to_documented_sixteen_slots(self):
        frames = [object() for _ in range(4)]
        slots = provider.native.expand_frame_slots(frames)
        self.assertEqual(len(slots), 16)
        self.assertEqual([frames.index(value) for value in slots], [i for i in range(4) for _ in range(4)])
        for count in (0, 3, 5, 8):
            with self.assertRaises(ValueError):
                provider.native.expand_frame_slots([object() for _ in range(count)])

    def test_preserves_prompt_pixels_order_candidates_and_no_gt(self):
        backend = FakeBackend()
        instance = provider.PaperDispiderProvider(backend)
        instance.predict(deepcopy(self.sample), self.taxonomy)
        call = backend.calls[0]
        self.assertEqual(call["system_prompt"], benchmark.PROMPT)
        self.assertEqual(call["context"], benchmark.visible_payload(self.sample, self.taxonomy))
        self.assertEqual(call["max_new_tokens"], 4096)
        self.assertEqual(call["temperature"], 0)
        self.assertEqual([frame.getpixel((0, 0))[0] for frame in call["frames"]], [0, 40, 80, 120])
        text = json.dumps(call["context"])
        for forbidden in ("sample_id", "s000", "image_path", "sha256", "gt", "frame-"):
            self.assertNotIn(forbidden, text)

    def test_silence_retains_raw_special_tokens_and_is_not_success(self):
        backend = FakeBackend(text="", silence=True)
        instance = provider.PaperDispiderProvider(backend)
        with self.assertRaises(provider.NativeSilenceError):
            instance.predict(self.sample, self.taxonomy)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(instance.traces[0]["raw_native_text"], "<keep_silent><|im_end|>")
        self.assertEqual(instance.traces[0]["generated_token_ids"], [1, 2, 3])
        self.assertEqual(instance.traces[0]["status"], "error")
        self.assertEqual(instance.traces[0]["finish_reason"], "stop")

    def test_truncation_is_not_repaired_or_retried(self):
        instance = provider.PaperDispiderProvider(FakeBackend(finish="length"))
        with self.assertRaises(ValueError):
            instance.predict(self.sample, self.taxonomy)
        self.assertEqual(len(instance.backend.calls), 1)
        self.assertEqual(instance.traces[0]["finish_reason"], "length")

    def test_invalid_json_is_not_repaired_or_retried(self):
        instance = provider.PaperDispiderProvider(FakeBackend(text='```json\n{"answer":"g0"}\n```'))
        with self.assertRaises(ValueError):
            instance.predict(self.sample, self.taxonomy)
        self.assertEqual(len(instance.backend.calls), 1)

    def test_changed_image_is_rejected_before_call(self):
        instance = provider.PaperDispiderProvider(FakeBackend())
        Path(self.sample["frames"][0]["image_path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            instance.predict(self.sample, self.taxonomy)
        self.assertEqual(instance.backend.calls, [])
        self.assertEqual(instance.traces, [])

    def test_backend_exception_records_one_failed_attempt(self):
        instance = provider.PaperDispiderProvider(FakeBackend(failure=RuntimeError("technical")))
        with self.assertRaises(RuntimeError):
            instance.predict(self.sample, self.taxonomy)
        self.assertEqual(len(instance.traces), 1)
        self.assertEqual(instance.traces[0]["error_code"], "RuntimeError")

    def test_infer_audits_silence_with_fixed_denominator(self):
        for name, value in (("manifest", self.manifest), ("taxonomy", self.taxonomy), ("protocol", {"frozen": True})):
            benchmark._write(self.root / (name + ".json"), value)
        instance = provider.PaperDispiderProvider(FakeBackend(text="", silence=True))
        output = self.root / "run"
        benchmark.infer(self.root / "manifest.json", self.root / "taxonomy.json", output, instance,
                        protocol_path=self.root / "protocol.json")
        audit = benchmark.audit_run(output)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["api_calls"], 1)
        self.assertEqual(audit["successful_samples"], 0)
        self.assertEqual(audit["failed_samples"], 1)


if __name__ == "__main__":
    unittest.main()
