"""CPU-only adapter tests; no weights, API, torch installation or GPU needed."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import satori_paper_benchmark as benchmark

_SPEC = importlib.util.spec_from_file_location("paper_vc3_provider", Path(__file__).with_name("provider.py"))
provider = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(provider)


class FakeTensor:
    def __init__(self, array):
        self.array = array

    def permute(self, *order):
        self.array = self.array.transpose(order)
        return self


class FakeProcessor:
    video_processor = SimpleNamespace(temporal_merge_size=2)

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "frozen-chat-template"

    def __call__(self, **kwargs):
        self.received = kwargs
        count = kwargs["videos"][0].array.shape[0]
        return {"video_grid_thw": np.array([[count, 2, 2]])}


class FakeBackend:
    def __init__(self, text=None, finish="stop", failure=None):
        self.text = text or '{"answer":"g0","evidence_frame_ids":["v000"]}'
        self.finish, self.failure, self.calls = finish, failure, []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        return {"text": self.text, "finish_reason": self.finish, "prompt_tokens": 100,
                "completion_tokens": 12, "model_revision": provider.native.MODEL_REVISION}


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
        self.metadata = {"fps": 30., "total_num_frames": 300, "duration": 10., "width": 8,
                         "height": 8, "frames_indices": [0, 30, 60, 90], "video_backend": "opencv"}
        self.registry = {"benchmark_id": "fixed", "samples": {"s000": self.metadata}}

    def tearDown(self):
        self.temp.cleanup()

    def make_provider(self, backend=None):
        return provider.PaperVideoChat3Provider(backend or FakeBackend(), registry=self.registry,
                                               manifest=self.manifest)

    def test_four_unique_frames_and_same_prompt_payload(self):
        backend = FakeBackend()
        instance = self.make_provider(backend)
        value = instance.predict(deepcopy(self.sample), self.taxonomy)
        call = backend.calls[0]
        self.assertEqual(call["system_prompt"], benchmark.PROMPT)
        self.assertEqual(call["context"], benchmark.visible_payload(self.sample, self.taxonomy))
        self.assertEqual(call["max_new_tokens"], 4096)
        self.assertEqual(len(call["frames"]), 4)
        self.assertEqual([frame.getpixel((0, 0))[0] for frame in call["frames"]], [0, 40, 80, 120])
        text = json.dumps(call["context"])
        for forbidden in ("sample_id", "s000", "image_path", "sha256", "gt", "frame-", "total_num_frames"):
            self.assertNotIn(forbidden, text)
        self.assertEqual(value["answer"], "g0")
        self.assertEqual(instance.traces[0]["status"], "success")

    def test_registry_rejects_source_id_and_arbitrary_backend_text(self):
        for name, value in (("source_video_id", "secret"), ("video_backend", "future label"),
                            ("width", "future label"), ("duration", float("nan"))):
            with self.subTest(name=name):
                bad = deepcopy(self.registry)
                bad["samples"]["s000"][name] = value
                with self.assertRaises((ValueError, TypeError)):
                    provider.validate_registry(bad, self.manifest)

    def test_registry_rejects_duplicates_future_and_misaligned_timestamps(self):
        for indices in ([0, 30, 30, 90], [0, 30, 60, 120], [0, 30, 60, 300]):
            with self.subTest(indices=indices):
                bad = deepcopy(self.registry)
                bad["samples"]["s000"]["frames_indices"] = indices
                with self.assertRaises(ValueError):
                    provider.validate_registry(bad, self.manifest)

    def test_sha_change_fails_before_model_attempt(self):
        instance = self.make_provider()
        Path(self.sample["frames"][0]["image_path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            instance.predict(self.sample, self.taxonomy)
        self.assertEqual(instance.backend.calls, [])
        self.assertEqual(instance.traces, [])

    def test_invalid_json_no_repair_no_retry(self):
        for text in ('```json\n{"answer":"g0","evidence_frame_ids":["v000"]}\n```',
                     '{"answer":"g0","answer":"g0","evidence_frame_ids":["v000"]}',
                     '{"answer":"g0","evidence_frame_ids":["v000"],"explanation":"x"}',
                     '{"answer":"unknown","evidence_frame_ids":["v000"]}'):
            with self.subTest(text=text):
                instance = self.make_provider(FakeBackend(text=text))
                with self.assertRaises(ValueError):
                    instance.predict(self.sample, self.taxonomy)
                self.assertEqual(len(instance.backend.calls), 1)
                self.assertEqual(instance.traces[0]["raw_json_text"], text)
                self.assertEqual(instance.traces[0]["status"], "error")

    def test_length_even_valid_json_is_failure(self):
        instance = self.make_provider(FakeBackend(finish="length"))
        with self.assertRaisesRegex(ValueError, "EOS"):
            instance.predict(self.sample, self.taxonomy)
        self.assertEqual(instance.traces[0]["finish_reason"], "length")
        self.assertIn("raw_json", instance.traces[0])

    def test_model_failure_preserves_single_attempt(self):
        instance = self.make_provider(FakeBackend(failure=RuntimeError("oom")))
        with self.assertRaises(RuntimeError):
            instance.predict(self.sample, self.taxonomy)
        self.assertEqual(len(instance.traces), 1)
        self.assertEqual(instance.traces[0]["status"], "error")

    def test_infer_and_existing_audit_contract(self):
        manifest_path = self.root / "manifest.json"
        taxonomy_path = self.root / "taxonomy.json"
        protocol_path = self.root / "protocol.json"
        registry_path = self.root / "registry.json"
        for path, value in ((manifest_path, self.manifest), (taxonomy_path, self.taxonomy),
                            (registry_path, self.registry)):
            benchmark._write(path, value)
        registry_raw = registry_path.read_bytes()
        benchmark._write(protocol_path, {"description": "CPU adapter test",
                                       "technical_metadata_registry_sha256": benchmark._sha(registry_raw)})
        instance = provider.PaperVideoChat3Provider(FakeBackend(), registry=self.registry,
                manifest=self.manifest, registry_sha256=benchmark._sha(registry_raw))
        result = benchmark.infer(manifest_path, taxonomy_path, self.root / "output", instance,
                                 protocol_path=protocol_path)
        (self.root / "output" / "metadata-registry.json").write_bytes(registry_raw)
        self.assertEqual(result["successful_samples"], 1)
        self.assertTrue(benchmark.audit_run(self.root / "output")["passed"])
        self.assertTrue(provider.audit_native_metadata(self.root / "output", self.registry)["passed"])
        changed = deepcopy(self.registry)
        changed["samples"]["s000"]["fps"] = 60.
        with self.assertRaises(ValueError):
            provider.audit_native_metadata(self.root / "output", changed)

    def test_cpu_native_processor_keeps_four_and_eight(self):
        instance = object.__new__(provider.PaperBackend)
        instance.processor = FakeProcessor()
        for count in (4, 8):
            with self.subTest(count=count), patch.dict(sys.modules, {"torch": SimpleNamespace(from_numpy=FakeTensor)}):
                metadata = deepcopy(self.metadata)
                metadata["frames_indices"] = [i * 30 for i in range(count)]
                context = {"frames": [{"frame_id": f"v{i:03d}", "timestamp_ms": i * 1000} for i in range(count)]}
                frames = [Image.new("RGB", (8, 8), (i, 0, 0)) for i in range(count)]
                _, prep = instance.prepare_inputs(system_prompt=benchmark.PROMPT, context=context,
                                                  frames=frames, frame_metadata=metadata)
                received = instance.processor.received
                self.assertEqual(received["videos"][0].array.shape, (count, 3, 8, 8))
                self.assertEqual(received["videos"][0].array[:, 0, 0, 0].tolist(), list(range(count)))
                self.assertFalse(received["videos_kwargs"]["do_sample_frames"])
                self.assertEqual(prep["original_input_frames"], count)
                self.assertEqual(prep["timestamps_seconds"], list(range(count)))
                self.assertEqual(instance.processor.messages[0]["content"], benchmark.PROMPT)

    def test_loader_requires_allocation_before_torch_import(self):
        with patch.dict(provider.native.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "requires_gpurun"):
                provider.PaperBackend("not_opened")


if __name__ == "__main__":
    unittest.main()
