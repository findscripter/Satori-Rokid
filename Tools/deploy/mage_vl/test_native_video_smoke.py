"""Dependency-free CPU checks; no model, CUDA, credentials, or network access."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("native_video_smoke.py")
SPEC = importlib.util.spec_from_file_location("native_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


class NativeSmokeTests(unittest.TestCase):
    def test_help_has_no_heavy_import(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--backend", result.stdout)
        self.assertNotIn("torch", smoke.__dict__)

    def test_size_dict_dataclass_bounds(self):
        @dataclasses.dataclass
        class Size:
            shortest_edge: int = 200704
            longest_edge: int = 1605632
            height: object = None
        p = types.SimpleNamespace(image_processor=types.SimpleNamespace(size=Size()), video_processor=types.SimpleNamespace(min_pixels=200704, max_pixels=1605632))
        smoke.configure_pixels(p)
        self.assertIsInstance(p.image_processor.size, Size)
        self.assertEqual(p.image_processor.size.shortest_edge, 3136)
        self.assertEqual(p.image_processor.size.longest_edge, 150000)
        self.assertEqual((p.video_processor.min_pixels, p.video_processor.max_pixels), (3136, 150000))

    def test_mapping_size_bounds(self):
        p = types.SimpleNamespace(image_processor=types.SimpleNamespace(size={"shortest_edge": 1, "longest_edge": 2}), video_processor=types.SimpleNamespace())
        smoke.configure_pixels(p)
        self.assertEqual(p.image_processor.size, {"shortest_edge": 3136, "longest_edge": 150000})

    def test_terminal_eos_means_stop(self):
        self.assertEqual(smoke.finish_reason([10, 151645], {151645}, 2), "stop")

    def test_truncation_is_not_stop(self):
        self.assertEqual(smoke.finish_reason([10, 11], {151645}, 2), "length")
        self.assertEqual(smoke.finish_reason([10], {151645}, 2), "unknown")
        self.assertEqual(smoke.finish_reason([], {151645}, 2), "unknown")

    def arguments(self, root):
        model = root / "model"; model.mkdir(); (model / "config.json").write_text("{}")
        video = root / "clip.mp4"; video.write_bytes(b"fixture-video-not-decoded")
        return smoke.parse_args(["--model-dir", str(model), "--video", str(video), "--backend", "frames", "--question", "What happens?", "--output", str(root / "result.json"), "--revision", "declared-test-revision"])

    def test_success_requires_stop_and_records_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            with patch.object(smoke, "infer", return_value={"finish_reason": "stop", "answer": "A hand moves.", "raw_answer": "A hand moves.<eos>"}):
                result = smoke.run_smoke(args)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["video_sha256"], hashlib.sha256(args.video.read_bytes()).hexdigest())
            self.assertEqual(result["model_revision_declared"], "declared-test-revision")
            self.assertFalse(result["revision_verified_by_this_script"])
            self.assertEqual(json.loads(args.output.read_text())["status"], "success")

    def test_length_result_preserves_answer_but_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            with patch.object(smoke, "infer", return_value={"finish_reason": "length", "answer": "Partial", "raw_answer": "Partial"}):
                result = smoke.run_smoke(args)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["raw_answer"], "Partial")

    def test_exception_does_not_leak_message(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            with patch.object(smoke, "infer", side_effect=RuntimeError("PRIVATE_TEST_SECRET")):
                result = smoke.run_smoke(args)
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("PRIVATE_TEST_SECRET", args.output.read_text())
            self.assertEqual(result["error_type"], "RuntimeError")

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory)); args.output.write_text("existing")
            with self.assertRaises(FileExistsError): smoke.run_smoke(args)
            self.assertEqual(args.output.read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
