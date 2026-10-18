"""CPU regression tests for time metadata and protocol boundaries."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("vc3_backend", Path(__file__).with_name("backend.py"))
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)


class MetadataTest(unittest.TestCase):
    def setUp(self):
        self.metadata = {"fps": 30.0, "total_num_frames": 120,
                         "duration": 4.0, "frames_indices": [0, 15, 30, 45, 60, 75, 90, 105]}

    def test_preserves_original_clock(self):
        self.assertEqual(backend.validate_metadata(self.metadata), self.metadata)

    def test_rejects_resampled_duplicate_frames(self):
        self.metadata["frames_indices"][2] = 15
        with self.assertRaises(ValueError):
            backend.validate_metadata(self.metadata)

    def test_rejects_source_identifier(self):
        self.metadata["source_video_id"] = "never_pass_source_ids"
        with self.assertRaises(ValueError):
            backend.validate_metadata(self.metadata)

    def test_rejects_fabricated_duration(self):
        self.metadata["duration"] = 8.0
        with self.assertRaises(ValueError):
            backend.validate_metadata(self.metadata)

    def test_rejects_missing_clock(self):
        del self.metadata["fps"]
        with self.assertRaises(ValueError):
            backend.validate_metadata(self.metadata)

    def test_eos_required_even_if_under_budget(self):
        self.assertEqual(backend.classify_finish([1, 2], 3), "length")
        self.assertEqual(backend.classify_finish([1, 2, 3], 3), "stop")
        self.assertEqual(backend.classify_finish([1, 2, 3], [3, 4]), "stop")

    def test_preserves_context_and_shared_frame_text(self):
        context = {"frames": [{"timestamp_ms": 500, "frame_id": "v000"}], "a": "中文"}
        self.assertEqual(backend.render_user_text(context),
                         '{"a":"中文","frames":[{"frame_id":"v000","timestamp_ms":500}]}\n'
                         'Frame ID: v000; timestamp_ms: 500')

    def test_model_loading_requires_scheduler_before_torch_import(self):
        with patch.dict(backend.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "requires_gpurun"):
                backend.Backend("not_opened")


if __name__ == "__main__":
    unittest.main()
