"""Offline checks for the nested, same-GT frame-coverage ablation."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location("prepare_gt_frame_ablation",Path(__file__).with_name("prepare_gt_frame_ablation.py"))
module=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=module;SPEC.loader.exec_module(module)


class FrameAblationTests(unittest.TestCase):
    def test_nested_unique_order_for_different_video_lengths(self):
        for count in (32,33,63,64,100,838,1321):
            old=module.midpoints(count,8);result=module.nested_indices(count,old)
            self.assertEqual(len(result),32);self.assertEqual(result,sorted(set(result)))
            self.assertTrue(set(old).issubset(result));self.assertTrue(all(0<=x<count for x in result))
            self.assertTrue(set(result)-set(old)<=set(module.midpoints(count,32)))

    def test_expected_remaining_candidate_midpoint_selection(self):
        count=838;old=module.midpoints(count,8);candidates=[i for i in module.midpoints(count,32) if i not in old]
        extras=[candidates[i] for i in module.midpoints(len(candidates),24)]
        self.assertEqual(module.nested_indices(count,old),sorted(old+extras))
        self.assertNotEqual(set(module.midpoints(count,32)),set(module.nested_indices(count,old)))

    def test_invalid_baseline_or_short_source_rejected(self):
        for count,old in ((20,list(range(8))),(100,[1]*8),(100,[0,1,2,3,4,5,6,100])):
            with self.assertRaises(ValueError):module.nested_indices(count,old)

    def test_old_jpeg_is_copied_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"old.jpg";target=root/"new.jpg";raw=b"unchanged original JPEG bytes"
            source.write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
            module.copy_baseline_frame(source,target,digest)
            self.assertEqual(target.read_bytes(),raw);self.assertEqual(source.read_bytes(),raw)
            with self.assertRaises(FileExistsError):module.copy_baseline_frame(source,target,digest)

    def test_bad_baseline_hash_is_not_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"old.jpg";source.write_bytes(b"changed")
            with self.assertRaises(ValueError):module.copy_baseline_frame(source,root/"new.jpg","0"*64)
            self.assertFalse((root/"new.jpg").exists())

    def test_gt_labels_and_video_ids_stay_exactly_unchanged(self):
        original={"schema_version":1,"benchmark_id":"old","labels":[{"sample_id":"e000","scene_id":"s001","action_ids":["c001","c008"],"source_video_id":"SOURCEEGO"}],"provenance":{"original":"proof"}}
        before=copy.deepcopy(original);result=module.ablation_gt(original,{"sha256":"x"})
        self.assertEqual(result["labels"],original["labels"]);self.assertEqual(original,before)
        self.assertEqual(result["benchmark_id"],"charades-ego-32frames-v1")
        result["labels"][0]["action_ids"].append("c002");self.assertEqual(original,before)


if __name__=="__main__":unittest.main()
