"""Offline checks for sparse tar indexing and frozen CharadesEgo selection."""
import importlib.util
from pathlib import Path
import sys
import tarfile
import unittest

SPEC=importlib.util.spec_from_file_location("prepare_charades_gt",Path(__file__).with_name("prepare_charades_gt.py"))
module=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=module;SPEC.loader.exec_module(module)


class PreparationTests(unittest.TestCase):
    def test_tar_header_and_padded_next_offset(self):
        item=tarfile.TarInfo("CharadesEgo_v1_480/ABCDEEGO.mp4");item.size=1234
        parsed=module.parse_header(item.tobuf(format=tarfile.GNU_FORMAT),1024,10000)
        self.assertEqual(parsed["video_id"],"ABCDEEGO")
        self.assertEqual(parsed["data_offset"],1536)
        self.assertEqual(parsed["next_header"],3072)

    def test_corrupt_or_out_of_archive_header_rejected(self):
        item=tarfile.TarInfo("clip.mp4");item.size=1234;raw=bytearray(item.tobuf(format=tarfile.GNU_FORMAT));raw[0]^=1
        self.assertIsNone(module.parse_header(bytes(raw),0,10000))
        self.assertIsNone(module.parse_header(item.tobuf(format=tarfile.GNU_FORMAT),0,600))

    def test_range_accepts_only_exact_bound_identity(self):
        headers={"Content-Range":"bytes 10-19/100","ETag":'"fixed"'}
        self.assertEqual(module.validate_range(206,headers,10,19,100,'"fixed"'),(100,'"fixed"'))
        for status,changes,total,etag in [(200,{},100,'"fixed"'),(206,{"Content-Range":"bytes 0-9/100"},100,'"fixed"'),(206,{},101,'"fixed"'),(206,{},100,'"other"'),(206,{"ETag":None},100,'"fixed"')]:
            with self.assertRaises(ValueError):module.validate_range(status,{**headers,**changes},10,19,total,etag)

    def test_midpoint_indices_keep_real_frame_time(self):
        indices=module.midpoint_indices(300,8)
        self.assertEqual(indices,[18,56,93,131,168,206,243,281])
        self.assertEqual(module.frame_timestamp_ms(18,30.0),600)
        with self.assertRaises(ValueError):module.midpoint_indices(3,8)

    def test_action_sets_do_not_rewrite_bad_time_intervals(self):
        self.assertEqual(module.action_ids("c001 5.0 3.0;c002 0 9;c001 1 2"),["c001","c002"])

    def test_selection_is_balanced_deterministic_and_independent_of_order(self):
        rows=[{"id":f"A{i}","scene":"A"} for i in range(10)]+[{"id":f"B{i}","scene":"B"} for i in range(10)]
        indexed={r["id"]:{"video_id":r["id"]} for r in rows}
        first=module.select_samples(rows,indexed,["A","B"],8,20261016)
        second=module.select_samples(list(reversed(rows)),indexed,["A","B"],8,20261016)
        self.assertEqual([r["id"] for r in first],[r["id"] for r in second]);self.assertEqual(len(first),16)
        self.assertEqual(sum(r["scene"]=="A" for r in first),8)
        with self.assertRaises(ValueError):module.select_samples(rows,{"A0":indexed["A0"]},["A","B"],8,20261016)


if __name__=="__main__":unittest.main()
