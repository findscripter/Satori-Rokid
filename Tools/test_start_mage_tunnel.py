"""Mock-only launcher checks: never create SSH, PowerShell, sockets or real jobs."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location("start_mage_tunnel",Path(__file__).with_name("start_mage_tunnel.py"))
launcher=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=launcher;SPEC.loader.exec_module(launcher)


class FakeProcess:
    pid=12345
    def __init__(self):self.returncode=None;self.terminated=False
    def poll(self):return self.returncode
    def terminate(self):self.terminated=True;self.returncode=1
    def wait(self,timeout=None):return self.returncode


class FakeOps:
    def __init__(self,root,*,initial=None,owner=12345,in_job=False,in_launcher_job=False,denied=False,early_exit=False,no_listener=False):
        self.executable=Path(root)/"ssh.exe";self.executable.write_bytes(b"not executed")
        self.initial=initial or [];self.owner=owner;self.in_job=in_job;self.denied=denied;self.early_exit=early_exit;self.no_listener=no_listener
        self.process=None;self.spawns=[];self.clock=0
        self.in_launcher_job=in_launcher_job
    def resolve_ssh(self):return self.executable
    def listeners(self):
        if self.process is None:return self.initial
        return [] if self.no_listener else [{"address":"127.0.0.1","port":18866,"pid":self.owner}]
    def spawn(self,argv,stdout,stderr,flags,cwd):
        self.spawns.append({"argv":argv,"flags":flags,"stdout":stdout.name,"stderr":stderr.name})
        if self.denied:raise PermissionError("mock breakaway denied")
        self.process=FakeProcess()
        if self.early_exit:self.process.returncode=255
        return self.process
    def is_in_job(self,pid):return self.in_job
    def launcher_job_snapshot(self):return {"launcher_pid":56789,"launcher_in_job":True,"process_ids":[56789,12345] if self.in_launcher_job else [56789]}
    def creation_time(self,pid):return {"utc":"2026-10-16T00:00:00+00:00","filetime":123}
    def monotonic(self):return self.clock
    def sleep(self,seconds):self.clock+=seconds


class TunnelLauncherTests(unittest.TestCase):
    def test_flags_and_fixed_forward(self):
        with tempfile.TemporaryDirectory() as root:
            ops=FakeOps(root);result=launcher.start_tunnel(Path(root),ops=ops)
            self.assertEqual(result["status"],"ready");self.assertTrue(result["detached_verified"])
            flags=ops.spawns[0]["flags"]
            for flag in (launcher.CREATE_BREAKAWAY_FROM_JOB,launcher.CREATE_NO_WINDOW,launcher.CREATE_NEW_PROCESS_GROUP):self.assertTrue(flags&flag)
            self.assertEqual(ops.spawns[0]["argv"][-1],"gpu-host")
            self.assertIn("127.0.0.1:18866:127.0.0.1:18866",ops.spawns[0]["argv"])
            self.assertIn("ServerAliveCountMax=6",ops.spawns[0]["argv"])
            self.assertFalse(ops.process.terminated)

    def test_unique_logs_for_each_attempt(self):
        with tempfile.TemporaryDirectory() as root:
            first=launcher.start_tunnel(Path(root),ops=FakeOps(root))
            second=launcher.start_tunnel(Path(root),ops=FakeOps(root))
            self.assertNotEqual(first["stdout_log"],second["stdout_log"])
            self.assertNotEqual(first["attempt_record"],second["attempt_record"])
            self.assertTrue(Path(first["stdout_log"]).exists())

    def test_existing_wrong_owner_never_spawned_or_killed(self):
        with tempfile.TemporaryDirectory() as root:
            ops=FakeOps(root,initial=[{"address":"127.0.0.1","port":18866,"pid":999}])
            result=launcher.start_tunnel(Path(root),ops=ops)
            self.assertEqual(result["error_code"],"port_already_occupied");self.assertEqual(ops.spawns,[])

    def test_breakaway_denied_has_no_fallback_and_preserves_record(self):
        with tempfile.TemporaryDirectory() as root:
            evidence=Path(root)/"Evidence";evidence.mkdir();record=evidence/"mage-vl-tunnel.json";record.write_bytes(b"previous record")
            ops=FakeOps(root,denied=True);result=launcher.start_tunnel(Path(root),ops=ops)
            self.assertEqual(result["status"],"failed");self.assertEqual(len(ops.spawns),1)
            self.assertEqual(record.read_bytes(),b"previous record")
            self.assertEqual(json.loads(Path(result["attempt_record"]).read_text())["status"],"failed")

    def test_launcher_job_bound_child_is_rejected_and_only_owned_child_stopped(self):
        with tempfile.TemporaryDirectory() as root:
            ops=FakeOps(root,in_job=True,in_launcher_job=True);result=launcher.start_tunnel(Path(root),ops=ops)
            self.assertEqual(result["error_code"],"child_still_in_launcher_job");self.assertTrue(ops.process.terminated)
            self.assertFalse((Path(root)/"Evidence/mage-vl-tunnel.json").exists())

    def test_separate_job_membership_is_not_false_claimed_as_no_job(self):
        with tempfile.TemporaryDirectory() as root:
            ops=FakeOps(root,in_job=True,in_launcher_job=False);result=launcher.start_tunnel(Path(root),ops=ops)
            self.assertEqual(result["status"],"ready");self.assertTrue(result["in_any_job"])
            self.assertFalse(result["in_launcher_job"]);self.assertTrue(result["detached_verified"])
            self.assertEqual(result["detachment_scope"],"launcher_windows_job");self.assertFalse(ops.process.terminated)

    def test_unverifiable_launcher_job_scope_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            ops=FakeOps(root);ops.launcher_job_snapshot=lambda:{"launcher_pid":56789,"launcher_in_job":True,"process_ids":[]}
            result=launcher.start_tunnel(Path(root),ops=ops)
            self.assertEqual(result["error_code"],"job_scope_unverified");self.assertTrue(ops.process.terminated)

    def test_post_spawn_wrong_listener_owner_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            ops=FakeOps(root,owner=999);result=launcher.start_tunnel(Path(root),ops=ops)
            self.assertEqual(result["error_code"],"listener_wrong_owner");self.assertTrue(ops.process.terminated)

    def test_early_exit_does_not_claim_ready(self):
        with tempfile.TemporaryDirectory() as root:
            ops=FakeOps(root,early_exit=True);result=launcher.start_tunnel(Path(root),ops=ops)
            self.assertEqual(result["error_code"],"ssh_exited");self.assertEqual(result["child_exit_code"],255)
            self.assertFalse(ops.process.terminated)

    def test_listener_timeout_stops_only_owned_child(self):
        with tempfile.TemporaryDirectory() as root:
            ops=FakeOps(root,no_listener=True);result=launcher.start_tunnel(Path(root),ops=ops,startup_timeout=1)
            self.assertEqual(result["error_code"],"listener_timeout");self.assertTrue(ops.process.terminated)


if __name__=="__main__":unittest.main()
