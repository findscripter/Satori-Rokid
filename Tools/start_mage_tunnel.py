"""Start this project's fixed Windows SSH tunnel outside the launcher's Job.

No credentials are read. Failure never replaces the current tunnel record and
can terminate only the Popen object created by this invocation. No fallback to
a launcher-job-bound child is permitted. Membership in a separate Job is logged,
not mistaken for inheritance from the launcher. Tests create no processes.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

CREATE_BREAKAWAY_FROM_JOB=0x01000000
CREATE_NO_WINDOW=0x08000000
CREATE_NEW_PROCESS_GROUP=0x00000200
CREATION_FLAGS=CREATE_BREAKAWAY_FROM_JOB|CREATE_NO_WINDOW|CREATE_NEW_PROCESS_GROUP
PORT=18866
HOST="gpu-host"
FORWARD="127.0.0.1:18866:127.0.0.1:18866"
LISTENER_QUERY="""$ErrorActionPreference='Stop'; [Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false); $items=@(Get-NetTCPConnection -LocalPort 18866 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { [pscustomobject]@{address=$_.LocalAddress;port=[int]$_.LocalPort;pid=[int]$_.OwningProcess} }); ConvertTo-Json -InputObject $items -Compress"""


class TunnelError(RuntimeError):
    def __init__(self,code,message):super().__init__(message);self.code=code


def utc_now():return datetime.now(timezone.utc).isoformat()


def ssh_arguments(executable):
    return [str(executable),"-N","-T","-n","-L",FORWARD,
            "-o","ExitOnForwardFailure=yes","-o","BatchMode=yes",
            "-o","ServerAliveInterval=30","-o","ServerAliveCountMax=6",
            "-o","ConnectTimeout=10",HOST]


def atomic_json(path,value):
    temporary=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with temporary.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


class WindowsOps:
    def __init__(self):
        if os.name!="nt":raise TunnelError("windows_required","This launcher requires Windows.")
        self.kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        self.kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];self.kernel.OpenProcess.restype=wintypes.HANDLE
        self.kernel.CloseHandle.argtypes=[wintypes.HANDLE];self.kernel.CloseHandle.restype=wintypes.BOOL
        self.kernel.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)];self.kernel.IsProcessInJob.restype=wintypes.BOOL
        self.kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME)]
        self.kernel.GetProcessTimes.restype=wintypes.BOOL
        self.kernel.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,wintypes.LPVOID,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD)]
        self.kernel.QueryInformationJobObject.restype=wintypes.BOOL
    def resolve_ssh(self):
        executable=shutil.which("ssh.exe")
        if not executable:raise TunnelError("ssh_missing","Windows ssh.exe was not found.")
        return Path(executable).resolve()
    def listeners(self):
        executable=shutil.which("powershell.exe")
        if not executable:raise TunnelError("listener_query_unavailable","PowerShell listener query is unavailable.")
        result=subprocess.run([executable,"-NoProfile","-NonInteractive","-Command",LISTENER_QUERY],
            stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            encoding="utf-8-sig",errors="replace",timeout=10,check=False,creationflags=CREATE_NO_WINDOW)
        if result.returncode!=0:raise TunnelError("listener_query_failed","Could not verify the local listener owner.")
        try:items=json.loads(result.stdout or "[]")
        except json.JSONDecodeError as error:raise TunnelError("listener_query_failed","Listener query returned invalid metadata.") from error
        if not isinstance(items,list):items=[items]
        for item in items:
            if not isinstance(item,dict) or not isinstance(item.get("pid"),int) or item.get("port")!=PORT or not isinstance(item.get("address"),str):
                raise TunnelError("listener_query_failed","Listener query metadata failed validation.")
        return items
    def spawn(self,argv,stdout,stderr,flags,cwd):
        return subprocess.Popen(argv,cwd=str(cwd),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            creationflags=flags,close_fds=True)
    def _handle(self,pid):
        handle=self.kernel.OpenProcess(0x1000,False,pid)
        if not handle:raise ctypes.WinError(ctypes.get_last_error())
        return handle
    def is_in_job(self,pid):
        handle=self._handle(pid)
        try:
            answer=wintypes.BOOL()
            if not self.kernel.IsProcessInJob(handle,None,ctypes.byref(answer)):raise ctypes.WinError(ctypes.get_last_error())
            return bool(answer.value)
        finally:self.kernel.CloseHandle(handle)
    def creation_time(self,pid):
        handle=self._handle(pid)
        try:
            created=wintypes.FILETIME();exited=wintypes.FILETIME();kernel=wintypes.FILETIME();user=wintypes.FILETIME()
            if not self.kernel.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(exited),ctypes.byref(kernel),ctypes.byref(user)):raise ctypes.WinError(ctypes.get_last_error())
            ticks=(int(created.dwHighDateTime)<<32)|int(created.dwLowDateTime)
            instant=datetime(1601,1,1,tzinfo=timezone.utc)+timedelta(microseconds=ticks//10)
            return {"utc":instant.isoformat(),"filetime":ticks}
        finally:self.kernel.CloseHandle(handle)
    def launcher_job_snapshot(self):
        launcher_pid=os.getpid()
        if not self.is_in_job(launcher_pid):
            return {"launcher_pid":launcher_pid,"launcher_in_job":False,"process_ids":[]}
        capacity=32;pointer_size=ctypes.sizeof(ctypes.c_size_t)
        while capacity<=65536:
            length=8+capacity*pointer_size;buffer=ctypes.create_string_buffer(length);returned=wintypes.DWORD()
            ok=self.kernel.QueryInformationJobObject(None,3,buffer,length,ctypes.byref(returned))
            assigned=ctypes.c_uint32.from_buffer(buffer,0).value;listed=ctypes.c_uint32.from_buffer(buffer,4).value
            if ok:
                if listed>capacity:raise TunnelError("job_scope_unverified","Launcher Job process list exceeded its verified buffer.")
                pids=[int(ctypes.c_size_t.from_buffer(buffer,8+i*pointer_size).value) for i in range(listed)]
                return {"launcher_pid":launcher_pid,"launcher_in_job":True,"process_ids":pids,"assigned_process_count":assigned}
            error=ctypes.get_last_error()
            if error not in (122,234):raise ctypes.WinError(error)
            capacity=max(capacity*2,int(assigned)+16,(int(returned.value)-8)//pointer_size+16)
        raise TunnelError("job_scope_unverified","Launcher Job process list could not be bounded and verified.")
    def monotonic(self):return time.monotonic()
    def sleep(self,seconds):time.sleep(seconds)


def verify_listener(listeners,pid):
    if any(item["pid"]!=pid for item in listeners):raise TunnelError("listener_wrong_owner","Port 18866 belongs to another process; it was not modified.")
    if listeners and not any(item["address"]=="127.0.0.1" for item in listeners):raise TunnelError("listener_wrong_address","The created listener is not the required IPv4 loopback endpoint.")
    if any(item["address"]!="127.0.0.1" for item in listeners):raise TunnelError("listener_wrong_address","The created listener also exposes an unexpected address.")
    return bool(listeners)


def verify_launcher_detachment(operations,pid,record):
    snapshot=operations.launcher_job_snapshot()
    if snapshot["launcher_in_job"] and snapshot["launcher_pid"] not in snapshot["process_ids"]:
        raise TunnelError("job_scope_unverified","Queried Job does not contain the launcher; detachment cannot be verified.")
    in_launcher_job=bool(snapshot["launcher_in_job"] and pid in snapshot["process_ids"])
    record.update(in_any_job=operations.is_in_job(pid),in_launcher_job=in_launcher_job,
        launcher_job_snapshot=snapshot,detachment_scope="launcher_windows_job")
    if in_launcher_job:
        raise TunnelError("child_still_in_launcher_job","SSH remains in the launcher's Windows Job; no fallback is accepted.")


def stop_owned_failed_child(process,record):
    """Use the owned process handle, never os.kill or a PID discovered elsewhere."""
    if process is None:return
    try:
        code=process.poll()
        if code is None:
            record["owned_failure_cleanup_attempted"]=True
            process.terminate();code=process.wait(timeout=5)
        record["child_exit_code"]=code
    except Exception as error:
        record["owned_failure_cleanup_error_type"]=type(error).__name__


def start_tunnel(project_root,*,ops=None,startup_timeout=20):
    project_root=Path(project_root).resolve();evidence=project_root/"Evidence"
    attempt_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex
    attempt=evidence/"mage-vl-tunnel-attempts"/attempt_id;attempt.mkdir(parents=True,exist_ok=False)
    stdout_path=attempt/"stdout.log";stderr_path=attempt/"stderr.log";attempt_record=attempt/"attempt.json"
    record={"status":"starting","attempt_id":attempt_id,"pid":None,"host":HOST,"local_address":"127.0.0.1","local_port":PORT,
            "remote_address":"127.0.0.1","remote_port":PORT,"started":utc_now(),"creation_flags":CREATION_FLAGS,
            "stdout_log":str(stdout_path),"stderr_log":str(stderr_path),"attempt_record":str(attempt_record),
            "detached_verified":False,"detachment_scope":"launcher_windows_job","in_any_job":None,"in_launcher_job":None,
            "listener_verified":False,"owned_failure_cleanup_attempted":False}
    process=None
    try:
        operations=ops if ops is not None else WindowsOps()
        if operations.listeners():raise TunnelError("port_already_occupied","Port 18866 is already occupied; no process was started or stopped.")
        executable=operations.resolve_ssh()
        if executable.name.lower()!="ssh.exe" or not executable.is_file():raise TunnelError("ssh_missing","Expected a local ssh.exe executable.")
        record["argv"]=ssh_arguments(executable);record["executable"]=str(executable)
        # Exclusive new log files prevent a reconnect from erasing an earlier failure.
        with stdout_path.open("xb") as stdout,stderr_path.open("xb") as stderr:
            try:process=operations.spawn(record["argv"],stdout,stderr,CREATION_FLAGS,project_root)
            except (OSError,PermissionError) as error:
                record["spawn_error_type"]=type(error).__name__;record["spawn_winerror"]=getattr(error,"winerror",None)
                raise TunnelError("breakaway_spawn_failed","Job-detached SSH creation failed; no job-bound fallback was attempted.") from error
            record["pid"]=process.pid;deadline=operations.monotonic()+startup_timeout
            while True:
                exit_code=process.poll()
                if exit_code is not None:record["child_exit_code"]=exit_code;raise TunnelError("ssh_exited","SSH exited before successful listener verification.")
                verify_launcher_detachment(operations,process.pid,record)
                if verify_listener(operations.listeners(),process.pid):break
                if operations.monotonic()>=deadline:raise TunnelError("listener_timeout","SSH did not establish the required owned listener before timeout.")
                operations.sleep(0.2)
            operations.sleep(0.5)
            if process.poll() is not None:record["child_exit_code"]=process.returncode;raise TunnelError("ssh_exited","SSH exited during the startup stabilization check.")
            verify_launcher_detachment(operations,process.pid,record)
            if not verify_listener(operations.listeners(),process.pid):raise TunnelError("listener_disappeared","The owned listener disappeared during startup verification.")
            created=operations.creation_time(process.pid)
            record.update(status="ready",detached_verified=True,listener_verified=True,
                process_creation_time_utc=created["utc"],process_creation_filetime=created["filetime"],verified_utc=utc_now())
            atomic_json(attempt_record,record)
            # Only a verified success can replace the project pointer. Failure
            # records stay in their unique attempt directory.
            atomic_json(evidence/"mage-vl-tunnel.json",record)
        return record
    except Exception as error:
        record.update(status="failed",detached_verified=False,failed_utc=utc_now(),
            error_code=error.code if isinstance(error,TunnelError) else "startup_verification_failed",
            error_type=type(error).__name__,message=str(error) if isinstance(error,TunnelError) else "Tunnel startup verification failed; inspect the unique attempt logs.")
        stop_owned_failed_child(process,record)
        atomic_json(attempt_record,record)
        return record


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--startup-timeout",type=float,default=20)
    args=parser.parse_args(argv)
    if not 1<=args.startup_timeout<=30:parser.error("startup-timeout must be between 1 and 30 seconds")
    try:result=start_tunnel(args.project_root,startup_timeout=args.startup_timeout)
    except Exception as error:
        print(json.dumps({"status":"failed","error_code":"evidence_unavailable","error_type":type(error).__name__,"message":"Could not create the local attempt record."}));return 1
    print(json.dumps(result,ensure_ascii=True))
    return 0 if result["status"]=="ready" else 1


if __name__=="__main__":raise SystemExit(main())
