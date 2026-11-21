"""Provision the glasses-only assistant over authorized ADB; no credential enters the APK or command line.

Use --serial DEVICE provision, launch, status, models, export --output DIR,
list-videos, export-videos --output DIR, list-screen-recordings, or export-screen-recordings --output DIR.
The existing BDI provider is reused. A plaintext import exists only briefly in the app's private sandbox,
is immediately encrypted with AndroidKeyStore, and is deleted before the configuration screen appears.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tarfile
from urllib import request, error, parse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Tools"))
from rokid_glasses import find_adb

PACKAGE = "com.satori.glasses.capture"
VIDEO_DIRECTORY = "files/first-person-videos"
SCREEN_DIRECTORY = "files/screen-recordings"
_FINAL_VIDEO = re.compile(r"recording-[A-Za-z0-9_-]{1,160}\.mp4\Z")
_SIDECAR = re.compile(r"recording-[A-Za-z0-9_-]{1,160}\.json\Z")
_INCOMPLETE_VIDEO = re.compile(r"recording-[A-Za-z0-9_-]{1,160}\.(?:pending|failed)\.mp4\Z")
_SCREEN_VIDEO = re.compile(r"screen-[0-9]{1,20}\.mp4\Z")
_SCREEN_SIDECAR = re.compile(r"screen-[0-9]{1,20}\.json\Z")
_SCREEN_INCOMPLETE = re.compile(r"screen-[0-9]{1,20}\.(?:pending|failed)\.mp4\Z")

# No caller-controlled shell text is executed. File names also pass the strict Python allowlist.
_VIDEO_LIST_SCRIPT = r"""# satori-video-list-v1
set -eu
directory='files/first-person-videos'
[ ! -L files ] && [ ! -L "$directory" ] || exit 3
[ -d "$directory" ] || { printf 'done\tvideo-list-v1\n'; exit 0; }
for path in "$directory"/recording-*.mp4; do
    [ -e "$path" ] || [ -L "$path" ] || continue
    name=${path##*/}
    case "$name" in *.pending.mp4|*.failed.mp4) continue ;; esac
    [ ! -L "$path" ] && [ -f "$path" ] || exit 3
    bytes=$(stat -c %s "$path") || exit 3
    printf 'video\t%s\t%s\n' "$name" "$bytes"
    sidecar="$directory/${name%.mp4}.json"
    [ ! -L "$sidecar" ] || exit 3
    if [ -e "$sidecar" ]; then
        [ -f "$sidecar" ] || exit 3
        bytes=$(stat -c %s "$sidecar") || exit 3
        printf 'sidecar\t%s\t%s\n' "${sidecar##*/}" "$bytes"
    fi
done
printf 'done\tvideo-list-v1\n'
"""


def _video_name(name, *, sidecar=False, screen=False):
    pattern = (_SCREEN_SIDECAR if sidecar else _SCREEN_VIDEO) if screen else (_SIDECAR if sidecar else _FINAL_VIDEO)
    if not isinstance(name, str) or pattern.fullmatch(name) is None:
        raise ValueError("unsafe or unfinished video filename")
    return name


def _video_script(name, *, read=False, screen=False):
    _video_name(name, sidecar=name.endswith(".json"), screen=screen)
    # Open first, then compare the opened fd's canonical path. A symlink/directory replacement
    # between checking the name and opening it cannot redirect cat to preferences or credentials.
    prefix = ("# satori-video-read-v1\n" if read else "# satori-video-info-v1\n") + f"export_name='{name}'\n"
    prefix += r"""set -eu
directory='files/first-person-videos'
[ ! -L files ] && [ ! -L "$directory" ] && [ -d "$directory" ] || exit 3
path="$directory/$export_name"
[ ! -L "$path" ] && [ -f "$path" ] || exit 3
app_root=$(pwd -P)
exec 3<"$path"
opened=$(readlink /proc/self/fd/0 <&3) || exit 3
[ "$opened" = "$app_root/$directory/$export_name" ] || exit 3
[ -f /proc/self/fd/3 ] || exit 3
"""
    if screen:
        prefix = prefix.replace(VIDEO_DIRECTORY, SCREEN_DIRECTORY)
    if read:
        if screen and name.endswith(".json"):
            return prefix + "exec head -c 65537 <&3\n"
        return prefix + "exec cat <&3\n"
    # Android mksh can mark fd3 close-on-exec. Explicitly duplicating it to stdin preserves the
    # same opened inode in every external tool, without reopening its potentially replaced path.
    return prefix + r"""before=$(stat -L -c %s /proc/self/fd/0 <&3) || exit 3
digest=$(sha256sum <&3) || exit 3
after=$(stat -L -c %s /proc/self/fd/0 <&3) || exit 3
[ "$before" = "$after" ] || exit 3
printf '%s\t%s\n' "$after" "${digest%% *}"
"""


def _run_video_script(adb, serial, script, *, stdout=subprocess.PIPE, timeout=30):
    # This device's customized shell channel expands CR/LF even with -T. Only raw exec-out
    # preserves video bytes. Pass fixed/allowlisted script text as one -c argument, without stdin
    # or another quoting layer: this installed adb already preserves that argument boundary.
    result = subprocess.run([str(adb), "-s", serial, "exec-out", "run-as", PACKAGE, "sh", "-c", script],
                            stdout=stdout, stderr=subprocess.PIPE,
                            timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError("ADB video operation failed; source files were not removed")
    if stdout is subprocess.PIPE:
        data = result.stdout or b""
        if len(data) > 1024 * 1024:
            raise RuntimeError("video metadata exceeds the safe response limit")
        return data
    return None


def list_videos(adb, serial, *, screen=False):
    """Only committed MP4 names and their optional committed JSON sidecars; no video bytes are read."""
    script = _VIDEO_LIST_SCRIPT.replace(VIDEO_DIRECTORY, SCREEN_DIRECTORY).replace("recording-*.mp4", "screen-*.mp4") if screen else _VIDEO_LIST_SCRIPT
    data = _run_video_script(adb, serial, script).decode("utf-8", errors="strict")
    lines = data.splitlines()
    if not lines or lines[-1] != "done\tvideo-list-v1":
        raise RuntimeError("device did not complete the video inventory")
    videos, sidecars = {}, {}
    for line in lines[:-1]:
        parts = line.split("\t")
        if len(parts) != 3 or parts[0] not in ("video", "sidecar") or not re.fullmatch(r"[0-9]{1,20}", parts[2]):
            raise ValueError("invalid video metadata returned by device")
        kind, name, raw_size = parts
        if kind == "video" and (_SCREEN_INCOMPLETE if screen else _INCOMPLETE_VIDEO).fullmatch(name):
            continue  # Defense in depth if an older device-side listing includes interrupted recordings.
        _video_name(name, sidecar=kind == "sidecar", screen=screen)
        bucket = videos if kind == "video" else sidecars
        if name in bucket:
            raise ValueError("duplicate video metadata returned by device")
        size = int(raw_size)
        if size <= 0:
            raise ValueError("empty final video or sidecar is not exportable")
        bucket[name] = size
    if any(name.removesuffix(".json") + ".mp4" not in videos for name in sidecars):
        raise ValueError("sidecar does not belong to a completed video")
    result = [dict(file=name, bytes=size, sidecar=(dict(file=name.removesuffix(".mp4") + ".json",
            bytes=sidecars[name.removesuffix(".mp4") + ".json"])
            if name.removesuffix(".mp4") + ".json" in sidecars else None))
            for name, size in sorted(videos.items())]
    if not screen:
        return result
    completed = []
    for item in result:
        sidecar = item["sidecar"]
        if sidecar is None:
            continue  # MP4 and sidecar commit separately; an incomplete pair is not exportable yet.
        if sidecar["bytes"] > 65536:
            raise ValueError("screen sidecar exceeds the safe metadata limit")
        raw = _run_video_script(adb, serial, _video_script(sidecar["file"], read=True, screen=True))
        if len(raw) != sidecar["bytes"] or len(raw) > 65536:
            raise ValueError("screen sidecar changed or exceeded its listed bounds")
        metadata = json.loads(raw)
        if (not isinstance(metadata, dict) or metadata.get("schema") != "satori.screen-recording/1" or
                type(metadata.get("finalized")) is not bool or
                not isinstance(metadata.get("path"), str) or
                not metadata["path"].endswith("/" + SCREEN_DIRECTORY + "/" + item["file"])):
            raise ValueError("screen sidecar does not describe this recording")
        if metadata.get("state") != "saved" or not metadata["finalized"]:
            continue
        sidecar["sha256"] = hashlib.sha256(raw).hexdigest()
        completed.append(item)
    return completed


def _video_timeout(size):
    # Per-file budgets allow even a slow wireless link; no whole recording is buffered in Python.
    return max(180, min(6 * 60 * 60, 60 + (size + 256 * 1024 - 1) // (256 * 1024)))


def _remote_video_info(adb, serial, name, *, timeout, screen=False):
    parts = _run_video_script(adb, serial, _video_script(name, screen=screen), timeout=timeout).decode("ascii").strip().split("\t")
    if len(parts) != 2 or re.fullmatch(r"[0-9]{1,20}", parts[0]) is None or re.fullmatch(r"[0-9a-fA-F]{64}", parts[1]) is None:
        raise ValueError("invalid device file digest")
    return int(parts[0]), parts[1].lower()


def _is_link(path):
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _file_identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _local_video_info(path):
    if _is_link(path):
        raise ValueError("export target must not be a symbolic link")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("export target must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if _file_identity(opened) != _file_identity(before):
            raise RuntimeError("export target changed while opening")
        digest = hashlib.sha256()
        count = 0
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            count += len(chunk)
        after = os.fstat(handle.fileno())
        if _file_identity(after) != _file_identity(opened) or count != opened.st_size:
            raise RuntimeError("export target changed while hashing")
    if _is_link(path) or _file_identity(path.lstat()) != _file_identity(after):
        raise RuntimeError("export target changed after hashing")
    return count, digest.hexdigest(), _file_identity(after)


def _promote_video_part(partial, target):
    # Windows rename refuses an existing target. POSIX link is the no-replace atomic equivalent;
    # unlike os.replace()/POSIX rename(), neither branch can overwrite a concurrently created file.
    if os.name == "nt":
        partial.rename(target)
    else:
        os.link(partial, target, follow_symlinks=False)
        partial.unlink()


def export_videos(adb, serial, output, *, screen=False):
    output = Path(output).expanduser()
    if _is_link(output):
        raise ValueError("export directory must not be a symbolic link")
    output.mkdir(parents=True, exist_ok=True)
    root = output.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("export destination must be a directory")
    videos = list_videos(adb, serial, screen=screen)
    files = []
    for video in videos:
        files.append(dict(file=video["file"], bytes=video["bytes"]))
        if video["sidecar"] is not None:
            files.append(video["sidecar"])
    copied = skipped = 0
    for index, entry in enumerate(files, 1):
        name, listed_size = entry["file"], entry["bytes"]
        timeout = _video_timeout(listed_size)
        print(f"[{index}/{len(files)}] Checking {name} ({listed_size} bytes)", flush=True)
        expected = _remote_video_info(adb, serial, name, timeout=timeout, screen=screen)
        if expected[0] != listed_size:
            raise RuntimeError(f"device file changed since listing: {name}")
        if entry.get("sha256") is not None and expected[1] != entry["sha256"]:
            raise RuntimeError(f"validated screen metadata changed since listing: {name}")
        target = root / name
        if _is_link(target):
            raise ValueError("export target must not be a symbolic link")
        if target.exists():
            size, digest, identity = _local_video_info(target)
            if (size, digest) != expected:
                raise FileExistsError(f"existing file differs; refusing to overwrite: {name}")
            if _remote_video_info(adb, serial, name, timeout=timeout, screen=screen) != expected or _file_identity(target.lstat()) != identity:
                raise RuntimeError(f"file changed during verification: {name}")
            print(f"[{index}/{len(files)}] Skip identical {name} (size and SHA256 verified)", flush=True)
            skipped += 1
            continue
        partial = root / (name + ".part")
        # Exclusive creation rejects both a prior failed copy and a pre-existing/dangling symlink.
        with partial.open("xb") as sink:
            print(f"[{index}/{len(files)}] Copying {name}", flush=True)
            _run_video_script(adb, serial, _video_script(name, read=True, screen=screen), stdout=sink, timeout=timeout)
            sink.flush()
            os.fsync(sink.fileno())
        size, digest, identity = _local_video_info(partial)
        after = _remote_video_info(adb, serial, name, timeout=timeout, screen=screen)
        if (size, digest) != expected or after != expected:
            raise RuntimeError(f"video verification failed; .part retained: {name}")
        if _is_link(partial) or _file_identity(partial.lstat()) != identity:
            raise RuntimeError(f"partial file changed before finalization: {name}")
        _promote_video_part(partial, target)
        print(f"[{index}/{len(files)}] Verified {name} ({size} bytes, SHA256 {digest})", flush=True)
        copied += 1
    print(json.dumps(dict(videos=len(videos), copied=copied, skipped=skipped, output=str(root)), ensure_ascii=False))


def _validate_records_tar(path):
    """Validate headers, scope, body bounds and the complete zero-block trailer; never extract."""
    if _is_link(path):
        raise RuntimeError("records archive must not be a symbolic link")
    size = path.stat().st_size
    if size < 1024 or size % 512:
        raise RuntimeError("records archive is incomplete; .part retained")
    count = end = 0
    try:
        with path.open("rb") as stream:
            with tarfile.open(fileobj=stream, mode="r:") as archive:
                for member in archive:
                    name = member.name.rstrip("/")
                    parts = name.split("/")
                    if (not parts or parts[0] != "standalone-runs" or "\\" in name or ":" in name or
                            any(part in ("", ".", "..") for part in parts) or
                            not (member.isdir() or member.isfile())):
                        raise RuntimeError("archive contains an unsafe path or non-record member; .part retained")
                    end = max(end, member.offset_data + ((member.size + 511) // 512) * 512)
                    if end > size:
                        raise RuntimeError("archive body is truncated; .part retained")
                    count += 1
            stream.seek(end)
            trailer = 0
            while chunk := stream.read(1024 * 1024):
                if any(chunk):
                    raise RuntimeError("archive has data after its end marker; .part retained")
                trailer += len(chunk)
            if count == 0 or trailer < 1024:
                raise RuntimeError("records archive has no complete end marker; .part retained")
    except (tarfile.TarError, EOFError, ValueError) as exc:
        raise RuntimeError("invalid or truncated records archive; .part retained") from exc
    return count


def export_records(adb, serial, output):
    output = Path(output).expanduser()
    if _is_link(output):
        raise ValueError("export directory must not be a symbolic link")
    output.mkdir(parents=True, exist_ok=True)
    root = output.resolve(strict=True)
    target = root / "standalone-runs.tar"
    partial = root / "standalone-runs.tar.part"
    if _is_link(target):
        raise ValueError("records target must not be a symbolic link")
    print("Streaming records to " + str(partial), flush=True)
    # Fixed private subtree only. The raw channel has already been checked against this device's
    # binary SHA behavior. Source capture should be paused/stopped; tar is not an atomic snapshot.
    with partial.open("xb") as sink:
        result = subprocess.run([str(adb), "-s", serial, "exec-out", "run-as", PACKAGE,
                                 "tar", "-C", "files", "-cf", "-", "standalone-runs"],
                                stdout=sink, stderr=subprocess.PIPE, timeout=6 * 60 * 60,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise RuntimeError("records transfer failed; .part retained and source files untouched")
        sink.flush()
        os.fsync(sink.fileno())
    members = _validate_records_tar(partial)
    size, digest, identity = _local_video_info(partial)
    if target.exists():
        old_size, old_digest, _ = _local_video_info(target)
        if (old_size, old_digest) != (size, digest):
            raise FileExistsError("existing records archive differs; .part retained, refusing to overwrite")
        partial.unlink()  # Only the new, verified identical copy is removed on this successful skip.
        print(f"Skip identical records archive (size and SHA256 verified): {target}", flush=True)
        return
    if _is_link(partial) or _file_identity(partial.lstat()) != identity:
        raise RuntimeError("records partial changed before finalization")
    _promote_video_part(partial, target)
    print(f"Exported records: {target} ({size} bytes, {members} members, SHA256 {digest})", flush=True)


def profile():
    selected = ROOT / "UserSettings/SatoriStandalone/provider.json"
    if not selected.is_file():
        selected = ROOT / "UserSettings/SatoriBDI/provider.json"
    config = json.loads(selected.read_text(encoding="utf-8-sig"))
    key_path = config.get("key_file")
    if config.get("key_store"):
        import satori_credentials
        key = satori_credentials.load(store=Path(config["key_store"]))
    elif key_path:
        key = Path(key_path).read_text(encoding="utf-8-sig").strip()
    else:
        import satori_credentials
        key = satori_credentials.load()
    if not key:
        raise ValueError("existing BDI API key is not configured")
    base = config["base_url"].rstrip("/")
    url = parse.urlsplit(base)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("provider must be a plain HTTPS endpoint")
    value = dict(base_url=base, api_key=key, planner=config.get("planner", "gpt-6.1-sol"),
                 monitor=config.get("monitor", "gpt-6-astra"), asr_model=config.get("asr_model", "whisper-1"),
                 tts_model=config.get("tts_model", "gpt-4o-mini-tts"), tts_voice=config.get("tts_voice", "alloy"))
    if "backup_key_store" in config:
        backup_store = config["backup_key_store"]
        if not isinstance(backup_store, str) or not backup_store.strip():
            raise ValueError("backup key store is not configured")
        import satori_credentials
        backup = satori_credentials.load(store=Path(backup_store))
        if (not isinstance(backup, str) or not 1 <= len(backup) <= 4096 or
                not all(33 <= ord(char) <= 126 for char in backup) or backup == key):
            raise ValueError("backup API key is missing, invalid or duplicates the primary")
        value["backup_api_key"] = backup
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("action", choices=("provision", "launch", "status", "models", "export", "list-videos", "export-videos",
                                           "list-screen-recordings", "export-screen-recordings"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    adb = str(find_adb())

    def run(*words, data=None, quiet=False):
        result = subprocess.run([adb, "-s", args.serial, *words], input=data, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise RuntimeError("ADB operation failed (details suppressed to protect configuration)")
        return b"" if quiet else result.stdout

    def launch():
        run("shell", "am", "start", "-a", "android.intent.action.MAIN", "-c", "android.intent.category.LAUNCHER",
            "-n", PACKAGE + "/.MainActivity", quiet=True)

    if args.action == "provision":
        value = profile()
        run("shell", "run-as", PACKAGE, "mkdir", "-p", "files", quiet=True)
        # tee receives the secret through stdin. Its echoed stdout is captured and discarded, never printed.
        run("shell", "run-as", PACKAGE, "tee", "files/standalone-import.json",
            data=json.dumps(value, ensure_ascii=False).encode(), quiet=True)
        run("shell", "am", "force-stop", PACKAGE, quiet=True)
        launch()
        print("Private configuration delivered; launcher started. Verify encrypted import with status.")
    elif args.action == "launch":
        launch()
        print("Standalone launcher started.")
    elif args.action == "status":
        files = run("shell", "run-as", PACKAGE, "ls", "files").decode()
        secure = run("shell", "run-as", PACKAGE, "cat", "shared_prefs/standalone_secure.xml").decode()
        state = json.loads(run("shell", "run-as", PACKAGE, "cat", "files/standalone-state.json"))
        print(json.dumps(dict(import_deleted="standalone-import.json" not in files,
                              encrypted_config="envelope" in secure and "api_key" not in secure,
                              phase=state.get("phase"), paused_reason=state.get("paused_reason"),
                              pid=run("shell", "pidof", PACKAGE).decode().strip()), ensure_ascii=False))
    elif args.action == "models":
        value = profile()
        class NoRedirect(request.HTTPRedirectHandler):
            def redirect_request(self, *unused):
                return None
        opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
        req = request.Request(value["base_url"] + "/models", headers={"Authorization": "Bearer " + value["api_key"]})
        try:
            with opener.open(req, timeout=25) as response:
                data = json.loads(response.read(2 * 1024 * 1024))
            names = [entry.get("id", "") for entry in data.get("data", [])]
            print(json.dumps(dict(count=len(names), audio_models=[name for name in names if re.search(r"tts|whisper|speech|asr|audio", name, re.I)]), ensure_ascii=False))
        except error.HTTPError as exc:
            print(json.dumps(dict(status=exc.code, models_available=False)))
    elif args.action == "export":
        if args.output is None:
            parser.error("export requires --output")
        export_records(adb, args.serial, args.output)
    elif args.action == "list-videos":
        print(json.dumps(dict(videos=list_videos(adb, args.serial)), ensure_ascii=False))
    elif args.action == "export-videos":
        if args.output is None:
            parser.error("export-videos requires --output")
        export_videos(adb, args.serial, args.output)
    elif args.action == "list-screen-recordings":
        print(json.dumps(dict(screen_recordings=list_videos(adb, args.serial, screen=True)), ensure_ascii=False))
    elif args.action == "export-screen-recordings":
        if args.output is None:
            parser.error("export-screen-recordings requires --output")
        export_videos(adb, args.serial, args.output, screen=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Operation failed: " + type(exc).__name__, file=sys.stderr)
        raise SystemExit(2) from None
