"""Verify and import this completed result archive without overwriting any run."""
from pathlib import Path
import hashlib
import tarfile

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "Evidence/satori-all-models-20261018/dispider"
archive_path = EVIDENCE / "dispider-paper-results-20261018.tar.gz"
expected = "cff7c0e0d6b5debcbbf05f8a5d39fb6d58c1180b12f09868e7616c4210b934ff"
if hashlib.sha256(archive_path.read_bytes()).hexdigest() != expected:
    raise ValueError("Downloaded result archive hash mismatch")
with tarfile.open(archive_path, "r:gz") as archive:
    mapped = []
    for member in archive.getmembers():
        parts = Path(member.name).parts
        if not parts or member.issym() or member.islnk() or Path(member.name).is_absolute() or ".." in parts:
            raise ValueError("Unsafe result archive member")
        if parts[0] == "runs":
            destination = ROOT / "Research/Backend" / member.name
            boundary = ROOT / "Research/Backend/runs"
        else:
            destination = EVIDENCE / "remote" / member.name
            boundary = EVIDENCE / "remote"
        if not destination.resolve().is_relative_to(boundary.resolve()):
            raise ValueError("Archive target escapes boundary")
        if member.isfile() and destination.exists():
            raise FileExistsError("Refusing to overwrite " + str(destination))
        mapped.append((member, destination))
    for member, destination in mapped:
        if member.isdir():
            destination.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.extractfile(member).read())
        else:
            raise ValueError("Unsupported archive member")
print("Verified and imported " + str(len(mapped)) + " archive members")
