"""Store or load a model API key, encrypted for the current Windows user (DPAPI).

The key lives outside every repository, in %LOCALAPPDATA%\\SatoriAR, and only
processes running as this Windows user can decrypt it. Nothing here prints the
key. CLI: `store` reads the key from stdin; `check` reports only whether one is
stored; `delete` removes it.
"""

import ctypes
import ctypes.wintypes as wintypes
import os
from pathlib import Path
import sys
import tempfile

STORE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "SatoriAR" / "provider.dpapi"
# Existing Windows profiles use this entropy to decrypt their configured key_store.
ENTROPY = b"satori-ar/axonera-api-key/v1"
CRYPTPROTECT_UI_FORBIDDEN = 0x01


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _blob(raw):
    buffer = ctypes.create_string_buffer(raw, len(raw))
    return _Blob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer


def _crypt(raw, protect):
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    data, keep_data = _blob(raw)
    entropy, keep_entropy = _blob(ENTROPY)
    out = _Blob()
    if protect:
        ok = crypt32.CryptProtectData(ctypes.byref(data), "Satori provider API key", ctypes.byref(entropy),
                                      None, None, CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out))
    else:
        ok = crypt32.CryptUnprotectData(ctypes.byref(data), None, ctypes.byref(entropy),
                                        None, None, CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out))
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)


def _valid(key):
    return bool(key) and key.isascii() and not any(char.isspace() for char in key)


def save(key, store=None):
    store = Path(store or STORE)
    if not _valid(key):
        raise ValueError("The key must be nonempty ASCII without whitespace")
    store.parent.mkdir(parents=True, exist_ok=True)
    encrypted = _crypt(key.encode("ascii"), protect=True)
    temp = tempfile.NamedTemporaryFile(dir=store.parent, delete=False)
    try:
        with temp:
            temp.write(encrypted)
        os.replace(temp.name, store)
    finally:
        Path(temp.name).unlink(missing_ok=True)


def load(store=None):
    """The stored key, or None when nothing is stored."""
    store = Path(store or STORE)
    if not store.exists():
        return None
    key = _crypt(store.read_bytes(), protect=False).decode("ascii")
    if not _valid(key):
        raise ValueError("The stored key is malformed")
    return key


def main(argv):
    command = argv[1] if len(argv) > 1 else "check"
    if command == "store":
        save(sys.stdin.readline().strip())
        print(f"stored, encrypted for this Windows user: {STORE}")
    elif command == "check":
        print(f"stored: {load() is not None} ({STORE})")
    elif command == "delete":
        STORE.unlink(missing_ok=True)
        print("deleted")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
