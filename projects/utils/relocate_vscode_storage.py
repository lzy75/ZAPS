"""Offline, zero-UNet VS Code storage relocation. Uses Python standard library."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time


def inventory(root):
    rows = {}
    for base, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            p = Path(base) / name
            s = p.lstat()
            key = str(p.relative_to(root))
            if stat.S_ISLNK(s.st_mode):
                rows[key] = ("link", os.readlink(p))
            elif stat.S_ISDIR(s.st_mode):
                rows[key] = ("dir", stat.S_IMODE(s.st_mode))
            elif stat.S_ISREG(s.st_mode):
                h = hashlib.sha256()
                with p.open("rb") as f:
                    for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                        h.update(block)
                rows[key] = ("file", s.st_size, stat.S_IMODE(s.st_mode), h.hexdigest())
            else:
                raise RuntimeError("Special file; original retained: " + str(p))
    return rows


def running_vscode(source):
    prefix = os.fsencode(str(source).rstrip("/") + "/")
    found = []
    for p in Path("/proc").glob("[0-9]*"):
        try:
            if p.stat().st_uid == os.getuid() and prefix in (p / "cmdline").read_bytes():
                found.append(int(p.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return found


def migrate(source, destination, *, check_processes=True, require_other_fs=True):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if source.is_symlink():
        if source.resolve().is_dir():
            print("ALREADY_LINKED:", source, "->", source.resolve(), flush=True)
            return
        raise RuntimeError("Broken source link; no changes")
    if not source.is_dir() or source.stat().st_uid != os.getuid():
        raise RuntimeError("Missing/not-owned source; no changes")
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("Destination exists; no overwrite")
    if source in destination.parents or destination in source.parents:
        raise RuntimeError("Nested source/destination refused")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if require_other_fs and source.stat().st_dev == destination.parent.stat().st_dev:
        raise RuntimeError("Destination is not on another filesystem")
    if check_processes and running_vscode(source):
        raise RuntimeError("VS Code processes still running: " + str(running_vscode(source)))
    print("1/4 Hashing original files...", flush=True)
    before = inventory(source)
    required = sum(r[1] for r in before.values() if r[0] == "file")
    if shutil.disk_usage(destination.parent).free < required + 1024 ** 3:
        raise RuntimeError("Destination needs copy size plus 1 GiB")
    print("2/4 Copying", required, "bytes to", destination, flush=True)
    shutil.copytree(source, destination, symlinks=True)
    print("3/4 Verifying destination and unchanged source...", flush=True)
    if inventory(destination) != before or inventory(source) != before:
        raise RuntimeError("Copy/source changed; original retained, no deletion")
    if check_processes and running_vscode(source):
        raise RuntimeError("VS Code restarted; original retained, no deletion")
    manifest = destination.with_name(destination.name + ".migration.json")
    record = dict(status="copy_verified", source=str(source), destination=str(destination),
                  bytes=required, inventory=before)
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    backup = source.with_name(source.name + ".verified_root_" + str(time.time_ns()))
    source.rename(backup)
    try:
        source.symlink_to(destination, target_is_directory=True)
        if not source.is_symlink() or source.resolve() != destination.resolve():
            raise RuntimeError("Link verification failed")
    except BaseException:
        if source.is_symlink():
            source.unlink()
        if not source.exists():
            backup.rename(source)
        raise
    print("4/4 Removing only the verified root duplicate...", flush=True)
    shutil.rmtree(backup)
    record["status"] = "complete"
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print("MIGRATION_PASS:", source, "->", destination, flush=True)
    print("Manifest:", manifest, flush=True)
    print("Original bytes preserved on data disk; new UNet NFE=0.", flush=True)


if __name__ == "__main__":
    migrate("/home/lzy/.vscode-server",
            "/data0/lzy/vscode-server-" + time.strftime("%Y%m%d_%H%M%S") + "-" + str(os.getpid()))
