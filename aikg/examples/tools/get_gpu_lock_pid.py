#!/usr/bin/env python3
"""Find processes that hold AIKG GPU flock files.

The worker creates files such as gpu_0.lock and keeps an exclusive flock on the
open file descriptor. The lock file does not contain a PID, so this tool matches
the file's device/inode against /proc/locks and then reads /proc/<pid>.
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_A100_LOCK_DIR = "/ssd/zhangzizheng/aikg_gpu_locks"
DEFAULT_LOCK_DIR = "/mnt/lustre-client/zhangzizheng/aikg_gpu_locks"
GPU_LOCK_RE = re.compile(r"gpu_(\d+)\.lock$")


@dataclass(frozen=True)
class LockRecord:
    lock_id: str
    waiting: bool
    lock_type: str
    mode: str
    pid: Optional[int]
    dev_major: int
    dev_minor: int
    inode: int
    raw: str


def detect_gpu_lock_dir() -> Path:
    configured_dir = os.environ.get("AIKG_GPU_LOCK_DIR")
    if configured_dir:
        return Path(configured_dir)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return Path(DEFAULT_LOCK_DIR)

    if result.returncode == 0 and "A100" in result.stdout:
        return Path(DEFAULT_A100_LOCK_DIR)
    return Path(DEFAULT_LOCK_DIR)


def parse_proc_lock_line(line: str) -> Optional[LockRecord]:
    parts = line.split()
    if len(parts) < 8:
        return None

    waiting = parts[1] == "->"
    offset = 1 if waiting else 0
    if len(parts) <= 5 + offset:
        return None

    dev_inode = parts[5 + offset]
    try:
        major_s, minor_s, inode_s = dev_inode.split(":", 2)
        pid = int(parts[4 + offset])
        return LockRecord(
            lock_id=parts[0].rstrip(":"),
            waiting=waiting,
            lock_type=parts[1 + offset],
            mode=parts[3 + offset],
            pid=pid if pid > 0 else None,
            dev_major=int(major_s, 16),
            dev_minor=int(minor_s, 16),
            inode=int(inode_s),
            raw=line.rstrip("\n"),
        )
    except (IndexError, ValueError):
        return None


def read_proc_locks(proc_locks: Path = Path("/proc/locks")) -> List[LockRecord]:
    records: List[LockRecord] = []
    with proc_locks.open("r", encoding="utf-8") as f:
        for line in f:
            record = parse_proc_lock_line(line)
            if record is not None:
                records.append(record)
    return records


def lock_key(path: Path) -> Tuple[int, int, int]:
    st = path.stat()
    return os.major(st.st_dev), os.minor(st.st_dev), st.st_ino


def records_for_path(path: Path, records: Iterable[LockRecord]) -> List[LockRecord]:
    dev_major, dev_minor, inode = lock_key(path)
    return [
        record
        for record in records
        if (
            record.dev_major == dev_major
            and record.dev_minor == dev_minor
            and record.inode == inode
        )
    ]


def read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def read_status_fields(pid: int) -> Dict[str, str]:
    status = read_text(Path("/proc") / str(pid) / "status")
    if not status:
        return {}

    fields: Dict[str, str] = {}
    for line in status.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key] = value.strip()
    return fields


def read_cmdline(pid: int) -> str:
    try:
        data = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return ""

    if not data:
        return ""
    return " ".join(
        part.decode("utf-8", errors="replace")
        for part in data.rstrip(b"\0").split(b"\0")
        if part
    )


def read_link(path: Path) -> Optional[str]:
    try:
        return os.readlink(path)
    except OSError:
        return None


def process_info(pid: Optional[int]) -> Optional[Dict[str, Any]]:
    if pid is None:
        return None

    proc_dir = Path("/proc") / str(pid)
    info: Dict[str, Any] = {"pid": pid}
    try:
        proc_stat = proc_dir.stat()
        info["user"] = pwd.getpwuid(proc_stat.st_uid).pw_name
    except (KeyError, OSError):
        info["user"] = None

    status = read_status_fields(pid)
    info["name"] = status.get("Name") or read_text(proc_dir / "comm")
    info["state"] = status.get("State")
    info["ppid"] = int(status["PPid"]) if status.get("PPid", "").isdigit() else None
    info["cmdline"] = read_cmdline(pid)
    info["cwd"] = read_link(proc_dir / "cwd")
    info["exe"] = read_link(proc_dir / "exe")
    return info


def lock_record_to_dict(record: LockRecord) -> Dict[str, Any]:
    return {
        "lock_id": record.lock_id,
        "waiting": record.waiting,
        "lock_type": record.lock_type,
        "mode": record.mode,
        "pid": record.pid,
        "dev": f"{record.dev_major:x}:{record.dev_minor:x}",
        "inode": record.inode,
        "raw": record.raw,
        "process": process_info(record.pid),
    }


def device_id_from_path(path: Path) -> Optional[int]:
    match = GPU_LOCK_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def inspect_lock(path: Path, all_records: List[LockRecord]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(path),
        "device_id": device_id_from_path(path),
        "exists": path.exists(),
    }
    if not result["exists"]:
        result["status"] = "missing"
        return result

    try:
        matched_records = records_for_path(path, all_records)
    except OSError as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        return result

    holders = [record for record in matched_records if not record.waiting]
    waiters = [record for record in matched_records if record.waiting]
    result["status"] = "locked" if holders else "free"
    result["holders"] = [lock_record_to_dict(record) for record in holders]
    result["waiters"] = [lock_record_to_dict(record) for record in waiters]
    return result


def lock_sort_key(path: Path) -> Tuple[int, Any]:
    device_id = device_id_from_path(path)
    if device_id is not None:
        return 0, device_id
    return 1, path.name


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    unique: List[Path] = []
    for path in paths:
        resolved = str(path.expanduser().resolve(strict=False))
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(Path(resolved))
    return unique


def collect_targets(args: argparse.Namespace, lock_dir: Path) -> List[Path]:
    targets: List[Path] = [Path(path) for path in args.lock_paths]

    for device_id in args.device_id or []:
        targets.append(lock_dir / f"gpu_{device_id}.lock")

    if not targets:
        if lock_dir.exists():
            targets.extend(sorted(lock_dir.glob("gpu_*.lock"), key=lock_sort_key))
        else:
            targets.append(lock_dir / "gpu_0.lock")

    return unique_paths(targets)


def ellipsize(value: str, max_len: int = 240) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def format_process(proc: Optional[Dict[str, Any]], pid: Optional[int]) -> str:
    if proc is None:
        return f"pid={pid if pid is not None else 'unknown'}"

    parts = [f"pid={proc['pid']}"]
    if proc.get("user"):
        parts.append(f"user={proc['user']}")
    if proc.get("ppid") is not None:
        parts.append(f"ppid={proc['ppid']}")
    if proc.get("state"):
        parts.append(f"state={proc['state']}")

    command = proc.get("cmdline") or proc.get("name") or "<cmdline unavailable>"
    parts.append(f"cmd={ellipsize(command)}")
    return " ".join(parts)


def print_text_report(
    results: List[Dict[str, Any]],
    lock_dir: Path,
    targets_from_scan: bool,
) -> None:
    print(f"GPU lock dir: {lock_dir}")
    if not results:
        print("No gpu_*.lock files found.")
        return

    for result in results:
        device = result.get("device_id")
        label = f"gpu_{device}.lock" if device is not None else Path(result["path"]).name
        print(f"\n{label}: {result['status'].upper()}")
        print(f"  path: {result['path']}")

        if result["status"] == "missing":
            if not targets_from_scan:
                print("  note: lock file does not exist.")
            continue
        if result["status"] == "error":
            print(f"  error: {result.get('error')}")
            continue
        if result["status"] == "free":
            print("  holder: none")
            continue

        for holder in result.get("holders", []):
            print(
                "  holder: "
                f"type={holder['lock_type']} mode={holder['mode']} "
                f"{format_process(holder.get('process'), holder.get('pid'))}"
            )
            cwd = (holder.get("process") or {}).get("cwd")
            if cwd:
                print(f"          cwd={cwd}")

        for waiter in result.get("waiters", []):
            print(
                "  waiter: "
                f"type={waiter['lock_type']} mode={waiter['mode']} "
                f"{format_process(waiter.get('process'), waiter.get('pid'))}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find the process that currently holds AIKG GPU lock files.",
    )
    parser.add_argument(
        "lock_paths",
        nargs="*",
        help="Specific lock file path(s). Defaults to scanning gpu_*.lock in the lock dir.",
    )
    parser.add_argument(
        "-d",
        "--device-id",
        action="append",
        type=int,
        help="Inspect one GPU id, for example -d 0. Can be repeated.",
    )
    parser.add_argument(
        "--lock-dir",
        type=Path,
        default=None,
        help="GPU lock directory. Defaults to AIKG_GPU_LOCK_DIR or local_worker.py detection.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    lock_dir = args.lock_dir.expanduser() if args.lock_dir else detect_gpu_lock_dir()
    targets = collect_targets(args, lock_dir)
    targets_from_scan = not args.lock_paths and not args.device_id

    try:
        all_records = read_proc_locks()
    except OSError as exc:
        print(f"Failed to read /proc/locks: {exc}", file=sys.stderr)
        return 2

    results = [inspect_lock(path, all_records) for path in targets]
    if args.json:
        print(json.dumps({"lock_dir": str(lock_dir), "locks": results}, indent=2))
    else:
        print_text_report(results, lock_dir, targets_from_scan)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
