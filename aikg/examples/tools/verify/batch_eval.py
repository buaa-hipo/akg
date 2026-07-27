#!/usr/bin/env python3
"""Batch wrapper around triton_speedup_eval.py for one evolve_database level."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from triton_speedup_eval import (
    GPUExclusiveSettings,
    SUPPORTED_ARCHS,
    infer_baseline_path,
    infer_op_name_from_path,
    prepare_baseline_dir,
    read_text,
    run_shared_baseline,
)


TOOL_DIR = Path(__file__).resolve().parent
SPEEDUP_EVAL = TOOL_DIR / "triton_speedup_eval.py"
DEFAULT_BATCH_ROOT = TOOL_DIR / "batch_eval_runs"
# DEFAULT_EVOLVE_DATABASE = Path("/home/zhangzizheng/AIKG/akg/aikg/evolve_database")
DEFAULT_EVOLVE_DATABASE = Path("/mnt/lustre-client/zhangzizheng/AIKG/save_data/gpt_5_5_h100/evolve_database")
# DEFAULT_EVOLVE_DATABASE = Path("/mnt/lustre-client/zhangzizheng/AIKG/save_data/ds_v4_flash_h100/evolve_database")


@dataclass
class OperatorResult:
    op_name: str
    op_dir: str
    output_dir: str
    status: str
    best_speedup: Optional[float]
    best_candidate_id: str = ""
    best_gen_time_us: Optional[float] = None
    base_time_us: Optional[float] = None
    selected_local_time_field: str = ""
    selected_local_time_us: Optional[float] = None
    selected_info_path: str = ""
    selected_old_gen_time_us: Optional[float] = None
    selected_old_speedup: Optional[float] = None
    returncode: Optional[int] = None
    error: str = ""


@dataclass
class QuickCandidate:
    candidate_id: str
    impl_path: str
    info_path: str
    task_id: str
    unique_dir: str
    arch: str
    profile_gen_time_us: float
    profile_base_time_us: Optional[float]
    profile_speedup: Optional[float]


def numeric_prefix(path: Path) -> Tuple[int, str]:
    match = re.match(r"(\d+)_", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def sanitize_name(name: str, max_len: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return (cleaned or "operator")[:max_len]


def normalize_level_name(level: str) -> str:
    level = level.strip()
    if not level:
        raise ValueError("--level cannot be empty")
    if level.isdigit():
        return f"level{level}"
    return level


def parse_range_spec(value: str) -> Tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)\s*", value)
    if not match:
        raise ValueError("--range must use the form START,END, for example: --range 2,12")
    start, end = int(match.group(1)), int(match.group(2))
    if start > end:
        raise ValueError(f"--range start must be <= end, got {start},{end}")
    return start, end


def resolve_level_dir(args: argparse.Namespace) -> Path:
    if args.level_dir is not None and args.level is not None:
        raise ValueError("Provide either positional level_dir or --level, not both.")
    if args.level_dir is not None:
        level_dir = args.level_dir.expanduser().resolve()
    elif args.level is not None:
        evolve_database = args.evolve_database.expanduser().resolve()
        level_dir = evolve_database / normalize_level_name(args.level)
    else:
        raise ValueError("Provide --level level1 or a positional level_dir.")
    if not level_dir.is_dir():
        raise FileNotFoundError(f"level_dir not found: {level_dir}")
    return level_dir


def discover_operator_dirs(level_dir: Path) -> List[Path]:
    op_dirs = [path for path in level_dir.iterdir() if path.is_dir()]
    return sorted(op_dirs, key=numeric_prefix)


def read_results_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_best_speedup(results_csv: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    rows = read_results_csv(results_csv)
    best_row: Dict[str, Any] = {}
    best_speedup: Optional[float] = None
    for row in rows:
        if row.get("status") != "ok":
            continue
        speedup = parse_float(row.get("speedup"))
        if speedup is None or not math.isfinite(speedup):
            continue
        if best_speedup is None or speedup > best_speedup:
            best_speedup = speedup
            best_row = row
    return best_speedup, best_row


def discover_quick_candidates(evolve_dir: Path, arch_override: Optional[str]) -> List[QuickCandidate]:
    candidates: List[QuickCandidate] = []
    for info_path in sorted(evolve_dir.rglob("impl_info.json")):
        impl_path = info_path.with_name("impl_code.py")
        if not impl_path.is_file():
            continue
        try:
            info = load_json(info_path)
        except Exception:
            continue
        task_info = info.get("task_info") or {}
        framework = (task_info.get("framework") or info.get("framework") or "").lower()
        dsl = (task_info.get("dsl") or info.get("dsl") or "").lower()
        backend = (task_info.get("backend") or info.get("backend") or "").lower()
        if framework != "torch" or dsl != "triton_cuda" or backend != "cuda":
            continue

        arch = (task_info.get("arch") or info.get("arch") or "h100").lower()
        if arch not in SUPPORTED_ARCHS:
            continue
        if arch_override is not None and arch != arch_override:
            continue

        profile = info.get("profile") or {}
        profile_gen_time_us = parse_float(profile.get("gen_time"))
        if profile_gen_time_us is None or not math.isfinite(profile_gen_time_us) or profile_gen_time_us <= 0:
            continue

        profile_base_time_us = parse_float(profile.get("base_time"))
        profile_speedup = parse_float(profile.get("speedup"))
        candidates.append(
            QuickCandidate(
                candidate_id=str(info.get("id") or info_path.parent.name),
                impl_path=str(impl_path),
                info_path=str(info_path),
                task_id=str(info.get("task_id") or task_info.get("task_id") or info_path.parent.name),
                unique_dir=str(info.get("unique_dir") or task_info.get("unique_dir") or info_path.parent.name),
                arch=arch,
                profile_gen_time_us=profile_gen_time_us,
                profile_base_time_us=profile_base_time_us,
                profile_speedup=profile_speedup,
            )
        )
    return candidates


def select_quick_candidate(candidates: Sequence[QuickCandidate]) -> QuickCandidate:
    if not candidates:
        raise FileNotFoundError("No valid candidates with impl_info.json profile.gen_time were found")
    return min(candidates, key=lambda item: (item.profile_gen_time_us, item.candidate_id, item.impl_path))


def operator_result_to_dict(result: OperatorResult) -> Dict[str, Any]:
    return {
        "op_name": result.op_name,
        "op_dir": result.op_dir,
        "output_dir": result.output_dir,
        "status": result.status,
        "best_speedup": result.best_speedup,
        "best_candidate_id": result.best_candidate_id,
        "best_gen_time_us": result.best_gen_time_us,
        "base_time_us": result.base_time_us,
        "selected_local_time_field": result.selected_local_time_field,
        "selected_local_time_us": result.selected_local_time_us,
        "selected_info_path": result.selected_info_path,
        "selected_old_gen_time_us": result.selected_old_gen_time_us,
        "selected_old_speedup": result.selected_old_speedup,
        "returncode": result.returncode,
        "error": result.error,
    }


def write_summary_csv(path: Path, results: Sequence[OperatorResult], include_header: bool) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if include_header:
            writer.writerow(["operator", "best_speedup"])
        for result in results:
            speedup = "" if result.best_speedup is None else f"{result.best_speedup:.6f}"
            writer.writerow([result.op_name, speedup])


def write_detail_json(path: Path, results: Sequence[OperatorResult], manifest: Dict[str, Any]) -> None:
    payload = {
        "manifest": manifest,
        "results": [operator_result_to_dict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_eval_command(args: argparse.Namespace, op_dir: Path, output_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(SPEEDUP_EVAL),
        str(op_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        str(args.device),
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        "--timeout",
        str(args.timeout),
        "--baseline_repeats",
        str(args.baseline_repeats),
        "--gpu_wait_timeout_minutes",
        str(args.gpu_wait_timeout_minutes),
        "--gpu_monitor_interval_seconds",
        str(args.gpu_monitor_interval_seconds),
        "--gpu_contamination_max_retries",
        str(args.gpu_contamination_max_retries),
        "--overwrite",
    ]
    if args.arch:
        cmd.extend(["--arch", args.arch])
    if args.torch_compile:
        cmd.append("--torch_compile")
    if args.torch_compile_warmup is not None:
        cmd.extend(["--torch_compile_warmup", str(args.torch_compile_warmup)])
    if args.torch_compile_mode:
        cmd.extend(["--torch_compile_mode", args.torch_compile_mode])
    if args.candidate_limit is not None:
        cmd.extend(["--limit", str(args.candidate_limit)])
    if args.eval_dry_run:
        cmd.append("--dry-run")
    return cmd


def run_operator(args: argparse.Namespace, op_dir: Path, op_output_dir: Path) -> OperatorResult:
    cmd = build_eval_command(args, op_dir, op_output_dir)
    op_output_dir.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if args.print_commands:
        print(" ".join(cmd))

    try:
        process = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            env=env,
            timeout=args.operator_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return OperatorResult(
            op_name=op_dir.name,
            op_dir=str(op_dir),
            output_dir=str(op_output_dir),
            status="failed",
            best_speedup=None,
            error=f"operator timeout after {args.operator_timeout}s: {exc}",
        )

    op_output_dir.mkdir(parents=True, exist_ok=True)
    (op_output_dir / "batch_stdout.log").write_text(process.stdout, encoding="utf-8")
    (op_output_dir / "batch_stderr.log").write_text(process.stderr, encoding="utf-8")
    (op_output_dir / "batch_command.json").write_text(
        json.dumps({"cmd": cmd, "returncode": process.returncode}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_speedup, best_row = select_best_speedup(op_output_dir / "results.csv")
    status = "ok" if best_speedup is not None else "failed"
    error = ""
    if status == "failed":
        if args.eval_dry_run and process.returncode == 0:
            status = "prepared"
        elif process.returncode != 0:
            error = f"triton_speedup_eval returned {process.returncode}"
        else:
            error = "no successful candidate in results.csv"

    return OperatorResult(
        op_name=op_dir.name,
        op_dir=str(op_dir),
        output_dir=str(op_output_dir),
        status=status,
        best_speedup=best_speedup,
        best_candidate_id=best_row.get("candidate_id", "") if best_row else "",
        best_gen_time_us=parse_float(best_row.get("gen_time_us")) if best_row else None,
        base_time_us=parse_float(best_row.get("base_time_us")) if best_row else None,
        returncode=process.returncode,
        error=error,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run triton_speedup_eval.py for operator directories in one evolve_database level.",
        epilog=(
            "Example: python batch_eval.py --level level1 --range 2,12 "
            "--csv level1_2_12_speedups.csv"
        ),
    )
    parser.add_argument(
        "level_dir",
        type=Path,
        nargs="?",
        help="Optional explicit path like .../evolve_database/level1.",
    )
    parser.add_argument(
        "--level",
        default=None,
        help="Level name under the evolve database, for example level1. Numeric values like 1 are also accepted.",
    )
    parser.add_argument(
        "--evolve-database",
        type=Path,
        default=DEFAULT_EVOLVE_DATABASE,
        help=f"Root evolve_database path used with --level. Default: {DEFAULT_EVOLVE_DATABASE}",
    )
    parser.add_argument(
        "--range",
        dest="range_spec",
        default=None,
        help="Closed numeric-prefix range START,END, for example --range 2,12.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Batch output directory. Defaults under examples/tools/verify/batch_eval_runs.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Summary CSV path. Defaults to <output-dir>/best_speedups.csv.",
    )
    parser.add_argument("--no-header", action="store_true", help="Write summary CSV without a header row.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", choices=["a100", "h100"], default=None)
    parser.add_argument("--warmup", type=int, default=5, help="do_bench warmup budget in ms.")
    parser.add_argument("--runs", type=int, default=50, help="do_bench benchmark budget in ms.")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout for each profile script.")
    parser.add_argument("--baseline_repeats", type=int, default=2)
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--torch_compile_warmup", type=int, default=3)
    parser.add_argument("--torch_compile_mode", default=None)
    parser.add_argument("--gpu_wait_timeout_minutes", type=int, default=120)
    parser.add_argument("--gpu_monitor_interval_seconds", type=float, default=1.0)
    parser.add_argument("--gpu_contamination_max_retries", type=int, default=10)
    parser.add_argument(
        "--candidate_limit",
        type=int,
        default=3,
        help=(
            "Per-operator candidate count passed to triton_speedup_eval.py in regular mode. "
            "Candidates are ranked by impl_info.json profile.speedup; use 0 to evaluate all."
        ),
    )
    parser.add_argument(
        "--quick_test",
        "--quick-test",
        action="store_true",
        help=(
            "Only benchmark the PyTorch baseline and select the single fastest "
            "candidate by impl_info.json profile.gen_time. Ignores --candidate_limit."
        ),
    )
    parser.add_argument(
        "--operator_limit",
        type=int,
        default=None,
        help="Only evaluate the first N operators after numeric-prefix sorting.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Only evaluate operators whose numeric prefix is >= this value. Prefer --range for new usage.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Only evaluate operators whose numeric prefix is <= this value. Prefer --range for new usage.",
    )
    parser.add_argument(
        "--eval-dry-run",
        action="store_true",
        help="Pass --dry-run to triton_speedup_eval.py; useful for validating paths without GPU benchmarking.",
    )
    parser.add_argument("--stop-on-error", action="store_true", help="Stop batch after the first failed operator.")
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument(
        "--operator-timeout",
        type=int,
        default=None,
        help="Optional wall-clock timeout in seconds for one operator-level triton_speedup_eval.py process.",
    )
    args = parser.parse_args(argv)
    if args.range_spec is not None:
        if args.start_index is not None or args.end_index is not None:
            parser.error("Use either --range or --start-index/--end-index, not both.")
        try:
            args.start_index, args.end_index = parse_range_spec(args.range_spec)
        except ValueError as exc:
            parser.error(str(exc))
    if args.level_dir is None and args.level is None:
        parser.error("Provide --level level1 or a positional level_dir.")
    if args.level_dir is not None and args.level is not None:
        parser.error("Provide either positional level_dir or --level, not both.")
    if args.candidate_limit is not None and args.candidate_limit < 0:
        parser.error("--candidate_limit must be >= 0; use 0 to evaluate all candidates.")
    if args.quick_test and args.eval_dry_run:
        parser.error("--quick_test cannot be combined with --eval-dry-run.")
    return args


def create_output_dir(args: argparse.Namespace, level_dir: Path) -> Path:
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_BATCH_ROOT / f"{stamp}_{sanitize_name(level_dir.name)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def filter_operator_dirs(args: argparse.Namespace, op_dirs: List[Path]) -> List[Path]:
    filtered = []
    for op_dir in op_dirs:
        prefix, _ = numeric_prefix(op_dir)
        if args.start_index is not None and prefix < args.start_index:
            continue
        if args.end_index is not None and prefix > args.end_index:
            continue
        filtered.append(op_dir)
    if args.operator_limit is not None:
        filtered = filtered[: args.operator_limit]
    return filtered


def run_quick_operator(args: argparse.Namespace, op_dir: Path, op_output_dir: Path) -> OperatorResult:
    candidates = discover_quick_candidates(op_dir, args.arch)
    selected = select_quick_candidate(candidates)

    baseline_path = infer_baseline_path(op_dir)
    baseline_code = read_text(baseline_path)
    op_name = infer_op_name_from_path(op_dir)
    baseline_dir = op_output_dir / "baseline"
    op_output_dir.mkdir(parents=True, exist_ok=True)

    gpu_settings = GPUExclusiveSettings(
        device_id=args.device,
        wait_timeout_minutes=args.gpu_wait_timeout_minutes,
        monitor_interval_seconds=max(0.2, args.gpu_monitor_interval_seconds),
        contamination_max_retries=args.gpu_contamination_max_retries,
    )
    baseline_arch = args.arch or selected.arch
    quick_manifest_path = op_output_dir / "quick_manifest.json"
    quick_manifest: Dict[str, Any] = {
        "op_name": op_name,
        "op_dir": str(op_dir),
        "baseline": str(baseline_path),
        "baseline_dir": str(baseline_dir),
        "baseline_mode": "torch_compile" if args.torch_compile else "torch_eager",
        "quick_test": True,
        "selection_metric": "impl_info.profile.gen_time",
        "candidate_count": len(candidates),
        "candidate_limit": args.candidate_limit,
        "candidate_limit_applied": False,
        "selected_candidate": {
            "candidate_id": selected.candidate_id,
            "impl_path": selected.impl_path,
            "info_path": selected.info_path,
            "task_id": selected.task_id,
            "unique_dir": selected.unique_dir,
            "arch": selected.arch,
            "profile_gen_time_us": selected.profile_gen_time_us,
            "profile_base_time_us": selected.profile_base_time_us,
            "profile_speedup": selected.profile_speedup,
        },
    }
    quick_manifest_path.write_text(
        json.dumps(quick_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    prepare_baseline_dir(
        op_name=op_name,
        arch=baseline_arch,
        baseline_dir=baseline_dir,
        baseline_code=baseline_code,
        warmup=args.warmup,
        runs=args.runs,
        device_id=args.device,
        torch_compile=args.torch_compile,
        torch_compile_warmup=args.torch_compile_warmup,
        torch_compile_mode=args.torch_compile_mode,
        baseline_repeats=args.baseline_repeats,
        gpu_settings=gpu_settings,
    )

    try:
        baseline_summary = run_shared_baseline(
            baseline_dir=baseline_dir,
            op_name=op_name,
            repeats=args.baseline_repeats,
            timeout=args.timeout,
            task_id=op_name,
            gpu_settings=gpu_settings,
        )
        baseline_time_us = baseline_summary.get("base_time_us")
        if baseline_time_us is None or not math.isfinite(baseline_time_us) or baseline_time_us <= 0:
            raise RuntimeError("shared baseline time is unavailable")
        best_speedup = baseline_time_us / selected.profile_gen_time_us
        quick_manifest["baseline_summary"] = baseline_summary
        quick_manifest_path.write_text(
            json.dumps(quick_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        result = OperatorResult(
            op_name=op_dir.name,
            op_dir=str(op_dir),
            output_dir=str(op_output_dir),
            status="ok",
            best_speedup=best_speedup,
            best_candidate_id=selected.candidate_id,
            best_gen_time_us=selected.profile_gen_time_us,
            base_time_us=baseline_time_us,
            selected_local_time_field="gen_time",
            selected_local_time_us=selected.profile_gen_time_us,
            selected_info_path=selected.info_path,
            selected_old_gen_time_us=selected.profile_gen_time_us,
            selected_old_speedup=selected.profile_speedup,
            returncode=0,
        )
    except Exception as exc:
        quick_manifest["baseline_error"] = f"{type(exc).__name__}: {exc}"
        quick_manifest_path.write_text(
            json.dumps(quick_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        result = OperatorResult(
            op_name=op_dir.name,
            op_dir=str(op_dir),
            output_dir=str(op_output_dir),
            status="failed",
            best_speedup=None,
            best_candidate_id=selected.candidate_id,
            best_gen_time_us=selected.profile_gen_time_us,
            base_time_us=None,
            selected_local_time_field="gen_time",
            selected_local_time_us=selected.profile_gen_time_us,
            selected_info_path=selected.info_path,
            selected_old_gen_time_us=selected.profile_gen_time_us,
            selected_old_speedup=selected.profile_speedup,
            returncode=None,
            error=f"{type(exc).__name__}: {exc}",
        )
        if op_output_dir.exists():
            (op_output_dir / "quick_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        return result
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    level_dir = resolve_level_dir(args)
    op_dirs = filter_operator_dirs(args, discover_operator_dirs(level_dir))
    if not op_dirs:
        raise FileNotFoundError(f"No operator directories found under {level_dir}")

    output_dir = create_output_dir(args, level_dir)
    csv_path = args.csv.expanduser().resolve() if args.csv else output_dir / "best_speedups.csv"
    detail_path = output_dir / "batch_results_detail.json"

    manifest = {
        "level_dir": str(level_dir),
        "level": args.level,
        "evolve_database": str(args.evolve_database.expanduser().resolve()),
        "range": args.range_spec,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "output_dir": str(output_dir),
        "csv": str(csv_path),
        "operator_count": len(op_dirs),
        "quick_test": args.quick_test,
        "selection_metric": "impl_info.profile.gen_time" if args.quick_test else "impl_info.profile.speedup",
        "candidate_limit": args.candidate_limit,
        "candidate_limit_applied": not args.quick_test,
        "candidate_selection": "min_impl_info_profile_gen_time" if args.quick_test else "top_impl_info_profile_speedup",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "torch_compile": args.torch_compile,
        "baseline_repeats": args.baseline_repeats,
        "device": args.device,
    }
    (output_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    results: List[OperatorResult] = []
    print(f"Level dir: {level_dir}")
    print(f"Operators: {len(op_dirs)}")
    print(f"Mode: {'quick_test' if args.quick_test else 'full_eval'}")
    print(f"Output dir: {output_dir}")
    print(f"CSV: {csv_path}")

    for idx, op_dir in enumerate(op_dirs, start=1):
        prefix, _ = numeric_prefix(op_dir)
        op_output_dir = output_dir / "operators" / f"{prefix:04d}_{sanitize_name(op_dir.name)}"
        print(f"[{idx}/{len(op_dirs)}] Evaluating {op_dir.name}")
        if args.quick_test:
            result = run_quick_operator(args, op_dir, op_output_dir)
        else:
            result = run_operator(args, op_dir, op_output_dir)
        results.append(result)

        write_summary_csv(csv_path, results, include_header=not args.no_header)
        write_detail_json(detail_path, results, manifest)

        if result.status == "prepared":
            print("  prepared")
        elif result.best_speedup is None:
            print(f"  failed: {result.error}")
            if args.stop_on_error:
                break
        else:
            if args.quick_test:
                print(
                    f"  selected={result.best_candidate_id} "
                    f"gen_time={result.best_gen_time_us:.6f}us "
                    f"speedup={result.best_speedup:.6f}"
                )
            else:
                print(f"  best_speedup={result.best_speedup:.6f} ({result.best_candidate_id})")

    failures = [item for item in results if item.status == "failed"]
    prepared = [item for item in results if item.status == "prepared"]
    ok_count = len([item for item in results if item.status == "ok"])
    print(f"Finished: ok={ok_count}/{len(results)}, prepared={len(prepared)}, failed={len(failures)}")
    print(f"Summary CSV: {csv_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
