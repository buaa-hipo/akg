#!/usr/bin/env python3
"""Standalone speedup evaluator for evolved PyTorch + Triton CUDA kernels.

This tool rebuilds the same profile-script layout used by KernelVerifier:
  {op_name}_torch.py
  {op_name}_triton_cuda.py
  profile_{op_name}_base.py
  profile_{op_name}_generation.py

Only PyTorch baseline + Triton CUDA kernels on NVIDIA A100/H100 are supported.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SUPPORTED_ARCHS = {"a100", "h100"}
TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = TOOL_DIR / "speedup_eval_runs"


def find_aikg_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "python" / "ai_kernel_generator").is_dir():
            return path
    raise RuntimeError(f"Cannot find aikg root from {start}")


AIKG_ROOT = find_aikg_root(TOOL_DIR)
AIKG_PYTHON = AIKG_ROOT / "python"
if str(AIKG_PYTHON) not in sys.path:
    sys.path.insert(0, str(AIKG_PYTHON))

existing_pythonpath = os.environ.get("PYTHONPATH", "")
pythonpath_parts = [str(AIKG_PYTHON)]
if existing_pythonpath:
    pythonpath_parts.append(existing_pythonpath)
os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)


def detect_gpu_lock_dir() -> str:
    configured_dir = os.environ.get("AIKG_GPU_LOCK_DIR")
    candidates: List[str] = []
    if configured_dir:
        candidates.append(configured_dir)
    else:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            result = None

        if result is not None and result.returncode == 0 and "A100" in result.stdout:
            candidates.append("/ssd/zhangzizheng/aikg_gpu_locks")
        else:
            candidates.append("/mnt/lustre-client/zhangzizheng/aikg_gpu_locks")

    candidates.append("/tmp/aikg_gpu_locks")
    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            return candidate
        except OSError:
            continue

    return "/tmp/aikg_gpu_locks"


GPU_LOCK_DIR = detect_gpu_lock_dir()


class GPUContaminationError(RuntimeError):
    """Raised when another process starts using the target GPU during a benchmark."""

    def __init__(self, device_id: int, external_pids: Set[int]):
        self.device_id = device_id
        self.external_pids = external_pids
        self.output_log = ""
        super().__init__(
            f"GPU_{device_id} was used by external pids during execution: "
            f"{sorted(external_pids)}"
        )


@dataclass
class Candidate:
    impl_path: Path
    rel_path: str
    candidate_id: str
    op_name: str
    task_id: str
    unique_dir: str
    round: Optional[int]
    arch: str
    info_path: Optional[Path]
    old_profile: Dict[str, Any] = field(default_factory=dict)
    raw_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    index: int
    candidate_id: str
    rel_path: str
    task_id: str
    unique_dir: str
    round: Optional[int]
    arch: str
    baseline_mode: str
    status: str
    base_time_us: Optional[float]
    gen_time_us: Optional[float]
    speedup: Optional[float]
    old_base_time_us: Optional[float]
    old_gen_time_us: Optional[float]
    old_speedup: Optional[float]
    workdir: str
    error: str = ""


@dataclass
class GPUExclusiveSettings:
    device_id: int
    wait_timeout_minutes: int
    monitor_interval_seconds: float
    contamination_max_retries: int


@asynccontextmanager
async def gpu_execution_lock(device_id: int, task_id: str):
    os.makedirs(GPU_LOCK_DIR, exist_ok=True)
    lock_path = os.path.join(GPU_LOCK_DIR, f"gpu_{device_id}.lock")
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        print(f"[{task_id}] Waiting for GPU_{device_id} lock: {lock_path}")
        await asyncio.to_thread(fcntl.flock, lock_file.fileno(), fcntl.LOCK_EX)
        print(f"[{task_id}] Acquired GPU_{device_id} lock")
        yield
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            print(f"[{task_id}] Released GPU_{device_id} lock")
        finally:
            lock_file.close()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_gpu_pid_output(output: str) -> Set[int]:
    pids = set()
    for line in output.splitlines():
        value = line.strip()
        if value.isdigit():
            pids.add(int(value))
    return pids


def query_gpu_compute_pids_sync(device_id: int) -> Set[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device_id}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi pid query failed: {result.stderr.strip()}")
    return parse_gpu_pid_output(result.stdout)


async def query_gpu_compute_pids(device_id: int, task_id: str) -> Set[int]:
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            f"--id={device_id}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        if process.returncode != 0:
            print(
                f"[{task_id}] nvidia-smi pid query failed for GPU_{device_id}: "
                f"{stderr.decode(errors='replace').strip()}",
                file=sys.stderr,
            )
            return set()
        return parse_gpu_pid_output(stdout.decode(errors="replace"))
    except asyncio.CancelledError:
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            with suppress(Exception):
                await process.wait()
        raise
    except asyncio.TimeoutError:
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            with suppress(Exception):
                await process.wait()
        print(f"[{task_id}] nvidia-smi pid query timed out for GPU_{device_id}", file=sys.stderr)
        return set()
    except FileNotFoundError:
        print(f"[{task_id}] nvidia-smi not found, GPU exclusivity monitor disabled", file=sys.stderr)
        return set()
    except Exception as exc:
        print(f"[{task_id}] Failed to query GPU_{device_id} compute pids: {exc}", file=sys.stderr)
        return set()


def wait_for_gpu_idle(device_id: int, timeout_minutes: int, poll_seconds: float = 3.0) -> None:
    max_count = max(1, int(timeout_minutes * 60 / poll_seconds))
    print(f"Waiting for GPU_{device_id} to become idle (timeout {timeout_minutes} min)")
    for attempt in range(max_count):
        try:
            compute_pids = query_gpu_compute_pids_sync(device_id)
            if not compute_pids:
                print(f"GPU_{device_id} is idle")
                return
            if attempt % 100 == 0:
                elapsed_min = attempt * poll_seconds / 60
                print(
                    f"GPU_{device_id} busy with pids {sorted(compute_pids)}; "
                    f"waited {elapsed_min:.2f} min"
                )
        except subprocess.TimeoutExpired:
            print(f"nvidia-smi timed out while waiting for GPU_{device_id}", file=sys.stderr)
        except FileNotFoundError:
            print("nvidia-smi not found; skipping GPU idle wait", file=sys.stderr)
            return
        except Exception as exc:
            print(f"GPU idle query failed: {exc}", file=sys.stderr)
        time.sleep(poll_seconds)
    raise TimeoutError(f"GPU_{device_id} did not become idle within {timeout_minutes} minutes")


def read_ppid_from_proc(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
            stat = f.read()
        fields = stat.rsplit(") ", 1)[1].split()
        if len(fields) > 1:
            return int(fields[1])
    except Exception:
        return None
    return None


def get_descendant_pids(root_pid: int) -> Set[int]:
    children_by_ppid: Dict[int, List[int]] = {}
    try:
        proc_entries = os.listdir("/proc")
    except Exception:
        return {root_pid}

    for entry in proc_entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        ppid = read_ppid_from_proc(pid)
        if ppid is not None:
            children_by_ppid.setdefault(ppid, []).append(pid)

    related_pids = {root_pid}
    stack = [root_pid]
    while stack:
        parent = stack.pop()
        for child in children_by_ppid.get(parent, []):
            if child in related_pids:
                continue
            related_pids.add(child)
            stack.append(child)
    return related_pids


def get_related_process_pids(root_pid: int) -> Set[int]:
    related_pids = get_descendant_pids(root_pid)
    related_pids.add(root_pid)

    try:
        root_pgid = os.getpgid(root_pid)
    except OSError:
        root_pgid = None

    if root_pgid is None:
        return related_pids

    try:
        proc_entries = os.listdir("/proc")
    except Exception:
        return related_pids

    for entry in proc_entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            if os.getpgid(pid) == root_pgid:
                related_pids.add(pid)
        except OSError:
            continue
    return related_pids


async def monitor_gpu_exclusivity(
    device_id: int,
    root_pid: int,
    task_id: str,
    operation_name: str,
    interval_seconds: float,
) -> None:
    while True:
        compute_pids = await query_gpu_compute_pids(device_id, task_id)
        if compute_pids:
            related_pids = get_related_process_pids(root_pid)
            external_pids = compute_pids - related_pids
            if external_pids:
                print(
                    f"[{task_id}] {operation_name} detected external GPU_{device_id} pids: "
                    f"{sorted(external_pids)} (own process group/tree: {sorted(related_pids)})",
                    file=sys.stderr,
                )
                raise GPUContaminationError(device_id, external_pids)
        await asyncio.sleep(interval_seconds)


def signal_process_group(process: asyncio.subprocess.Process, sig: signal.Signals, reason: str) -> None:
    if process.returncode is not None:
        return
    try:
        process_pgid = os.getpgid(process.pid)
    except OSError:
        process_pgid = None

    if process_pgid is not None and process_pgid != os.getpgrp():
        try:
            os.killpg(process_pgid, sig)
            return
        except ProcessLookupError:
            return
        except OSError as exc:
            print(f"Failed to signal process group {process_pgid} for {reason}: {exc}", file=sys.stderr)

    try:
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


async def stop_process_and_collect_output(
    process: asyncio.subprocess.Process,
    communicate_task: asyncio.Task,
    reason: str,
) -> Tuple[bytes, bytes]:
    signal_process_group(process, signal.SIGTERM, reason)
    try:
        return await asyncio.wait_for(asyncio.shield(communicate_task), timeout=5)
    except asyncio.TimeoutError:
        signal_process_group(process, signal.SIGKILL, reason)
        try:
            return await asyncio.wait_for(asyncio.shield(communicate_task), timeout=5)
        except Exception:
            return b"", b""
    except Exception:
        return b"", b""


async def run_monitored_gpu_subprocess_once(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    device_id: int,
    task_id: str,
    timeout: int,
    operation_name: str,
    monitor_interval_seconds: float,
) -> Tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    communicate_task = asyncio.create_task(process.communicate())
    monitor_task = asyncio.create_task(
        monitor_gpu_exclusivity(
            device_id,
            process.pid,
            task_id,
            operation_name,
            monitor_interval_seconds,
        )
    )

    try:
        done, _ = await asyncio.wait(
            {communicate_task, monitor_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            await stop_process_and_collect_output(process, communicate_task, f"{operation_name} timeout")
            raise asyncio.TimeoutError()

        if monitor_task in done:
            try:
                monitor_task.result()
            except GPUContaminationError as exc:
                stdout, stderr = await stop_process_and_collect_output(
                    process,
                    communicate_task,
                    f"{operation_name} GPU contamination",
                )
                exc.output_log = stdout.decode(errors="replace") + "\n" + stderr.decode(errors="replace")
                raise

        stdout, stderr = await communicate_task
        return process.returncode, stdout, stderr
    finally:
        if not monitor_task.done():
            monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await monitor_task
        if not communicate_task.done():
            await stop_process_and_collect_output(process, communicate_task, f"{operation_name} cleanup")


async def run_monitored_gpu_subprocess_with_retry(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    device_id: int,
    task_id: str,
    timeout: int,
    operation_name: str,
    wait_timeout_minutes: int,
    monitor_interval_seconds: float,
    contamination_max_retries: int,
    cleanup_on_contamination: Optional[Callable[[], None]] = None,
) -> Tuple[int, bytes, bytes]:
    last_error: Optional[GPUContaminationError] = None

    async with gpu_execution_lock(device_id, task_id):
        for attempt in range(contamination_max_retries + 1):
            if attempt > 0:
                print(
                    f"[{task_id}] Retrying {operation_name} after GPU contamination "
                    f"({attempt}/{contamination_max_retries})"
                )
            wait_for_gpu_idle(device_id, wait_timeout_minutes)
            try:
                return await run_monitored_gpu_subprocess_once(
                    cmd,
                    cwd,
                    env,
                    device_id,
                    task_id,
                    timeout,
                    operation_name,
                    monitor_interval_seconds,
                )
            except GPUContaminationError as exc:
                last_error = exc
                if cleanup_on_contamination is not None:
                    cleanup_on_contamination()
                if attempt >= contamination_max_retries:
                    raise
                print(
                    f"[{task_id}] {operation_name} will restart after GPU contamination: {exc}",
                    file=sys.stderr,
                )

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{operation_name} did not start")


def resolve_evolve_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        return path
    if not path.name.endswith("_"):
        alt = path.with_name(path.name + "_")
        if alt.is_dir():
            return alt
    raise FileNotFoundError(f"evolve_dir not found: {path}")


def strip_numeric_prefix(name: str) -> str:
    if "10_3D_tensor_matrix_multiplication" in name:
        return "three_D_tensor_matrix_multiplication"
    elif "11_4D_tensor_matrix_multiplication" in name:
        return "four_D_tensor_matrix_multiplication"
    return re.sub(r"^\d+_", "", name)


def infer_op_name_from_path(evolve_dir: Path) -> str:
    return strip_numeric_prefix(evolve_dir.name)


def detect_level_name(evolve_dir: Path) -> Optional[str]:
    parts = evolve_dir.parts
    for idx, part in enumerate(parts):
        if part == "evolve_database" and idx + 1 < len(parts):
            return parts[idx + 1]
    for part in parts:
        if re.fullmatch(r"level\d+", part):
            return part
    return None


def find_aikg_workspace_root(start: Path) -> Optional[Path]:
    for path in [start, *start.parents]:
        if (path / "KernelBench" / "KernelBench").is_dir():
            return path
    return None


def infer_baseline_path(evolve_dir: Path) -> Path:
    workspace_root = find_aikg_workspace_root(evolve_dir)
    if workspace_root is None:
        raise FileNotFoundError(
            "Cannot infer baseline path. Please pass --baseline explicitly."
        )

    level_name = detect_level_name(evolve_dir)
    if level_name is None:
        raise FileNotFoundError(
            "Cannot infer KernelBench level from evolve_dir. Please pass --baseline explicitly."
        )

    kernelbench_dir = workspace_root / "KernelBench" / "KernelBench" / level_name
    candidates = [
        kernelbench_dir / f"{evolve_dir.name}.py",
    ]
    if not evolve_dir.name.endswith("_"):
        candidates.append(kernelbench_dir / f"{evolve_dir.name}_.py")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    expected = kernelbench_dir / f"{evolve_dir.name}.py"
    raise FileNotFoundError(
        f"Cannot infer baseline path. Tried {expected}; please pass --baseline."
    )


def discover_impl_files(evolve_dir: Path) -> List[Path]:
    impl_files = sorted(evolve_dir.rglob("impl_code.py"))
    if impl_files:
        return impl_files

    skipped_prefixes = ("profile_", "verify_")
    result = []
    for path in sorted(evolve_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.name == "__init__.py":
            continue
        if path.name.startswith(skipped_prefixes):
            continue
        if path.name.endswith("_torch.py") or path.name.endswith("_triton_cuda.py"):
            continue
        result.append(path)
    return result


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    values = [int(x) for x in re.findall(r"\d+", text)]
    return tuple(values) if values else (10**9,)


def candidate_sort_key(candidate: Candidate) -> Tuple[Any, ...]:
    round_key = candidate.round if candidate.round is not None else 10**9
    return (round_key, parse_int_tuple(candidate.task_id), candidate.rel_path)


def candidate_old_speedup(candidate: Candidate) -> Optional[float]:
    speedup = to_float(candidate.old_profile.get("speedup"))
    if speedup is None or math.isnan(speedup):
        return None
    return speedup


def candidate_rank_key(candidate: Candidate) -> Tuple[Any, ...]:
    old_speedup = candidate_old_speedup(candidate)
    if old_speedup is None:
        speedup_key = (1, 0.0)
    else:
        speedup_key = (0, -old_speedup)
    return (*speedup_key, *candidate_sort_key(candidate))


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isinf(number) or math.isnan(number):
        return number
    return number


def load_candidate_info(impl_path: Path) -> Tuple[Optional[Path], Dict[str, Any]]:
    info_path = impl_path.with_name("impl_info.json")
    if not info_path.is_file():
        return None, {}
    return info_path, load_json(info_path)


def make_candidate(
    impl_path: Path,
    evolve_dir: Path,
    fallback_op_name: str,
    arch_override: Optional[str],
) -> Candidate:
    info_path, info = load_candidate_info(impl_path)
    task_info = info.get("task_info") or {}

    framework = (task_info.get("framework") or info.get("framework") or "torch").lower()
    dsl = (task_info.get("dsl") or info.get("dsl") or "triton_cuda").lower()
    backend = (task_info.get("backend") or info.get("backend") or "cuda").lower()
    if framework != "torch" or dsl != "triton_cuda" or backend != "cuda":
        raise ValueError(
            f"{impl_path}: only framework=torch, dsl=triton_cuda, backend=cuda are supported; "
            f"got framework={framework}, dsl={dsl}, backend={backend}"
        )

    arch = (arch_override or task_info.get("arch") or info.get("arch") or "h100").lower()
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"{impl_path}: only arch in {sorted(SUPPORTED_ARCHS)} is supported; got {arch}")

    op_name = info.get("op_name") or task_info.get("op_name") or fallback_op_name
    task_id = str(info.get("task_id") or task_info.get("task_id") or impl_path.parent.name)
    unique_dir = str(info.get("unique_dir") or task_info.get("unique_dir") or impl_path.parent.name)
    
    round_value = info.get("round")
    if round_value is not None:
        try:
            round_value = int(round_value)
        except (TypeError, ValueError):
            round_value = None

    fallback_candidate_id = impl_path.parent.name if impl_path.name == "impl_code.py" else impl_path.stem
    candidate_id = str(info.get("id") or fallback_candidate_id)
    old_profile = info.get("profile") or {}
    rel_path = str(impl_path.relative_to(evolve_dir))

    return Candidate(
        impl_path=impl_path,
        rel_path=rel_path,
        candidate_id=candidate_id,
        op_name=op_name,
        task_id=task_id,
        unique_dir=unique_dir,
        round=round_value,
        arch=arch,
        info_path=info_path,
        old_profile=old_profile,
        raw_info=info,
    )


def sanitize_name(name: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return (cleaned or "candidate")[:max_len]


def generate_base_profile_script(
    op_name: str,
    arch: str,
    warmup: int,
    runs: int,
    device_id: int,
    is_dynamic_shape: bool,
    torch_compile: bool,
    torch_compile_warmup: int,
    torch_compile_mode: Optional[str],
) -> str:
    input_import = "get_inputs_dyn_list" if is_dynamic_shape else "get_inputs"
    static_or_dynamic = "dynamic" if is_dynamic_shape else "static"
    dynamic_block = f"""
    inputs_list = get_inputs_dyn_list()
    all_execution_times = []
    for case_idx, inputs in enumerate(inputs_list):
        inputs = [process_input(x) for x in inputs]
        execution_time, method = run_benchmark(inputs)
        all_execution_times.append(execution_time)
        print(f"[{op_name}] Case {{case_idx + 1}} execution time: {{execution_time * 1000:.4f}} us")

    avg_execution_time = sum(all_execution_times) / len(all_execution_times)
    result_data = {{
        "execution_time_ms": avg_execution_time,
        "execution_time_us": avg_execution_time * 1000,
        "case_count": len(inputs_list),
        "case_times": all_execution_times,
        "bench_warmup_ms": {warmup},
        "bench_rep_ms": {runs},
        "bench_time_unit": "ms",
        "method": method,
        "shape_type": "{static_or_dynamic}",
        "baseline_mode": baseline_mode,
        "torch_compile": torch_compile_enabled,
        "torch_compile_warmup_calls": torch_compile_warmup_calls,
        "torch_compile_mode": torch_compile_mode,
    }}
    print(f"[{op_name}] Average execution time: {{avg_execution_time * 1000:.4f}} us")
"""
    static_block = f"""
    inputs = get_inputs()
    inputs = [process_input(x) for x in inputs]
    execution_time, method = run_benchmark(inputs)
    result_data = {{
        "execution_time_ms": execution_time,
        "execution_time_us": execution_time * 1000,
        "bench_warmup_ms": {warmup},
        "bench_rep_ms": {runs},
        "bench_time_unit": "ms",
        "method": method,
        "shape_type": "{static_or_dynamic}",
        "baseline_mode": baseline_mode,
        "torch_compile": torch_compile_enabled,
        "torch_compile_warmup_calls": torch_compile_warmup_calls,
        "torch_compile_mode": torch_compile_mode,
    }}
    print(f"[{op_name}] Base execution time: {{execution_time * 1000:.4f}} us")
"""
    run_block = dynamic_block if is_dynamic_shape else static_block

    return f'''import os
import json
import numpy as np
from typing import Any, List, Literal, Tuple, Union

import torch
from {op_name}_torch import Model as FrameworkModel, get_init_inputs, {input_import}

TensorType = torch.Tensor


def run_base_implementations():
    backend = "cuda"
    arch = "{arch}"
    dsl = "triton_cuda"
    torch_compile_enabled = {torch_compile}
    torch_compile_warmup_calls = {torch_compile_warmup}
    torch_compile_mode = {torch_compile_mode!r}
    baseline_mode = "torch_compile" if torch_compile_enabled else "torch_eager"

    os.environ["CUDA_VISIBLE_DEVICES"] = str({device_id})
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.manual_seed(0)

    init_params = get_init_inputs()
    framework_model = FrameworkModel(*init_params)
    framework_model = framework_model.to(device)
    framework_model.eval()
    if torch_compile_enabled:
        if not hasattr(torch, "compile"):
            raise RuntimeError("Current PyTorch does not provide torch.compile")
        compile_kwargs = {{}}
        if torch_compile_mode is not None:
            compile_kwargs["mode"] = torch_compile_mode
        framework_model = torch.compile(framework_model, **compile_kwargs)

    def process_input(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)
        if isinstance(x, (list, tuple)):
            return type(x)(process_input(item) for item in x)
        if isinstance(x, (int, float, bool, type(None))):
            return x
        try:
            return x.to(device)
        except (AttributeError, TypeError):
            return x

    def run_benchmark(inputs):
        import triton.testing

        def base_benchmark_fn():
            return framework_model(*inputs)

        if torch_compile_enabled:
            for _ in range(torch_compile_warmup_calls):
                base_benchmark_fn()
            torch.cuda.synchronize()

        execution_time_ms = triton.testing.do_bench(
            base_benchmark_fn,
            warmup={warmup},
            rep={runs},
            return_mode="median",
        )
        method = "triton_do_bench_torch_compile" if torch_compile_enabled else "triton_do_bench"
        return execution_time_ms, method
{run_block}
    with open("base_profile_result.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)


if __name__ == "__main__":
    run_base_implementations()
'''


def generate_generation_profile_script(
    op_name: str,
    arch: str,
    warmup: int,
    runs: int,
    device_id: int,
    is_dynamic_shape: bool,
) -> str:
    input_import = "get_inputs_dyn_list" if is_dynamic_shape else "get_inputs"
    static_or_dynamic = "dynamic" if is_dynamic_shape else "static"
    dynamic_block = f"""
    inputs_list = get_inputs_dyn_list()
    all_execution_times = []
    for case_idx, inputs in enumerate(inputs_list):
        inputs = [process_input(x) for x in inputs]
        execution_time, method = run_benchmark(inputs, case_idx=case_idx)
        all_execution_times.append(execution_time)
        print(f"[{op_name}] Case {{case_idx + 1}} execution time: {{execution_time * 1000:.4f}} us")

    avg_execution_time = sum(all_execution_times) / len(all_execution_times)
    result_data = {{
        "execution_time_ms": avg_execution_time,
        "execution_time_us": avg_execution_time * 1000,
        "case_count": len(inputs_list),
        "case_times": all_execution_times,
        "bench_warmup_ms": {warmup},
        "bench_rep_ms": {runs},
        "bench_time_unit": "ms",
        "method": method,
        "shape_type": "{static_or_dynamic}",
    }}
    print(f"[{op_name}] Average execution time: {{avg_execution_time * 1000:.4f}} us")
"""
    static_block = f"""
    inputs = get_inputs()
    inputs = [process_input(x) for x in inputs]
    execution_time, method = run_benchmark(inputs, case_idx=0)
    result_data = {{
        "execution_time_ms": execution_time,
        "execution_time_us": execution_time * 1000,
        "bench_warmup_ms": {warmup},
        "bench_rep_ms": {runs},
        "bench_time_unit": "ms",
        "method": method,
        "shape_type": "{static_or_dynamic}",
    }}
    print(f"[{op_name}] Generation execution time: {{execution_time * 1000:.4f}} us")
"""
    run_block = dynamic_block if is_dynamic_shape else static_block

    return f'''import os
import json
import numpy as np
from typing import Any, List, Literal, Tuple, Union

import torch
import triton
import triton.language as tl
from {op_name}_torch import get_init_inputs, {input_import}
from {op_name}_triton_cuda import ModelNew

try:
    import nvtx
except Exception:
    nvtx = None

TensorType = torch.Tensor


def run_generation_implementations():
    backend = "cuda"
    arch = "{arch}"
    dsl = "triton_cuda"

    os.environ["CUDA_VISIBLE_DEVICES"] = str({device_id})
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.manual_seed(0)

    init_params = get_init_inputs()
    torch.manual_seed(0)
    impl_model = ModelNew(*init_params)
    impl_model = impl_model.to(device)

    def process_input(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)
        if isinstance(x, (list, tuple)):
            return type(x)(process_input(item) for item in x)
        if isinstance(x, (int, float, bool, type(None))):
            return x
        try:
            return x.to(device)
        except (AttributeError, TypeError):
            return x

    def run_benchmark(inputs, case_idx=0):
        import triton.testing

        def triton_benchmark_fn():
            return impl_model(*inputs)

        triton_benchmark_fn()
        nvtx_range = nvtx.start_range("target_kernel", color="red") if nvtx else None
        try:
            execution_time_ms = triton.testing.do_bench(
                triton_benchmark_fn,
                warmup={warmup},
                rep={runs},
                return_mode="median",
            )
        finally:
            if nvtx_range is not None:
                nvtx.end_range(nvtx_range)
        method = "triton_do_bench"
        return execution_time_ms, method
{run_block}
    with open("generation_profile_result.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)


if __name__ == "__main__":
    run_generation_implementations()
'''


def read_profile_time_us(case_dir: Path, filename: str) -> float:
    result_path = case_dir / filename
    if not result_path.is_file():
        raise FileNotFoundError(f"Profile result not found: {result_path}")
    data = load_json(result_path)
    value = data.get("avg_time_us") or data.get("execution_time_us")
    if value is None:
        raise ValueError(f"{result_path} has no avg_time_us or execution_time_us")
    return float(value)


def run_profile_script(
    case_dir: Path,
    script_name: str,
    result_json_name: str,
    timeout: int,
    task_id: str,
    gpu_settings: GPUExclusiveSettings,
) -> None:
    result_json_path = case_dir / result_json_name

    def cleanup_partial_result() -> None:
        with suppress(FileNotFoundError):
            result_json_path.unlink()

    cleanup_partial_result()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    operation_name = f"Profile script {script_name}"
    returncode, stdout, stderr = asyncio.run(
        run_monitored_gpu_subprocess_with_retry(
            [sys.executable, script_name],
            case_dir,
            env,
            gpu_settings.device_id,
            task_id,
            timeout,
            operation_name,
            gpu_settings.wait_timeout_minutes,
            gpu_settings.monitor_interval_seconds,
            gpu_settings.contamination_max_retries,
            cleanup_on_contamination=cleanup_partial_result,
        )
    )
    log_path = case_dir / f"{script_name}.log"
    write_text(
        log_path,
        "STDOUT:\n"
        + stdout.decode(errors="replace")
        + "\nSTDERR:\n"
        + stderr.decode(errors="replace")
        + f"\nRETURN_CODE: {returncode}\n",
    )
    if returncode != 0:
        raise RuntimeError(f"{script_name} failed with return code {returncode}; see {log_path}")


def run_generation_profile_and_collect_result(
    case_dir: Path,
    op_name: str,
    timeout: int,
    task_id: str,
    gpu_settings: GPUExclusiveSettings,
) -> float:
    run_profile_script(
        case_dir,
        f"profile_{op_name}_generation.py",
        "generation_profile_result.json",
        timeout,
        task_id,
        gpu_settings,
    )
    gen_time_us = read_profile_time_us(case_dir, "generation_profile_result.json")
    return gen_time_us


def prepare_baseline_dir(
    op_name: str,
    arch: str,
    baseline_dir: Path,
    baseline_code: str,
    warmup: int,
    runs: int,
    device_id: int,
    torch_compile: bool,
    torch_compile_warmup: int,
    torch_compile_mode: Optional[str],
    baseline_repeats: int,
    gpu_settings: GPUExclusiveSettings,
) -> None:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    write_text(baseline_dir / f"{op_name}_torch.py", baseline_code)

    is_dynamic_shape = "get_inputs_dyn_list" in baseline_code
    write_text(
        baseline_dir / f"profile_{op_name}_base.py",
        generate_base_profile_script(
            op_name,
            arch,
            warmup,
            runs,
            device_id,
            is_dynamic_shape,
            torch_compile,
            torch_compile_warmup,
            torch_compile_mode,
        ),
    )

    manifest = {
        "op_name": op_name,
        "arch": arch,
        "bench_warmup_ms": warmup,
        "bench_rep_ms": runs,
        "bench_time_unit": "ms",
        "device_id": device_id,
        "baseline_repeats": baseline_repeats,
        "baseline_mode": "torch_compile" if torch_compile else "torch_eager",
        "torch_compile": torch_compile,
        "torch_compile_warmup_calls": torch_compile_warmup,
        "torch_compile_mode": torch_compile_mode,
        "gpu_wait_timeout_minutes": gpu_settings.wait_timeout_minutes,
        "gpu_monitor_interval_seconds": gpu_settings.monitor_interval_seconds,
        "gpu_contamination_max_retries": gpu_settings.contamination_max_retries,
    }
    write_text(baseline_dir / "baseline_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))


def run_shared_baseline(
    baseline_dir: Path,
    op_name: str,
    repeats: int,
    timeout: int,
    task_id: str,
    gpu_settings: GPUExclusiveSettings,
) -> Dict[str, Any]:
    times_us: List[float] = []
    script_name = f"profile_{op_name}_base.py"

    for repeat_idx in range(1, repeats + 1):
        run_profile_script(
            baseline_dir,
            script_name,
            "base_profile_result.json",
            timeout,
            f"{task_id}_baseline_{repeat_idx}",
            gpu_settings,
        )
        base_time_us = read_profile_time_us(baseline_dir, "base_profile_result.json")
        times_us.append(base_time_us)

        result_path = baseline_dir / "base_profile_result.json"
        log_path = baseline_dir / f"{script_name}.log"
        if result_path.is_file():
            shutil.copy2(result_path, baseline_dir / f"base_profile_result_run_{repeat_idx}.json")
        if log_path.is_file():
            shutil.copy2(log_path, baseline_dir / f"{script_name}.run_{repeat_idx}.log")

        print(f"[baseline {repeat_idx:02d}/{repeats:02d}] {base_time_us:.3f}us")

    mean_us = sum(times_us) / len(times_us)
    sorted_times = sorted(times_us)
    mid = len(sorted_times) // 2
    if len(sorted_times) % 2:
        median_us = sorted_times[mid]
    else:
        median_us = (sorted_times[mid - 1] + sorted_times[mid]) / 2

    variance = sum((value - mean_us) ** 2 for value in times_us) / len(times_us)
    summary = {
        "base_time_us": mean_us,
        "mean_us": mean_us,
        "median_us": median_us,
        "min_us": min(times_us),
        "max_us": max(times_us),
        "std_us": math.sqrt(variance),
        "repeats": repeats,
        "times_us": times_us,
    }
    write_text(baseline_dir / "baseline_summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def prepare_case_dir(
    candidate: Candidate,
    case_dir: Path,
    baseline_code: str,
    warmup: int,
    runs: int,
    device_id: int,
    baseline_mode: str,
    baseline_dir: Path,
    baseline_repeats: int,
    gpu_settings: GPUExclusiveSettings,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)

    framework_file = case_dir / f"{candidate.op_name}_torch.py"
    impl_file = case_dir / f"{candidate.op_name}_triton_cuda.py"
    write_text(framework_file, baseline_code)

    impl_code = read_text(candidate.impl_path)
    write_text(impl_file, "import torch\nimport triton\nimport triton.language as tl\n\n" + impl_code)

    is_dynamic_shape = "get_inputs_dyn_list" in baseline_code
    write_text(
        case_dir / f"profile_{candidate.op_name}_generation.py",
        generate_generation_profile_script(
            candidate.op_name,
            candidate.arch,
            warmup,
            runs,
            device_id,
            is_dynamic_shape,
        ),
    )

    manifest = {
        "source_impl": str(candidate.impl_path),
        "source_info": str(candidate.info_path) if candidate.info_path else None,
        "candidate_id": candidate.candidate_id,
        "op_name": candidate.op_name,
        "task_id": candidate.task_id,
        "unique_dir": candidate.unique_dir,
        "round": candidate.round,
        "arch": candidate.arch,
        "warmup": warmup,
        "runs": runs,
        "bench_warmup_ms": warmup,
        "bench_rep_ms": runs,
        "bench_time_unit": "ms",
        "device_id": device_id,
        "baseline_mode": baseline_mode,
        "shared_baseline_dir": str(baseline_dir),
        "baseline_repeats": baseline_repeats,
        "gpu_wait_timeout_minutes": gpu_settings.wait_timeout_minutes,
        "gpu_monitor_interval_seconds": gpu_settings.monitor_interval_seconds,
        "gpu_contamination_max_retries": gpu_settings.contamination_max_retries,
    }
    write_text(case_dir / "candidate_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))


def run_candidate(
    index: int,
    candidate: Candidate,
    case_dir: Path,
    dry_run: bool,
    timeout: int,
    baseline_mode: str,
    base_time_us: Optional[float],
    gpu_settings: GPUExclusiveSettings,
) -> EvalResult:
    old_profile = candidate.old_profile
    old_base = to_float(old_profile.get("base_time"))
    old_gen = to_float(old_profile.get("gen_time"))
    old_speedup = to_float(old_profile.get("speedup"))

    if dry_run:
        return EvalResult(
            index=index,
            candidate_id=candidate.candidate_id,
            rel_path=candidate.rel_path,
            task_id=candidate.task_id,
            unique_dir=candidate.unique_dir,
            round=candidate.round,
            arch=candidate.arch,
            baseline_mode=baseline_mode,
            status="prepared",
            base_time_us=base_time_us,
            gen_time_us=None,
            speedup=None,
            old_base_time_us=old_base,
            old_gen_time_us=old_gen,
            old_speedup=old_speedup,
            workdir=str(case_dir),
        )

    try:
        if base_time_us is None or not math.isfinite(base_time_us) or base_time_us <= 0:
            raise RuntimeError("shared baseline time is unavailable")
        gen_time = run_generation_profile_and_collect_result(
            case_dir,
            candidate.op_name,
            timeout,
            candidate.task_id,
            gpu_settings,
        )
        speedup = base_time_us / gen_time if gen_time and gen_time > 0 else 0.0
        status = "ok" if math.isfinite(gen_time) and gen_time > 0 else "failed"
        error = "" if status == "ok" else "profile script returned non-finite timing"
    except Exception as exc:  # Keep batch runs moving and record the failure.
        gen_time = None
        speedup = None
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        write_text(case_dir / "error.log", traceback.format_exc())

    return EvalResult(
        index=index,
        candidate_id=candidate.candidate_id,
        rel_path=candidate.rel_path,
        task_id=candidate.task_id,
        unique_dir=candidate.unique_dir,
        round=candidate.round,
        arch=candidate.arch,
        baseline_mode=baseline_mode,
        status=status,
        base_time_us=base_time_us,
        gen_time_us=gen_time,
        speedup=speedup,
        old_base_time_us=old_base,
        old_gen_time_us=old_gen,
        old_speedup=old_speedup,
        workdir=str(case_dir),
        error=error,
    )


def result_to_dict(result: EvalResult) -> Dict[str, Any]:
    return {
        "index": result.index,
        "candidate_id": result.candidate_id,
        "rel_path": result.rel_path,
        "task_id": result.task_id,
        "unique_dir": result.unique_dir,
        "round": result.round,
        "arch": result.arch,
        "baseline_mode": result.baseline_mode,
        "status": result.status,
        "base_time_us": result.base_time_us,
        "gen_time_us": result.gen_time_us,
        "speedup": result.speedup,
        "old_base_time_us": result.old_base_time_us,
        "old_gen_time_us": result.old_gen_time_us,
        "old_speedup": result.old_speedup,
        "workdir": result.workdir,
        "error": result.error,
    }


def write_csv(path: Path, results: Sequence[EvalResult]) -> None:
    rows = [result_to_dict(result) for result in results]
    fieldnames = [
        "index",
        "candidate_id",
        "rel_path",
        "task_id",
        "unique_dir",
        "round",
        "arch",
        "baseline_mode",
        "status",
        "base_time_us",
        "gen_time_us",
        "speedup",
        "old_base_time_us",
        "old_gen_time_us",
        "old_speedup",
        "workdir",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_successes(results: Iterable[EvalResult]) -> List[EvalResult]:
    return [
        result
        for result in results
        if result.status == "ok"
        and result.speedup is not None
        and result.gen_time_us is not None
        and math.isfinite(result.speedup)
        and math.isfinite(result.gen_time_us)
    ]


def build_summary(results: Sequence[EvalResult]) -> Dict[str, Any]:
    successes = finite_successes(results)
    failures = [result for result in results if result.status == "failed"]
    summary: Dict[str, Any] = {
        "total": len(results),
        "ok": len(successes),
        "failed": len(failures),
    }
    if successes:
        best_speedup = max(successes, key=lambda item: item.speedup or 0.0)
        best_time = min(successes, key=lambda item: item.gen_time_us or float("inf"))
        avg_speedup = sum(item.speedup or 0.0 for item in successes) / len(successes)
        summary.update(
            {
                "avg_speedup": avg_speedup,
                "best_speedup": result_to_dict(best_speedup),
                "best_gen_time": result_to_dict(best_time),
            }
        )
    return summary


def write_speed_record(path: Path, op_name: str, results: Sequence[EvalResult]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            if result.status != "ok" or result.base_time_us is None or result.gen_time_us is None:
                continue
            f.write(f"op_name: {op_name}, task_id: {result.task_id}, unique_dir: {result.unique_dir}, ")
            f.write(f"base_time: {result.base_time_us:.6f} us, generation_time: {result.gen_time_us:.6f} us, ")
            f.write(f"speedup: {(result.speedup or 0.0):.6f}x\n")


def print_progress(result: EvalResult) -> None:
    prefix = f"[{result.index:04d}] {result.candidate_id} [{result.baseline_mode}]"
    if result.status == "ok":
        print(
            f"{prefix} speedup={result.speedup:.4f}x "
            f"base={result.base_time_us:.3f}us gen={result.gen_time_us:.3f}us"
        )
    elif result.status == "prepared":
        print(f"{prefix} prepared -> {result.workdir}")
    else:
        print(f"{prefix} failed: {result.error}")


def env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run speedup evaluation for saved evolved Triton CUDA kernels "
            "against a PyTorch KernelBench baseline."
        )
    )
    parser.add_argument(
        "evolve_dir",
        type=Path,
        help="Operator storage directory, e.g. .../evolve_database/level1/2_Standard_matrix_multiplication_",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="PyTorch baseline file. If omitted, inferred from evolve_database level and operator folder name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated profile scripts and result files. Defaults under examples/tools/verify.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id to expose to profile scripts.")
    parser.add_argument(
        "--arch",
        choices=sorted(SUPPORTED_ARCHS),
        default=None,
        help="Override candidate metadata architecture. If omitted, use impl_info.json arch or h100.",
    )
    parser.add_argument(
        "--warmup",
        "--warmup-ms",
        dest="warmup",
        type=int,
        default=5,
        help="Warmup time budget in milliseconds passed to triton.testing.do_bench.",
    )
    parser.add_argument(
        "--runs",
        "--rep-ms",
        dest="runs",
        type=int,
        default=50,
        help="Benchmark time budget in milliseconds passed to triton.testing.do_bench.",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for each profile script.")
    parser.add_argument(
        "--torch_compile",
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for the PyTorch baseline before benchmarking.",
    )
    parser.add_argument(
        "--torch_compile_warmup",
        type=int,
        default=3,
        help="Extra unmeasured baseline calls before do_bench when --torch_compile is enabled.",
    )
    parser.add_argument(
        "--torch_compile_mode",
        default=None,
        help="Optional mode passed to torch.compile, e.g. reduce-overhead or max-autotune.",
    )
    parser.add_argument(
        "--baseline_repeats",
        type=int,
        default=5,
        help="How many times to run the shared baseline before evaluating any Triton candidate.",
    )
    parser.add_argument(
        "--gpu_wait_timeout_minutes",
        type=int,
        default=120,
        help="Maximum minutes to wait for the target GPU to become idle before each profile run.",
    )
    parser.add_argument(
        "--gpu_monitor_interval_seconds",
        type=float,
        default=env_float("AIKG_GPU_MONITOR_INTERVAL_SECONDS", 1.0),
        help="Seconds between nvidia-smi checks while monitoring GPU exclusivity.",
    )
    parser.add_argument(
        "--gpu_contamination_max_retries",
        type=int,
        default=env_int("AIKG_GPU_CONTAMINATION_MAX_RETRIES", 10),
        help="Maximum retries after detecting another process using the target GPU.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help=(
            "Evaluate only the top N candidates ranked by impl_info.json profile.speedup. "
            "Use 0 to evaluate all candidates."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate profile workdirs and manifests, but do not run benchmarks.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch when a candidate fails during preparation or profiling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output-dir first when it already exists.",
    )
    return parser.parse_args(argv)


def create_output_dir(args: argparse.Namespace, evolve_dir: Path) -> Path:
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        baseline_mode = "torch_compile" if args.torch_compile else "torch_eager"
        output_dir = DEFAULT_RUN_ROOT / f"{stamp}_{sanitize_name(evolve_dir.name)}_{baseline_mode}"

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir already exists: {output_dir} (use --overwrite)")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if (
        args.warmup < 0
        or args.runs <= 0
        or args.timeout <= 0
        or args.torch_compile_warmup < 0
        or args.baseline_repeats <= 0
        or args.gpu_wait_timeout_minutes <= 0
        or args.gpu_monitor_interval_seconds <= 0
        or args.gpu_contamination_max_retries < 0
        or (args.limit is not None and args.limit < 0)
    ):
        raise ValueError(
            "--warmup must be >= 0, --runs must be > 0, --timeout must be > 0, "
            "--torch_compile_warmup must be >= 0, --baseline_repeats must be > 0, "
            "--gpu_wait_timeout_minutes must be > 0, --gpu_monitor_interval_seconds must be > 0, "
            "--gpu_contamination_max_retries must be >= 0, and --limit must be >= 0"
        )

    evolve_dir = resolve_evolve_dir(args.evolve_dir)
    baseline_path = args.baseline.expanduser().resolve() if args.baseline else infer_baseline_path(evolve_dir)
    if not baseline_path.is_file():
        raise FileNotFoundError(f"baseline not found: {baseline_path}")

    fallback_op_name = infer_op_name_from_path(evolve_dir)
    baseline_code = read_text(baseline_path)
    impl_files = discover_impl_files(evolve_dir)
    if not impl_files:
        raise FileNotFoundError(f"No generated .py implementation files found under {evolve_dir}")

    candidates: List[Candidate] = []
    for impl_file in impl_files:
        candidates.append(make_candidate(impl_file, evolve_dir, fallback_op_name, args.arch))
    total_candidates = len(candidates)
    candidates.sort(key=candidate_rank_key)
    if args.limit is not None and args.limit > 0:
        candidates = candidates[: args.limit]

    output_dir = create_output_dir(args, evolve_dir)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    baseline_mode = "torch_compile" if args.torch_compile else "torch_eager"
    gpu_settings = GPUExclusiveSettings(
        device_id=args.device,
        wait_timeout_minutes=args.gpu_wait_timeout_minutes,
        monitor_interval_seconds=max(0.2, args.gpu_monitor_interval_seconds),
        contamination_max_retries=args.gpu_contamination_max_retries,
    )

    run_manifest = {
        "evolve_dir": str(evolve_dir),
        "baseline": str(baseline_path),
        "output_dir": str(output_dir),
        "device": args.device,
        "arch_override": args.arch,
        "warmup": args.warmup,
        "runs": args.runs,
        "bench_warmup_ms": args.warmup,
        "bench_rep_ms": args.runs,
        "bench_time_unit": "ms",
        "timeout": args.timeout,
        "baseline_mode": baseline_mode,
        "torch_compile": args.torch_compile,
        "torch_compile_warmup_calls": args.torch_compile_warmup,
        "torch_compile_mode": args.torch_compile_mode,
        "baseline_repeats": args.baseline_repeats,
        "gpu_wait_timeout_minutes": gpu_settings.wait_timeout_minutes,
        "gpu_monitor_interval_seconds": gpu_settings.monitor_interval_seconds,
        "gpu_contamination_max_retries": gpu_settings.contamination_max_retries,
        "dry_run": args.dry_run,
        "candidate_total_before_limit": total_candidates,
        "candidate_count": len(candidates),
        "candidate_limit": args.limit,
        "candidate_selection": "top_impl_info_profile_speedup",
        "selected_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "rel_path": candidate.rel_path,
                "task_id": candidate.task_id,
                "unique_dir": candidate.unique_dir,
                "old_speedup": candidate_old_speedup(candidate),
            }
            for candidate in candidates
        ],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_text(output_dir / "run_manifest.json", json.dumps(run_manifest, indent=2, ensure_ascii=False))

    print(f"Evolve dir: {evolve_dir}")
    print(f"Baseline:   {baseline_path}")
    print(f"Mode:       {baseline_mode}")
    print(f"Output dir: {output_dir}")
    print(f"Candidates: {len(candidates)} selected from {total_candidates}")
    print("")

    results: List[EvalResult] = []
    op_name_for_record = candidates[0].op_name if candidates else fallback_op_name
    baseline_dir = output_dir / "baseline"
    prepare_baseline_dir(
        op_name=op_name_for_record,
        arch=candidates[0].arch if candidates else (args.arch or "h100"),
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

    if args.dry_run:
        baseline_summary = {
            "status": "prepared",
            "baseline_repeats": args.baseline_repeats,
            "baseline_mode": baseline_mode,
            "baseline_dir": str(baseline_dir),
        }
    else:
        baseline_summary = run_shared_baseline(
            baseline_dir=baseline_dir,
            op_name=op_name_for_record,
            repeats=args.baseline_repeats,
            timeout=args.timeout,
            task_id=op_name_for_record,
            gpu_settings=gpu_settings,
        )

    for index, candidate in enumerate(candidates, start=1):
        case_name = f"{index:04d}_{sanitize_name(candidate.unique_dir or candidate.candidate_id)}"
        case_dir = cases_dir / case_name
        try:
            prepare_case_dir(
                candidate=candidate,
                case_dir=case_dir,
                baseline_code=baseline_code,
                warmup=args.warmup,
                runs=args.runs,
                device_id=args.device,
                baseline_mode=baseline_mode,
                baseline_dir=baseline_dir,
                baseline_repeats=args.baseline_repeats,
                gpu_settings=gpu_settings,
            )
            result = run_candidate(
                index,
                candidate,
                case_dir,
                dry_run=args.dry_run,
                timeout=args.timeout,
                baseline_mode=baseline_mode,
                base_time_us=baseline_summary.get("base_time_us") if isinstance(baseline_summary, dict) else None,
                gpu_settings=gpu_settings,
            )
        except Exception as exc:
            if case_dir.exists():
                write_text(case_dir / "error.log", traceback.format_exc())
            result = EvalResult(
                index=index,
                candidate_id=candidate.candidate_id,
                rel_path=candidate.rel_path,
                task_id=candidate.task_id,
                unique_dir=candidate.unique_dir,
                round=candidate.round,
                arch=candidate.arch,
                baseline_mode=baseline_mode,
                status="failed",
                base_time_us=baseline_summary.get("base_time_us") if isinstance(baseline_summary, dict) else None,
                gen_time_us=None,
                speedup=None,
                old_base_time_us=to_float(candidate.old_profile.get("base_time")),
                old_gen_time_us=to_float(candidate.old_profile.get("gen_time")),
                old_speedup=to_float(candidate.old_profile.get("speedup")),
                workdir=str(case_dir),
                error=f"{type(exc).__name__}: {exc}",
            )

        results.append(result)
        print_progress(result)

        write_csv(output_dir / "results.csv", results)
        write_text(
            output_dir / "results.json",
            json.dumps([result_to_dict(item) for item in results], indent=2, ensure_ascii=False),
        )
        write_speed_record(output_dir / "speed_up_record.txt", op_name_for_record, results)
        summary = build_summary(results)
        summary["shared_baseline"] = baseline_summary
        write_text(output_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))

        if args.stop_on_error and result.status == "failed":
            print("Stopped after failure because --stop-on-error was set.")
            break

    summary = build_summary(results)
    summary["shared_baseline"] = baseline_summary
    print("")
    print(f"Finished: ok={summary['ok']}/{summary['total']}, failed={summary['failed']}")
    if "best_speedup" in summary:
        best = summary["best_speedup"]
        print(
            "Best speedup: "
            f"{best['speedup']:.4f}x ({best['candidate_id']}, gen={best['gen_time_us']:.3f}us)"
        )
    print(f"Results: {output_dir / 'results.csv'}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
