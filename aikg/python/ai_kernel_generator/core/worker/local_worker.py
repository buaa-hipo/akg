import asyncio
import fcntl
import os
import signal
import shutil
import subprocess
import tempfile
import tarfile
import logging
import sys
import io
import json
from typing import Callable, List, Tuple, Dict, Any, Union, Optional, Sequence, Set
from contextlib import ExitStack, asynccontextmanager, suppress

from pathlib import Path

import pandas as pd
import numpy as np
import math

from .interface import WorkerInterface
from ..async_pool.device_pool import DevicePool
from ..verifier.profiler_utils import (
    run_profile_scripts_and_collect_results,
    read_profile_result_from_json,
    run_msprof,
    analyze_prof_data,
    run_nsys,
    analyze_nsys_data
)

logger = logging.getLogger(__name__)


class GPUContaminationError(RuntimeError):
    """Raised when an external process starts using the target GPU during a run."""

    def __init__(self, device_id: int, external_pids: Set[int]):
        self.device_id = device_id
        self.external_pids = external_pids
        self.output_log = ""
        super().__init__(
            f"GPU_{device_id} was used by external pids during execution: "
            f"{sorted(external_pids)}"
        )


def _detect_gpu_lock_dir() -> str:
    configured_dir = os.environ.get("AIKG_GPU_LOCK_DIR")
    if configured_dir:
        return configured_dir

    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("Failed to detect GPU model for lock dir selection: %s", exc)
        return "/mnt/lustre-client/zhangzizheng/aikg_gpu_locks"

    if result.returncode != 0:
        logger.warning(
            "nvidia-smi --list-gpus failed when selecting lock dir: %s",
            result.stderr.strip(),
        )
        return "/mnt/lustre-client/zhangzizheng/aikg_gpu_locks"

    if "A100" in result.stdout:
        return "/ssd/zhangzizheng/aikg_gpu_locks"
    return "/mnt/lustre-client/zhangzizheng/aikg_gpu_locks"


GPU_LOCK_DIR = _detect_gpu_lock_dir()


NCU_METRICS = ",".join([
    "sm__cycles_active.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__registers_per_thread",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
])


# List version for convenient header selection
NCU_METRIC_COLUMNS: List[str] = [s.strip() for s in NCU_METRICS.split(",")]


def load_ncu_metrics(
    csv_path: Union[str, Path] = "ncu_temp.csv",
    columns: Optional[Sequence[str]] = None,
    extra_keep: Optional[Sequence[str]] = ("Kernel Name",),
    coerce_numeric: bool = True,
    name_list: Optional[Sequence[str]] = None,  # New: multiple kernel names
    select: str = "last",                       # Selection policy when multiple rows per name
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, comment="=", low_memory=False)

    metric_cols = list(columns) if columns is not None else NCU_METRIC_COLUMNS
    keep_cols: List[str] = []
    if extra_keep:
        keep_cols.extend([c for c in extra_keep if c in df.columns])
    keep_cols.extend([c for c in metric_cols if c in df.columns])
    if not keep_cols:
        raise ValueError("No requested columns found in the CSV header.")

    sub = df[keep_cols].copy()

    # Drop the units row
    if len(sub) > 0:
        first_row_str = sub.iloc[0].map(lambda v: "" if pd.isna(v) else str(v).lower())
        unit_tokens = ("%", "inst", "cycle", "block", "register", "register/thread")
        if first_row_str.apply(lambda x: any(tok in x for tok in unit_tokens)).any():
            sub = sub.iloc[1:].reset_index(drop=True)

    # Coerce metrics to numeric
    if coerce_numeric:
        metric_in_sub = [c for c in metric_cols if c in sub.columns]
        sub[metric_in_sub] = (
            sub[metric_in_sub]
            .replace({",": "", "%": ""}, regex=True)
            .apply(pd.to_numeric, errors="coerce")
        )

    # ========== Extract by kernel name list ==========
    if name_list:
        results = []
        for name in name_list:
            # Use contains match instead of exact equality
            matched = sub[sub["Kernel Name"].astype(str).str.contains(name, regex=False, na=False)]
            if matched.empty:
                continue
            if len(matched) > 1:
                if select == "first":
                    row = matched.iloc[[0]]
                elif select == "last":
                    row = matched.iloc[[-1]]
                elif select == "max_cycles" and "sm__cycles_active.avg" in matched.columns:
                    row = matched.sort_values("sm__cycles_active.avg", ascending=False).head(1)
                else:
                    row = matched.iloc[[-1]]  # fallback
            else:
                row = matched
            results.append(row)

        if results:
            sub = pd.concat(results, ignore_index=True)
        else:
            sub = pd.DataFrame(columns=keep_cols)

    return sub


def metrics_to_prompt(
    df: pd.DataFrame,
    title: str = "Here are the GPU NCU profiling metrics:",  # Placeholder, not emitted
    key_by: str = "Kernel Name",
    round_digits: Optional[int] = 3,
    compact: bool = False,
    keep_cols: Optional[List[str]] = None,
) -> str:
    """
    Return **only** the data section as a JSON string:
    {
      "<key>": { "<metric>": <value>, ... }  OR
      "<key>": [{...}, {...}]  # list if there are multiple rows for the same key
    }
    If the key column doesn't exist, return a list of rows: [ {col: val, ...}, ... ]
    """

    def _safe(v: Any) -> Any:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, (pd.Timestamp, pd.Timedelta, pd.Interval)):
            return str(v)
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if isinstance(v, float) and round_digits is not None:
            return round(v, round_digits)
        return v

    # Empty table
    if df is None or df.empty:
        return "{}"

    cols = list(df.columns)

    # Round numeric columns
    if round_digits is not None:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            df = df.copy()
            df[num_cols] = df[num_cols].round(round_digits)

    # If key column is missing, return a list of rows
    if key_by not in cols:
        rows = [{k: _safe(v) for k, v in rec.items()} for rec in df.to_dict(orient="records")]
        return json.dumps(rows, ensure_ascii=False, indent=None if compact else 2)

    # Determine value columns
    value_cols = [c for c in cols if c != key_by]
    if keep_cols is not None:
        value_cols = [c for c in value_cols if c in keep_cols]

    data: Dict[str, Any] = {}
    for rec in df[[key_by] + value_cols].to_dict(orient="records"):
        k = str(rec.pop(key_by))
        val_obj = {ck: _safe(cv) for ck, cv in rec.items()}
        ## convert json - value to json - list
        for val_obj_k in val_obj.keys():
            val_obj[val_obj_k] = [val_obj[val_obj_k]]
        if k in data:
            # if isinstance(data[k], list):
            #     data[k].append(val_obj)
            # else:
            #     data[k] = [data[k], val_obj]
            for val_obj_k in val_obj.keys():
                data[k][val_obj_k].extend(val_obj[val_obj_k])
        else:
            data[k] = val_obj
    return json.dumps(data, ensure_ascii=False, indent=None if compact else 2)

def ncu_json_get_mean(ncu_json: dict) -> Dict[str, float]:
    """
    从 NCU JSON 结果中提取数值型指标的平均值，返回一个字典 {metric: mean_value}。
    ncu_json 的结构为 { kernel_name: { metric_name: [ values ... ] ... } ... }，这里我们对所有 metric 的数值列表取平均。
    """
    metric_value_list = {}
    metric_value_mean = {}
    for kernel, metric_dict in ncu_json.items():
        for metric, values in metric_dict.items():
            if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
                if metric not in metric_value_list:
                    metric_value_list[metric] = []
                metric_value_list[metric].extend(values)
    for metric, values in metric_value_list.items():
        metric_value_mean[metric] = sum(values) / len(values)
    return metric_value_mean


def collect_json_artifacts(directory: str) -> Dict[str, str]:
    """
    收集目录中所有 JSON/JSONL 文件的原始内容。

    Args:
        directory: 要扫描的目录路径

    Returns:
        Dict[str, str]: 文件相对路径 -> 文件原始内容（字符串）
        例如: {"autotune_info_case_0.json": "{...}", "subdir/result.jsonl": "..."}
    """
    artifacts = {}
    if not os.path.exists(directory):
        return artifacts

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.json') or filename.endswith('.jsonl'):
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, directory)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        artifacts[rel_path] = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read file {rel_path}: {e}")
    return artifacts

class LocalWorker(WorkerInterface):
    """
    Local implementation of WorkerInterface.
    Executes verification tasks in a local subprocess, managing devices via DevicePool.
    """
    def __init__(self, device_pool: DevicePool, backend: str = "cuda"):
        self.device_pool = device_pool
        self.backend = backend

    def _resolve_device_id(self, device_id: Optional[int] = None) -> int:
        if device_id is not None:
            return int(device_id)
        return int(os.environ.get('CUDA_VISIBLE_DEVICES', 0))

    def waiting_for_resources(self, device_id: Optional[int] = None, timeout_minutes: int = 120):
        import time
        import subprocess
        from typing import Optional

        actual_device_id = self._resolve_device_id(device_id)
        cnt = 0
        max_count = int(timeout_minutes * 60 / 3)  # 3秒一次

        logger.info(f"开始等待 GPU_{actual_device_id} 资源，超时时间 {timeout_minutes} 分钟")

        while cnt < max_count:
            if cnt % 100 == 0:
                elapsed_min = cnt * 3 / 60
                logger.info(f"已等待 GPU_{actual_device_id} 资源 {elapsed_min:.2f} 分钟 ...")

            try:
                # 运行 nvidia-smi，捕获所有异常
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={actual_device_id}",
                        "--query-compute-apps=pid",
                        "--format=csv,noheader"
                    ],
                    encoding="utf-8",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10  # 防止命令卡死
                )

                # 命令执行失败
                if result.returncode != 0:
                    err = result.stderr.strip()
                    logger.warning(f"nvidia-smi 执行失败 (code={result.returncode}): {err}")
                    cnt += 1
                    time.sleep(3)
                    continue

                output = result.stdout.strip()

                # 没有进程 → 退出等待
                if not output:
                    logger.info(f"GPU_{actual_device_id} 资源已释放，开始执行任务")
                    return

            except subprocess.TimeoutExpired:
                logger.error("nvidia-smi 命令执行超时，重试...")
            except Exception as e:
                logger.error(f"查询 GPU 资源异常: {str(e)}", exc_info=True)

            cnt += 1
            time.sleep(3)

        # 超时退出
        logger.error(f"等待 GPU_{actual_device_id} 资源超时 {timeout_minutes} 分钟，退出等待")
        raise TimeoutError(f"GPU_{actual_device_id} 资源等待超时")

    def _gpu_contamination_max_retries(self) -> int:
        raw_value = os.environ.get("AIKG_GPU_CONTAMINATION_MAX_RETRIES", "10")
        try:
            return max(0, int(raw_value))
        except ValueError:
            logger.warning(
                "Invalid AIKG_GPU_CONTAMINATION_MAX_RETRIES=%r, using default 10",
                raw_value,
            )
            return 3

    def _gpu_monitor_interval_seconds(self) -> float:
        raw_value = os.environ.get("AIKG_GPU_MONITOR_INTERVAL_SECONDS", "1.0")
        try:
            return max(0.2, float(raw_value))
        except ValueError:
            logger.warning(
                "Invalid AIKG_GPU_MONITOR_INTERVAL_SECONDS=%r, using default 1.0",
                raw_value,
            )
            return 1.0

    @staticmethod
    def _parse_gpu_pid_output(output: str) -> Set[int]:
        pids = set()
        for line in output.splitlines():
            value = line.strip()
            if value.isdigit():
                pids.add(int(value))
        return pids

    async def _query_gpu_compute_pids(self, device_id: int, task_id: str) -> Set[int]:
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
                logger.warning(
                    "[%s] nvidia-smi pid query failed for GPU_%s: %s",
                    task_id,
                    device_id,
                    stderr.decode(errors='replace').strip(),
                )
                return set()
            return self._parse_gpu_pid_output(stdout.decode(errors='replace'))
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
            logger.warning("[%s] nvidia-smi pid query timed out for GPU_%s", task_id, device_id)
            return set()
        except FileNotFoundError:
            logger.warning("[%s] nvidia-smi not found, GPU exclusivity monitor disabled", task_id)
            return set()
        except Exception as exc:
            logger.warning(
                "[%s] Failed to query GPU_%s compute pids: %s",
                task_id,
                device_id,
                exc,
                exc_info=True,
            )
            return set()

    @staticmethod
    def _read_ppid_from_proc(pid: int) -> Optional[int]:
        try:
            with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
                stat = f.read()
            fields = stat.rsplit(") ", 1)[1].split()
            if len(fields) > 1:
                return int(fields[1])
        except Exception:
            return None
        return None

    def _get_descendant_pids(self, root_pid: int) -> Set[int]:
        children_by_ppid: Dict[int, List[int]] = {}
        try:
            proc_entries = os.listdir("/proc")
        except Exception:
            return {root_pid}

        for entry in proc_entries:
            if not entry.isdigit():
                continue
            pid = int(entry)
            ppid = self._read_ppid_from_proc(pid)
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

    def _get_related_process_pids(self, root_pid: int) -> Set[int]:
        related_pids = self._get_descendant_pids(root_pid)
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

    async def _monitor_gpu_exclusivity(
        self,
        device_id: int,
        root_pid: int,
        task_id: str,
        operation_name: str,
    ):
        interval_seconds = self._gpu_monitor_interval_seconds()
        while True:
            compute_pids = await self._query_gpu_compute_pids(device_id, task_id)
            if compute_pids:
                related_pids = self._get_related_process_pids(root_pid)
                external_pids = compute_pids - related_pids
                if external_pids:
                    logger.warning(
                        "[%s] %s detected external GPU_%s pids: %s "
                        "(own process group/tree: %s)",
                        task_id,
                        operation_name,
                        device_id,
                        sorted(external_pids),
                        sorted(related_pids),
                    )
                    raise GPUContaminationError(device_id, external_pids)
            await asyncio.sleep(interval_seconds)

    def _signal_process_group(
        self,
        process: asyncio.subprocess.Process,
        sig: signal.Signals,
        task_id: str,
        reason: str,
    ):
        if process.returncode is not None:
            return

        try:
            process_pgid = os.getpgid(process.pid)
        except OSError:
            process_pgid = None

        logger.warning("[%s] Stopping process pid=%s due to %s", task_id, process.pid, reason)

        if process_pgid is not None and process_pgid != os.getpgrp():
            try:
                os.killpg(process_pgid, sig)
                return
            except ProcessLookupError:
                return
            except OSError as exc:
                logger.warning(
                    "[%s] Failed to signal process group %s: %s",
                    task_id,
                    process_pgid,
                    exc,
                )

        try:
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except ProcessLookupError:
            pass

    async def _stop_process_and_collect_output(
        self,
        process: asyncio.subprocess.Process,
        communicate_task: asyncio.Task,
        task_id: str,
        reason: str,
    ) -> Tuple[bytes, bytes]:
        self._signal_process_group(process, signal.SIGTERM, task_id, reason)
        try:
            return await asyncio.wait_for(asyncio.shield(communicate_task), timeout=5)
        except asyncio.TimeoutError:
            self._signal_process_group(process, signal.SIGKILL, task_id, reason)
            try:
                return await asyncio.wait_for(asyncio.shield(communicate_task), timeout=5)
            except Exception:
                return b"", b""
        except Exception:
            return b"", b""

    async def _run_monitored_gpu_subprocess_once(
        self,
        cmd: List[str],
        cwd: str,
        env: Dict[str, str],
        actual_device_id: int,
        task_id: str,
        timeout: int,
        operation_name: str,
    ) -> Tuple[int, bytes, bytes]:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        communicate_task = asyncio.create_task(process.communicate())
        monitor_task = asyncio.create_task(
            self._monitor_gpu_exclusivity(
                actual_device_id,
                process.pid,
                task_id,
                operation_name,
            )
        )

        try:
            done, _ = await asyncio.wait(
                {communicate_task, monitor_task},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                await self._stop_process_and_collect_output(
                    process,
                    communicate_task,
                    task_id,
                    f"{operation_name} timeout",
                )
                raise asyncio.TimeoutError()

            if monitor_task in done:
                try:
                    monitor_task.result()
                except GPUContaminationError as exc:
                    stdout, stderr = await self._stop_process_and_collect_output(
                        process,
                        communicate_task,
                        task_id,
                        f"{operation_name} GPU contamination",
                    )
                    exc.output_log = (
                        stdout.decode(errors='replace')
                        + "\n"
                        + stderr.decode(errors='replace')
                    )
                    raise

            stdout, stderr = await communicate_task
            return process.returncode, stdout, stderr
        finally:
            if not monitor_task.done():
                monitor_task.cancel()
                with suppress(asyncio.CancelledError):
                    await monitor_task
            if not communicate_task.done():
                await self._stop_process_and_collect_output(
                    process,
                    communicate_task,
                    task_id,
                    f"{operation_name} cleanup",
                )

    async def _run_gpu_exclusive_subprocess(
        self,
        cmd: List[str],
        cwd: str,
        env: Dict[str, str],
        device_id: Optional[int],
        task_id: str,
        timeout: int,
        operation_name: str,
        cleanup_on_contamination: Optional[Callable[[], None]] = None,
    ) -> Tuple[int, bytes, bytes]:
        max_retries = self._gpu_contamination_max_retries()
        last_error: Optional[GPUContaminationError] = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.info(
                    "[%s] Retrying %s after GPU contamination (%s/%s)",
                    task_id,
                    operation_name,
                    attempt,
                    max_retries,
                )

            async with self.gpu_execution_lock(device_id, task_id) as actual_device_id:
                try:
                    return await self._run_monitored_gpu_subprocess_once(
                        cmd,
                        cwd,
                        env,
                        actual_device_id,
                        task_id,
                        timeout,
                        operation_name,
                    )
                except GPUContaminationError as exc:
                    last_error = exc
                    if cleanup_on_contamination is not None:
                        try:
                            cleanup_on_contamination()
                        except Exception as cleanup_exc:
                            logger.warning(
                                "[%s] Failed to clean partial %s artifacts: %s",
                                task_id,
                                operation_name,
                                cleanup_exc,
                            )
                    if attempt >= max_retries:
                        logger.error(
                            "[%s] %s aborted after %s GPU contamination retries: %s",
                            task_id,
                            operation_name,
                            max_retries,
                            exc,
                        )
                        raise
                    logger.warning(
                        "[%s] %s will restart after GPU contamination: %s",
                        task_id,
                        operation_name,
                        exc,
                    )

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"{operation_name} did not start")

    async def _run_monitored_gpu_subprocess_with_retry_on_device(
        self,
        cmd: List[str],
        cwd: str,
        env: Dict[str, str],
        actual_device_id: int,
        task_id: str,
        timeout: int,
        operation_name: str,
        cleanup_on_contamination: Optional[Callable[[], None]] = None,
    ) -> Tuple[int, bytes, bytes]:
        max_retries = self._gpu_contamination_max_retries()
        last_error: Optional[GPUContaminationError] = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.info(
                    "[%s] Retrying %s after GPU contamination (%s/%s)",
                    task_id,
                    operation_name,
                    attempt,
                    max_retries,
                )

            try:
                return await self._run_monitored_gpu_subprocess_once(
                    cmd,
                    cwd,
                    env,
                    actual_device_id,
                    task_id,
                    timeout,
                    operation_name,
                )
            except GPUContaminationError as exc:
                last_error = exc
                if cleanup_on_contamination is not None:
                    try:
                        cleanup_on_contamination()
                    except Exception as cleanup_exc:
                        logger.warning(
                            "[%s] Failed to clean partial %s artifacts: %s",
                            task_id,
                            operation_name,
                            cleanup_exc,
                        )
                if attempt >= max_retries:
                    logger.error(
                        "[%s] %s aborted after %s GPU contamination retries: %s",
                        task_id,
                        operation_name,
                        max_retries,
                        exc,
                    )
                    raise
                logger.warning(
                    "[%s] %s will restart after GPU contamination: %s",
                    task_id,
                    operation_name,
                    exc,
                )
                await asyncio.to_thread(self.waiting_for_resources, actual_device_id)

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"{operation_name} did not start")

    async def _run_profile_script_monitored_on_device(
        self,
        extract_dir: str,
        script_name: str,
        result_json_name: str,
        task_id: str,
        actual_device_id: int,
        timeout: int = 300,
    ) -> float:
        script_path = os.path.join(extract_dir, script_name)
        if not os.path.exists(script_path):
            logger.error(f"[{task_id}] Profile script {script_name} not found.")
            return float('inf')

        result_json_path = os.path.join(extract_dir, result_json_name)

        def cleanup_partial_profile_result():
            with suppress(FileNotFoundError):
                os.remove(result_json_path)

        cleanup_partial_profile_result()

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        try:
            returncode, stdout, stderr = await self._run_monitored_gpu_subprocess_with_retry_on_device(
                [sys.executable, script_name],
                extract_dir,
                env,
                actual_device_id,
                task_id,
                timeout,
                f"Profile script {script_name}",
                cleanup_on_contamination=cleanup_partial_profile_result,
            )
        except asyncio.TimeoutError:
            cleanup_partial_profile_result()
            logger.error(f"[{task_id}] Profile script {script_name} timed out.")
            return float('inf')
        except GPUContaminationError as exc:
            cleanup_partial_profile_result()
            logger.error(f"[{task_id}] Profile script {script_name} aborted due to repeated GPU contamination: {exc}")
            return float('inf')

        output_log = stdout.decode(errors='replace') + "\n" + stderr.decode(errors='replace')
        if returncode != 0:
            logger.error(f"[{task_id}] Profile script {script_name} failed with log:\n{output_log}")
            return float('inf')

        return read_profile_result_from_json(extract_dir, result_json_name)

    async def _run_profile_scripts_and_collect_results_monitored(
        self,
        extract_dir: str,
        op_name: str,
        task_id: str,
        actual_device_id: int,
    ) -> Tuple[float, float]:
        base_time = await self._run_profile_script_monitored_on_device(
            extract_dir,
            f"profile_{op_name}_base.py",
            "base_profile_result.json",
            task_id,
            actual_device_id,
        )
        if math.isinf(base_time):
            return float('inf'), float('inf')

        gen_time = await self._run_profile_script_monitored_on_device(
            extract_dir,
            f"profile_{op_name}_generation.py",
            "generation_profile_result.json",
            task_id,
            actual_device_id,
        )
        if math.isinf(gen_time):
            return float('inf'), float('inf')

        logger.info(f"[{task_id}] Profile results: base={base_time:.2f} us, gen={gen_time:.2f} us")
        return base_time, gen_time

    @staticmethod
    def _cleanup_nsys_outputs(script_dir: str, output_name: str):
        for path in Path(script_dir).glob(f"{output_name}*"):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except FileNotFoundError:
                continue
            except Exception as exc:
                logger.warning("Failed to remove partial nsys artifact %s: %s", path, exc)

    async def _run_nsys_monitored_on_device(
        self,
        script_path: str,
        op_name: str,
        task_id: str,
        actual_device_id: int,
        timeout: int = 600,
    ) -> Tuple[bool, str, Optional[str]]:
        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)
        output_name = "nsys_report_" + script_name.replace(".py", "")
        report_path = os.path.join(script_dir, output_name + ".nsys-rep")

        def cleanup_partial_nsys_outputs():
            self._cleanup_nsys_outputs(script_dir, output_name)

        cleanup_partial_nsys_outputs()

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        cmd = ["nsys", "profile", f"--output={output_name}", sys.executable, script_name]
        logger.debug(f"[{task_id}:{op_name}] Running monitored nsys profile: {' '.join(cmd)}")

        try:
            returncode, stdout, stderr = await self._run_monitored_gpu_subprocess_with_retry_on_device(
                cmd,
                script_dir,
                env,
                actual_device_id,
                task_id,
                timeout,
                f"nsys profile {script_name}",
                cleanup_on_contamination=cleanup_partial_nsys_outputs,
            )
        except FileNotFoundError as exc:
            return False, f"nsys not found: {exc}", None
        except asyncio.TimeoutError:
            cleanup_partial_nsys_outputs()
            return False, f"nsys profile timed out after {timeout} seconds", None
        except GPUContaminationError as exc:
            cleanup_partial_nsys_outputs()
            return False, str(exc), None

        output_log = stdout.decode(errors='replace') + "\n" + stderr.decode(errors='replace')
        if returncode != 0:
            cleanup_partial_nsys_outputs()
            return False, output_log, None
        if os.path.exists(report_path):
            return True, "", report_path
        return False, f"未找到nsys报告文件: {report_path}\n{output_log}", None

    async def _run_nsys_profiling_monitored(
        self,
        extract_dir: str,
        op_name: str,
        task_id: str,
        warmup_times: int,
        run_times: int,
        actual_device_id: int,
    ) -> Tuple[float, float]:
        base_script = os.path.join(extract_dir, f"profile_{op_name}_base.py")
        success, error, base_rep_path = await self._run_nsys_monitored_on_device(
            base_script,
            op_name,
            task_id,
            actual_device_id,
        )
        if not success or not base_rep_path:
            logger.error(f"[{task_id}] Base nsys failed: {error}")
            return float('inf'), float('inf')

        gen_script = os.path.join(extract_dir, f"profile_{op_name}_generation.py")
        success, error, gen_rep_path = await self._run_nsys_monitored_on_device(
            gen_script,
            op_name,
            task_id,
            actual_device_id,
        )
        if not success or not gen_rep_path:
            logger.error(f"[{task_id}] Generation nsys failed: {error}")
            return float('inf'), float('inf')

        success, error, base_time = await asyncio.to_thread(
            analyze_nsys_data,
            base_rep_path,
            warmup_times,
            run_times,
            "base",
            op_name,
            task_id,
        )
        if not success:
            logger.error(f"[{task_id}] Base nsys analysis failed: {error}")
            return float('inf'), float('inf')

        success, error, gen_time = await asyncio.to_thread(
            analyze_nsys_data,
            gen_rep_path,
            warmup_times,
            run_times,
            "generation",
            op_name,
            task_id,
        )
        if not success:
            logger.error(f"[{task_id}] Generation nsys analysis failed: {error}")
            return float('inf'), float('inf')

        return base_time, gen_time

    @asynccontextmanager
    async def gpu_execution_lock(self, device_id: Optional[int], task_id: str):
        actual_device_id = self._resolve_device_id(device_id)
        os.makedirs(GPU_LOCK_DIR, exist_ok=True)
        lock_path = os.path.join(GPU_LOCK_DIR, f"gpu_{actual_device_id}.lock")
        lock_file = open(lock_path, "a+", encoding="utf-8")
        try:
            logger.info(f"[{task_id}] Waiting for GPU_{actual_device_id} lock: {lock_path}")
            await asyncio.to_thread(fcntl.flock, lock_file.fileno(), fcntl.LOCK_EX)
            logger.info(f"[{task_id}] Acquired GPU_{actual_device_id} lock")
            await asyncio.to_thread(self.waiting_for_resources, actual_device_id)
            yield actual_device_id
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                logger.info(f"[{task_id}] Released GPU_{actual_device_id} lock")
            finally:
                lock_file.close()

    async def verify(
        self,
        package_data: Union[bytes, str],
        task_id: str,
        op_name: str,
        timeout: int = 300,
        device_id: Optional[int] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute verification task locally.

        注意：device 的管理（acquire/release）由调用方负责
        这个方法只负责执行已经生成好的脚本（脚本中已包含正确的 device_id）

        Args:
            package_data: 验证包数据（bytes 或目录路径）
            task_id: 任务ID
            op_name: 算子名称
            timeout: 超时时间
            device_id: GPU设备ID，用于跨进程文件锁

        Returns:
            Tuple[bool, str, Dict[str, Any]]: (success, log, artifacts)
        """
        try:
            with ExitStack() as stack:
                if isinstance(package_data, (bytes, bytearray)):
                    temp_dir = stack.enter_context(tempfile.TemporaryDirectory(dir=os.getcwd()))
                    tar_path = os.path.join(temp_dir, "package.tar")
                    with open(tar_path, "wb") as f:
                        f.write(package_data)
                    extract_dir = os.path.join(temp_dir, "extract")
                    os.makedirs(extract_dir, exist_ok=True)
                    try:
                        with tarfile.open(tar_path, 'r') as tar_ref:
                            tar_ref.extractall(extract_dir)
                    except Exception as e:
                        return False, f"Failed to extract package: {e}", {}
                elif isinstance(package_data, str):
                    extract_dir = package_data
                else:
                    return False, "Unsupported package_data type for LocalWorker.verify", {}

                script_name = f"verify_{op_name}.py"
                script_path = os.path.join(extract_dir, script_name)
                if not os.path.exists(script_path):
                    return False, f"Verification script {script_name} not found.", {}

                # 注意：脚本中的 device_id 已在生成时设置正确
                # worker 只负责执行脚本，不管理设备分配
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'

                python_exe = sys.executable
                cmd = [python_exe, script_name]
                logger.info(f"[{task_id}] Running verification for {op_name}")
                try:
                    returncode, stdout, stderr = await self._run_gpu_exclusive_subprocess(
                        cmd,
                        extract_dir,
                        env,
                        device_id,
                        task_id,
                        timeout,
                        "Verification",
                    )
                except asyncio.TimeoutError:
                    logger.error(f"[{task_id}] Verification timed out.")
                    return False, f"Verification timed out after {timeout} seconds.", {}
                except GPUContaminationError as exc:
                    logger.error(f"[{task_id}] Verification aborted due to repeated GPU contamination: {exc}")
                    return False, str(exc), {}

                output_log = stdout.decode(errors='replace') + "\n" + stderr.decode(errors='replace')
                success = (returncode == 0)

                # 收集执行过程中生成的 JSON 文件
                artifacts = collect_json_artifacts(extract_dir)
                if artifacts:
                    logger.info(f"[{task_id}] Collected {len(artifacts)} artifact files: {list(artifacts.keys())}")

                if success:
                    logger.info(f"[{task_id}] Verification passed.")
                else:
                    logger.error(f"[{task_id}] Verification failed with log:\n{output_log}")

                return success, output_log, artifacts

        except Exception as e:
            logger.error(f"[{task_id}] LocalWorker verification failed: {e}", exc_info=True)
            return False, str(e), {}

    async def profile(
        self,
        package_data: bytes,
        task_id: str,
        op_name: str,
        profile_settings: Dict[str, Any],
        device_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute profiling task locally.

        注意：device 的管理（acquire/release）由调用方负责
        这个方法只负责执行已经生成好的 profile 脚本

        Returns:
            Dict[str, Any]: 包含 gen_time, base_time, speedup, artifacts 等字段
        """
        try:
            # 2. Create temp directory and extract
            with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dir:
                # Extract package
                tar_path = os.path.join(temp_dir, "package.tar")
                with open(tar_path, "wb") as f:
                    f.write(package_data)

                extract_dir = os.path.join(temp_dir, "extract")
                os.makedirs(extract_dir, exist_ok=True)

                try:
                    with tarfile.open(tar_path, 'r') as tar_ref:
                        tar_ref.extractall(extract_dir)
                except Exception as e:
                    return {'gen_time': float('inf'), 'base_time': 0.0, 'speedup': 0.0, 'artifacts': {}, 'error': str(e)}

                # 注意：profile 脚本中的 device_id 应该在生成时就已经设置正确
                # （与 verify 类似，通过预先获取设备ID）
                # 3. Get settings
                backend = profile_settings.get('backend', self.backend)
                dsl = profile_settings.get('dsl', '')
                run_times = profile_settings.get('run_times', 50)
                warmup_times = profile_settings.get('warmup_times', 5)

                # 4. Execute profiling based on backend/dsl
                try:
                    if "triton_cuda" in dsl:
                        async with self.gpu_execution_lock(device_id, task_id) as actual_device_id:
                            base_time, gen_time = await self._run_profile_scripts_and_collect_results_monitored(
                                extract_dir,
                                op_name,
                                task_id,
                                actual_device_id,
                            )
                    elif "triton_ascend" in dsl or backend == "cpu":
                        async with self.gpu_execution_lock(device_id, task_id):
                            # Triton Ascend/CPU: keep the existing synchronous helper path.
                            loop = asyncio.get_running_loop()
                            base_time, gen_time = await loop.run_in_executor(
                                None,
                                run_profile_scripts_and_collect_results,
                                extract_dir, op_name, task_id
                            )
                            logger.info(f"[{task_id}] Profile results: base={base_time:.2f} us, gen={gen_time:.2f} us")
                    elif backend == "ascend":
                        async with self.gpu_execution_lock(device_id, task_id):
                            # Ascend: use msprof
                            loop = asyncio.get_running_loop()
                            base_time, gen_time = await loop.run_in_executor(
                                None,
                                self._run_msprof_profiling,
                                extract_dir, op_name, task_id, warmup_times, run_times
                            )
                    elif backend == "cuda":
                        async with self.gpu_execution_lock(device_id, task_id) as actual_device_id:
                            # CUDA: use nsys under the same runtime GPU exclusivity monitor.
                            base_time, gen_time = await self._run_nsys_profiling_monitored(
                                extract_dir,
                                op_name,
                                task_id,
                                warmup_times,
                                run_times,
                                actual_device_id,
                            )
                    else:
                        logger.warning(f"[{task_id}] Unsupported backend for profiling: {backend}")
                        return {'gen_time': float('inf'), 'base_time': 0.0, 'speedup': 0.0, 'artifacts': {}}

                    # 5. Calculate speedup
                    speedup = base_time / gen_time if gen_time > 0 else 0.0

                    # 6. 收集执行过程中生成的 JSON 文件
                    artifacts = collect_json_artifacts(extract_dir)
                    if artifacts:
                        logger.info(f"[{task_id}] Collected {len(artifacts)} artifact files: {list(artifacts.keys())}")

                    return {
                        'gen_time': gen_time,
                        'base_time': base_time,
                        'speedup': speedup,
                        'artifacts': artifacts
                    }

                except Exception as e:
                    logger.error(f"[{task_id}] Profiling execution failed: {e}", exc_info=True)
                    return {'gen_time': float('inf'), 'base_time': 0.0, 'speedup': 0.0, 'artifacts': {}, 'error': str(e)}

        except Exception as e:
            logger.error(f"[{task_id}] LocalWorker profiling failed: {e}", exc_info=True)
            return {'gen_time': float('inf'), 'base_time': 0.0, 'speedup': 0.0, 'artifacts': {}, 'error': str(e)}

    async def ncu_profile(
        self,
        package_data: str,
        task_id: str,
        op_name: str,
        timeout: int = 300,
        device_id: Optional[int] = None
    ) -> Dict[str, Any]:
        extract_dir = package_data
        script_name = f"profile_{op_name}_generation.py"
        script_path = os.path.join(extract_dir, script_name)    # extract_dir /home/zhangzizheng/aikg_logs/Task_74gs0nc1/Square_matrix_multiplication_/I1_0_0_S02_verify
        if not os.path.exists(script_path):
            return False, f"NCU profile script {script_name} not found.", {}, "{}"

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        ncu_bin = '/mnt/lustre-client/zhangzizheng/gpu5_install/NVIDIA-Nsight-Compute-2025.4/ncu'
        ncu_bin = ncu_bin if os.path.exists(ncu_bin) else '/home/zhangzizheng/h100_install/NVIDIA-Nsight-Compute-2025.4/ncu'

        cmd = [
            ncu_bin,
            "--nvtx",
            "--nvtx-include", "target_kernel",
            "--csv",
            "--page=raw",
            "--kernel-name-base=demangled",
            "--target-processes=all",
            "--replay-mode=kernel",
            "--profile-from-start=on",
            f"--log-file={extract_dir}/ncu_temp.csv",
            f"--metrics={NCU_METRICS}",
            "--launch-skip=10",
            "--launch-count=30",
            "--force-overwrite",
            sys.executable, script_name,
            "--repeat 10",
        ]

        logger.info(f"[{task_id}] Running ncu profiling for {op_name}")
        ncu_csv_path = os.path.join(extract_dir, "ncu_temp.csv")

        def cleanup_partial_ncu_csv():
            with suppress(FileNotFoundError):
                os.remove(ncu_csv_path)

        try:
            returncode, stdout, stderr = await self._run_gpu_exclusive_subprocess(
                cmd,
                extract_dir,
                env,
                device_id,
                task_id,
                timeout * 10,
                "NCU Profile",
                cleanup_on_contamination=cleanup_partial_ncu_csv,
            )
        except asyncio.TimeoutError:
            logger.error(f"[{task_id}] NCU Profile timed out.")
            return False, f"NCU Profile timed out after {timeout} seconds.", {}, "{}"
        except GPUContaminationError as exc:
            cleanup_partial_ncu_csv()
            logger.error(f"[{task_id}] NCU Profile aborted due to repeated GPU contamination: {exc}")
            return False, str(exc), {}, "{}"

        output_log = stdout.decode(errors='replace') + "\n" + stderr.decode(errors='replace')
        success = (returncode == 0)
        ncu_json = "{}"

        # 收集执行过程中生成的 JSON 文件
        artifacts = collect_json_artifacts(extract_dir)
        if artifacts:
            logger.info(f"[{task_id}] Collected {len(artifacts)} artifact files: {list(artifacts.keys())}")

        if success:
            logger.info(f"[{task_id}] NCU Profile passed.")
            ncu_df = load_ncu_metrics(f"{extract_dir}/ncu_temp.csv", None)
            ncu_json = metrics_to_prompt(ncu_df)
        else:
            logger.error(f"[{task_id}] NCU Profile failed with log:\n{output_log}")

        return success, output_log, artifacts, ncu_json


    def _run_msprof_profiling(self, extract_dir: str, op_name: str, task_id: str, warmup_times: int, run_times: int) -> Tuple[float, float]:
        """Run msprof profiling for Ascend backend (synchronous)"""
        try:
            # Run msprof for base script
            base_script = os.path.join(extract_dir, f"profile_{op_name}_base.py")
            success, error, base_prof_path = run_msprof(base_script, op_name, task_id)
            if not success or not base_prof_path:
                logger.error(f"[{task_id}] Base msprof failed: {error}")
                return float('inf'), float('inf')

            # Run msprof for generation script
            gen_script = os.path.join(extract_dir, f"profile_{op_name}_generation.py")
            success, error, gen_prof_path = run_msprof(gen_script, op_name, task_id)
            if not success or not gen_prof_path:
                logger.error(f"[{task_id}] Generation msprof failed: {error}")
                return float('inf'), float('inf')

            # Analyze prof data
            success, error, base_time = analyze_prof_data(base_prof_path, warmup_times, run_times, op_name, task_id)
            if not success:
                logger.error(f"[{task_id}] Base prof analysis failed: {error}")
                return float('inf'), float('inf')

            success, error, gen_time = analyze_prof_data(gen_prof_path, warmup_times, run_times, op_name, task_id)
            if not success:
                logger.error(f"[{task_id}] Generation prof analysis failed: {error}")
                return float('inf'), float('inf')

            return base_time, gen_time

        except Exception as e:
            logger.error(f"[{task_id}] msprof profiling failed: {e}", exc_info=True)
            return float('inf'), float('inf')

    def _run_nsys_profiling(self, extract_dir: str, op_name: str, task_id: str, warmup_times: int, run_times: int) -> Tuple[float, float]:
        """Run nsys profiling for CUDA backend (synchronous)"""
        try:
            # Run nsys for base script
            base_script = os.path.join(extract_dir, f"profile_{op_name}_base.py")
            success, error, base_rep_path = run_nsys(base_script, op_name, task_id)
            if not success or not base_rep_path:
                logger.error(f"[{task_id}] Base nsys failed: {error}")
                return float('inf'), float('inf')

            # Run nsys for generation script
            gen_script = os.path.join(extract_dir, f"profile_{op_name}_generation.py")
            success, error, gen_rep_path = run_nsys(gen_script, op_name, task_id)
            if not success or not gen_rep_path:
                logger.error(f"[{task_id}] Generation nsys failed: {error}")
                return float('inf'), float('inf')

            # Analyze nsys data
            success, error, base_time = analyze_nsys_data(base_rep_path, warmup_times, run_times, "base", op_name, task_id)
            if not success:
                logger.error(f"[{task_id}] Base nsys analysis failed: {error}")
                return float('inf'), float('inf')

            success, error, gen_time = analyze_nsys_data(gen_rep_path, warmup_times, run_times, "generation", op_name, task_id)
            if not success:
                logger.error(f"[{task_id}] Generation nsys analysis failed: {error}")
                return float('inf'), float('inf')

            return base_time, gen_time

        except Exception as e:
            logger.error(f"[{task_id}] nsys profiling failed: {e}", exc_info=True)
            return float('inf'), float('inf')
