import argparse
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

EXAMPLES_DIR = Path(os.getcwd())
KERNELBENCH_DIR = Path("/mnt/lustre-client/zhangzizheng/AIKG/KernelBench/KernelBench")


@dataclass(frozen=True)
class KernelTask:
    index: int
    level: str
    full_op_name: str
    op_name: str
    task_desc: Path
    evolve_database: str
    log_path: Path


@dataclass
class RunningTask:
    task: KernelTask
    process: subprocess.Popen
    start_time: float
    use_cuda: bool


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def parse_range(range_text: str) -> tuple[int, int]:
    try:
        start_text, end_text = range_text.split(",", maxsplit=1)
        start = int(start_text)
        end = int(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("range must be formatted like 3,18") from exc

    if start <= 0 or end <= 0:
        raise argparse.ArgumentTypeError("range values must be positive integers")
    if start > end:
        raise argparse.ArgumentTypeError("range start must be <= range end")
    return start, end


def parse_kernel_index(path: Path) -> int | None:
    prefix = path.stem.split("_", maxsplit=1)[0]
    if not prefix.isdigit():
        return None
    return int(prefix)


def build_kernel_tasks(level: str, index_range: tuple[int, int]) -> list[KernelTask]:
    level_dir = KERNELBENCH_DIR / level
    if not level_dir.is_dir():
        raise FileNotFoundError(f"level directory not found: {level_dir}")

    start, end = index_range
    kernels_by_index: dict[int, Path] = {}
    for kernel_file in level_dir.glob("*.py"):
        index = parse_kernel_index(kernel_file)
        if index is None or index < start or index > end:
            continue
        if index in kernels_by_index:
            raise ValueError(
                f"duplicate kernel index {index}: {kernels_by_index[index]} and {kernel_file}"
            )
        kernels_by_index[index] = kernel_file

    missing = [index for index in range(start, end + 1) if index not in kernels_by_index]
    if missing:
        log(f"skip missing indexes: {','.join(map(str, missing))}")

    tasks: list[KernelTask] = []
    for index in sorted(kernels_by_index):
        kernel_file = kernels_by_index[index]
        full_op_name = kernel_file.stem
        op_name = full_op_name.split("_", maxsplit=1)[1]
        if "3D_tensor_matrix_multiplication" in op_name:
            op_name = "three_D_tensor_matrix_multiplication"
        elif "4D_tensor_matrix_multiplication" in op_name:
            op_name = "four_D_tensor_matrix_multiplication"
        
        evolve_database = f"{level}/{full_op_name}"
        log_path = EXAMPLES_DIR / "log" / level / f"{full_op_name}.log"
        tasks.append(
            KernelTask(
                index=index,
                level=level,
                full_op_name=full_op_name,
                op_name=op_name,
                task_desc=kernel_file,
                evolve_database=evolve_database,
                log_path=log_path,
            )
        )

    if not tasks:
        raise ValueError(f"no kernel tasks found for {level} range {start},{end}")
    return tasks


def submit_task(task: KernelTask, config_name: str, use_cuda: bool) -> RunningTask:
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if use_cuda:
        env["AIKG_MODEL_DEVICE"] = "cuda"
    else:
        env.pop("AIKG_MODEL_DEVICE", None)

    cmd = [
        sys.executable,
        "run_torch_evolve_triton.py",
        "--config_name",
        config_name,
        "--op-name",
        task.op_name,
        "--task-desc",
        str(task.task_desc),
        "--evolve-database",
        task.evolve_database,
    ]

    with open(task.log_path, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=EXAMPLES_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    device_tag = "cuda" if use_cuda else "default"
    log(
        f"submitted {task.level}/{task.full_op_name} pid={process.pid} "
        f"device={device_tag} log={task.log_path}"
    )
    return RunningTask(
        task=task,
        process=process,
        start_time=time.time(),
        use_cuda=use_cuda,
    )


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def run_scheduler(
    tasks: list[KernelTask],
    max_running: int,
    cuda_limit: int,
    poll_interval: int,
    config_name: str,
) -> int:
    pending = deque(tasks)
    running: list[RunningTask] = []
    completed: list[KernelTask] = []
    failed: list[tuple[KernelTask, int]] = []
    cuda_running = 0

    log(
        f"start scheduler total={len(tasks)} max_running={max_running} "
        f"cuda_limit={cuda_limit} range={tasks[0].index}-{tasks[-1].index}"
    )

    try:
        while pending or running:
            still_running: list[RunningTask] = []
            for running_task in running:
                return_code = running_task.process.poll()
                if return_code is None:
                    still_running.append(running_task)
                    continue

                elapsed = format_duration(time.time() - running_task.start_time)
                task = running_task.task
                if running_task.use_cuda:
                    cuda_running -= 1
                if return_code == 0:
                    completed.append(task)
                    log(f"completed {task.level}/{task.full_op_name} elapsed={elapsed}")
                else:
                    failed.append((task, return_code))
                    log(
                        f"failed {task.level}/{task.full_op_name} "
                        f"return_code={return_code} elapsed={elapsed}"
                    )
            running = still_running

            while pending and len(running) < max_running:
                use_cuda = cuda_running < cuda_limit
                running.append(submit_task(pending.popleft(), config_name, use_cuda))
                if use_cuda:
                    cuda_running += 1

            if pending or running:
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        log("scheduler interrupted; running task processes were not stopped")
        for running_task in running:
            task = running_task.task
            log(f"still running {task.level}/{task.full_op_name} pid={running_task.process.pid}")
        return 130

    log(f"all submitted tasks finished completed={len(completed)} failed={len(failed)}")
    if failed:
        for task, return_code in failed:
            log(f"failed task {task.level}/{task.full_op_name} return_code={return_code}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit KernelBench tasks with a fixed task-level concurrency limit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--level",
        choices=["level1", "level2", "level3"],
        required=True,
        help="KernelBench level to submit.",
    )
    parser.add_argument(
        "--range",
        dest="index_range",
        type=parse_range,
        required=True,
        help="Inclusive kernel index range, for example 3,18.",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=6,
        help="Maximum number of tasks running at the same time.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Maximum number of running tasks that use AIKG_MODEL_DEVICE=cuda.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--config-name",
        default="evolve_openai.yaml",
        help="Config file name passed to run_torch_evolve_triton.py.",
    )
    args = parser.parse_args()

    if args.task <= 0:
        parser.error("--task must be a positive integer")
    if args.cuda < 0 or args.cuda > args.task:
        parser.error("--cuda must satisfy 0 <= cuda <= task")
    if args.interval <= 0:
        parser.error("--interval must be a positive integer")

    os.chdir(EXAMPLES_DIR)
    tasks = build_kernel_tasks(args.level, args.index_range)
    return run_scheduler(tasks, args.task, args.cuda, args.interval, args.config_name)


if __name__ == "__main__":
    raise SystemExit(main())
