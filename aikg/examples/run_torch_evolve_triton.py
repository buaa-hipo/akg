# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import argparse
# 导入evolve函数和必要的模块
from ai_kernel_generator.core.evolve import new_evolve
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.worker.manager import register_worker
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.utils.environment_check import check_env_for_task
from ai_kernel_generator.utils.evolve.runner_manager import RunnerConfig, print_evolve_config, print_evolution_result
from ai_kernel_generator import get_project_root
from pathlib import Path


def get_op_name():
    return 'aikg_relu'


def get_task_desc():
    return '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return torch.relu(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
'''
def get_task_desc(file: str):
    return ''.join(open(file, 'r').readlines())

async def run_torch_evolve_triton(op_name: str, task_desc: str, evolve_database: str, worker_mode="local", worker_url=None, config_name: str = "evolve_deepseek.yaml"):
    """
    运行Triton进化示例
    
    Args:
        worker_mode: "local" 或 "remote"，指定使用本地还是远程 Worker
        worker_url: 当 worker_mode="remote" 时，指定远程 Worker Service 的 URL
    """
    # 创建配置对象并设置硬编码参数
    config = RunnerConfig()

    # 基础配置
    config.dsl = "triton_cuda"  # 使用triton_cuda替代通用的triton
    config.framework = "torch"
    config.backend = "cuda"
    config.arch = "a100"

    # 进化参数
    config.max_rounds = 10
    config.parallel_num = 1         # coder_parallel_num 和 parallel_num 不可同时选，当前开发 coder_parallel_num
    config.coder_parallel_num = 3   # 每次岛屿任务并行探索Coder的数量

    # 岛屿模型参数
    config.num_islands = 1
    config.migration_interval = 2
    config.elite_size = 5
    config.parent_selection_prob = 0.5

    # 设备配置
    config.device_list = [1]

    # 配置文件路径
    config.config_path = str(Path(get_project_root()) / "config" / config_name)

    # 选择要运行的任务
    # config.op_name = get_op_name()
    # config.task_desc = get_task_desc()
    config.op_name = op_name  # any kernel name
    config.task_desc = get_task_desc(task_desc)
    config.evolve_database = evolve_database # must be with task_desc levelx/xxx

    # 打印配置信息
    print_evolve_config(config.op_name, config)
    
    # 打印 Worker 模式
    print(f"\n{'='*60}")
    print(f"Worker 模式: {worker_mode.upper()}")
    if worker_mode == "remote":
        worker_url = worker_url or os.getenv("AIKG_WORKER_URL")
        if worker_url:
            print(f"Remote Worker URL: {worker_url}")
        else:
            print(f"Remote Worker URL: 将从环境变量 AIKG_WORKER_URL 读取")
    print(f"{'='*60}\n")

    # 初始化资源池
    task_pool = TaskPool(max_concurrency=config.coder_parallel_num)     # 让coder后续可以并行跑起来
    
    # 根据 worker_mode 设置 worker
    if worker_mode == "remote":
        target_worker_url = worker_url or os.getenv("AIKG_WORKER_URL")
        print(f"🔗 注册 RemoteWorker (url={target_worker_url or 'AIKG_WORKER_URL'})")
        await register_worker(
            backend=config.backend,
            arch=config.arch,
            worker_url=target_worker_url
        )
    else:
        print(f"🔗 注册 LocalWorker: devices={config.device_list}")
        await register_worker(
            backend=config.backend,
            arch=config.arch,
            device_ids=config.device_list
        )
    print()

    # 加载配置并检查环境
    loaded_config = load_config(config_path=config.config_path)
    
    # Remote 模式跳过硬件检查
    is_remote = (worker_mode == "remote")
    check_env_for_task(
        config.framework, 
        config.backend, 
        config.dsl, 
        loaded_config,
        is_remote=is_remote
    )

    # 调用evolve函数
    print("开始进化过程...")
    evolution_result = await new_evolve(
        op_name=config.op_name,
        task_desc=config.task_desc,
        evolve_database=config.evolve_database,
        dsl=config.dsl,
        framework=config.framework,
        backend=config.backend,
        arch=config.arch,
        config=loaded_config,
        task_pool=task_pool,
        max_rounds=config.max_rounds,
        parallel_num=config.parallel_num,
        coder_parallel_num=config.coder_parallel_num,
        num_islands=config.num_islands,
        migration_interval=config.migration_interval,
        elite_size=config.elite_size,
        parent_selection_prob=config.parent_selection_prob
    )

    # 打印结果
    print_evolution_result(evolution_result, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="运行 Triton 进化示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用本地 Worker（默认）
  python run_torch_evolve_triton.py
  
  # 使用远程 Worker（通过环境变量）
  export AIKG_WORKER_URL=http://localhost:9001
  python run_torch_evolve_triton.py --worker remote
  
  # 使用远程 Worker（指定 URL）
  python run_torch_evolve_triton.py --worker remote --worker-url http://192.168.1.100:9001
        """
    )
    parser.add_argument(
        "--worker",
        choices=["local", "remote"],
        default="local",
        help="Worker 模式: local (本地) 或 remote (远程)，默认: local"
    )
    parser.add_argument(
        "--worker-url",
        type=str,
        default=None,
        help="远程 Worker Service 的 URL（仅 remote 模式需要）。也可通过环境变量 AIKG_WORKER_URL 设置"
    )
    
    parser.add_argument(
        "--config_name",
        type=str,
        default="evolve_deepseek.yaml",
    )
    
    aikg_debug_mode = os.environ.get("AIKG_DEBUG_MODE", False) is not False
    parser.add_argument(
        "--op-name",
        type=str,
        default="Matmul_with_transposed_both" if aikg_debug_mode is False else "Matmul_with_transposed_both",
    )
    parser.add_argument(
        "--task-desc",
        type=str,
        default="/mnt/lustre-client/zhangzizheng/AIKG/KernelBench/KernelBench/level1/18_Matmul_with_transposed_both.py" if aikg_debug_mode is False else "/mnt/lustre-client/zhangzizheng/AIKG/KernelBench/KernelBench/level1/18_Matmul_with_transposed_both.py",
    )
    parser.add_argument(
        "--evolve-database",
        type=str,
        default="test" if aikg_debug_mode is False else "test",     # 同样会被用作检查点路径名 evolve_database/xxx
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Triton 进化示例")
    print("=" * 60)
    
    if args.worker == "remote":
        worker_url = args.worker_url or os.getenv("AIKG_WORKER_URL")
        if worker_url:
            print(f"\n⚠️  Remote Worker 模式")
            print(f"   确保远程 Worker Service 正在运行: {worker_url}")
            print(f"   如果使用 SSH 隧道，确保隧道已建立")
        else:
            print(f"\n⚠️  Remote Worker 模式")
            print(f"   请设置环境变量 AIKG_WORKER_URL 或使用 --worker-url 参数")
        print()
    
    asyncio.run(run_torch_evolve_triton(args.op_name, args.task_desc, args.evolve_database, worker_mode=args.worker, worker_url=args.worker_url, config_name=args.config_name))
