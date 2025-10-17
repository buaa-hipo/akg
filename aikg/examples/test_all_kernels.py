# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.task import Task
import asyncio
import os

os.environ['AIKG_STREAM_OUTPUT'] = 'on'

KERNEL_DIR = "/workspace/aikg_zkg/aikg/thirdparty/KernelBench/KernelBench/level1"


def get_op_name(file_path: str) -> str:
    """从文件名提取 kernel 名称"""
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    # 文件名格式: 1_Square_matrix_multiplication_.py
    # 去掉开头序号和尾部下划线
    parts = name.split("_", 1)
    if len(parts) > 1:
        kernel_name = parts[1].strip("_")
    else:
        kernel_name = name
    return kernel_name


def get_torch_task_desc(file_path: str) -> str:
    """读取 kernel 文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


async def run_mindspore_triton_single():
    # 遍历目录中的所有 py 文件
    files = [os.path.join(KERNEL_DIR, f) for f in os.listdir(KERNEL_DIR) if f.endswith(".py")]
    # print(files)
    # print(len(files))
    files.sort()  # 保证顺序一致
    files = files[50:]
    print(files)
    print(len(files))
    # exit(0)

    task_pool = TaskPool()
    device_pool = DevicePool([0])
    config = load_config(config_path="./python/ai_kernel_generator/config/vllm_triton_coderonly_config.yaml")

    for idx, file_path in enumerate(files):
        op_name = get_op_name(file_path)
        task_desc = get_torch_task_desc(file_path)

        task = Task(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(idx),
            dsl="triton",
            backend="cuda",
            arch="a100",
            config=config,
            device_pool=device_pool,
            framework="torch",
            workflow="coder_only_workflow"
        )

        task_pool.create_task(task.run)

        # 等待所有任务完成
        results = await task_pool.wait_all()
        for op_name, result, _ in results:
            if result:
                print(f"Task {op_name} passed")
            else:
                print(f"Task {op_name} failed")


if __name__ == "__main__":
    asyncio.run(run_mindspore_triton_single())
