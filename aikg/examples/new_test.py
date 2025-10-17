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

from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.utils.environment_check import check_env_for_task
import asyncio
import os
os.environ['AIKG_STREAM_OUTPUT'] = 'on'


def get_op_name():
    return 'relu'


def get_task_desc():
    return '''
import mindspore as ms
from mindspore import nn


class Model(nn.Cell):
    """
    ReLU激活函数模型
    """
    def __init__(self):
        super(Model, self).__init__()

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """
        计算ReLU激活函数
        Args:
            x: 输入张量
        Returns:
            ReLU激活后的张量
        """
        return ms.ops.relu(x)


batch_size = 16
dim = 16384


def get_inputs():
    x = ms.ops.randn(batch_size, dim, dtype=ms.float16)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed
'''

def get_torch_task_desc():
    return '''
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ReLU激活函数模型
    """
    def __init__(self):
        super(Model, self).__init__()
        # PyTorch中可以直接使用nn.ReLU()层
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算ReLU激活函数
        Args:
            x: 输入张量
        Returns:
            ReLU激活后的张量
        """
        return self.relu(x)
        # 也可以使用函数式API: return torch.relu(x)


batch_size = 16
dim = 16384


def get_inputs():
    # 生成符合正态分布的随机张量，类型为float16
    x = torch.randn(batch_size, dim, dtype=torch.float16)
    return [x]


def get_init_inputs():
    return []  # 不需要特殊的初始化输入

'''

async def run_mindspore_triton_single():
    op_name = get_op_name()
    # task_desc = get_task_desc()
    task_desc = get_torch_task_desc()

    task_pool = TaskPool()
    device_pool = DevicePool([0])
    # config = load_config(config_path="/home/zzz/akg/aikg/python/ai_kernel_generator/config/default_triton_config.yaml")  # use offical deepseek api
    # config = load_config(config_path="./python/ai_kernel_generator/config/vllm_triton_coderonly_config.yaml")
    config = load_config(config_path="./python/ai_kernel_generator/config/vllm_triton_coderonly_config.yaml")
    # check_env_for_task("mindspore", "ascend", "triton", config)
    
    # check_env_for_task("torch", "cuda", "triton", config)
    task = Task(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        dsl="triton",
        backend="cuda",
        arch="a100",
        config=config,
        device_pool=device_pool,
        framework="torch",
        workflow="coder_only_workflow"
    )
    
    task_pool.create_task(task.run)
    results = await task_pool.wait_all()
    for op_name, result, _ in results:
        if result:
            print(f"Task {op_name} passed")
        else:
            print(f"Task {op_name} failed")

if __name__ == "__main__":
    asyncio.run(run_mindspore_triton_single())
