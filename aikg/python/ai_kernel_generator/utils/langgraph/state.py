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

"""State definition for LangGraph-based workflow."""

from typing import TypedDict, Annotated, Optional, List, Dict, Any
from operator import add

def dict_reducer(left, right):
    """合并并排序列表中的值"""
    # 确保 left 是列表
    if not isinstance(left, list):
        left = [left]

    # 确保 right 是列表
    if not isinstance(right, list):
        right = [right]

    # 合并list中的dict
    # 1. 先将两个列表中的所有字典提取出来并合并
    merged_dict = {}
    
    # 遍历左列表中的所有字典，合并到merged_dict
    for d in left:
        if isinstance(d, dict):
            merged_dict.update(d)
    
    # 遍历右列表中的所有字典，合并到merged_dict
    for d in right:
        if isinstance(d, dict):
            merged_dict.update(d)
    
    # 2. 返回包含合并后字典的列表
    return [merged_dict]


class KernelGenState(TypedDict, total=False):
    """Complete state definition for kernel generation workflow."""
    
    # === 基础信息 ===
    op_name: str
    task_desc: str
    task_id: str
    dsl: str
    framework: str
    backend: str
    arch: str
    task_type: str  # 任务类型：profile/precision_only
    parent_id: str
    
    # === Agent 输出字段 ===
    designer_code: Optional[str]
    designer_prompt: Optional[str]
    designer_reasoning: Optional[str]
    
    filter_or_not: Optional[bool]  # filter 的决策结果
    code_feat: Optional[str]  # filter 中 feature extractor 的输出特征信息
    
    coder_code: Optional[str]
    coder_prompt: Optional[str]
    coder_reasoning: Optional[str]
    
    space_config_code: Optional[str]  # 动态形状参数空间
    
    verifier_result: bool
    verifier_error: str
    profile_res: Optional[Dict[str, Any]]
    optimize_history: str
    ncu_profile_result: str
    ncu_profile_metric: str
    
    # === 多 case 验证 ===
    multi_case_error: Optional[str]
    
    # === 流程控制 ===
    iteration: int
    max_iterations: int
    step_count: int
    
    # === 历史记录（累积）===
    history_attempts: Annotated[List[Dict[str, Any]], add]
    agent_history: Annotated[List[str], add]
    inefficiency_programs: Optional[List[dict[str, Any]]]
    
    # === Conductor 建议 ===
    conductor_suggestion: Optional[str]
    conductor_decision: Optional[str]  # Conductor 的决策结果
    
    # === 额外配置 ===
    inspirations: Optional[List[str]]
    meta_prompts: Optional[str]
    handwrite_suggestions: Optional[List[Dict[str, str]]]
    
    # === Base Doc 字段 ===
    dsl_api_doc: Optional[str]
    framework_api_doc: Optional[str]
    backend_api_doc: Optional[str]
    workflow_name: Optional[str]

    # === Pruner 相关字段 ===
    last_filtered_sketch: Optional[str]
    filter_retry_count: int
