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

"""Default workflow: Designer → Coder ↔ Verifier"""

from langgraph.graph import StateGraph, END
from ai_kernel_generator.workflows.base_workflow import BaseWorkflow
from ai_kernel_generator.utils.langgraph.state import KernelGenState
from ai_kernel_generator.utils.langgraph.nodes import NodeFactory
from ai_kernel_generator.utils.langgraph.routers import RouterFactory


class EvolveWorkflow(BaseWorkflow):
    """进化 Workflow
    
    Flow:
        designer -> filter -> coder_0 -> verifier_0 -> profiler_0 -> END
            |_________|    -> coder_1 -> verifier_1 -> profiler_1 -> END
                           -> coder_n -> verifier_n -> profiler_n -> END
                                ^            |
                                | conductor_0|
                                | conductor_1|
                                | conductor_n|
                                |____________|
    """
    
    def build_graph(self) -> StateGraph:
        """构建默认工作流图"""
        workflow = StateGraph(KernelGenState)
        
        # 检查必需的 Agent
        required_agents = ['designer', 'filter', 'coder', 'verifier']
        for agent_name in required_agents:
            if agent_name not in self.agents:
                raise RuntimeError(f"Required agent '{agent_name}' is not available. "
                                 f"Available agents: {list(self.agents.keys())}")
        para_coge_gen_num = self.config.get("para_coge_gen_num", 3) # Task 内并行 coder 的数量，默认 3
        
        # 创建节点
        designer_node = NodeFactory.create_designer_node(
            self.agents['designer'], 
            self.trace,
            self.config,
            para_coge_gen_num
        )
        filter_node = NodeFactory.create_filter_node(
            self.agents['filter']
        )
        
        # 添加节点
        workflow.add_node("designer", designer_node)
        workflow.add_node("filter", filter_node)
        for i in range(para_coge_gen_num):
            coder_node = NodeFactory.create_coder_node(
                self.agents['coder'], 
                self.trace,
                i
            )
            verifier_node = NodeFactory.create_verifier_node(
                self.agents['verifier'], 
                self.device_pool, 
                self.trace,
                self.config,
                self.private_worker,
                self.worker_manager,
                self.backend,
                self.arch,
                i
            )
            conductor_node = NodeFactory.create_conductor_node(
                self.trace,
                self.config,
                self.conductor_template,
                i
            )
            workflow.add_node(f"coder_{i}", coder_node)  # 添加多个 coder 节点
            workflow.add_node(f"verifier_{i}", verifier_node)
            workflow.add_node(f"conductor_{i}", conductor_node)
        
        # 添加边
        workflow.add_edge("designer", "filter")
        
        # 条件边：filter 后的路由（选择 designer 或者 coder）
        filter_router = RouterFactory.create_filter_router(self.config)
        workflow.add_conditional_edges(
            "filter",
            filter_router,
            {
                "designer": "designer",  # 需要重新设计
                "coder": [ f"coder_{i}" for i in range(para_coge_gen_num) ]       # 直接进入编码
            }
        )
        
        for i in range(para_coge_gen_num):
            workflow.add_edge(f"coder_{i}", f"verifier_{i}")
        
        # 条件边：verifier 后的路由（验证通过跳过 conductor）
        for i in range(para_coge_gen_num):
            verifier_router = RouterFactory.create_verifier_router_with_conductor(
                self.config,
                i
            )
            workflow.add_conditional_edges(
                f"verifier_{i}",
                verifier_router,
                {
                    "conductor": f"conductor_{i}",  # 验证失败 → Conductor 分析
                    "finish": END              # 验证通过 → 直接结束
                }
            )
        
        # 条件变：Conductor 后的路由
        for i in range(para_coge_gen_num):
            conductor_router = RouterFactory.create_conductor_router(self.config, i)
            workflow.add_conditional_edges(
                f"conductor_{i}",
                conductor_router,
                {
                    "coder": f"coder_{i}",
                    "finish": END
                }
            )
        
        # 设置入口
        workflow.set_entry_point("designer")
        
        return workflow