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


class DesignerOnlyWorkflow(BaseWorkflow):
    """进化 Designer Only Workflow
    
    Flow:
        designer -> filter 
            |_________|    
    """
    
    def build_graph(self) -> StateGraph:
        """构建 Designer Only 工作流图"""
        workflow = StateGraph(KernelGenState)
        
        # 检查必需的 Agent
        required_agents = ['designer', 'filter']
        for agent_name in required_agents:
            if agent_name not in self.agents:
                raise RuntimeError(f"Required agent '{agent_name}' is not available. "
                                 f"Available agents: {list(self.agents.keys())}")
        
        # 创建节点
        designer_node = NodeFactory.create_designer_node(
            self.agents['designer'], 
            self.trace,
            self.config
        )
        filter_node = NodeFactory.create_filter_node(
            self.agents['filter']
        )
        
        # 添加节点
        workflow.add_node("designer", designer_node)
        workflow.add_node("filter", filter_node)
        
        # 添加边
        workflow.add_edge("designer", "filter")
        
        # 条件边：filter 后的路由（选择 designer 或者 coder）
        filter_router = RouterFactory.create_filter_router(self.config)
        workflow.add_conditional_edges(
            "filter",
            filter_router,
            {
                "designer": "designer",  # 需要重新设计
                "coder": END       # 直接进入编码  Designer Only 里直接结束
            }
        )

        # 设置入口
        workflow.set_entry_point("designer")
        
        return workflow