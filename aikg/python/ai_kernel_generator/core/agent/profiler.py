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

import logging
from typing import Tuple
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.utils.hardware_utils import get_hardware_doc
from ai_kernel_generator.utils.common_utils import ParserFactory, get_md5_hash
logger = logging.getLogger(__name__)


class Profiler(AgentBase):
    """
    使用 NCU 等硬件分析工具测评 impl code 性能
    """

    def __init__(self, config: dict, op_name: str, framework: str = "", task_desc: str = "", impl_code: str = "", dsl: str = "", ncu_json: str = "",
                 optimize_history: str="", backend: str="", arch: str=""):
        self.model_config = config.get("agent_model_config", {})
        self.op_name = op_name
        self.impl_code = impl_code
        self.framework = framework
        self.task_desc = task_desc
        self.dsl = dsl
        self.ncu_json = ncu_json
        self.optimize_history = optimize_history
        self.backend = backend
        self.arch = arch

        context = {
            "agent_name": "profiler",
        }
        super().__init__(context=context)

        # 初始化解析器
        from ai_kernel_generator.utils.parser_loader import create_agent_parser
        self.code_parser = create_agent_parser("profiler")
        if not self.code_parser:
            raise ValueError(
                "Failed to create Profiler parser. Please check your parser_config.yaml configuration."
            )
        self.format_instructions = self.code_parser.get_format_instructions()

        # 初始化模板
        self.gen_profile_suggestion_template = self.load_template("profiler/gen_profile_suggestion_level1_opt2.j2")

        self.gen_profile_suggestion_input = {
            "op_name": self.op_name,
            "framework": self.framework,
            "task_desc": self.task_desc,
            "impl_code": self.impl_code,
            "dsl": self.dsl,
            "ncu_profile_res": self.ncu_json.strip(),
            "format_instructions": self.format_instructions,
            "hardware_doc": get_hardware_doc(self.backend, self.arch),
        }

    async def run(self) -> Tuple[str, str, str]:
        # 执行LLM生成前更新context，确保正确性
        hash = get_md5_hash(impl_code=self.impl_code)
        to_update_context = {
            "impl_code": self.impl_code,
            # "optimize_history": self.optimize_history,
            "hash": hash,
        }
        self.context.update(to_update_context)
        logging.info("NCU Profiler calling LLM ...")
        
        # DEBUG MODE
        import os
        if os.environ.get("AIKG_DEBUG_MODE", False):
            llm_content = ''.join(open('/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/examples/debug_io/example_output/ncu_profile_res.txt', 'r').readlines())
            formatted_prompt = ''
            reasoning = ''
        else:
            llm_content, formatted_prompt, reasoning = await self.run_llm(self.gen_profile_suggestion_template, self.gen_profile_suggestion_input, self.model_config.get("profiler", "deepseek_r1_default"))
        
        try:
            if self.code_parser:
                parsed_result = ParserFactory.robust_parse(llm_content, self.code_parser)
                if parsed_result:
                    llm_content = getattr(parsed_result, 'code', llm_content)
        except Exception as e:
            logger.warning(f"Profiler LLM 解析 parser 失败 {e}，使用原始输出")

        # import json
        # suggestion = json.loads(llm_content).get("code", "No LLM profile suggestion.")
        # if suggestion == "No LLM profile suggestion.":
        #     logger.warning("Profiler LLM has no suggestions")
        return llm_content, formatted_prompt, reasoning
        # return await self.run_llm(self.gen_profile_suggestion_template, self.gen_profile_suggestion_input, self.model_config.get("profiler", "deepseek_r1_default"))
