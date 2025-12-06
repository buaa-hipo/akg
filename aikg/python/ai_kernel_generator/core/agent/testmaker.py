import os
import importlib.util
import torch
import random
from typing import List, Tuple, Optional
import re
from ai_kernel_generator.core.agent.agent_base import AgentBase
from langchain.prompts import PromptTemplate
from ai_kernel_generator.utils.common_utils import ParserFactory, remove_copyright_from_text
import json
import logging

logger = logging.getLogger(__name__)

def load_module_from_opname(op_name, framework="torch"):
    """获取 KernelBench 任务描述"""
    current_file_path = os.path.abspath(__file__)
    m = re.match(r'^(.*?/aikg)', current_file_path)
    if m:
        aikg_path = m.group(1)
    else:
        raise RuntimeError("无法定位 aikg 根目录路径")

    if framework == "torch":
        # Path for torch benchmarks from the KernelBench submodule.
        # The submodule is at `aikg/thirdparty/KernelBench`, and benchmark files are inside `KernelBench/level1/` subdirectory.
        base_dir = os.path.join(
            aikg_path, 'thirdparty', 'KernelBench', 'KernelBench', 'level1')
        # Files are directly in level1 directory with naming pattern: {number}_{name}.py
        task_path = os.path.join(base_dir, op_name + '.py')
    else:
        # Original path for mindspore and numpy benchmarks
        base_dir = os.path.join(aikg_path, 'benchmark',
                                'kernelbench', framework)
        task_path = os.path.join(
            base_dir, op_name, op_name + f'_{framework}.py')

    # 检查文件是否存在
    if not os.path.exists(task_path):
        if framework == "torch":
            _raise_submodule_error("KernelBench 任务文件", task_path)
        else:
            _raise_submodule_error(f"{framework} benchmark 任务文件", task_path)

    module_name = op_name
    spec = importlib.util.spec_from_file_location(module_name, task_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_kernelbench_model_from_mod(mod, device=None):
    # 2. 调用 get_init_inputs() 获取 Model 初始化参数
    if not hasattr(mod, "get_init_inputs"):
        raise RuntimeError(f"{task_py_path} 中未找到 get_init_inputs()")
    init_args = mod.get_init_inputs()
    if init_args is None:
        init_args = []
    elif not isinstance(init_args, (list, tuple)):
        raise RuntimeError("get_init_inputs() 返回值必须为 list/tuple 或 None")

    # 3. 获取 Model
    if not hasattr(mod, "Model"):
        raise RuntimeError(f"{task_py_path} 中未找到 class Model")
    ModelClass = mod.Model

    # 4. 初始化模型
    model = ModelClass(*init_args)
    model.eval()

    # 是否转 GPU
    if device is not None:
        model = model.to(device)

    # 5. 获取推理输入
    if not hasattr(mod, "get_inputs"):
        raise RuntimeError(f"{task_py_path} 中未找到 get_inputs()")
    inputs = mod.get_inputs()
    if not isinstance(inputs, (list, tuple)):
        raise RuntimeError("get_inputs() 返回值必须是 list/tuple")

    if device is not None:
        inputs = [x.to(device) for x in inputs]

    # 6. 执行模型
    with torch.no_grad():
        outputs = model(*inputs)

    return model, inputs, outputs
# def load_module_from_path(path: str):
#     """动态从路径加载 python 模块"""
#     path = os.path.abspath(path)
#     module_name = os.path.splitext(os.path.basename(path))[0]
#     spec = importlib.util.spec_from_file_location(module_name, path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module


def run_kernelbench_model(op_name: str, device=None):
    # 1. 动态加载模块
    mod = load_module_from_opname(op_name=op_name)
    # mod = load_module_from_path(task_py_path)

    # 2. 调用 get_init_inputs() 获取 Model 初始化参数
    if not hasattr(mod, "get_init_inputs"):
        raise RuntimeError(f"{task_py_path} 中未找到 get_init_inputs()")
    init_args = mod.get_init_inputs()
    if init_args is None:
        init_args = []
    elif not isinstance(init_args, (list, tuple)):
        raise RuntimeError("get_init_inputs() 返回值必须为 list/tuple 或 None")

    # 3. 获取 Model
    if not hasattr(mod, "Model"):
        raise RuntimeError(f"{task_py_path} 中未找到 class Model")
    ModelClass = mod.Model

    # 4. 初始化模型
    model = ModelClass(*init_args)
    model.eval()

    # 是否转 GPU
    if device is not None:
        model = model.to(device)

    # 5. 获取推理输入
    if not hasattr(mod, "get_inputs"):
        raise RuntimeError(f"{task_py_path} 中未找到 get_inputs()")
    inputs = mod.get_inputs()
    if not isinstance(inputs, (list, tuple)):
        raise RuntimeError("get_inputs() 返回值必须是 list/tuple")

    if device is not None:
        inputs = [x.to(device) for x in inputs]

    # 6. 执行模型
    with torch.no_grad():
        outputs = model(*inputs)

    return model, inputs, outputs

def _choose_small_shape(orig_shape: torch.Size, rng: random.Random) -> Tuple[int, ...]:
    nd = len(orig_shape)
    if nd == 0:
        return tuple()  # 标量不改
    elif nd == 1:
        return (8,)
    elif nd == 2:
        candidates = [(2, 4), (4, 2)]
        return candidates[rng.randrange(len(candidates))]
    elif nd == 3:
        base = [2, 4, 8]
        rng.shuffle(base)
        return tuple(base)
    else:
        base = [2**(i+1) for i in range(nd)]
        rng.shuffle(base)
        return tuple(base)

def rand_generator(shape: List[int], device=None, seed: int = 0):
    return torch.rand(shape, device=device)

def ones_generator(shape: List[int], device=None, seed: int = 0):
    return torch.ones(shape, device=device)

def one_hot_generator(shape: List[int], device=None, seed: int = 0):
    """
    生成 one-hot 输入。
    默认假设最后一维是类别维，即 shape = [..., C]
    输出中沿最后一维每个位置只会有一个 1.
    """
    if len(shape) == 0:
        return torch.tensor(1.0, device=device)

    C = shape[-1]
    total = int(torch.tensor(shape).prod().item() // C)  # 元素组数 = 除最后一维外的乘积

    # 每组随机选一个索引
    idx = torch.randint(low=0, high=C, size=(total,), device=device)

    # 生成 one-hot (total, C)
    oh = torch.zeros(total, C, device=device)
    oh.scatter_(1, idx.unsqueeze(1), 1.0)

    # reshape 回原始形状
    return oh.reshape(shape)

def arange_generator(shape: List[int], device=None, seed: int = 0):
    """
    等差序列输入
    """
    total = int(torch.tensor(shape).prod().item())
    return torch.arange(total, dtype=torch.float32, device=device).reshape(shape)

def small_int_generator(shape: List[int], device=None, seed: int = 0):
    """
    小整数序列输入
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    return torch.randint(0, 4, shape, device=device).float()

def eye_generator(shape: List[int], device=None, seed: int = 0):
    """
    单位矩阵输入（适用于 MatMul / Linear 验证映射逻辑）
    """
    if len(shape) != 2:
        return torch.rand(shape, device=device)
    n = min(shape)
    E = torch.eye(n, device=device)
    return E.expand(*shape)  # 自动 broadcast / 切片

def pattern_generator(shape: List[int], device=None, seed: int = 0):
    #周期 Pattern 输入（适用于检查 broadcasting、stride correctness）
    base = torch.tensor([-1, 0, 1, 2, 1], device=device, dtype=torch.float32)
    out = base
    while out.numel() < torch.prod(torch.tensor(shape)):
        out = torch.cat([out, base], dim=0)
    return out[:torch.prod(torch.tensor(shape))].reshape(shape)

def _shrink_inputs_by_shape(inputs: List[torch.Tensor], device=None, seed: int = 0, generator=rand_generator):
    rng = random.Random(seed)
    new_inputs: List[torch.Tensor] = []
    for x in inputs:
        if not isinstance(x, torch.Tensor):
            new_inputs.append(x)
            continue
        small_shape = _choose_small_shape(x.shape, rng)
        x_small = generator(shape=small_shape, device=device, seed=seed)
        new_inputs.append(x_small)
    return new_inputs

def run_minimal_kernelbench_model(op_name: str, device=None, generator=rand_generator):
    # 1. 动态加载模块
    mod = load_module_from_opname(op_name=op_name)
    # mod = load_module_from_path(task_py_path)

    # 2. 调用 get_init_inputs() 获取 Model 初始化参数
    if not hasattr(mod, "get_init_inputs"):
        raise RuntimeError(f"{task_py_path} 中未找到 get_init_inputs()")
    init_args = mod.get_init_inputs()
    if init_args is None:
        init_args = []
    elif not isinstance(init_args, (list, tuple)):
        raise RuntimeError("get_init_inputs() 返回值必须为 list/tuple 或 None")

    # 3. 获取 Model
    if not hasattr(mod, "Model"):
        raise RuntimeError(f"{task_py_path} 中未找到 class Model")
    ModelClass = mod.Model

    # 4. 初始化模型
    model = ModelClass(*init_args)
    model.eval()

    # 是否转 GPU
    if device is not None:
        model = model.to(device)

    # 5. 获取推理输入
    if not hasattr(mod, "get_inputs"):
        raise RuntimeError(f"{task_py_path} 中未找到 get_inputs()")
    inputs = mod.get_inputs()
    if not isinstance(inputs, (list, tuple)):
        raise RuntimeError("get_inputs() 返回值必须是 list/tuple")

    # 重新生成小输入
    small_inputs = _shrink_inputs_by_shape(list(inputs), device=device, seed=123, generator=generator)

    # 6. 执行模型
    with torch.no_grad():
        outputs = model(*small_inputs)

    return model, small_inputs, outputs

class TestMaker(AgentBase):
    def __init__(self, 
                 op_name: str,
                 task_desc: str,
                 dsl: str,
                 framework: str,
                 backend: str,
                 arch: str = "",
                 workflow_config_path: str = None,
                 config: dict = None):
        self.op_name = op_name
        self.task_desc = remove_copyright_from_text(task_desc)
        self.dsl = dsl
        self.framework = framework
        self.backend = backend
        self.arch = arch
        self.workflow_config_path = workflow_config_path
        self.config = config
        self.codegen_step_count = 0
        self.api_step_count = 0

        if config:
            self.model_config = config.get("agent_model_config", {})
            self.database_config = config.get("database_config", {})
        else:
            raise ValueError("config is required for MyAgent")

        context = {
            "dsl": dsl,
            "op_name": op_name,
            "framework": framework,
            "backend": backend,
            "arch": arch,
            "task_desc": task_desc,
        }
        super().__init__(context=context, config=config)

        self.model_config = config.get("agent_model_config", {})

        # 加载模板（相对prompts目录）
        self.testmaker_prompt: PromptTemplate = self.load_template("testmaker/testgen.j2")
        ParserFactory.register_parser(
            "python_code_parser",
            {
                "output_fields": {
                    "python_test_generator_code": {
                        "field_type": "str",
                        "mandatory": True,
                        "field_description": "生成的最小测试输入生成函数实现"
                    }
                }
            }
        )



    async def run(self, task_info: dict, device: Optional[int] = None) -> Tuple[str, str, str]:
        """执行代码生成

        Args:
            task_info: 任务信息字典，包含当前所有代码和状态
            device: 设备ID（来自 DevicePool），如 0、1。用于选择运行最小模型/输入生成时的设备。

        Returns:
            Tuple[str, str, str]: 生成的代码、提示信息和推理过程
        """
        try:
            #Langchain's get_format_instructions
            # api_parser = ParserFactory.get_api_parser()
            # format_api_instructions = api_parser.get_format_instructions()
            python_code_parser = ParserFactory.get_parser("python_code_parser")
            format_instructions = python_code_parser.get_format_instructions()

            input_data = {
                "task_desc": self.context.get("task_desc", ""),
                "extra": task_info.get("extra", ""),
                "code_file_content": self.task_desc,
                "format_instructions": format_instructions
            }
        
            # 返回一个json, 内部包含
            testmaker_res, testmaker_prompt, testmaker_reasoning  = await self.run_llm(self.testmaker_prompt, input_data, self.model_config.get("testmaker", "default"))
            # =================== for test ====================
            # testmaker_res = """
            # {
            #     "get_input_mini_code": "def get_input_mini():\\n    batch_size_mini = 1\\n    dim_mini = 2\\n    x = torch.linspace(-1, 1, steps=batch_size_mini * dim_mini).reshape(batch_size_mini, dim_mini)\\n    return [x]"
            # }
            # """
            # testmaker_prompt, testmaker_reasoning = "", ""

            json_res = json.loads(testmaker_res)    # convert str to json
            weights_dict = None
            testmaker_path = ""
            if json_res['get_input_mini_code']:
                testmaker_path = "input_mini"
                tmp_code_file_content = self.task_desc + "\n\n" + json_res['get_input_mini_code']
                spec = importlib.util.spec_from_loader('input_mini_module', loader=None)
                module = importlib.util.module_from_spec(spec)
                exec(tmp_code_file_content, module.__dict__)
                input_mini = module.get_input_mini()
                ModelClass = module.Model
                init_args = module.get_init_inputs()
                model = ModelClass(*init_args)
            # elif json_res['key_index_of_inouts']:
            #     spec = importlib.util.spec_from_loader("ori_module", loader=None)
            #     module = importlib.util.module_from_spec(spec)
            #     exec(json_res['key_index_of_inouts'], module.__dict__)
            #     input_mini = module.get_inputs()
            #     ModelClass = module.Model
            #     init_args = module.get_init_inputs()
            #     model = ModelClass(*init_args)
            elif json_res['minimized_code']:
                testmaker_path = "model_mini"
                spec = importlib.util.spec_from_loader('minimized_code', loader=None)
                module = importlib.util.module_from_spec(spec)
                exec(json_res['minimized_code'], module.__dict__)
                input_mini = module.get_inputs()
                ModelClass = module.Model
                init_args = module.get_init_inputs()
                model = ModelClass(*init_args)
                weights_dict = {}
                for name, param in model.named_parameters():
                    weights_dict[name] = param.detach().cpu().tolist()
                if len(weights_dict) == 0:
                    weights_dict = None
            else:
                raise RuntimeError("testmaker agent 未返回任何有效结果")

            model.eval()
            # 将设备ID转换为具体的 torch 设备（与 verifier 一致地使用 device_id）
            target_device = None
            try:
                if device is not None:
                    if self.backend in ["cuda", "gpu", "nvidia"] and torch.cuda.is_available():
                        target_device = torch.device(f"cuda:{int(device)}")
                    elif self.backend in ["ascend", "npu"] and hasattr(torch, "npu"):
                        # 对于 Ascend/NPU，若使用 torch-npu，设备形如 npu:0
                        target_device = torch.device(f"npu:{int(device)}")
                    else:
                        target_device = torch.device("cpu")
            except Exception:
                target_device = torch.device("cpu")

            # 安全地迁移到设备
            if target_device is not None:
                try:
                    model = model.to(target_device)
                    if isinstance(input_mini, (list, tuple)):
                        input_mini = [x.to(target_device) if isinstance(x, torch.Tensor) else x for x in input_mini]
                except Exception:
                    # 设备不可用时，退回CPU
                    target_device = torch.device("cpu")
                    model = model.to(target_device)
                    if isinstance(input_mini, (list, tuple)):
                        input_mini = [x.to(target_device) if isinstance(x, torch.Tensor) else x for x in input_mini]

            # 运行一次以校验最小输入可执行
            with torch.no_grad():
                outputs = model(*input_mini)
            try:
                out_shape_str = outputs.shape if hasattr(outputs, 'shape') else 'N/A'
            except Exception:
                out_shape_str = 'N/A'
            print(f"using minimized inputs, in shapes: {[x.shape for x in input_mini if hasattr(x, 'shape')]}, out shapes: {out_shape_str}")
                
            # 统一输出字段，确保包含一个名称中包含 "code" 的主字段，便于通用解析器逻辑保存为 testmaker_code
            # 优先选择 LLM 返回的可执行代码片段；若无则置空字符串
            # primary_code = (
            #     json_res.get('get_input_mini_code')
            #     or json_res.get('key_index_of_inouts')
            #     or json_res.get('minimized_model')
            #     or ""
            # )
            # to show, convert [x] to x when only one input
            if len(input_mini) == 1 :
                decomposed_input = input_mini[0].tolist()
            else:
                decomposed_input = [x.tolist() for x in input_mini]
            handled_res = {
                # 'code': primary_code,  # 关键字段：用于 ResultProcessor 保存为 task_info['testmaker_code']
                'ori_testmaker_response': testmaker_res,
                'testmaker_path': testmaker_path,
                'embedding_inputs': decomposed_input,
                'model_weights': weights_dict,
                'embedding_outputs': outputs.tolist()
            }

            return json.dumps(handled_res), testmaker_prompt, testmaker_reasoning
        except Exception as e:
            logger.error(f"Exception in testmaker.run: {type(e).__name__}: {e}")
            raise

if __name__ == "__main__":
    op_name = '36_RMSNorm_'
    # model, inputs, outputs = run_kernelbench_model(op_name, "cuda")
    op_name = '54_conv_standard_3D__square_input__square_kernel'
    # generator judged by kernel type:
    # matmul kernel: eye
    # dim reduction kernel: one-hot/ 
    model, inputs, outputs = run_minimal_kernelbench_model(op_name, "cuda", generator=small_int_generator)

    for x in inputs:
        shapes = [i for i in x.shape]
    print(shapes)
    print("输入形状:", [x.shape for x in inputs])
    print(inputs)
    print("输出类型:", type(outputs))
    print(outputs)
    if hasattr(outputs, "shape"):
        print("输出形状:", outputs.shape)