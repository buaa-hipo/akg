# Triton 编程基础教程

本文档介绍 Triton 的核心概念和标准编程模式，通过详细示例帮助理解如何构建内核。

## 1. 核心概念

### 内核 (Kernel)
- **定义**: 使用 `@triton.jit` 装饰的 Python 函数，编译后在硬件加速器上并行执行
- **特点**: 每个内核实例处理数据的一个子集，通过程序 ID 区分

### 网格 (Grid) 与块 (Block)
- **网格**: 内核启动时的并行维度配置，如 `(num_blocks_x, num_blocks_y)`
- **块**: 每个程序实例处理的数据块大小，如 `BLOCK_SIZE = 1024`
- **关系**: `grid_size = ceil(total_elements / block_size)`

### 内存层次
- **全局内存**: 主内存，所有程序可访问，延迟高
- **共享内存**: 块内共享，延迟低，容量有限
- **寄存器**: 每个线程私有，最快访问

## 2. 标准内核结构

所有 Triton 内核都遵循相同的五步结构模式：

```python
@triton.jit
def standard_kernel(
    output_ptr, input_ptr, n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 获取程序 ID 和计算偏移
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 2. 创建边界掩码
    mask = offsets < n_elements
    
    # 3. 加载数据
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 4. 执行计算
    result = compute_function(data)
    
    # 5. 存储结果
    tl.store(output_ptr + offsets, result, mask=mask)
```

### 内核启动方式
```python
def launch_kernel(input_tensor, output_tensor):
    BLOCK_SIZE = 1024  
    grid = (triton.cdiv(input_tensor.numel(), BLOCK_SIZE),)
    
    kernel[grid](
        output_tensor, input_tensor, input_tensor.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
```

## 4. 边界处理示例

### 使用 mask 处理边界
```python
# 基本边界检查
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
data = tl.load(ptr + offsets, mask=mask, other=0.0)
```

### 条件计算
```python
# 使用 tl.where 进行条件选择
result = tl.where(condition, true_value, false_value)

# 复杂条件的掩码组合
valid_mask = (offsets < n_elements) & (offsets >= 0)
data = tl.load(ptr + offsets, mask=valid_mask, other=0.0)
```
