# Triton 常见错误补充

## tl.load/tl.store
### 描述
`tl.load(pointer, ...)/tl.store(pointer, ...)`等指针相关的函数，需要传入指针类型。
`x_ptr = x.data_ptr()`得到的是一个整数类型，如果直接传给`tl.load/tl.store`，将会编译错误，例如:
```python
Unsupported ptrtype <['512'],int32> in tl.load
```
### 修复方法
不要使用`data_ptr()`获取指针，直接将tensor传入triton kernel会自动获取其指针。
```
kernel[grid](x,...)
```

## 索引访问张量
### 描述
triton中不能通过`[index]`访问tensor元素。`tl.tensor`不是数组，而是 SIMD 向量 / 矢量寄存器的抽象。
### 修复方法
将`for`循环+逐元素访问，改为使用`mask`的向量化操作。

## 精度问题/输出不一致
### 描述
出现不是编译错误，而是输出不一致导致的问题。
### 修复方法
优先分析以下几个方面：
1. weight等参数tensor是否是先在host端创建，后传递到device端。
2. 参数shape是否与算子描述的相同。
3. tensor的dtype（精度）是否与算子描述的相同。
4. 关键累加步骤使用 `float32`；必要时显式转换类型。

## tl.make_block_ptr
### 描述
`tl.make_block_ptr(base: tensor, shape, strides, offsets, block_shape, order)`，没有`mask`，`other`参数。
### 修复方法
传入正确的参数。

## tl.arange
### 描述
`tl.arange(start, end)`，传入的参数需要为`constexpr`。
### 修复方法
传入参数为常量/常量表达式。

## tl.reshape
### 描述
`tl.reshape(input, *shape, can_reorder=False)`，input为输入的tensor，shape是改变后的形状，需要保证改变前后tensor的元素总数保持一致。
### 修复方法
确保正确的shape，保证reshape前后tensor元素总数保持一致。

## tl.program_id
### 描述
`tl.program_id(axis)`，triton的axis只能为0，1，2。即triton是3D启动网格。
### 修复方法
确保正确的`axis`值。

## 其他方面
- **constexpr**: 仅限 Kernel 签名中的编译时常量，Host 端不可调用。
- **Ascend 特化**: 避免在 `tl.load/store` 中使用 `tl.where` 进行动态偏移计算，改用 Host 端静态逻辑。
- **内存**: 是否都有 `mask` 或 `boundary_check`？Stride 是否匹配张量布局？
- **控制流**: 是否误用了禁止的 Python 语法？
- **初始化**: Host 端初始化是否在 CPU 完成且精度与输入一致？
- **原子操作**: 涉及并发写入时是否使用了 `tl.atomic_add/max`？

