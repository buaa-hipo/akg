# Triton 开发优化与避坑指南

## 1. 性能优化 (Performance)
- **块大小 (BLOCK_SIZE)**: 优先取 2 的幂 (256~1024)；Ascend 后端可取 16 的倍数。
- **内存访问**: 2D 数据首选 `tl.make_block_ptr`；确保 stride 正确以实现内存合并 (Coalescing)。
- **切分策略**: 复杂算子拆分为多个简单 Kernel；单次网格启动 (Grid) 不超过 65535。

## 2. 数值稳定性 (Stability)
- **防溢出**: 计算 Softmax/Exp 前先减去 `tl.max(x)`。
- **防止非法输入**: `tl.sqrt` 前确保数值 $\ge 0$ (使用 `tl.maximum(x, eps)`)。
- **精度提升**: 关键累加步骤使用 `float32`；必要时显式转换类型。

## 3. 核心禁忌与替代 (DOs & DON'Ts)
- **语法禁令**: 严禁 `return`, `break`, `continue`, `lambda`，`data_ptr()`。
- **控制流**: 使用 `mask` 代替逻辑判断；Device 端尽量消除 `if`。
- **constexpr**: 仅限 Kernel 签名中的编译时常量，Host 端不可调用。
- **Ascend 特化**: 避免在 `tl.load/store` 中使用 `tl.where` 进行动态偏移计算，改用 Host 端静态逻辑。

## 4. 调试排查清单 (Troubleshooting)
- [ ] **内存**: 是否都有 `mask` 或 `boundary_check`？Stride 是否匹配张量布局？
- [ ] **控制流**: 是否误用了禁止的 Python 语法？
- [ ] **初始化**: Host 端初始化是否在 CPU 完成且精度与输入一致？
- [ ] **原子操作**: 涉及并发写入时是否使用了 `tl.atomic_add/max`？

## 5. 常见错误速查
| 错误类型 | 症状 | 核心对策 |
| :--- | :--- | :--- |
| **越界/非法** | 结果 NaN 或崩溃 | 补全 mask / 检查边界条件 |
| **编译失败** | 提示控制流错误 | 移除 return，改用指针偏移或 mask |
| **精度偏移** | 计算结果微小误差 | 提升累加器类型至 fp32 |
| **数据重叠** | 结果随机不确定 | 添加原子操作 `atomic_*` |

## 6. 代码风格
- **语义化**: 变量名需反映物理含义 (如 `m_offsets`, `k_ptr`)。
- **无 if 化**: Device 代码通过 mask 实现逻辑分支。
- **显式初始化**: Tensor 先在 CPU 创建并在 Host 端统一步长，再移至 Device。
- 直接把张量传进 kernel，不要 `.data_ptr()`。

