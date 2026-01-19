# UnifiedSketch 设计 251228

## 1. 目标与原则

用最小 DSL 表达算子设计意图，便于 LLM 理解和 Coder 实现。

### 原则

- **极简原语**：只有少数核心操作（alloc/load/store/compute/...）
- **统一语法**：所有操作都是函数调用风格，无语法差异
- **标准控制流**：使用 Python for/range 语法，不发明新语法
- **hint 分离**：复杂优化用 hint 表达，不影响主逻辑清晰性
- **硬件抽象**：草图描述逻辑并行度与访存模式，不涉及具体的寄存器分配、同步指令

## 2. 核心语法元素

### 2.1 结构声明

```python
sketch <op_name> {
  symbols: M, N, K;                    # 符号变量声明
  tensors: A[M, K]: f16; B[K, N]: f16; C[M, N]: f32;  # 张量声明
  constexpr: m0, k0, n0
}
```

### 2.2 内存管理 IR：alloc 操作

```python
tile = alloc([shape], llm_hint=["存储要求", "用途说明", "性能要求"])
```

**hint 设计原则：语义化描述**

- **存储要求**: `"fastest"` (Register), `"fast"` (Shared/L1), `"medium"` (L2), `"slow"` (Global)
- **用途说明**: `"accumulator"`, `"input_cache"`, `"output_buffer"`, `"temp_workspace"`
- **初始化**: `"init_zero"`, `"init_neg_inf"`, `"no_init"`

### 2.3 数据流 IR：Load/Store 操作

```python
load(tensor[slice] -> tile, mask=mask_tile, llm_hint=["..."])
store(tile -> tensor[slice], mask=mask_tile, llm_hint=["..."])
```

**关键 llm_hint**:
- **Parallel**: `"cooperative_load"`, `"cooperative_store"`
- **Vectorized**: `"vectorized"` (连续), `"strided"`, `"broadcast"`
- **Boundary**: `"mask_boundary"`, `"assume_aligned"`
- **Access**: `"block_ptr"` (强制使用块指针优化，需满足线性索引条件), `"atomic_add"` (原子加)

### 2.4 计算 IR

- **基础**: `add`, `sub`, `mul`, `div`, `exp`, `log`, `sqrt`, `max`, `min`, `clamp`, `where`, `abs`
- **线性代数**: `gemm(a,b,dst)`, `dot`, `outer_product`
- **归约**: `reduce_sum(src, axis)`, `reduce_max`, `reduce_min`, `reduce_argmax`
- **扫描**: `local_cumsum`, `local_cumprod`
- **复合**: `softmax`, `relu`, `gelu`, `sigmoid`

### 2.5 @llm_hint 装饰器

```python
@llm_hint("parallel", "grididx.x/y/z")  # 映射到 Grid 维度
@llm_hint("pipeline")                   # 启用软件流水线
@llm_hint("vectorize")                  # 强制向量化
@llm_hint("unroll")                     # 循环展开
```


## 3. 常见 IR 示例

### 3.1 For 循环结构表达

```python
# GPU 风格：Grid 并行
@llm_hint("parallel", "grididx.x")
for i in range(0, M, BM):
    @llm_hint("parallel", "grididx.y") 
    for j in range(0, N, BN):               
        a_tile = alloc([BM, BK], llm_hint=["fast", "input_cache"])
        b_tile = alloc([BK, BN], llm_hint=["fast", "input_cache"])
        
        @llm_hint("pipeline")
        for k in range(0, K, BK):
            load(...)
            gemm(...)
```

## 4. 元提示优化 (Meta Prompt Optimization)

元提示优化被建模为对 UnifiedSketch IR 的一系列 Pass。

### 4.1 元优化 Pass 清单 (Optimization Passes)

元提示优化是将**硬件无关的 UnifiedSketch** 映射到**特定硬件代码**的变换过程。主要包含以下 Pass：

1.  **Hierarchy Binding (并行层级映射)**: 将抽象循环 `for` 绑定到物理并行层级 (Grid/Block/Warp/Lane)。
2.  **Memory Scope Promotion (存储层级提升)**: 将数据 `alloc` 从 Global 提升至 Shared (L1) 或 Register 以复用数据。
3.  **Tiling (分块策略)**: 将大循环切分为适合 Cache 大小的 Tile，并构造 Block/Thread 级并行。
4.  **Loop Transformation (循环变换)**:
    -   **Reordering**: 交换循环次序 (Permutation) 以优化连续访存 (Coalescing)。
    -   **Fusion**: 融合相邻 Elementwise 循环以减少 Launch 开销和访存。
    -   **Unrolling**: 显式展开循环以增加指令级并行度 (ILP)。
    -   **Flattening (展平)**: 将多层嵌套循环合并为 1D 线性循环，消除维度碎片，最大化 Parallel Occupancy（适用于 Pointwise/Reduction）。
5.  **Dataflow & Instruction Behavior (数据流与指令行为)**:
    -   **Hoisting (代码外提)**: 将计算无关的地址计算或常量判断移出内层循环 (Loop Invariant Code Motion)。
    -   **Synchronization (同步插入)**: 在 Shared Memory 读写存在 RAW 依赖时显式插入 Barrier (`__syncthreads`)。
    -   **Precision Adaptation (精度适配)**: 处理混合精度场景（如 f16 load -> f32 accum -> f16 store）。
6.  **Intrinsic Mapping (指令映射)**: 将标准算子 (`gemm`, `reduce`) 映射到硬件专用指令 (TensorCore MMA, SIMD, AtomicAdd)。
7.  **Software Pipelining (软流水)**: 插入 Prefetch 指令构造 `Load -> Compute` 并行流水线，掩盖访存延迟。
8.  **Layout Transformation (布局变换)**: Padding (消除 Bank Conflict) 或 Permutation (NHWC <-> NCHW)。
9.  **Parallel Strategy (并行策略变换)**:
    -   **Split-K**: 将归约维度 K 切分为多个分片分配给不同 Block，利用原子操作汇总，解决 Wave 浪费问题。
    -   **Batching**: 将多个小任务打包处理（如 Gemm Batched）以增加 Occupancy。

### 4.2 典型 Pass 示例

#### 1. Hierarchy & Tiling 示例
```python
# Pass: Tiling(BM, BN) + Binding(Grid)
@llm_hint("parallel", "grididx.y")
for i in range(0, M, BM):
    @llm_hint("parallel", "grididx.x")
    for j in range(0, N, BN):
        ...
```

#### 2. Memory Promotion 示例
```python
# Pass: Global -> Shared/Register Promotion
tile_g = load(A[...])
tile_s = alloc(..., llm_hint=["fast"]) # Shared
store(tile_g -> tile_s)
# Barrier Injection for synchronization
```

#### 3. Loop Reordering 示例
```python
# Before (Stride Access)
for i in range(M): for j in range(N): x = A[j, i]
# Pass: Reorder for Coalescing
for j in range(N): for i in range(M): x = A[j, i]
```

#### 4. Intrinsic Mapping 示例
```python
# Before: Naive Loop
for k in range(K): C += A[k] * B[k]
# Pass: Map to TensorCore (e.g., mma.sync)
# gemm() call mapped to wmma intriniscs
gemm(frag_a, frag_b, frag_c) 
```

#### 5. Software Pipeline 示例
```python
# Pass: Prefetch Injection
load(Next_Tile)
compute(Current_Tile)
```

#### 6. Split-K Optimization 示例
```python
# Before: Serial Reduction (Low Parallelism for small M/N large K)
for k in range(K): acc += A[k] * B[k]
# Pass: Parallel Reduction (Split-K)
@llm_hint("parallel", "grididx.z") # Map K-parts to Grid Z
for k_part in range(k_start, k_end, BK):
    local_acc += ...
    # Global Atomic Reduction
    store(local_acc -> Final_Output, llm_hint=["atomic_add"])
```

#### 7. Loop Flattening 示例
```python
# Before: Nested 3D Loop (Small dims limit parallelism)
for b in range(B): for h in range(H): for w in range(W): 
    out[b,h,w] = relu(in[b,h,w])
# Pass: Flattening -> 1D Linear Kernel
# Total N = B*H*W, fully maximize Grid Z/Y/X
@llm_hint("parallel", "grididx.x")
for i in range(0, N, BN):
   # Coalesced Linear Access
   val = load(in[i:i+BN])
   store(relu(val) -> out[i:i+BN])
```

### 4.3 设计思维流与自检 (Design Thinking Protocol)

1.  **并行度分析**：`Total_Blocks` 是否充足？若不足，**必须**使用 `Split-K` 或 `Batch Parallelism`。
2.  **访存对齐**：识别 `Stride=1` 维度，**强制**内层循环沿该维度向量化。
3.  **计算下移 (Hoisting)**：循环不变量（如 Norm 均值、Conv 坐标判定）必须移出最内层循环。
4.  **展平决策 (Flattening)**：逐元素或全局归约算子，**强制**将多维张量展平为 1D 线性索引。
5.  **物理布局校验**：
    - 3D (NCDHW): W 最快。解码顺序 `w <- h <- d <- n`。
    - 2D (NCHW): W 最快。解码顺序 `w <- h <- n`。
6.  **物理拓扑隔离 (Conv)**：**绝对严禁**将空间维度 (R, S, T) 展平进 `GEMM_K`。`GEMM_K` 必须仅包含通道维度。
7.  **自检清单**：
    - [ ] `GEMM_K` 是否仅包含通道维度？
    - [ ] 空间窗口循环是否嵌套在通道循环 (BK) 之外？
    - [ ] 是否消除了 BK 循环内的 `%` 和 `//` 运算？
    - [ ] 访存是否标记了 `llm_hint=['block_ptr']` (针对线性索引)？

## 5. 硬件 specific 优化

- **GPU (NVIDIA/AMD)**:
  - `llm_hint="fast"` -> Shared Memory
  - `llm_hint="fastest"` -> Register File
  - Grid Mapping: 映射 `grididx.x/y/z` 到 Grid Block。
- **NPU (Ascend)**:
  - `llm_hint="fast"` -> L1 Buffer / Unified Buffer
  - `llm_hint="coreidx"` -> AI Core 并行

## 6. 性能禁忌 (Performance Anti-Patterns)

1.  **内循环标量化**：禁止单标量 Load，必须 `arange` 向量化。
2.  **K 维展平 (K-Flattening)**：**绝对严禁**将 R, S, T 合并入 GEMM_K。
3.  **GEMM_K 定义错误**：GEMM_K 必须仅包含通道维度。
4.  **Alloc 滥用**：严禁分配大面积工作空间用于 Input Cache，导致 Register Spilling。
5.  **3D 索引还原错误**：必须遵循 W (最快) -> H -> D -> N 的顺序。
6.  **标量线程映射**：严禁 One-thread-per-pixel 映射，必须 Block-based。
7.  **手书归约逻辑**：严禁手写 Tree-reduction，使用 `reduce_sum`。
8.  **多步扫描同步**：严禁使用全局多步同步扫描。
9.  **Tile 内标量循环**：严禁 `for i in range(BLOCK_SIZE)`。
10. **二阶段串行归约**：**严禁** "Partial Sum -> Global Sum" 的二阶段写法，使用 Atomic。
11. **手动线程 ID**：**严禁**使用 `thread_idx` 或 Grid-Stride Loop。
12. **手动 Shared Memory**：**严禁**手动管理 SMEM 进行归约。
13. **过度分配中间缓冲**：**严禁**为每个算术步骤 alloc tile，使用 Math Fusion (直接表达式)。
14. **损失函数多 Pass**：**严禁**将 Loss 拆分为“写回 + 求和”，必须 Atomic。
15. **过度 Tiling (Over-Tiling)**：在 Grid 并行度充足时 (如 B*C 很大)，严禁在 Block 内部对 Batch/Instance 维度再次 Tiling (如 `for bc in range(0, B*C, BC)`)，应直接映射 `range(0, B*C, 1)` 到 Grid (Step=1)。
