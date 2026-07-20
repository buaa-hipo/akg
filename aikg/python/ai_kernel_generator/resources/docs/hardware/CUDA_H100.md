# GPU硬件说明 - NVIDIA H100 PCIe

## memory_system
├── gm (GlobalMemory/DeviceMemory):
|    ├── size: 80GB HBM2e (本机H100 PCIe型号；H100 SXM型号通常为80GB HBM3)
|    ├── memory_bus_width: 5120-bit
|    └── L2_cache: 50MB
├── shared_memory (per SM): // H100 PCIe有114个SM流多处理器，Compute Capability 9.0
|    └── shared_memory_block:
|         ├── 大小: 228KB (每SM最大shared memory容量)
|         ├── 单block最大可用: 227KB (动态shared memory超过48KB需要显式opt-in)
|         ├── from_data: ["gm"]
|         └── to_data: ["gm"]
└── registers (per SM):
     ├── 数量: 65536个32位寄存器
     ├── 最大线程数: 2048 (每SM，64个warp)
     ├── 最大blocks: 32 (每SM)
     └── 最大寄存器数: 255 (每thread)

## compute_system
├── cuda_cores: // CUDA计算核心
|    ├── 数量: 14592个 (128个/SM × 114个SM)
|    ├── 约束: 每个warp 32个线程同步执行
|    ├── 约束: 分支发散会降低效率
|    └── 功能: 标准浮点运算、整数运算、比较运算
└── tensor_cores: // 第四代Tensor Core (Hopper架构)
     ├── 数量: 456个 (4个/SM × 114个SM)
     ├── 支持精度: FP16, BF16, TF32, FP64, FP8, INT8
     ├── 约束: 需要特定的矩阵尺寸对齐，优先使用Tensor Core友好的tile形状
     └── 功能: 矩阵乘法累加(GEMM)操作，支持Hopper Transformer Engine/FP8工作负载

## 搬移Pipeline
- **Grid-Block-Thread层次**: Grid划分为Block，Block划分为Thread；Hopper额外支持Thread Block Cluster层次
- **内存合并访问**: 连续线程访问连续内存地址效率最高，非合并访问会降低global memory有效带宽
- **shared memory使用**: 大块动态shared memory需要通过cudaFuncSetAttribute显式opt-in；注意bank conflict和occupancy之间的权衡
- **异步搬移**: Hopper支持Tensor Memory Accelerator (TMA)，适合在global memory和shared memory之间搬移1D到5D tensor数据
- **同步约束**: Block内可同步；使用Thread Block Cluster时，cluster内可通过Distributed Shared Memory进行更细粒度协作

## 执行逻辑
全部warp并行执行，同一warp内线程SIMT同步执行。线程间同步通过__syncthreads()实现，用户需要显式管理线程块内的同步。

H100基于Hopper架构，单SM资源相较A100增加了shared memory容量，并提供TMA、Thread Block Cluster、Distributed Shared Memory、第四代Tensor Core和FP8能力。优化时需要同时关注寄存器占用、shared memory占用、warp occupancy、global memory合并访问和Tensor Core tile对齐。

前置需求：
请你思考，并理解上面的GPU系统，包括存储结构、线程层次、内存访问模式等。

