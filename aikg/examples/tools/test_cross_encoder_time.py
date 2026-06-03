import time
import gc
import torch
import multiprocessing as mp
from ai_kernel_generator.core.agent.filter import CrossEncoderSimilarity


text = """
sketch Matmul_Swish_Sum_GroupNorm {\n  symbols: B, K, C, G;\n  tensors: x[B, K]: f16; weight[K, C]: f16; linear_bias[C]: f32; extra_bias[C]: f32; gn_weight[C]: f32; gn_bias[C]: f32; output[B, C]: f32;\n  constexpr: BLOCK_B=4, BLOCK_K=64, CPG=C//G, EPS=1e-5;\n\n  @llm_hint(\"parallel\", \"grididx.x\")\n  for b_start in range(0, B, BLOCK_B):\n    offs_b = arange(0, BLOCK_B) + b_start\n    mask_b = offs_b < B\n    @llm_hint(\"parallel\", \"grididx.y\")\n    for c_start in range(0, C, CPG):\n      offs_c = arange(0, CPG) + c_start\n      mask_c = offs_c < C\n\n      # Shared memory for weight tile\n      w_shared = alloc([BLOCK_K, CPG], llm_hint=[\"fast\", \"input_cache\"])\n      a_tile = alloc([BLOCK_B, BLOCK_K], llm_hint=[\"fast\", \"input_cache\"])\n      acc = alloc([BLOCK_B, CPG], llm_hint=[\"fast\", \"accumulator\", \"init_zero\"])\n\n      @llm_hint(\"pipeline\")\n      for k_start in range(0, K, BLOCK_K):\n        offs_k = arange(0, BLOCK_K) + k_start\n        mask_k = offs_k < K\n\n        # Cooperative load weight tile to shared memory\n        mask_w_load = mask_k[:, None] & mask_c[None, :]\n        load(weight[offs_k, offs_c] -> w_shared, mask=mask_w_load, llm_hint=[\"cooperative_load\", \"vectorized\", \"mask_boundary\"])\n\n        # Load x tile (from global)\n        mask_a_load = mask_b[:, None] & mask_k[None, :]\n        load(x[offs_b, offs_k] -> a_tile, mask=mask_a_load, llm_hint=[\"vectorized\", \"mask_boundary\"])\n\n        # Synchronize before using shared memory\n        @llm_hint(\"syncthreads\")\n\n        # Compute GEMM using w_shared\n        gemm(a_tile, w_shared, acc)\n\n        # Optional syncthreads before next iteration (not needed if not overwriting w_shared)\n        @llm_hint(\"syncthreads\")\n\n      # linear bias\n      bias_tile = alloc([CPG], llm_hint=[\"fast\", \"input_cache\"])\n      load(linear_bias[offs_c] -> bias_tile, mask=mask_c, llm_hint=[\"vectorized\"])\n      acc = acc + bias_tile[None, :]\n\n      # Swish\n      acc = sigmoid(acc) * acc\n\n      # extra bias\n      extra_tile = alloc([CPG], llm_hint=[\"fast\", \"input_cache\"])\n      load(extra_bias[offs_c] -> extra_tile, mask=mask_c, llm_hint=[\"vectorized\"])\n      acc = acc + extra_tile[None, :]\n\n      # GroupNorm\n      valid_c = reduce_sum(mask_c, axis=0)\n      mean = reduce_sum(acc * mask_c[None, :], axis=1) / valid_c\n      diff = acc - mean[:, None]\n      var = reduce_sum(diff * diff * mask_c[None, :], axis=1) / valid_c\n      rstd = 1.0 / sqrt(var + EPS)\n      acc = diff * rstd[:, None]\n\n      # gn affine\n      gn_w_tile = alloc([CPG], llm_hint=[\"fast\", \"input_cache\"])\n      load(gn_weight[offs_c] -> gn_w_tile, mask=mask_c, llm_hint=[\"vectorized\"])\n      gn_b_tile = alloc([CPG], llm_hint=[\"fast\", \"input_cache\"])\n      load(gn_bias[offs_c] -> gn_b_tile, mask=mask_c, llm_hint=[\"vectorized\"])\n      acc = acc * gn_w_tile[None, :] + gn_b_tile[None, :]\n\n      # store\n      mask_out = mask_b[:, None] & mask_c[None, :]\n      store(acc -> output[offs_b, offs_c], mask=mask_out, llm_hint=[\"vectorized\"])\n\n  @llm_hint(\"available_tiling\")\n  available_tiling:\n    BLOCK_B=4, BLOCK_K=64\n    BLOCK_B=8, BLOCK_K=64\n    BLOCK_B=2, BLOCK_K=128\n}
"""

# 子进程里真正跑的函数
def worker_func(queue):
    # 所有 GPU 相关操作都在子进程里
    start = time.time()
    cross_encoder = CrossEncoderSimilarity()
    print(f"加载 CrossEncoder 时间 {time.time() - start:.2f}s\n")
    exist_code_ir = [text for _ in range(20)]
    designer_ir = text

    print("子进程开始计算...")
    start = time.time()
    ir_similarity_scores = [
        cross_encoder.calculate_similarity(designer_ir, exist_ir)
        for exist_ir in exist_code_ir
    ]
    end = time.time()
    print(f"子进程耗时: {end - start:.4f} s")

    # 最后再做一次清理
    del cross_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # 结果传回主进程
    queue.put(ir_similarity_scores)

def main():
    # 用 Queue 把子进程结果拿回主进程
    queue = mp.Queue()

    # 创建子进程
    p = mp.Process(target=worker_func, args=(queue,))
    p.start()

    # 等待子进程跑完
    p.join()

    # 获取结果
    scores = queue.get()
    print("主进程拿到结果：", scores)

    # 子进程已经销毁，GPU 完全释放
    print("子进程已销毁，GPU 显存已释放")

if __name__ == "__main__":
    main()
    # 这里再进 pdb，主进程无任何 GPU 占用
    import pdb; pdb.set_trace()
