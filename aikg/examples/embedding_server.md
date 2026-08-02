可以独立出来，而且我已经按“最小侵入”的方式接好了：

改动在这几处：
- [model_service.py](/home/zhangzizheng/AIKG/akg/aikg/python/ai_kernel_generator/server/model_service.py): 新增 FastAPI 服务，提供 cross-encoder 批量相似度和 `CoderDatabase.samples` 查询。
- [model_service_client.py](/home/zhangzizheng/AIKG/akg/aikg/python/ai_kernel_generator/client/model_service_client.py): 新增远程 client，接口形状对齐原来的 `CrossEncoderSimilarity` / `CoderDatabase`。
- [shared_resources.py](/home/zhangzizheng/AIKG/akg/aikg/python/ai_kernel_generator/core/agent/shared_resources.py): 配了服务地址就返回远程对象，否则保持本地加载。
- [filter.py](/home/zhangzizheng/AIKG/akg/aikg/python/ai_kernel_generator/core/agent/filter.py): 相似度改成批量计算，并加了 batch size，避免一次塞爆显存。

启动服务端示例：

```bash
cd /home/zhangzizheng/AIKG/akg/aikg
export PYTHONPATH=$PWD/python
export AIKG_MODEL_DEVICE=cuda
export AIKG_CROSS_ENCODER_BATCH_SIZE=8
export AIKG_CODER_DATABASE_MAX_CONCURRENCY=1

python3 -m ai_kernel_generator.server.model_service \
  --host 0.0.0.0 \
  --port 8010 \
  --config-path config/vllm_triton_cuda_evolve_config.yaml \
  --database-path /home/zhangzizheng/AIKG/triton_database/nvidia/TritonBench_G_v1
```

客户端脚本只需要：

```bash
export AIKG_MODEL_SERVICE_URL=http://服务机器IP:8010
```

也支持拆开配：

```bash
export AIKG_CROSS_ENCODER_SERVICE_URL=http://服务机器IP:8010
export AIKG_CODER_DATABASE_SERVICE_URL=http://服务机器IP:8010
```

我跑了 `python3 -m py_compile`，四个文件静态检查通过；远程 client 也能导入。当前这个 shell 里缺 `fastapi`，所以我没有实际拉起服务端做 HTTP 端到端验证，安装/使用仓库 requirements 后即可启动。
