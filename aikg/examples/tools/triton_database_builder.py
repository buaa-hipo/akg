from ai_kernel_generator.database.build_database import build_coder_database
import asyncio
from pathlib import Path

if __name__ == '__main__':
    import asyncio
    triton_file_dir = Path('/mnt/lustre-client/zhangzizheng/AIKG/TritonBench/data/TritonBench_G_v1')
    step = 3
    asyncio.run(build_coder_database(triton_file_dir, step))
