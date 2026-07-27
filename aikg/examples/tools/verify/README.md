## 这是一个自动验证加速比的工具

### 单个进化算子集验证

`triton_speedup_eval.py`工具用来验证单个生成算子进化文件夹

`python triton_speedup_eval.py <path-to>/evolve_database/level1/2_xxx --device 0 --limit 3`

表示验证`<path-to>/evolve_database/level1/2_xxx`文件夹下的加速比最高的3个算子，使用设备0。

`--limit`参数表示验证算子实例数量，最高的`--limit`个算子会被验证，设0则表示全量验证。

`--torch_compile`表示baseline是否使用torch.compile编译，默认不使用。

`--warmup --runs`表示验证时的预热轮数和运行时间，默认预热时间50，运行时间为300。

### 批量进化算子集验证

`batch_eval.py`工具用来批量验证多个生成算子进化文件夹

`python batch_eval.py --level level1 --range 2,12 `表示验证level1下\[2,12\]号的算子进化文件夹，其他默认参数同上。

**--quick_test模式**：实际测pytorch baseline，然后取本地的最快算子做加速比计算。