#!/bin/bash

# 检查参数数量
if [ $# -ne 1 ]; then
    echo "Usage: $0 <level_dir/op_name_with_prefix>"
    exit 1
fi

INPUT=$1

# 提取目录部分（如 level2）
DIR=$(echo "$INPUT" | cut -d'/' -f1)

# 提取完整文件名（如 76_Gemm_Add_ReLU）
FULL_OP_NAME=$(echo "$INPUT" | cut -d'/' -f2)

# 去掉前面的数字下划线得到 op-name（如 Gemm_Add_ReLU）
# 正则表达式匹配开头的一个或多个数字，然后是下划线，将其删除
OP_NAME=$(echo "$FULL_OP_NAME" | sed 's/^[0-9]*_//')

# 创建日志目录（如果不存在）
mkdir -p "log/$DIR"

# 执行命令
nohup python run_torch_evolve_triton.py \
    --config_name "evolve_openai.yaml" \
    --op-name "$OP_NAME" \
    --task-desc "/mnt/lustre-client/zhangzizheng/AIKG/KernelBench/KernelBench/$DIR/$FULL_OP_NAME.py" \
    --evolve-database "$DIR/$FULL_OP_NAME" \
    >> "log/$DIR/$FULL_OP_NAME.log" 2>&1 &

echo "Submitted: $INPUT"
echo "Log file: log/$DIR/$FULL_OP_NAME.log"