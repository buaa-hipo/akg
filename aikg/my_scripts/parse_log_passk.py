import re
import sys
from pathlib import Path

def parse_log_file(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 按任务分割
    task_blocks = re.split(r"🚀 正在处理任务:", content)
    results = {"success": {}, "failed": {}}

    for block in task_blocks[1:]:
        # 提取任务名
        task_name_match = re.match(r"\s*(\S+)", block)
        if not task_name_match:
            continue
        task_name = task_name_match.group(1).strip()

        # 检测是否成功
        success = bool(re.search(r"(✓ SUCCESS|成功任务数:\s*[1-9])", block))
        failed = bool(re.search(r"(✗ FAILED|成功任务数:\s*0)", block))

        # 提取核心信息
        info = {}
        for key in ["实现类型", "框架", "后端", "架构"]:
            m = re.search(fr"{key}:\s*(\S+)", block)
            if m:
                info[key] = m.group(1)

        # 提取错误日志
        if failed:
            error_lines = []
            for line in block.splitlines():
                if any(kw in line for kw in ["ERROR", "Exception", "APITimeoutError"]):
                    error_lines.append(line.strip())
            results["failed"][task_name] = {
                "info": info,
                "errors": error_lines or ["无具体错误信息"]
            }

        elif success:
            results["success"][task_name] = info

    return results


def print_summary(results):
    print("=" * 80)
    print("任务结果汇总")
    print("=" * 80)
    print("✅ 成功任务:")
    for name in results["success"]:
        print(f"  - {name}")
    print()
    print("❌ 失败任务:")
    for name in results["failed"]:
        print(f"  - {name}")
    print("=" * 80)
    print("\n详细错误日志:")
    for name, data in results["failed"].items():
        print(f"\n🧩 任务: {name}")
        for line in data["errors"]:
            print(f"  {line}")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python parse_log.py <log_file>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"错误: 找不到文件 {log_path}")
        sys.exit(1)

    results = parse_log_file(log_path)
    print_summary(results)
