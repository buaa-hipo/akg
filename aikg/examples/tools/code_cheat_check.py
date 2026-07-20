# -*- coding: utf-8 -*-
"""
算子代码作弊检查工具
====================

使用 MAKG 项目的 CodeChecker 静态分析 Triton 算子代码，检测是否存在以下作弊行为：
  1. 直接调用 torch.nn.*/torch.nn.functional.* 等高层 API 替代 Triton kernel 实现
  2. 使用 eval/exec/getattr 等动态机制绕过白名单检测
  3. 通过通配符导入 (from torch.nn import *) 绕过别名追踪
  4. 导入原始 *_torch.py 参考实现作为 fallback
  5. 没有定义任何 @triton.jit kernel 函数
  6. 定义了 kernel 但从未通过 kernel[grid](...) 语法调用
  7. Python 语法错误 / 编译错误
  8. 文件为空或只包含中文注释

用法：
    # 1) 在 MAKG 目录下直接运行（脚本能自动找到 MAKG/python）
    python examples/code_cheat_check.py <路径>

    # 2) 指定单个 .py 文件
    python examples/code_cheat_check.py /path/to/your_impl_code.py

    # 3) 指定一个目录（递归扫描所有 *.py）
    python examples/code_cheat_check.py /path/to/evolve_database

    # 4) 在其他项目目录（如 akg）下运行时，需要 --makg-path 参数
    python /path/to/code_cheat_check.py /path/to/target --makg-path /mnt/lustre-client/zhangzizheng/AIKG/MAKG

参数：
    --backend    目标后端 (cuda / ascend), 默认 cuda
    --dsl        DSL 类型 (triton_cuda / triton_ascend), 默认 triton_cuda
    --makg-path  MAKG 项目根目录路径，用于定位 MAKG/python
    --summary    只输出汇总报告，不输出每个文件的详细错误
    --only-cheat 只输出存在作弊问题的文件（忽略语法/编译/空文件等其他错误）
"""

import asyncio
import argparse
import json
import sys
import ast
import logging
import py_compile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Path 配置：自动定位 MAKG/python
# ---------------------------------------------------------------------------

def resolve_makg_python_path(explicit_makg_root: Optional[str] = None) -> Path:
    """
    定位 MAKG/python 目录，优先级：
      1. 命令行传入的 --makg-path
      2. 环境变量 MAKG_ROOT
      3. 脚本所在目录向上查找 MAKG
      4. 当前工作目录向上查找 MAKG
    """
    candidates: List[Path] = []

    if explicit_makg_root:
        candidates.append(Path(explicit_makg_root).resolve())

    script_dir = Path(__file__).resolve().parent
    # 脚本在 MAKG/examples/ -> 向上一级就是 MAKG 根
    if (script_dir.parent / "python" / "akg_agents").is_dir():
        candidates.append(script_dir.parent.resolve())
    # 脚本直接在 MAKG 根目录
    if (script_dir / "python" / "akg_agents").is_dir():
        candidates.append(script_dir.resolve())

    import os
    if "MAKG_ROOT" in os.environ:
        candidates.append(Path(os.environ["MAKG_ROOT"]).resolve())

    # 从 CWD 向上查找
    cur = Path.cwd().resolve()
    while cur.parent != cur:
        if (cur / "python" / "akg_agents").is_dir():
            candidates.append(cur)
            break
        cur = cur.parent

    for candidate in candidates:
        python_dir = candidate / "python"
        if python_dir.is_dir() and (python_dir / "akg_agents" / "op" / "utils" / "code_checker").is_dir():
            return python_dir

    raise RuntimeError(
        "无法定位 MAKG 项目根目录。请通过 --makg-path 参数指定 MAKG 根路径，\n"
        "或将脚本放到 MAKG/examples/ 目录下运行。\n"
        "已尝试路径: \n  " + "\n  ".join(str(p) for p in candidates)
    )


# ---------------------------------------------------------------------------
# 主检查逻辑
# ---------------------------------------------------------------------------

class CheatCheckReport:
    """单个文件的检查报告"""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.errors: List[Dict] = []              # 来自 CodeChecker 的阻塞错误
        self.diagnostic_errors: List[Dict] = []   # Triton 诊断问题
        self.syntax_error: Optional[str] = None
        self.compile_error: Optional[str] = None
        self.is_empty: bool = False
        self.has_cheat_issue: bool = False        # 是否有作弊相关问题
        self.check_success: bool = False          # CodeChecker 是否成功运行

    @property
    def has_any_issue(self) -> bool:
        return (
            bool(self.errors)
            or bool(self.diagnostic_errors)
            or self.syntax_error is not None
            or self.compile_error is not None
            or self.is_empty
        )


_CHEAT_ERROR_TYPES: Set[str] = {
    # Triton DSL 合规性 (作弊)
    "no_triton_kernel",
    "triton_kernel_not_called",
    "framework_model_import_fallback",
    "torch_wildcard_import",
    "torch_api_not_allowlisted",
    "torch_api_dynamic_lookup",
    "torch_api_dynamic_bypass",
    "sub_expr_local_test_in_impl",
    # 动态执行
    "dynamic_python_execution",
    "dynamic_import",
    # Triton 其他语义风险（也视为作弊线索）
    "triton_dot_explicit_precision",
    "triton_block_ptr_while_loop",
    "triton_api_missing",
    "triton_api_bad_kwarg",
}


def _classify_error_type(err_type: str) -> str:
    """把各种错误类型分类到可读的中文类别"""
    if err_type in ("no_triton_kernel", "triton_kernel_not_called"):
        return "Triton kernel 缺失"
    if err_type in ("framework_model_import_fallback",):
        return "导入参考 torch 实现 (fallback 作弊)"
    if err_type == "torch_wildcard_import":
        return "torch 通配符导入"
    if err_type in ("torch_api_not_allowlisted",):
        return "调用 torch/nn/F 高层 API (作弊)"
    if err_type in ("torch_api_dynamic_lookup",):
        return "动态查找 torch API (绕过白名单)"
    if err_type in ("dynamic_python_execution",):
        return "eval/exec 动态执行"
    if err_type in ("dynamic_import",):
        return "动态 import 绕过"
    if err_type == "sub_expr_local_test_in_impl":
        return "impl 中包含测试代码 (if __name__ == '__main__')"
    if err_type == "syntax_error":
        return "Python 语法错误"
    if err_type == "compile_error":
        return "Python 编译错误"
    if err_type == "import_error":
        return "导入模块不可用"
    if err_type.startswith("triton_"):
        return "Triton 语义 / API 风险"
    return err_type


async def check_single_file(
    file_path: Path,
    checker_obj,
    logger: logging.Logger,
) -> CheatCheckReport:
    report = CheatCheckReport(file_path)

    try:
        code = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        report.syntax_error = f"读取文件失败: {e}"
        return report

    if not code.strip():
        report.is_empty = True
        return report

    # 先用 ast.parse 快速语法检查
    try:
        ast.parse(code)
    except SyntaxError as e:
        report.syntax_error = f"SyntaxError at line {e.lineno}: {e.msg}"
        return report

    # 用 py_compile 检查
    try:
        py_compile.compile(str(file_path), doraise=True)
    except py_compile.PyCompileError as e:
        report.compile_error = f"PyCompileError: {e.msg}"

    # 调用 MAKG CodeChecker
    try:
        passed, _err_msg, errors = await checker_obj.check(code)
        report.check_success = True
        report.errors = list(errors)
        report.has_cheat_issue = any(
            (e.get("error_type") or "") in _CHEAT_ERROR_TYPES for e in errors
        )
        # Triton 诊断（非阻塞）
        if hasattr(checker_obj, "last_diagnostic_errors") and checker_obj.last_diagnostic_errors:
            for diag in checker_obj.last_diagnostic_errors:
                # Issue 对象可能是 dataclass，也可能是 dict
                if isinstance(diag, dict):
                    report.diagnostic_errors.append(diag)
                else:
                    try:
                        report.diagnostic_errors.append({
                            "rule_id": getattr(diag, "rule_id", ""),
                            "severity": getattr(diag, "severity", ""),
                            "title": getattr(diag, "title", ""),
                            "message": getattr(diag, "message", ""),
                            "line": getattr(getattr(diag, "location", None), "lineno", -1),
                        })
                    except Exception:
                        report.diagnostic_errors.append({"detail": str(diag)})
    except Exception as e:
        logger.warning(f"CodeChecker 抛出异常 ({file_path}): {type(e).__name__}: {e}")

    return report


def find_python_files(target: Path) -> List[Path]:
    if target.is_file():
        if target.suffix == ".py":
            return [target]
        return []
    if not target.is_dir():
        return []
    # 排除常见无关目录，只检查看起来像算子实现的文件名
    skip_dirs = {"__pycache__", ".venv", "venv", "site-packages", "node_modules", ".git"}
    files: List[Path] = []
    for p in sorted(target.rglob("*.py")):
        if any(seg in skip_dirs for seg in p.parts):
            continue
        files.append(p)
    return files


# ---------------------------------------------------------------------------
# 输出格式
# ---------------------------------------------------------------------------

def _format_error_detail(err: Dict) -> str:
    line = err.get("line", 0) or 0
    detail = err.get("detail") or err.get("message") or ""
    suggestion = err.get("suggestion") or ""
    rule = err.get("error_type") or err.get("rule_id") or ""
    snippet = err.get("code_snippet") or ""

    lines = [f"  [{rule}] 第 {line} 行: {detail}"]
    if suggestion:
        lines.append(f"         建议: {suggestion}")
    if snippet:
        lines.append(f"         代码: {snippet}")
    return "\n".join(lines)


def print_report(report: CheatCheckReport, only_cheat: bool, summary_only: bool):
    if only_cheat and not report.has_cheat_issue:
        return

    if summary_only and not report.has_any_issue:
        return

    status_icon = "✓" if not report.has_any_issue else ("🔥" if report.has_cheat_issue else "✗")
    status_text = "PASS" if not report.has_any_issue else ("CHEAT" if report.has_cheat_issue else "FAIL")

    print(f"\n{status_icon} [{status_text}]  {report.file_path}")

    if summary_only and report.has_any_issue:
        cheat_count = sum(1 for e in report.errors if (e.get("error_type") or "") in _CHEAT_ERROR_TYPES)
        other_count = len(report.errors) - cheat_count + len(report.diagnostic_errors)
        extra = []
        if report.is_empty:
            extra.append("空文件")
        if report.syntax_error:
            extra.append("语法错误")
        if report.compile_error:
            extra.append("编译错误")
        parts = []
        if cheat_count:
            parts.append(f"作弊问题 {cheat_count} 个")
        if other_count:
            parts.append(f"其他问题 {other_count} 个")
        if extra:
            parts.append(", ".join(extra))
        if parts:
            print(f"    摘要: " + "; ".join(parts))
        return

    if report.is_empty:
        print("    -> 文件为空")
        return

    if report.syntax_error:
        print(f"    -> {report.syntax_error}")

    if report.compile_error:
        print(f"    -> {report.compile_error}")

    # 按类别分组输出
    by_category: Dict[str, List[Dict]] = {}
    for err in report.errors:
        cat = _classify_error_type(err.get("error_type", ""))
        by_category.setdefault(cat, []).append(err)

    if by_category:
        for cat, errs in sorted(by_category.items()):
            print(f"    -- {cat} (共 {len(errs)} 处) --")
            for err in errs:
                print(_format_error_detail(err))

    if report.diagnostic_errors:
        print(f"    -- Triton 诊断 (共 {len(report.diagnostic_errors)} 处) --")
        for diag in report.diagnostic_errors:
            line = diag.get("line", -1)
            rule = diag.get("rule_id", "")
            title = diag.get("title", "")
            message = diag.get("message", "")
            print(f"      [{rule}] 第 {line} 行: {title}")
            if message:
                print(f"               {message}")


def print_overall_summary(reports: List[CheatCheckReport], total_files: int):
    cheat_files = sum(1 for r in reports if r.has_cheat_issue)
    failed_files = sum(1 for r in reports if r.has_any_issue and not r.has_cheat_issue)
    pass_files = sum(1 for r in reports if not r.has_any_issue)

    cheat_error_total = sum(
        sum(1 for e in r.errors if (e.get("error_type") or "") in _CHEAT_ERROR_TYPES)
        for r in reports
    )
    other_error_total = sum(
        len(r.errors) + len(r.diagnostic_errors) for r in reports
    ) - cheat_error_total

    bar_width = 40
    total = len(reports) or 1
    cheat_pct = int(cheat_files / total * bar_width)
    fail_pct = int(failed_files / total * bar_width)
    pass_pct = bar_width - cheat_pct - fail_pct

    print("\n" + "=" * 70)
    print("检查汇总")
    print("=" * 70)
    print(f"  总文件数          : {total_files}")
    print(f"  成功解析          : {len(reports)}")
    print(f"  存在作弊问题      : {cheat_files}")
    print(f"  其他错误          : {failed_files}")
    print(f"  通过              : {pass_files}")
    print()
    print(
        "  状态条: "
        + "🔥" * cheat_pct
        + "✗" * fail_pct
        + "✓" * pass_pct
    )
    print(f"  作弊类错误总数    : {cheat_error_total}")
    print(f"  其他错误 / 诊断   : {other_error_total}")

    if cheat_files:
        print("\n存在作弊嫌疑的文件列表:")
        for r in reports:
            if r.has_cheat_issue:
                print(f"  - {r.file_path}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 MAKG CodeChecker 检查算子代码是否存在作弊行为",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="要检查的 .py 文件或目录（递归扫描目录下所有 *.py）",
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        choices=["cuda", "ascend"],
        help="目标后端 (默认: cuda)",
    )
    parser.add_argument(
        "--dsl",
        default="triton_cuda",
        choices=["triton_cuda", "triton_ascend"],
        help="DSL 类型 (默认: triton_cuda)",
    )
    parser.add_argument(
        "--makg-path",
        default=None,
        help="MAKG 项目根目录路径，用于定位 MAKG/python。若脚本在 MAKG 内则可省略",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="只输出汇总报告，不输出每个文件的详细错误",
    )
    parser.add_argument(
        "--only-cheat",
        action="store_true",
        help="只输出存在作弊嫌疑的文件，忽略纯语法 / 编译 / 导入错误",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        metavar="OUTPUT_JSON",
        help="将详细检查结果保存为 JSON 文件",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="输出调试日志",
    )
    return parser


async def _run_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    logger = logging.getLogger("code_cheat_check")

    # 1. 定位 MAKG/python 并注入 sys.path
    makg_python = resolve_makg_python_path(args.makg_path)
    if str(makg_python) not in sys.path:
        sys.path.insert(0, str(makg_python))
    logger.info(f"使用 MAKG/python: {makg_python}")

    # 2. 导入 CodeChecker
    try:
        from akg_agents.op.utils.code_checker import CodeChecker  # type: ignore
    except ImportError as e:
        print(f"[ERROR] 无法导入 MAKG CodeChecker: {e}", file=sys.stderr)
        print("请确认 --makg-path 指向正确的 MAKG 项目根目录。", file=sys.stderr)
        return 2

    # 3. 构建 CodeChecker 实例
    checker = CodeChecker(backend=args.backend, dsl=args.dsl)

    # 4. 扫描目标文件
    target_path = Path(args.target).resolve()
    files = find_python_files(target_path)
    if not files:
        print(f"[WARN] 在 {target_path} 未找到任何 .py 文件")
        return 1

    print(f"扫描目标: {target_path}")
    print(f"待检查文件: {len(files)}")
    print(f"后端/DSL   : {args.backend} / {args.dsl}")

    # 5. 逐个检查
    reports: List[CheatCheckReport] = []
    for file_path in files:
        report = await check_single_file(file_path, checker, logger)
        reports.append(report)
        print_report(report, only_cheat=args.only_cheat, summary_only=args.summary)

    # 6. 汇总
    print_overall_summary(reports, len(files))

    # 7. 可选 JSON 输出
    if args.json_output:
        out_path = Path(args.json_output).resolve()
        json_data = []
        for r in reports:
            json_data.append({
                "file": str(r.file_path),
                "is_empty": r.is_empty,
                "syntax_error": r.syntax_error,
                "compile_error": r.compile_error,
                "has_cheat_issue": r.has_cheat_issue,
                "check_success": r.check_success,
                "errors": r.errors,
                "diagnostic_errors": r.diagnostic_errors,
            })
        out_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n详细 JSON 报告已保存: {out_path}")

    # 返回值：0=无任何问题，1=有作弊，2=工具错误
    if any(r.has_cheat_issue for r in reports):
        return 1
    if any(r.has_any_issue for r in reports):
        return 0  # 非作弊错误不视为失败（与用户需求对齐）
    return 0


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(_run_async(args))
    except KeyboardInterrupt:
        print("\n[已中止]", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())