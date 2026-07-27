# -*- coding: utf-8 -*-
"""
算子代码作弊检查工具
====================

这个脚本是 MAKG CodeChecker 的项目内包装器。它不重新实现 MAKG 的
检查规则，而是负责：

1. 自动定位 MAKG/python 并导入最新版 CodeChecker
2. 扫描单个 .py 文件，或目录中的 impl_code.py / *_triton_*.py 实现文件
3. 调用 MAKG 的 blocking base checkers 与非阻塞 Triton diagnostics
4. 将 MAKG 返回的错误整理成适合人工排查的中文报告

当前 MAKG CodeChecker 的主要逻辑：

- blocking base_checkers:
  empty_code, python_syntax, py_compile, import_availability,
  stray_chinese, triton_dsl_compliance
- non-blocking triton_checkers:
  api_signature, high_confidence_semantics，Ascend 下额外包含 ascend_semantics

用法：
    python code_cheat_check.py /path/to/impl_code.py
    python code_cheat_check.py /path/to/evolve_database --only-cheat
    python code_cheat_check.py /path/to/target --makg-path /home/zhangzizheng/AIKG/MAKG

返回码：
    0  未发现作弊类 blocking 问题
    1  发现作弊类 blocking 问题
    2  工具自身错误，例如无法导入 MAKG CodeChecker
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import importlib.machinery
import importlib.util
import json
import logging
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


# ---------------------------------------------------------------------------
# MAKG 路径定位
# ---------------------------------------------------------------------------

_CODE_CHECKER_REL = Path("akg_agents/op/utils/code_checker")
_DEFAULT_IMPL_PATTERNS = (
    "impl_code.py",
    "*_triton.py",
    "*_triton_cuda.py",
    "*_triton_ascend.py",
)


def _makg_python_from_candidate(candidate: Path) -> Optional[Path]:
    """Return MAKG/python if candidate looks like MAKG root or python dir."""
    candidate = candidate.resolve()
    if (candidate / _CODE_CHECKER_REL).is_dir():
        return candidate

    python_dir = candidate / "python"
    if (python_dir / _CODE_CHECKER_REL).is_dir():
        return python_dir.resolve()

    return None


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen: Set[str] = set()
    out: List[Path] = []
    for path in paths:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _ancestor_candidates(seed: Path) -> Iterable[Path]:
    """Yield likely MAKG roots around a file/directory path."""
    try:
        seed = seed.resolve()
    except OSError:
        return

    starts = [seed]
    if seed.is_file():
        starts.append(seed.parent)

    for start in starts:
        yield start
        for parent in start.parents:
            yield parent
            # Project layout used here:
            #   /home/zhangzizheng/AIKG/akg/...
            #   /home/zhangzizheng/AIKG/MAKG/...
            yield parent / "MAKG"
            if parent.name == "akg":
                yield parent.parent / "MAKG"


def resolve_makg_python_path(explicit_makg_path: Optional[str] = None) -> Path:
    """
    定位 MAKG/python 目录。

    支持 --makg-path 传 MAKG 根目录或 MAKG/python 目录；没有显式参数时，
    会尝试环境变量和当前项目常见的 sibling MAKG 布局。
    """
    candidates: List[Path] = []

    if explicit_makg_path:
        candidates.append(Path(explicit_makg_path))

    if os.environ.get("MAKG_PYTHON"):
        candidates.append(Path(os.environ["MAKG_PYTHON"]))
    if os.environ.get("MAKG_ROOT"):
        candidates.append(Path(os.environ["MAKG_ROOT"]))

    script_path = Path(__file__).resolve()
    candidates.extend(_ancestor_candidates(script_path))
    candidates.extend(_ancestor_candidates(Path.cwd()))

    tried: List[Path] = []
    for candidate in _dedupe_paths(candidates):
        tried.append(candidate)
        python_dir = _makg_python_from_candidate(candidate)
        if python_dir is not None:
            return python_dir

    tried_text = "\n  ".join(str(p) for p in tried[:30])
    if len(tried) > 30:
        tried_text += f"\n  ... 共尝试 {len(tried)} 个路径"
    raise RuntimeError(
        "无法定位 MAKG CodeChecker。请通过 --makg-path 指定 MAKG 根目录或 MAKG/python。\n"
        f"已尝试路径:\n  {tried_text}"
    )


def prepend_python_paths(paths: Sequence[Path]) -> None:
    for path in reversed([p.resolve() for p in paths if p]):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def _install_namespace_package(name: str, package_dir: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(package_dir)]
    module.__package__ = name
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_dir)]
    module.__spec__ = spec
    sys.modules[name] = module
    return module


def _purge_akg_agents_modules() -> None:
    for name in list(sys.modules):
        if name == "akg_agents" or name.startswith("akg_agents."):
            sys.modules.pop(name, None)


def _import_code_checker_lightweight(makg_python: Path):
    """Load only akg_agents.op.utils.code_checker, bypassing top-level deps."""
    _purge_akg_agents_modules()

    akg_agents_dir = makg_python / "akg_agents"
    op_dir = akg_agents_dir / "op"
    utils_dir = op_dir / "utils"
    checker_dir = utils_dir / "code_checker"
    init_file = checker_dir / "__init__.py"
    if not init_file.is_file():
        raise RuntimeError(f"找不到 CodeChecker __init__.py: {init_file}")

    akg_pkg = _install_namespace_package("akg_agents", akg_agents_dir)
    op_pkg = _install_namespace_package("akg_agents.op", op_dir)
    utils_pkg = _install_namespace_package("akg_agents.op.utils", utils_dir)
    akg_pkg.op = op_pkg
    op_pkg.utils = utils_pkg

    package_name = "akg_agents.op.utils.code_checker"
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(checker_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建 CodeChecker import spec: {init_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    utils_pkg.code_checker = module
    spec.loader.exec_module(module)
    return module.CodeChecker


def import_code_checker_class(makg_python: Path, logger: logging.Logger):
    """
    Import MAKG CodeChecker.

    Normal import executes akg_agents/__init__.py, which may require workflow/LLM
    dependencies that are irrelevant for this static checker. If that import fails,
    fall back to isolated loading of the code_checker package only.
    """
    try:
        from akg_agents.op.utils.code_checker import CodeChecker  # type: ignore

        return CodeChecker, "normal", ""
    except ImportError as normal_exc:
        logger.warning("正常导入 MAKG CodeChecker 失败，尝试轻量加载: %s", normal_exc)
        try:
            return _import_code_checker_lightweight(makg_python), "lightweight", str(normal_exc)
        except Exception as fallback_exc:
            raise RuntimeError(
                "无法导入 MAKG CodeChecker。\n"
                f"正常导入错误: {type(normal_exc).__name__}: {normal_exc}\n"
                f"轻量加载错误: {type(fallback_exc).__name__}: {fallback_exc}"
            ) from fallback_exc


# ---------------------------------------------------------------------------
# 报告模型与分类
# ---------------------------------------------------------------------------

_CHEAT_ERROR_TYPES: Set[str] = {
    # MAKG triton_dsl_compliance_checker blocking errors
    "framework_model_import_fallback",
    "torch_wildcard_import",
    "sub_expr_local_test_in_impl",
    "no_triton_kernel",
    "triton_kernel_not_called",
    "triton_block_ptr_while_loop",
    "triton_dot_explicit_precision",
    "torch_api_not_allowlisted",
    "torch_api_dynamic_lookup",
    "torch_api_dynamic_bypass",  # old MAKG compatibility
    "dynamic_python_execution",
    "dynamic_import",
}

_CATEGORY_BY_ERROR_TYPE: Dict[str, str] = {
    "empty_code": "空代码",
    "syntax_error": "Python 语法错误",
    "compile_error": "Python 编译错误",
    "import_error": "导入模块不可用",
    "stray_chinese_text": "疑似中文描述混入",
    "framework_model_import_fallback": "导入参考 torch 实现",
    "torch_wildcard_import": "torch 通配符导入",
    "sub_expr_local_test_in_impl": "impl 中包含本地测试代码",
    "no_triton_kernel": "Triton kernel 缺失",
    "triton_kernel_not_called": "Triton kernel 未启动",
    "triton_block_ptr_while_loop": "Triton DSL 已知失败模式",
    "triton_dot_explicit_precision": "tl.dot 显式精度参数",
    "torch_api_not_allowlisted": "host 侧 torch/nn/F 非白名单调用",
    "torch_api_dynamic_lookup": "动态查找 torch API",
    "torch_api_dynamic_bypass": "动态绕过 torch API 白名单",
    "dynamic_python_execution": "eval/exec/compile 动态执行",
    "dynamic_import": "动态 import 绕过",
}

_CATEGORY_BY_RULE_ID: Dict[str, str] = {
    "TRITON_API_MISSING": "Triton API 缺失",
    "TRITON_API_BAD_KWARG": "Triton API 参数不兼容",
    "TRITON_API_TOO_MANY_POSITIONAL_ARGS": "Triton API 位置参数过多",
    "TRITON_UNSUPPORTED_CONTROL_FLOW": "Triton kernel 控制流风险",
    "TRITON_RUNTIME_PY_IF": "Triton kernel 数据依赖 Python if",
    "TRITON_PROGRAM_ID_AXIS": "tl.program_id axis 非法",
    "TRITON_STATIC_RANGE_NON_CONSTEXPR": "tl.static_range 非 constexpr",
    "TRITON_DYNAMIC_SHAPE_IN_ALLOC": "Triton 动态 shape 分配",
    "TRITON_EXPAND_DIMS_TUPLE_AXIS": "tl.expand_dims axis 不兼容",
    "TRITON_CONSTEXPR_TO_METHOD": "constexpr 标量调用 .to()",
    "TRITON_DUPLICATE_KERNEL_ARGUMENT": "kernel 启动参数重复",
    "TRITON_ASCEND_MIXED_SCALAR_SLICE_INDEX": "Ascend Triton 混合索引风险",
    "TRITON_DIAGNOSTIC_CHECKER_INTERNAL_ERROR": "Triton 诊断器内部错误",
}


@dataclass
class CheatCheckReport:
    """单个文件的检查报告。"""

    file_path: Path
    passed: bool = False
    error_message: str = ""
    errors: List[Dict[str, Any]] = field(default_factory=list)
    diagnostic_passed: bool = True
    diagnostic_errors: List[Dict[str, Any]] = field(default_factory=list)
    diagnostic_error_message: str = ""
    read_error: Optional[str] = None
    checker_exception: Optional[str] = None

    @property
    def has_blocking_issue(self) -> bool:
        return bool(self.errors) or self.read_error is not None or self.checker_exception is not None

    @property
    def has_diagnostic_issue(self) -> bool:
        return bool(self.diagnostic_errors)

    @property
    def has_any_issue(self) -> bool:
        return self.has_blocking_issue or self.has_diagnostic_issue

    @property
    def has_cheat_issue(self) -> bool:
        return any(is_cheat_error(err) for err in self.errors)

    @property
    def has_tool_error(self) -> bool:
        return self.read_error is not None or self.checker_exception is not None


def is_cheat_error(err: Dict[str, Any]) -> bool:
    return str(err.get("error_type") or "") in _CHEAT_ERROR_TYPES


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_error(err: Any) -> Dict[str, Any]:
    if isinstance(err, dict):
        return dict(err)
    return {
        "line": _as_int(getattr(err, "line", 0)),
        "error_type": str(getattr(err, "error_type", "unknown")),
        "detail": str(getattr(err, "detail", err)),
        "suggestion": str(getattr(err, "suggestion", "")),
        "code_snippet": str(getattr(err, "code_snippet", "")),
    }


def _normalize_diagnostic(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        diag = dict(item)
    else:
        location = getattr(item, "location", None)
        diag = {
            "line": getattr(location, "lineno", -1),
            "column": getattr(location, "col", 0),
            "end_line": getattr(location, "end_lineno", -1),
            "end_column": getattr(location, "end_col", 0),
            "severity": getattr(item, "severity", ""),
            "rule_id": getattr(item, "rule_id", ""),
            "title": getattr(item, "title", ""),
            "detail": getattr(item, "message", ""),
            "suggestion": getattr(item, "hint", ""),
            "tags": sorted(getattr(item, "tags", set()) or []),
        }

    if "detail" not in diag and "message" in diag:
        diag["detail"] = diag.get("message")
    if "suggestion" not in diag and "hint" in diag:
        diag["suggestion"] = diag.get("hint")
    return diag


def _line_label(line: Any, column: Any = None) -> str:
    line_num = _as_int(line, 0)
    if line_num <= 0:
        return "全局"
    col_num = _as_int(column, -1)
    if col_num >= 0:
        return f"第 {line_num} 行:{col_num}"
    return f"第 {line_num} 行"


def _error_category(error_type: str) -> str:
    if error_type in _CATEGORY_BY_ERROR_TYPE:
        return _CATEGORY_BY_ERROR_TYPE[error_type]
    if error_type in _CATEGORY_BY_RULE_ID:
        return _CATEGORY_BY_RULE_ID[error_type]
    if error_type.startswith("TRITON_"):
        return "Triton 诊断"
    if error_type.startswith("triton_"):
        return "Triton DSL 合规问题"
    return error_type or "unknown"


# ---------------------------------------------------------------------------
# 检查执行
# ---------------------------------------------------------------------------


def build_checker_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build an explicit config matching MAKG's current default checker groups."""
    base_checkers: Any = "all"
    if args.skip_import_check:
        base_checkers = [
            "empty_code",
            "python_syntax",
            "py_compile",
            "stray_chinese",
            "triton_dsl_compliance",
        ]

    diagnostic_cfg: Dict[str, Any] = {
        "enabled": not args.no_diagnostics,
        "only_errors": True,
        "dedup": True,
    }
    if args.diagnostic_blocking:
        diagnostic_cfg["blocking"] = True

    return {
        "code_checker": {
            "base_checkers": base_checkers,
            "triton_checkers": "all",
        },
        "code_diagnostic_checker": diagnostic_cfg,
    }


async def check_single_file(
    file_path: Path,
    checker_obj: Any,
    logger: logging.Logger,
) -> CheatCheckReport:
    report = CheatCheckReport(file_path=file_path)

    try:
        code = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        report.read_error = f"{type(exc).__name__}: {exc}"
        return report

    try:
        passed, error_message, errors = await checker_obj.check(
            code,
            task_info={"task_id": file_path.name, "file_path": str(file_path)},
        )
        report.passed = bool(passed)
        report.error_message = error_message or ""
        report.errors = [_normalize_error(err) for err in (errors or [])]
        report.diagnostic_passed = bool(getattr(checker_obj, "last_diagnostic_passed", True))
        report.diagnostic_errors = [
            _normalize_diagnostic(item)
            for item in (getattr(checker_obj, "last_diagnostic_errors", []) or [])
        ]
        report.diagnostic_error_message = str(
            getattr(checker_obj, "last_diagnostic_error_message", "") or ""
        )
    except Exception as exc:
        report.checker_exception = f"{type(exc).__name__}: {exc}"
        logger.exception("CodeChecker failed for %s", file_path)

    return report


def _matches_patterns(path: Path, root: Path, patterns: Sequence[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    return any(
        fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(rel, pattern)
        for pattern in patterns
    )


def find_python_files(
    target: Path,
    *,
    include_patterns: Sequence[str],
    all_python: bool,
) -> List[Path]:
    if target.is_file():
        return [target] if target.suffix == ".py" else []
    if not target.is_dir():
        return []

    skip_dirs = {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
        "site-packages",
        "venv",
    }
    files: List[Path] = []
    for path in sorted(target.rglob("*.py")):
        if any(part in skip_dirs for part in path.parts):
            continue
        files.append(path)

    patterns = list(include_patterns)
    if not patterns and not all_python:
        patterns = list(_DEFAULT_IMPL_PATTERNS)
    if not patterns:
        return files

    return [path for path in files if _matches_patterns(path, target, patterns)]


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _format_error_detail(err: Dict[str, Any]) -> str:
    error_type = str(err.get("error_type") or "unknown")
    detail = str(err.get("detail") or err.get("message") or err.get("title") or "")
    suggestion = str(err.get("suggestion") or err.get("hint") or "")
    snippet = str(err.get("code_snippet") or "")

    lines = [f"      [{error_type}] {_line_label(err.get('line'))}: {detail}"]
    if suggestion:
        lines.append(f"        建议: {suggestion}")
    if snippet:
        lines.append(f"        代码: {snippet}")
    return "\n".join(lines)


def _format_diagnostic_detail(diag: Dict[str, Any]) -> str:
    rule_id = str(diag.get("rule_id") or "unknown")
    severity = str(diag.get("severity") or "")
    title = str(diag.get("title") or "")
    detail = str(diag.get("detail") or diag.get("message") or "")
    suggestion = str(diag.get("suggestion") or diag.get("hint") or "")
    tags = diag.get("tags") or []
    tag_text = f" tags={','.join(map(str, tags))}" if tags else ""

    head = f"      [{rule_id}] {_line_label(diag.get('line'), diag.get('column'))}"
    if severity:
        head += f" {severity}"
    if title:
        head += f": {title}"
    if tag_text:
        head += tag_text

    lines = [head]
    if detail:
        lines.append(f"        {detail}")
    if suggestion:
        lines.append(f"        建议: {suggestion}")
    return "\n".join(lines)


def _report_status(report: CheatCheckReport) -> str:
    if report.has_tool_error:
        return "TOOL_ERROR"
    if report.has_cheat_issue:
        return "CHEAT"
    if report.errors:
        return "FAIL"
    if report.diagnostic_errors:
        return "DIAG"
    return "PASS"


def _group_errors(errors: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for err in errors:
        error_type = str(err.get("error_type") or "unknown")
        grouped.setdefault(_error_category(error_type), []).append(err)
    return grouped


def _group_diagnostics(diagnostics: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for diag in diagnostics:
        rule_id = str(diag.get("rule_id") or "unknown")
        grouped.setdefault(_error_category(rule_id), []).append(diag)
    return grouped


def print_report(report: CheatCheckReport, *, only_cheat: bool, summary_only: bool) -> None:
    if only_cheat and not report.has_cheat_issue:
        return
    if summary_only and not report.has_any_issue:
        return

    status = _report_status(report)
    print(f"\n[{status}] {report.file_path}")

    cheat_count = sum(1 for err in report.errors if is_cheat_error(err))
    other_blocking_count = len(report.errors) - cheat_count
    diag_count = len(report.diagnostic_errors)

    if summary_only:
        parts: List[str] = []
        if report.read_error:
            parts.append("读取失败")
        if report.checker_exception:
            parts.append("checker 异常")
        if cheat_count:
            parts.append(f"作弊/DSL 合规问题 {cheat_count} 个")
        if other_blocking_count:
            parts.append(f"其他 blocking 问题 {other_blocking_count} 个")
        if diag_count:
            parts.append(f"Triton 诊断 {diag_count} 个")
        print("    摘要: " + ("; ".join(parts) if parts else "无问题"))
        return

    if report.read_error:
        print(f"    读取失败: {report.read_error}")
    if report.checker_exception:
        print(f"    CodeChecker 异常: {report.checker_exception}")

    for category, errors in _group_errors(report.errors).items():
        print(f"    -- {category} (共 {len(errors)} 处) --")
        for err in errors:
            print(_format_error_detail(err))

    if report.diagnostic_errors:
        print(f"    -- Triton 非阻塞诊断 (共 {len(report.diagnostic_errors)} 处) --")
        for category, diagnostics in _group_diagnostics(report.diagnostic_errors).items():
            print(f"       {category} (共 {len(diagnostics)} 处)")
            for diag in diagnostics:
                print(_format_diagnostic_detail(diag))


def print_overall_summary(reports: List[CheatCheckReport], total_files: int) -> None:
    cheat_files = sum(1 for report in reports if report.has_cheat_issue)
    tool_error_files = sum(1 for report in reports if report.has_tool_error)
    blocking_only_files = sum(
        1 for report in reports
        if report.errors and not report.has_cheat_issue and not report.has_tool_error
    )
    diagnostic_only_files = sum(
        1 for report in reports
        if report.diagnostic_errors and not report.has_blocking_issue
    )
    pass_files = sum(1 for report in reports if not report.has_any_issue)

    cheat_error_total = sum(
        sum(1 for err in report.errors if is_cheat_error(err))
        for report in reports
    )
    other_blocking_total = sum(len(report.errors) for report in reports) - cheat_error_total
    diagnostic_total = sum(len(report.diagnostic_errors) for report in reports)

    print("\n" + "=" * 72)
    print("检查汇总")
    print("=" * 72)
    print(f"  总文件数              : {total_files}")
    print(f"  完成检查              : {len(reports)}")
    print(f"  作弊/DSL 合规问题文件 : {cheat_files}")
    print(f"  其他 blocking 问题文件: {blocking_only_files}")
    print(f"  仅 Triton 诊断文件    : {diagnostic_only_files}")
    print(f"  工具/读取错误文件     : {tool_error_files}")
    print(f"  完全通过              : {pass_files}")
    print()
    print(f"  作弊/DSL 合规错误总数 : {cheat_error_total}")
    print(f"  其他 blocking 错误总数: {other_blocking_total}")
    print(f"  Triton 诊断总数       : {diagnostic_total}")

    if cheat_files:
        print("\n存在作弊/DSL 合规问题的文件:")
        for report in reports:
            if report.has_cheat_issue:
                print(f"  - {report.file_path}")

    if tool_error_files:
        print("\n工具/读取错误文件:")
        for report in reports:
            if report.has_tool_error:
                print(f"  - {report.file_path}")

    print("=" * 72)


def write_json_report(path: Path, reports: List[CheatCheckReport]) -> None:
    data = []
    for report in reports:
        data.append(
            {
                "file": str(report.file_path),
                "status": _report_status(report),
                "passed": report.passed,
                "has_cheat_issue": report.has_cheat_issue,
                "has_blocking_issue": report.has_blocking_issue,
                "has_diagnostic_issue": report.has_diagnostic_issue,
                "read_error": report.read_error,
                "checker_exception": report.checker_exception,
                "errors": report.errors,
                "diagnostic_passed": report.diagnostic_passed,
                "diagnostic_errors": report.diagnostic_errors,
            }
        )
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用最新版 MAKG CodeChecker 检查 Triton 算子实现是否存在作弊/DSL 合规问题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="要检查的 .py 文件或目录（目录默认只扫描实现文件模式）",
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        choices=["cuda", "ascend"],
        help="目标后端，默认 cuda",
    )
    parser.add_argument(
        "--dsl",
        default="triton_cuda",
        choices=["triton_cuda", "triton_ascend"],
        help="DSL 类型，默认 triton_cuda",
    )
    parser.add_argument(
        "--makg-path",
        default=None,
        help="MAKG 根目录或 MAKG/python 目录；默认会自动探测 sibling MAKG",
    )
    parser.add_argument(
        "--extra-python-path",
        action="append",
        default=[],
        help="额外加入 sys.path 的路径，可重复传；用于减少本地模块 import_error 误报",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "目录扫描时指定实现文件 glob，可重复传；默认 "
            "impl_code.py,*_triton.py,*_triton_cuda.py,*_triton_ascend.py"
        ),
    )
    parser.add_argument(
        "--all-python",
        action="store_true",
        help="目录扫描时检查所有 *.py（会包含 *_torch.py、verify_*.py、profile_*.py）",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="只输出异常文件的一行摘要和总汇总",
    )
    parser.add_argument(
        "--only-cheat",
        action="store_true",
        help="只输出存在作弊/DSL 合规问题的文件",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="关闭 MAKG 的非阻塞 Triton diagnostics，只跑 blocking checks",
    )
    parser.add_argument(
        "--skip-import-check",
        action="store_true",
        help="跳过 import_availability；适合未安装 torch/triton 的轻量环境",
    )
    parser.add_argument(
        "--diagnostic-blocking",
        action="store_true",
        help="将 MAKG Triton diagnostics 也作为 blocking errors 返回（默认不阻塞）",
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
        level=logging.DEBUG if args.verbose else logging.ERROR,
        format="%(levelname)s: %(message)s",
    )
    logger = logging.getLogger("code_cheat_check")

    try:
        makg_python = resolve_makg_python_path(args.makg_path)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    extra_paths = [Path(item) for item in args.extra_python_path]
    prepend_python_paths([makg_python, *extra_paths])
    logger.info("使用 MAKG/python: %s", makg_python)

    try:
        CodeChecker, import_mode, import_note = import_code_checker_class(makg_python, logger)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print(f"        已加入 sys.path: {makg_python}", file=sys.stderr)
        return 2

    checker_config = build_checker_config(args)
    checker = CodeChecker(backend=args.backend, dsl=args.dsl, config=checker_config)

    target_path = Path(args.target).resolve()
    files = find_python_files(
        target_path,
        include_patterns=args.include,
        all_python=args.all_python,
    )
    if not files:
        if target_path.is_dir() and not args.all_python and not args.include:
            pattern_text = ",".join(_DEFAULT_IMPL_PATTERNS)
            print(f"[WARN] 在 {target_path} 未找到匹配默认实现文件模式的 .py: {pattern_text}")
            print("       如需扫描所有 .py，请加 --all-python")
        else:
            print(f"[WARN] 在 {target_path} 未找到任何匹配条件的 .py 文件")
        return 0

    print(f"扫描目标    : {target_path}")
    print(f"待检查文件  : {len(files)}")
    print(f"MAKG/python : {makg_python}")
    print(f"导入模式    : {import_mode}")
    print(f"后端/DSL    : {args.backend} / {args.dsl}")
    if target_path.is_dir():
        if args.all_python:
            print("文件过滤    : 所有 *.py")
        else:
            patterns = args.include or list(_DEFAULT_IMPL_PATTERNS)
            print(f"文件过滤    : {','.join(patterns)}")
    print(f"诊断检查    : {'关闭' if args.no_diagnostics else '开启'}")
    if args.skip_import_check:
        print("import 检查 : 跳过")
    if args.diagnostic_blocking:
        print("诊断模式    : blocking")
    if import_note and args.verbose:
        print(f"正常导入失败: {import_note}")

    reports: List[CheatCheckReport] = []
    for file_path in files:
        report = await check_single_file(file_path, checker, logger)
        if args.diagnostic_blocking:
            report.diagnostic_errors = []
        reports.append(report)
        print_report(report, only_cheat=args.only_cheat, summary_only=args.summary)

    print_overall_summary(reports, total_files=len(files))

    if args.json_output:
        out_path = Path(args.json_output).resolve()
        write_json_report(out_path, reports)
        print(f"\n详细 JSON 报告已保存: {out_path}")

    if any(report.has_tool_error for report in reports):
        return 2
    if any(report.has_cheat_issue for report in reports):
        return 1
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
