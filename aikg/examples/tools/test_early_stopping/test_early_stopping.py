#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _ensure_project_python_path() -> None:
    """Make the script runnable without installing ai_kernel_generator."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        python_dir = parent / "python"
        marker = python_dir / "ai_kernel_generator" / "database" / "early_stopping.py"
        if marker.exists():
            sys.path.insert(0, str(python_dir))
            return


_ensure_project_python_path()

from ai_kernel_generator.database.early_stopping import (  # noqa: E402
    NCU_METRIC_LIST,
    BranchEarlyStoppingJudge,
    EarlyStoppingConfig,
    EarlyStoppingDecision,
    IterationRecord,
)


@dataclass
class KernelNode:
    impl_id: str
    parent_id: Optional[str]
    dir_path: Path
    impl_info: Dict[str, Any]


@dataclass
class NodeDecision:
    node: KernelNode
    branch: List[KernelNode]
    decision: EarlyStoppingDecision
    ncu_parse_error: Optional[str] = None


def _short_id(value: Optional[str], width: int = 8) -> str:
    if not value:
        return "-"
    return str(value)[:width]


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(number) or math.isinf(number):
        return "-"
    return f"{number:.{digits}f}"


def _truncate(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    if max_len <= 3:
        return value[:max_len]
    return value[: max_len - 3] + "..."


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_island_nodes(island_dir: Path) -> Tuple[Dict[str, KernelNode], List[str]]:
    nodes: Dict[str, KernelNode] = {}
    warnings: List[str] = []

    if not island_dir.exists():
        raise FileNotFoundError(f"island dir not found: {island_dir}")
    if not island_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {island_dir}")

    for impl_info_path in sorted(island_dir.glob("*/impl_info.json")):
        try:
            info = _load_json(impl_info_path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"skip invalid json: {impl_info_path} ({exc})")
            continue

        impl_id = info.get("id")
        if not impl_id:
            warnings.append(f"skip impl_info without id: {impl_info_path}")
            continue

        if impl_id in nodes:
            warnings.append(
                "duplicate impl id "
                f"{impl_id}; keep {nodes[impl_id].dir_path}, skip {impl_info_path.parent}"
            )
            continue

        nodes[impl_id] = KernelNode(
            impl_id=str(impl_id),
            parent_id=info.get("parent_id") or None,
            dir_path=impl_info_path.parent,
            impl_info=info,
        )

    if not nodes:
        raise ValueError(f"no */impl_info.json found under: {island_dir}")

    return nodes, warnings


def build_children_map(nodes: Dict[str, KernelNode]) -> Dict[str, List[str]]:
    children: Dict[str, List[str]] = defaultdict(list)
    for node in nodes.values():
        if node.parent_id and node.parent_id in nodes:
            children[node.parent_id].append(node.impl_id)

    for child_ids in children.values():
        child_ids.sort(key=lambda node_id: _node_sort_key(nodes[node_id]))
    return dict(children)


def _node_sort_key(node: KernelNode) -> Tuple[int, str, str]:
    round_value = node.impl_info.get("round")
    try:
        round_key = int(round_value)
    except (TypeError, ValueError):
        round_key = 10**12
    task_id = str(node.impl_info.get("task_id") or "")
    return round_key, task_id, node.impl_id


def branch_from_root(nodes: Dict[str, KernelNode], target_id: str) -> List[KernelNode]:
    branch: List[KernelNode] = []
    seen = set()
    current_id: Optional[str] = target_id

    while current_id:
        if current_id in seen:
            raise ValueError(f"cycle detected while tracing parent chain at {current_id}")
        seen.add(current_id)

        current = nodes.get(current_id)
        if current is None:
            break

        branch.append(current)
        current_id = current.parent_id

    branch.reverse()
    return branch


def ncu_json_get_mean(ncu_json: Dict[str, Any]) -> Dict[str, float]:
    """
    Local copy of the worker helper.

    Expected shape:
        {kernel_name: {metric_name: [values ...]}}
    """
    metric_values: Dict[str, List[float]] = {}
    for metric_dict in ncu_json.values():
        if not isinstance(metric_dict, dict):
            continue
        for metric, values in metric_dict.items():
            if not isinstance(values, list):
                continue
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if len(numeric_values) != len(values):
                continue
            metric_values.setdefault(metric, []).extend(float(v) for v in numeric_values)

    return {
        metric: sum(values) / len(values)
        for metric, values in metric_values.items()
        if values
    }


def parse_ncu_profile_metric(raw_metric: Any) -> Tuple[Dict[str, float], Optional[str]]:
    if raw_metric in (None, ""):
        return {}, None

    try:
        if isinstance(raw_metric, str):
            raw_metric = json.loads(raw_metric)
        if not isinstance(raw_metric, dict):
            return {}, f"ncu_profile_metric is {type(raw_metric).__name__}, expected dict/json string"
        metric_mean = ncu_json_get_mean(raw_metric)
        return {
            key: metric_mean[key]
            for key in metric_mean.keys()
            if key in NCU_METRIC_LIST
        }, None
    except Exception as exc:  # noqa: BLE001
        return {}, str(exc)


def extract_speedup(info: Dict[str, Any]) -> float:
    try:
        return float(info.get("profile", {}).get("speedup", 0.0))
    except (TypeError, ValueError):
        return 0.0


def make_iteration_records(branch: List[KernelNode]) -> Tuple[List[IterationRecord], Optional[str]]:
    records: List[IterationRecord] = []
    parse_errors: List[str] = []

    for depth, node in enumerate(branch):
        ncu_profile, parse_error = parse_ncu_profile_metric(
            node.impl_info.get("ncu_profile_metric")
        )
        if parse_error:
            parse_errors.append(f"{_short_id(node.impl_id)}: {parse_error}")

        records.append(
            IterationRecord(
                step=depth,
                speedup=extract_speedup(node.impl_info),
                profile=ncu_profile,
            )
        )

    return records, "; ".join(parse_errors) if parse_errors else None


def judge_all_nodes(
    nodes: Dict[str, KernelNode],
    config: EarlyStoppingConfig,
) -> List[NodeDecision]:
    judge = BranchEarlyStoppingJudge(config)
    decisions: List[NodeDecision] = []

    for node in sorted(nodes.values(), key=_node_sort_key):
        branch = branch_from_root(nodes, node.impl_id)
        records, ncu_parse_error = make_iteration_records(branch)
        decision = judge.judge(records)
        decisions.append(
            NodeDecision(
                node=node,
                branch=branch,
                decision=decision,
                ncu_parse_error=ncu_parse_error,
            )
        )

    return decisions


def detail_value(decision: EarlyStoppingDecision, section: str, key: str, default: Any = None) -> Any:
    return decision.details.get(section, {}).get(key, default)


def make_summary_rows(decisions: List[NodeDecision], island_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for item in decisions:
        node = item.node
        decision = item.decision
        reasons = "; ".join(decision.reasons)
        if item.ncu_parse_error:
            reasons = (reasons + "; " if reasons else "") + "NCU parse warning"

        rows.append(
            {
                "round": str(node.impl_info.get("round", "-")),
                "depth": str(len(item.branch)),
                "stop": "Y" if decision.stop else "N",
                "score": _fmt_float(decision.score, 3),
                "speedup": _fmt_float(extract_speedup(node.impl_info), 3),
                "gain": _fmt_float(detail_value(decision, "gain", "score"), 2),
                "slope": _fmt_float(detail_value(decision, "slope", "score"), 2),
                "ncu": _fmt_float(detail_value(decision, "ncu", "score"), 2),
                "low": str(detail_value(decision, "stagnation", "consecutive_low_gain_rounds", "-")),
                "id": _short_id(node.impl_id),
                "parent": _short_id(node.parent_id),
                "dir": str(node.dir_path.relative_to(island_dir)),
                "reasons": _truncate(reasons, 72),
            }
        )

    return rows


def print_table(rows: List[Dict[str, str]], columns: List[str]) -> None:
    if not rows:
        print("(no rows)")
        return

    widths = {
        column: max(len(column), max(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }

    header = "  ".join(column.ljust(widths[column]) for column in columns)
    sep = "  ".join("-" * widths[column] for column in columns)
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def decision_to_jsonable(item: NodeDecision) -> Dict[str, Any]:
    node = item.node
    return {
        "id": node.impl_id,
        "parent_id": node.parent_id,
        "dir": str(node.dir_path),
        "round": node.impl_info.get("round"),
        "task_id": node.impl_info.get("task_id"),
        "speedup": extract_speedup(node.impl_info),
        "branch": [
            {
                "id": branch_node.impl_id,
                "parent_id": branch_node.parent_id,
                "round": branch_node.impl_info.get("round"),
                "speedup": extract_speedup(branch_node.impl_info),
                "dir": str(branch_node.dir_path),
            }
            for branch_node in item.branch
        ],
        "early_stopping": dataclasses.asdict(item.decision),
        "ncu_parse_error": item.ncu_parse_error,
    }


def print_node_detail(item: NodeDecision) -> None:
    node = item.node
    decision = item.decision

    print()
    print(f"[detail] id={node.impl_id}")
    print(f"dir: {node.dir_path}")
    print(f"parent: {node.parent_id or '-'}")
    print(f"round/task: {node.impl_info.get('round', '-')}/{node.impl_info.get('task_id', '-')}")
    print(f"branch_depth: {len(item.branch)}")
    print("branch:")
    for depth, branch_node in enumerate(item.branch):
        print(
            "  "
            f"{depth:02d} id={_short_id(branch_node.impl_id, 12)} "
            f"speedup={_fmt_float(extract_speedup(branch_node.impl_info), 4)} "
            f"round={branch_node.impl_info.get('round', '-')}"
        )

    print(
        "decision: "
        f"stop={decision.stop}, score={_fmt_float(decision.score, 6)}, "
        f"reasons={decision.reasons}"
    )
    if item.ncu_parse_error:
        print(f"ncu_parse_error: {item.ncu_parse_error}")

    print("details:")
    print(json.dumps(decision.details, ensure_ascii=False, indent=2))


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_config_overrides(
    config: EarlyStoppingConfig,
    config_json: Optional[Path],
    set_values: Iterable[str],
) -> EarlyStoppingConfig:
    updates: Dict[str, Any] = {}

    if config_json:
        loaded = _load_json(config_json)
        if not isinstance(loaded, dict):
            raise ValueError(f"config json must be an object: {config_json}")
        updates.update(loaded)

    for item in set_values:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set key is empty: {item}")
        updates[key] = parse_value(value.strip())

    config_fields = {field.name for field in dataclasses.fields(config)}
    for key, value in updates.items():
        if "." in key:
            top_key, nested_key = key.split(".", 1)
            if top_key != "ncu_metric_thresholds":
                raise ValueError(f"unsupported nested config key: {key}")
            config.ncu_metric_thresholds[nested_key] = float(value)
            continue

        if key not in config_fields:
            valid = ", ".join(sorted(config_fields))
            raise ValueError(f"unknown config key: {key}; valid keys: {valid}")
        setattr(config, key, value)

    return config


def filter_decisions(
    decisions: List[NodeDecision],
    stop_only: bool,
    min_score: Optional[float],
    detail_id: Optional[str],
) -> List[NodeDecision]:
    filtered = decisions

    if stop_only:
        filtered = [item for item in filtered if item.decision.stop]

    if min_score is not None:
        filtered = [item for item in filtered if item.decision.score >= min_score]

    if detail_id:
        filtered = [
            item
            for item in filtered
            if item.node.impl_id == detail_id or item.node.impl_id.startswith(detail_id)
        ]

    return filtered


def sort_decisions(decisions: List[NodeDecision], sort_by: str) -> List[NodeDecision]:
    if sort_by == "score":
        return sorted(decisions, key=lambda item: item.decision.score, reverse=True)
    if sort_by == "speedup":
        return sorted(decisions, key=lambda item: extract_speedup(item.node.impl_info), reverse=True)
    if sort_by == "depth":
        return sorted(decisions, key=lambda item: len(item.branch), reverse=True)
    if sort_by == "id":
        return sorted(decisions, key=lambda item: item.node.impl_id)
    return sorted(decisions, key=lambda item: _node_sort_key(item.node))


def print_dataset_summary(
    island_dir: Path,
    nodes: Dict[str, KernelNode],
    children_map: Dict[str, List[str]],
    decisions: List[NodeDecision],
    warnings: List[str],
) -> None:
    leaf_count = sum(1 for node_id in nodes if not children_map.get(node_id))
    missing_parent_count = sum(
        1 for node in nodes.values() if node.parent_id and node.parent_id not in nodes
    )
    stop_count = sum(1 for item in decisions if item.decision.stop)
    max_depth = max((len(item.branch) for item in decisions), default=0)
    max_speedup = max((extract_speedup(item.node.impl_info) for item in decisions), default=0.0)

    print(f"island_dir: {island_dir}")
    print(
        "loaded: "
        f"{len(nodes)} kernels, {leaf_count} leaves, "
        f"{missing_parent_count} missing-parent links"
    )
    print(
        "early_stopping: "
        f"{stop_count}/{len(decisions)} stop, max_depth={max_depth}, "
        f"max_speedup={_fmt_float(max_speedup, 4)}"
    )
    for warning in warnings[:10]:
        print(f"[warn] {warning}")
    if len(warnings) > 10:
        print(f"[warn] ... {len(warnings) - 10} more warnings")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline early-stopping inspector for one evolve_database island directory."
        )
    )
    parser.add_argument(
        "--island_dir",
        default=Path("/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/evolve_database/level2/62_Matmul_GroupNorm_LeakyReLU_Sum/island_0"),
        type=Path,
        help="Path like .../evolve_database/level2/99_Matmul_GELU_Softmax/island_0",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional JSON object used to override EarlyStoppingConfig fields.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override EarlyStoppingConfig, e.g. --set min_history=4 "
            "--set stop_score_threshold=0.8 "
            "--set ncu_metric_thresholds.sm__cycles_active.avg=0.03"
        ),
    )
    parser.add_argument(
        "--sort-by",
        choices=["round", "score", "speedup", "depth", "id"],
        default="round",
        help="Sort summary rows.",
    )
    parser.add_argument(
        "--stop-only",
        action="store_true",
        help="Only print nodes judged as stop=True.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Only print nodes whose early-stopping score is at least this value.",
    )
    parser.add_argument(
        "--detail-id",
        type=str,
        default=None,
        help="Print full details for an exact id or id prefix.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print full details for every displayed row.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write all decisions and detailed sub-scores to a JSON file.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = apply_config_overrides(
        EarlyStoppingConfig(),
        config_json=args.config_json,
        set_values=args.set,
    )
    nodes, warnings = load_island_nodes(args.island_dir)
    children_map = build_children_map(nodes)
    decisions = judge_all_nodes(nodes, config)
    decisions = sort_decisions(decisions, args.sort_by)

    print_dataset_summary(args.island_dir, nodes, children_map, decisions, warnings)
    print()
    print("config:")
    print(json.dumps(dataclasses.asdict(config), ensure_ascii=False, indent=2))
    print()

    displayed = filter_decisions(
        decisions,
        stop_only=args.stop_only,
        min_score=args.min_score,
        detail_id=args.detail_id,
    )
    rows = make_summary_rows(displayed, args.island_dir)
    print_table(
        rows,
        columns=[
            "round",
            "depth",
            "stop",
            "score",
            "speedup",
            "gain",
            "slope",
            "ncu",
            "low",
            "id",
            "parent",
            "dir",
            "reasons",
        ],
    )

    if args.detail_id or args.details:
        if args.detail_id and not displayed:
            print(f"\n[warn] no node matched detail id/prefix: {args.detail_id}")
        for item in displayed:
            print_node_detail(item)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "island_dir": str(args.island_dir),
                    "config": dataclasses.asdict(config),
                    "decisions": [decision_to_jsonable(item) for item in decisions],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\njson written to: {args.json_out}")


if __name__ == "__main__":
    main()
