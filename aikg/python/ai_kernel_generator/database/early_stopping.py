from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math



EPS = 1e-12

NCU_METRIC_LIST = [
    # Overall work/runtime behavior.
    "sm__cycles_active.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed.sum",
    # Compute-pipe utilization.
    "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    # DRAM traffic and pressure.
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    # Cache behavior.
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_active",
    # Major stall reasons for dense/pointwise/reduction kernels.
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
]

@dataclass
class IterationRecord:
    """
    表示某个分支上一次迭代/一次生成的算子结果。
    """
    step: int
    speedup: float
    profile: Dict[str, float] = field(default_factory=dict)


@dataclass
class EarlyStoppingConfig:
    """
    收敛判定配置
    """

    # -------- 基本长度要求 --------
    min_history: int = 3              # 至少多少轮后才开始判停
    patience: int = 4                 # 连续多少轮低收益 / 平台后可触发

    # -------- 增量收益阈值 --------
    abs_improve_threshold: float = 0.05   # speedup绝对提升阈值，例如 0.01 表示提升不到1%
    rel_improve_threshold: float = 0.01  # 相对提升阈值，例如 0.5%

    # -------- 趋势斜率 --------
    moving_avg_window: int = 3            # 求平均窗口大小 -> 窗口内求得一个平滑点
    slope_window: int = 4                 # 用最近多少个平滑点估计趋势
    slope_threshold: float = 0.2        # 平均每步提升斜率阈值

    # -------- 波动噪声 --------
    noise_tolerance: float = 0.000        # 允许的小波动，可根据测量噪声调

    # -------- NCU 平台化判定 --------
    ncu_stable_window: int = 5            # 最近几轮看NCU是否平台化
    ncu_metric_thresholds: Dict[str, float] = field(default_factory=lambda: {
        # Thresholds are relative changes. For example, 0.03 means 3%.
        "sm__cycles_active.avg": 0.03,
        "sm__warps_active.avg.pct_of_peak_sustained_active": 0.03,
        "sm__inst_executed.sum": 0.03,
        "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active": 0.05,
        "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active": 0.05,
        "dram__bytes_read.sum": 0.03,
        "dram__bytes_write.sum": 0.03,
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": 0.05,
        "l1tex__t_sector_hit_rate.pct": 0.05,
        "l1tex__throughput.avg.pct_of_peak_sustained_active": 0.05,
        "lts__t_sector_hit_rate.pct": 0.05,
        "lts__throughput.avg.pct_of_peak_sustained_active": 0.05,
        "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct": 0.08,
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": 0.08,
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": 0.08,
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct": 0.08,
    })
    ncu_plateau_ratio: float = 0.65       # 超过多少比例关键指标都平台化，则认为NCU平台化

    # -------- 收敛得分权重 --------
    weight_gain: float = 0.45
    weight_slope: float = 0.45
    weight_ncu: float = 0.10
    # weight_stagnation: float = 0.15

    # -------- 最终停止阈值 --------
    stop_score_threshold: float = 0.7

    # -------- 安全限制 --------
    max_depth: Optional[int] = 20               # 超出 max_depth 直接判停
    hard_no_improve_rounds: Optional[int] = 3   # 若连续4轮几乎无提升，直接停，无需和其他指标加权


@dataclass
class EarlyStoppingDecision:
    stop: bool
    score: float
    reasons: List[str]
    details: Dict[str, Any] = field(default_factory=dict)


class BranchEarlyStoppingJudge:
    def __init__(self, config: Optional[EarlyStoppingConfig] = None):
        self.cfg = config or EarlyStoppingConfig()

    # =========================
    # 对外主接口
    # =========================
    def judge(
        self,
        history: List[IterationRecord]
    ) -> EarlyStoppingDecision:
        """
        输入某个分支的历史记录，输出是否应停止当前分支探索。
        """
    
        reasons = []
        details = {}

        # 1) 基础检查
        valid_history = [r for r in history if r.speedup is not None]
        if len(valid_history) < self.cfg.min_history:
            # 如果分支深度不够长则无需判停
            return EarlyStoppingDecision(
                stop=False,
                score=0.0,
                reasons=[f"history too short: {len(valid_history)} < min_history={self.cfg.min_history}"],
                details={}
            )

        current_depth = len(history)
        if self.cfg.max_depth is not None and current_depth is not None:
            if current_depth >= self.cfg.max_depth:
                return EarlyStoppingDecision(
                    stop=True,
                    score=1.0,
                    reasons=[f"reach max_depth={self.cfg.max_depth}"],
                    details={"current_depth": current_depth}
                )

        speedups = [r.speedup for r in valid_history]

        # 2) 各子判定
        gain_info = self._evaluate_gain(speedups)
        slope_info = self._evaluate_slope(speedups)
        stagnation_info = self._evaluate_stagnation(speedups)
        ncu_info = self._evaluate_ncu_plateau(valid_history)

        details["gain"] = gain_info
        details["slope"] = slope_info
        details["stagnation"] = stagnation_info
        details["ncu"] = ncu_info

        # 3) 硬判停规则
        if self.cfg.hard_no_improve_rounds is not None:
            if stagnation_info["consecutive_low_gain_rounds"] >= self.cfg.hard_no_improve_rounds:
                reasons.append(
                    f"hard stop: consecutive low-gain rounds = "
                    f"{stagnation_info['consecutive_low_gain_rounds']}"
                )
                return EarlyStoppingDecision(
                    stop=True,
                    score=1.0,
                    reasons=reasons,
                    details=details
                )

        # 4) 多信号组合得分
        score = (
            self.cfg.weight_gain * gain_info["score"] +
            self.cfg.weight_slope * slope_info["score"] +
            self.cfg.weight_ncu * ncu_info["score"]
        )

        if gain_info["low_gain"]:
            reasons.append("recent improvement is below threshold")
        if slope_info["flat_trend"]:
            reasons.append("smoothed trend slope is near zero")
        if ncu_info["plateau"]:
            reasons.append("key NCU metrics are plateaued")
        if stagnation_info["stagnating"]:
            reasons.append("branch has stagnated for multiple rounds")

        stop = score >= self.cfg.stop_score_threshold

        if stop and not reasons:
            reasons.append("EarlyStopping score exceeds threshold")

        return EarlyStoppingDecision(
            stop=stop,
            score=score,
            reasons=reasons,
            details=details
        )

    # =========================
    # 子逻辑1：增量收益
    # =========================
    def _evaluate_gain(self, speedups: List[float]) -> Dict[str, Any]:
        recent_pairs = list(zip(speedups[:-1], speedups[1:]))
        if not recent_pairs:
            return {
                "score": 0.0,
                "low_gain": False,
                "recent_abs_improvements": [],
                "recent_rel_improvements": [],
            }

        recent_pairs = recent_pairs[-self.cfg.patience:]
        abs_improvements = [b - a for a, b in recent_pairs]
        rel_improvements = [(b - a) / (abs(a) + EPS) for a, b in recent_pairs]

        low_abs = [x <= self.cfg.abs_improve_threshold + self.cfg.noise_tolerance for x in abs_improvements]
        low_rel = [x <= self.cfg.rel_improve_threshold + self.cfg.noise_tolerance for x in rel_improvements]

        low_gain_count = sum(1 for a, b in zip(low_abs, low_rel) if a and b)
        low_gain = low_gain_count >= max(1, math.ceil(len(recent_pairs) * 0.75))

        score = low_gain_count / max(1, len(recent_pairs))

        return {
            "score": min(1.0, score),
            "low_gain": low_gain,
            "recent_abs_improvements": abs_improvements,
            "recent_rel_improvements": rel_improvements,
            "low_gain_count": low_gain_count,
        }

    # =========================
    # 子逻辑2：平滑趋势斜率
    # =========================
    def _moving_average(self, arr: List[float], window: int) -> List[float]:
        if window <= 1:
            return arr[:]
        if len(arr) < window:
            return [sum(arr) / len(arr)] if arr else []
        out = []
        for i in range(window - 1, len(arr)):
            seg = arr[i - window + 1 : i + 1]
            out.append(sum(seg) / len(seg))
        return out

    def _linear_slope(self, y: List[float]) -> float:
        """
        求斜率
        对 y = a*x + b 做最小二乘，返回 slope=a
        """
        n = len(y)
        if n <= 1:
            return 0.0
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(y) / n
        num = sum((x - mean_x) * (v - mean_y) for x, v in zip(xs, y))
        den = sum((x - mean_x) ** 2 for x in xs) + EPS
        return num / den

    def _evaluate_slope(self, speedups: List[float]) -> Dict[str, Any]:
        ma = self._moving_average(speedups, self.cfg.moving_avg_window)
        if len(ma) < 2:
            return {
                "score": 0.0,
                "flat_trend": False,
                "moving_average": ma,
                "slope": 0.0,
            }

        recent_ma = ma[-self.cfg.slope_window:] if len(ma) >= self.cfg.slope_window else ma
        slope = self._linear_slope(recent_ma)

        flat_trend = slope <= self.cfg.slope_threshold + self.cfg.noise_tolerance

        # slope低于阈值表示收益增长不足，越倾向停止；超过阈值越多，早停分越低。
        # 使用以 slope_limit 为中心的 S 型曲线，避免阈值以下全部变成 1.0。
        slope_limit = self.cfg.slope_threshold + self.cfg.noise_tolerance
        slope_scale = abs(slope_limit) + EPS
        normalized_delta = (slope - slope_limit) / slope_scale
        normalized_delta = max(-60.0, min(60.0, normalized_delta))
        score = 1.0 / (1.0 + math.exp(normalized_delta))
        return {
            "score": score,
            "flat_trend": flat_trend,
            "moving_average": ma,
            "recent_moving_average": recent_ma,
            "slope": slope,
        }

    # =========================
    # 子逻辑3：连续停滞
    # =========================
    def _evaluate_stagnation(self, speedups: List[float]) -> Dict[str, Any]:
        consecutive = 0
        for i in range(len(speedups) - 1, 0, -1):
            prev_v = speedups[i - 1]
            curr_v = speedups[i]
            abs_gain = curr_v - prev_v
            rel_gain = abs_gain / (abs(prev_v) + EPS)

            is_low = (
                abs_gain <= self.cfg.abs_improve_threshold + self.cfg.noise_tolerance and
                rel_gain <= self.cfg.rel_improve_threshold + self.cfg.noise_tolerance
            )
            if is_low:
                consecutive += 1
            else:
                break

        stagnating = consecutive >= self.cfg.patience
        score = min(1.0, consecutive / max(1, self.cfg.patience))

        return {
            "score": score,
            "stagnating": stagnating,
            "consecutive_low_gain_rounds": consecutive,
        }

    # =========================
    # 子逻辑4：NCU平台化
    # =========================
    def _evaluate_ncu_plateau(self, history: List[IterationRecord]) -> Dict[str, Any]:
        metric_thresholds = self.cfg.ncu_metric_thresholds
        if len(history) < self.cfg.ncu_stable_window:
            return {
                "score": 0.0,
                "plateau": False,
                "metric_results": {},
            }

        recent = history[-self.cfg.ncu_stable_window:]
        metric_results = {}
        plateau_count = 0
        valid_metric_count = 0

        for metric_name, threshold in metric_thresholds.items():
            vals = []
            for r in recent:
                if metric_name in r.profile and r.profile[metric_name] is not None:
                    vals.append(r.profile[metric_name])

            # 至少需要2个点才判断变化
            if len(vals) < 2:
                continue

            abs_changes = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
            rel_changes = [
                abs(vals[i] - vals[i - 1]) / (abs(vals[i - 1]) + EPS)
                for i in range(1, len(vals))
            ]
            avg_abs_change = sum(abs_changes) / len(abs_changes)
            avg_rel_change = sum(rel_changes) / len(rel_changes)
            stable = avg_rel_change <= threshold

            valid_metric_count += 1
            if stable:
                plateau_count += 1

            metric_results[metric_name] = {
                "values": vals,
                "avg_abs_change": avg_abs_change,
                "avg_rel_change": avg_rel_change,
                "threshold": threshold,
                "stable": stable,
            }

        if valid_metric_count == 0:
            return {
                "score": 0.0,
                "plateau": False,
                "metric_results": metric_results,
            }

        plateau_ratio = plateau_count / valid_metric_count
        plateau = plateau_ratio >= self.cfg.ncu_plateau_ratio

        return {
            "score": plateau_ratio,
            "plateau": plateau,
            "plateau_ratio": plateau_ratio,
            "plateau_count": plateau_count,
            "valid_metric_count": valid_metric_count,
            "metric_results": metric_results,
        }
        
if __name__ == '__main__':
    # test early stopping
    import random
    iter_record_list = [IterationRecord(i, random.random(), { key:random.random() for key in NCU_METRIC_LIST }) for i in range(10)]
    early_stopping_judge = BranchEarlyStoppingJudge()
    print(early_stopping_judge.judge(iter_record_list))
