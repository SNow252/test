"""
Noisy constrained optimization experiment
Problem: min f(x)+epsilon(x), x in R^2, feasible region x1>=0, x2>=0, x1^2+x2^2<=1.
Algorithms:
  1) SA: Simulated Annealing with exponential cooling and Gaussian neighbor.
  2) GA: Real-coded Genetic Algorithm with elitism and adaptive mutation step.
  3) ARSA: Adaptive Resampling Simulated Annealing, a noise-tolerant method.

Run:
  python noisy_constrained_optimization.py --runs 50 --budget 2000 --outdir results

Optional parameter sensitivity:
  python noisy_constrained_optimization.py --runs 50 --budget 2000 --sensitivity --sens-runs 15 --outdir results

Dependencies:
  numpy pandas matplotlib scipy
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


# -----------------------------
# Problem definition
# -----------------------------

def true_f(x: np.ndarray) -> float:
    x1, x2 = float(x[0]), float(x[1])
    return (x1 - 1.0) ** 2 + (x2 - 1.0) ** 2 + math.sin(3.0 * math.pi * x1) * math.cos(2.0 * math.pi * x2)


def noise_sigma(x: np.ndarray) -> float:
    x1, x2 = abs(float(x[0])), abs(float(x[1]))
    return 0.1 + 0.2 * (x1 + x2) / (1.0 + x1 + x2)


def noisy_F(x: np.ndarray, rng: np.random.Generator) -> float:
    return true_f(x) + rng.normal(0.0, noise_sigma(x))


def feasible(x: np.ndarray, tol: float = 1e-12) -> bool:
    return bool(x[0] >= -tol and x[1] >= -tol and x[0] ** 2 + x[1] ** 2 <= 1.0 + tol)


def project_to_feasible(x: np.ndarray) -> np.ndarray:
    """Projection-like repair: first clip to first quadrant, then radial projection to unit disk."""
    y = np.maximum(np.asarray(x, dtype=float), 0.0)
    norm = np.linalg.norm(y)
    if norm > 1.0:
        y = y / norm
    return y


def random_feasible(rng: np.random.Generator) -> np.ndarray:
    """Uniform sample from the first-quadrant unit disk."""
    r = math.sqrt(rng.random())
    theta = rng.uniform(0.0, math.pi / 2.0)
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=float)


class EvalCounter:
    def __init__(self, budget: int, rng: np.random.Generator):
        self.budget = int(budget)
        self.rng = rng
        self.count = 0

    def can_eval(self, n: int = 1) -> bool:
        return self.count + n <= self.budget

    def eval_noisy(self, x: np.ndarray) -> float:
        if not self.can_eval(1):
            raise RuntimeError("Evaluation budget exhausted")
        self.count += 1
        return noisy_F(x, self.rng)


@dataclass
class RunResult:
    algorithm: str
    run: int
    best_true: float
    best_x1: float
    best_x2: float
    feasible: bool
    evals: int


# -----------------------------
# Utilities
# -----------------------------

def update_best(candidate: np.ndarray, best_x: np.ndarray | None, best_true: float) -> Tuple[np.ndarray, float]:
    if feasible(candidate):
        val = true_f(candidate)
        if best_x is None or val < best_true:
            return candidate.copy(), val
    return best_x, best_true


def append_curve(curve: List[Tuple[int, float]], evals: int, best_true: float) -> None:
    if np.isfinite(best_true):
        curve.append((evals, best_true))


# -----------------------------
# Algorithm 1: Standard SA
# -----------------------------

def run_sa(
    rng: np.random.Generator,
    budget: int = 2000,
    T0: float = 1.0,
    alpha: float = 0.995,
    step0: float = 0.20,
) -> Tuple[np.ndarray, float, List[Tuple[int, float]]]:
    counter = EvalCounter(budget, rng)
    x = random_feasible(rng)
    y = counter.eval_noisy(x)
    best_x, best_true = update_best(x, None, float("inf"))
    curve: List[Tuple[int, float]] = []
    append_curve(curve, counter.count, best_true)

    k = 0
    while counter.can_eval(1):
        T = max(T0 * (alpha ** k), 1e-8)
        step = step0 * math.sqrt(max(T / T0, 1e-8))
        cand = project_to_feasible(x + rng.normal(0.0, step, size=2))
        yc = counter.eval_noisy(cand)
        delta = yc - y
        if delta <= 0.0 or rng.random() < math.exp(-delta / T):
            x, y = cand, yc
        best_x, best_true = update_best(x, best_x, best_true)
        append_curve(curve, counter.count, best_true)
        k += 1

    return best_x, best_true, curve


# -----------------------------
# Algorithm 2: Real-coded GA
# -----------------------------

def tournament_select(pop: np.ndarray, noisy_fit: np.ndarray, rng: np.random.Generator, k: int = 3) -> np.ndarray:
    idx = rng.choice(len(pop), size=k, replace=False)
    return pop[idx[np.argmin(noisy_fit[idx])]].copy()


def blend_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    lam = rng.random()
    return lam * p1 + (1.0 - lam) * p2, lam * p2 + (1.0 - lam) * p1


def run_ga(
    rng: np.random.Generator,
    budget: int = 2000,
    pop_size: int = 40,
    elite_frac: float = 0.10,
    crossover_prob: float = 0.90,
    mutation_step0: float = 0.20,
    mutation_prob: float = 0.30,
    adapt_window: int = 5,
) -> Tuple[np.ndarray, float, List[Tuple[int, float]]]:
    counter = EvalCounter(budget, rng)
    pop = np.array([random_feasible(rng) for _ in range(pop_size)])
    noisy_fit = np.array([counter.eval_noisy(ind) for ind in pop])

    true_vals = np.array([true_f(ind) for ind in pop])
    best_idx = int(np.argmin(true_vals))
    best_x = pop[best_idx].copy()
    best_true = float(true_vals[best_idx])
    curve: List[Tuple[int, float]] = [(counter.count, best_true)]

    elite_num = max(1, int(pop_size * elite_frac))
    mutation_step = mutation_step0
    no_improve = 0
    last_gen_best = best_true

    while counter.can_eval(pop_size):
        order = np.argsort(noisy_fit)
        elites = pop[order[:elite_num]].copy()

        new_pop = [e.copy() for e in elites]
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, noisy_fit, rng)
            p2 = tournament_select(pop, noisy_fit, rng)
            if rng.random() < crossover_prob:
                c1, c2 = blend_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            for c in (c1, c2):
                if rng.random() < mutation_prob:
                    c += rng.normal(0.0, mutation_step, size=2)
                c = project_to_feasible(c)
                new_pop.append(c)
                if len(new_pop) >= pop_size:
                    break

        pop = np.array(new_pop[:pop_size])
        noisy_fit = np.array([counter.eval_noisy(ind) for ind in pop])

        gen_true_vals = np.array([true_f(ind) for ind in pop])
        gen_best_idx = int(np.argmin(gen_true_vals))
        gen_best_true = float(gen_true_vals[gen_best_idx])
        if gen_best_true < best_true:
            best_true = gen_best_true
            best_x = pop[gen_best_idx].copy()

        # Adaptive mutation step: enlarge if stagnating, shrink if improving.
        if gen_best_true < last_gen_best - 1e-6:
            mutation_step = max(0.03, mutation_step * 0.90)
            no_improve = 0
            last_gen_best = gen_best_true
        else:
            no_improve += 1
            if no_improve >= adapt_window:
                mutation_step = min(0.50, mutation_step * 1.25)
                no_improve = 0

        append_curve(curve, counter.count, best_true)

    # Spend remaining budget on re-evaluating current population if budget not divisible by pop_size.
    # These evaluations influence noisy fitness only; best_true is always measured by true_f.
    while counter.can_eval(1):
        i = int(rng.integers(0, pop_size))
        noisy_fit[i] = counter.eval_noisy(pop[i])
        append_curve(curve, counter.count, best_true)

    return best_x, best_true, curve


# -----------------------------
# Algorithm 3: Adaptive Resampling SA
# -----------------------------

def adaptive_mean_eval(
    counter: EvalCounter,
    x: np.ndarray,
    min_rep: int,
    max_rep: int,
) -> Tuple[float, int]:
    vals = []
    reps = 0
    while reps < min_rep and counter.can_eval(1):
        vals.append(counter.eval_noisy(x))
        reps += 1
    return float(np.mean(vals)), reps


def compare_with_resampling(
    counter: EvalCounter,
    x: np.ndarray,
    cand: np.ndarray,
    base_reps: int,
    max_reps: int,
    uncertainty_scale: float,
) -> Tuple[float, float]:
    """Estimate noisy means. If the two means are close relative to uncertainty, resample more."""
    vals_x: List[float] = []
    vals_c: List[float] = []
    for _ in range(base_reps):
        if not counter.can_eval(2):
            break
        vals_x.append(counter.eval_noisy(x))
        vals_c.append(counter.eval_noisy(cand))

    while len(vals_x) < max_reps and counter.can_eval(2):
        mx, mc = float(np.mean(vals_x)), float(np.mean(vals_c))
        sx = noise_sigma(x) / math.sqrt(len(vals_x))
        sc = noise_sigma(cand) / math.sqrt(len(vals_c))
        threshold = uncertainty_scale * math.sqrt(sx * sx + sc * sc)
        if abs(mc - mx) > threshold:
            break
        vals_x.append(counter.eval_noisy(x))
        vals_c.append(counter.eval_noisy(cand))

    return float(np.mean(vals_x)), float(np.mean(vals_c))


def run_arsa(
    rng: np.random.Generator,
    budget: int = 2000,
    T0: float = 0.8,
    alpha: float = 0.996,
    step0: float = 0.18,
    base_reps: int = 2,
    max_reps: int = 8,
    uncertainty_scale: float = 1.5,
) -> Tuple[np.ndarray, float, List[Tuple[int, float]]]:
    """
    ARSA: Adaptive Resampling Simulated Annealing.
    Key idea: because F is noisy and heteroscedastic, compare candidate/current by repeated sampling.
    More samples are allocated only when the noisy means are statistically close.
    """
    counter = EvalCounter(budget, rng)
    x = random_feasible(rng)
    best_x, best_true = update_best(x, None, float("inf"))
    curve: List[Tuple[int, float]] = []
    append_curve(curve, counter.count, best_true)

    k = 0
    while counter.can_eval(2 * base_reps):
        T = max(T0 * (alpha ** k), 1e-8)
        # Adapt step with both temperature and local noise. Higher local noise -> slightly larger exploration.
        local_noise_factor = 1.0 + noise_sigma(x)
        step = step0 * math.sqrt(max(T / T0, 1e-8)) * local_noise_factor
        cand = project_to_feasible(x + rng.normal(0.0, step, size=2))

        y_x, y_c = compare_with_resampling(
            counter, x, cand, base_reps=base_reps, max_reps=max_reps, uncertainty_scale=uncertainty_scale
        )
        delta = y_c - y_x
        # Conservative acceptance under uncertainty: use temperature plus noise standard error.
        eff_T = T + 0.5 * math.sqrt(noise_sigma(x) ** 2 + noise_sigma(cand) ** 2)
        if delta <= 0.0 or rng.random() < math.exp(-delta / max(eff_T, 1e-8)):
            x = cand
        best_x, best_true = update_best(x, best_x, best_true)
        append_curve(curve, counter.count, best_true)
        k += 1

    # Use tiny remaining budget for local random search around best_x.
    while counter.can_eval(1):
        cand = project_to_feasible(best_x + rng.normal(0.0, 0.03, size=2))
        _ = counter.eval_noisy(cand)
        best_x, best_true = update_best(cand, best_x, best_true)
        append_curve(curve, counter.count, best_true)

    return best_x, best_true, curve


# -----------------------------
# Experiment, statistics, plots
# -----------------------------

def run_all(runs: int, budget: int, seed: int) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[int, float]]]]:
    algorithms: Dict[str, Callable[[np.random.Generator, int], Tuple[np.ndarray, float, List[Tuple[int, float]]]]] = {
        "SA": lambda rng, b: run_sa(rng, budget=b),
        "GA": lambda rng, b: run_ga(rng, budget=b),
        "ARSA": lambda rng, b: run_arsa(rng, budget=b),
    }
    records: List[RunResult] = []
    curves: Dict[str, List[Tuple[int, float]]] = {name: [] for name in algorithms}

    master = np.random.default_rng(seed)
    for run_id in range(runs):
        for alg_name, alg_fn in algorithms.items():
            rng = np.random.default_rng(int(master.integers(0, 2**32 - 1)))
            best_x, best_true, curve = alg_fn(rng, budget)
            records.append(
                RunResult(
                    algorithm=alg_name,
                    run=run_id,
                    best_true=float(best_true),
                    best_x1=float(best_x[0]),
                    best_x2=float(best_x[1]),
                    feasible=feasible(best_x),
                    evals=budget,
                )
            )
            # Store with algorithm/run encoded through list; aggregation happens later.
            curves[alg_name].append(curve)
    return pd.DataFrame([asdict(r) for r in records]), curves


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alg, g in df.groupby("algorithm"):
        q1 = g["best_true"].quantile(0.25)
        q3 = g["best_true"].quantile(0.75)
        rows.append(
            {
                "algorithm": alg,
                "median_best_true": g["best_true"].median(),
                "Q1": q1,
                "Q3": q3,
                "IQR": q3 - q1,
                "mean_best_true": g["best_true"].mean(),
                "std_best_true": g["best_true"].std(ddof=1),
                "feasible_ratio": g["feasible"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("median_best_true")


def wilcoxon_tests(df: pd.DataFrame) -> pd.DataFrame:
    algs = sorted(df["algorithm"].unique())
    rows = []
    pivot = df.pivot(index="run", columns="algorithm", values="best_true")
    for i in range(len(algs)):
        for j in range(i + 1, len(algs)):
            a, b = algs[i], algs[j]
            stat, p = wilcoxon(pivot[a], pivot[b], alternative="two-sided", zero_method="wilcox")
            rows.append({"alg_A": a, "alg_B": b, "wilcoxon_stat": stat, "p_value": p})
    return pd.DataFrame(rows)


def mean_curve(curves_for_alg: List[List[Tuple[int, float]]], budget: int, grid_size: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(1, budget, grid_size)
    mat = []
    for curve in curves_for_alg:
        if len(curve) == 0:
            continue
        xs = np.array([p[0] for p in curve], dtype=float)
        ys = np.array([p[1] for p in curve], dtype=float)
        # Stepwise best-so-far interpolation.
        vals = []
        idx = 0
        current = ys[0]
        for g in grid:
            while idx + 1 < len(xs) and xs[idx + 1] <= g:
                idx += 1
                current = ys[idx]
            vals.append(current)
        mat.append(vals)
    return grid, np.mean(np.asarray(mat), axis=0)


def plot_results(df: pd.DataFrame, curves: Dict[str, List[List[Tuple[int, float]]]], budget: int, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Boxplot
    plt.figure(figsize=(7, 4.5))
    labels = sorted(df["algorithm"].unique())
    data = [df[df["algorithm"] == alg]["best_true"].values for alg in labels]
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Best true f(x), lower is better")
    plt.title("Final performance over independent runs")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "boxplot_final_true_f.png"), dpi=200)
    plt.close()

    # Mean convergence curve
    plt.figure(figsize=(7, 4.5))
    for alg in labels:
        grid, mean_vals = mean_curve(curves[alg], budget)
        plt.plot(grid, mean_vals, label=alg)
    plt.xlabel("Number of objective evaluations")
    plt.ylabel("Mean best-so-far true f(x)")
    plt.title("Convergence curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "convergence_curves.png"), dpi=200)
    plt.close()

    # Scatter of best solutions
    plt.figure(figsize=(5.5, 5.0))
    theta = np.linspace(0, math.pi / 2, 200)
    plt.plot(np.cos(theta), np.sin(theta), linestyle="--", linewidth=1)
    for alg in labels:
        g = df[df["algorithm"] == alg]
        plt.scatter(g["best_x1"], g["best_x2"], s=18, alpha=0.7, label=alg)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Best solutions found")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "best_solution_scatter.png"), dpi=200)
    plt.close()


def parameter_sensitivity(sens_runs: int, budget: int, seed: int, outdir: str) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)
    master = np.random.default_rng(seed + 12345)
    rows = []

    # SA grid: initial temperature and cooling rate.
    T0_values = [0.3, 0.8, 1.5, 3.0]
    alpha_values = [0.990, 0.995, 0.998]
    for T0 in T0_values:
        for alpha in alpha_values:
            vals = []
            for _ in range(sens_runs):
                rng = np.random.default_rng(int(master.integers(0, 2**32 - 1)))
                _, best_true, _ = run_sa(rng, budget=budget, T0=T0, alpha=alpha)
                vals.append(best_true)
            rows.append({"algorithm": "SA", "param1": "T0", "value1": T0, "param2": "alpha", "value2": alpha,
                         "median_best_true": float(np.median(vals)), "IQR": float(np.quantile(vals, .75) - np.quantile(vals, .25))})

    # GA grid: crossover probability and mutation step.
    pc_values = [0.6, 0.8, 0.9, 1.0]
    mut_values = [0.08, 0.15, 0.25, 0.40]
    for pc in pc_values:
        for ms in mut_values:
            vals = []
            for _ in range(sens_runs):
                rng = np.random.default_rng(int(master.integers(0, 2**32 - 1)))
                _, best_true, _ = run_ga(rng, budget=budget, crossover_prob=pc, mutation_step0=ms)
                vals.append(best_true)
            rows.append({"algorithm": "GA", "param1": "crossover_prob", "value1": pc, "param2": "mutation_step0", "value2": ms,
                         "median_best_true": float(np.median(vals)), "IQR": float(np.quantile(vals, .75) - np.quantile(vals, .25))})

    sens = pd.DataFrame(rows)
    sens.to_csv(os.path.join(outdir, "parameter_sensitivity.csv"), index=False)

    def heatmap(sub: pd.DataFrame, title: str, filename: str) -> None:
        xvals = sorted(sub["value1"].unique())
        yvals = sorted(sub["value2"].unique())
        z = np.zeros((len(yvals), len(xvals)))
        for i, y in enumerate(yvals):
            for j, x in enumerate(xvals):
                z[i, j] = sub[(sub["value1"] == x) & (sub["value2"] == y)]["median_best_true"].iloc[0]
        plt.figure(figsize=(6.5, 4.8))
        im = plt.imshow(z, origin="lower", aspect="auto")
        plt.colorbar(im, label="Median best true f(x)")
        plt.xticks(range(len(xvals)), [str(v) for v in xvals])
        plt.yticks(range(len(yvals)), [str(v) for v in yvals])
        plt.xlabel(sub["param1"].iloc[0])
        plt.ylabel(sub["param2"].iloc[0])
        plt.title(title)
        for i in range(len(yvals)):
            for j in range(len(xvals)):
                plt.text(j, i, f"{z[i, j]:.3f}", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, filename), dpi=200)
        plt.close()

    heatmap(sens[sens["algorithm"] == "SA"], "SA parameter sensitivity", "sensitivity_SA_heatmap.png")
    heatmap(sens[sens["algorithm"] == "GA"], "GA parameter sensitivity", "sensitivity_GA_heatmap.png")
    return sens


def write_report_template(outdir: str) -> None:
    text = """# 实验报告模板：带噪声的约束优化问题与自适应算法设计

## 1. 问题描述
本实验研究二维约束优化问题：在第一象限单位圆盘内最小化带异方差高斯噪声的目标函数 F(x)=f(x)+epsilon(x)。实验评估时，算法选择过程使用带噪声目标 F(x)，但最终性能统计使用无噪声真实目标 f(x)。

## 2. 算法设计
### 2.1 标准模拟退火 SA
- 指数降温：T_k = T0 * alpha^k。
- 邻域：二维高斯扰动。
- 约束处理：先将坐标截断到非负，再将圆盘外点径向投影回单位圆盘。
- 接受准则：Metropolis 准则。

### 2.2 实数编码遗传算法 GA
- 种群实数编码，每个个体为 x=(x1,x2)。
- 选择：锦标赛选择。
- 交叉：blend crossover。
- 变异：高斯变异。
- 精英保留：保留前 10% 个体。
- 自适应变异步长：若停滞则增大步长，若改进则缩小步长。

### 2.3 自适应噪声容忍算法 ARSA
ARSA 结合模拟退火和自适应重采样。由于目标函数评估存在异方差噪声，ARSA 在比较当前解和候选解时，先分别进行少量重采样；若两者均值差异相对估计不确定性较小，则增加采样次数。这样可以将评估预算集中在“难以判断优劣”的候选比较上，从而降低噪声误导接受决策的概率。

## 3. 参数设置
- 每种算法独立运行 50 次。
- 每次运行预算为 2000 次目标函数评估。
- SA 默认参数：T0=1.0, alpha=0.995, step0=0.20。
- GA 默认参数：population=40, crossover_prob=0.90, mutation_step0=0.20。
- ARSA 默认参数：T0=0.8, alpha=0.996, base_reps=2, max_reps=8。

## 4. 实验结果
请插入以下代码生成的结果：
- summary.csv：中位数、IQR、均值、标准差、可行解比例。
- wilcoxon_tests.csv：算法两两 Wilcoxon 符号秩检验结果。
- boxplot_final_true_f.png：最终真实目标值箱线图。
- convergence_curves.png：平均收敛曲线。
- best_solution_scatter.png：最优解位置分布。

## 5. 参数敏感性分析
使用 --sensitivity 参数运行后，插入：
- sensitivity_SA_heatmap.png
- sensitivity_GA_heatmap.png

分析建议：观察 SA 的初始温度和降温速率是否导致过早收敛或过度随机搜索；观察 GA 的交叉概率与变异步长是否影响种群多样性和局部搜索能力。

## 6. 开放性探究：在线估计噪声并动态调整参数
本文选择“设计一种在线估计噪声水平并动态调整算法参数”的开放题。ARSA 使用已知噪声函数 sigma(x) 近似评估候选比较的不确定性，并根据不确定性动态决定重采样次数。当两个解的噪声均值差异较小时，算法增加采样次数；当差异明显时，减少额外采样，以节省预算。实验中可以通过比较 ARSA 与 SA 的中位数、IQR、收敛曲线和 Wilcoxon 检验 p 值来验证其有效性。

## 7. 结论
根据 summary.csv 和图表总结哪种算法更稳定、哪种算法中位数更低、噪声容忍策略是否显著改善性能。若 Wilcoxon 检验 p<0.05，可认为两种算法在最终结果上存在统计显著差异。
"""
    with open(os.path.join(outdir, "report_template.md"), "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--sens-runs", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df, curves = run_all(args.runs, args.budget, args.seed)
    summary = summarize_results(df)
    tests = wilcoxon_tests(df)

    df.to_csv(os.path.join(args.outdir, "raw_results.csv"), index=False)
    summary.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
    tests.to_csv(os.path.join(args.outdir, "wilcoxon_tests.csv"), index=False)
    plot_results(df, curves, args.budget, args.outdir)
    write_report_template(args.outdir)

    if args.sensitivity:
        parameter_sensitivity(args.sens_runs, args.budget, args.seed, args.outdir)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print("\n=== Wilcoxon signed-rank tests ===")
    print(tests.to_string(index=False))
    print(f"\nSaved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
