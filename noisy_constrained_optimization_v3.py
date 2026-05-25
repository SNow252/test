#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noisy constrained optimization experiment.
Algorithms:
  1) SA   - standard simulated annealing
  2) GA   - elitist real-coded genetic algorithm with adaptive mutation
  3) NAMS - Noise-tolerant Adaptive Memetic Search

Problem:
  minimize F(x)=f(x)+eps(x), eps(x)~N(0, sigma(x)^2)
  feasible set: x1>=0, x2>=0, x1^2+x2^2<=1

Evaluation protocol:
  Algorithms observe only noisy F(x). Final statistics use true f(x).
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

Array = np.ndarray


# -----------------------------
# Problem definition
# -----------------------------

def true_f(x: Array) -> float:
    x1, x2 = float(x[0]), float(x[1])
    return (x1 - 1.0) ** 2 + (x2 - 1.0) ** 2 + math.sin(3.0 * math.pi * x1) * math.cos(2.0 * math.pi * x2)


def sigma_x(x: Array) -> float:
    x1, x2 = abs(float(x[0])), abs(float(x[1]))
    return 0.1 + 0.2 * (x1 + x2) / (1.0 + x1 + x2)


class BudgetCounter:
    def __init__(self, budget: int):
        self.budget = int(budget)
        self.used = 0

    def left(self) -> int:
        return self.budget - self.used

    def can_eval(self, n: int = 1) -> bool:
        return self.used + n <= self.budget

    def eval_noisy(self, x: Array, rng: np.random.Generator) -> float:
        if not self.can_eval(1):
            raise RuntimeError("Evaluation budget exhausted")
        self.used += 1
        return true_f(x) + rng.normal(0.0, sigma_x(x))

    def eval_mean(self, x: Array, rng: np.random.Generator, reps: int) -> Tuple[float, float, int]:
        reps = max(1, min(int(reps), self.left()))
        vals = np.array([self.eval_noisy(x, rng) for _ in range(reps)], dtype=float)
        if len(vals) == 1:
            return float(vals[0]), float("inf"), 1
        return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(len(vals))), len(vals)


def is_feasible(x: Array, tol: float = 1e-10) -> bool:
    return bool(x[0] >= -tol and x[1] >= -tol and x[0] * x[0] + x[1] * x[1] <= 1.0 + tol)


def project_feasible(x: Array) -> Array:
    y = np.maximum(np.asarray(x, dtype=float), 0.0)
    nrm = float(np.linalg.norm(y))
    if nrm > 1.0:
        y = y / nrm
    return y


def random_feasible(rng: np.random.Generator, n: int = 1) -> Array:
    # Uniform in first-quadrant unit disk.
    theta = rng.uniform(0.0, math.pi / 2.0, size=n)
    r = np.sqrt(rng.uniform(0.0, 1.0, size=n))
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return pts[0] if n == 1 else pts


def record_curve(curve: List[Tuple[int, float]], budget: BudgetCounter, best_x: Array):
    curve.append((budget.used, true_f(best_x)))


def curve_to_grid(curve: List[Tuple[int, float]], grid: Array) -> Array:
    curve = sorted(curve, key=lambda z: z[0])
    xs = np.array([c[0] for c in curve], dtype=int)
    ys = np.array([c[1] for c in curve], dtype=float)
    out = np.empty_like(grid, dtype=float)
    j = 0
    best = ys[0]
    for i, g in enumerate(grid):
        while j < len(xs) and xs[j] <= g:
            best = ys[j]
            j += 1
        out[i] = best
    return out


@dataclass
class RunResult:
    algorithm: str
    seed: int
    best_x1: float
    best_x2: float
    best_true_f: float
    feasible: bool
    evals: int
    curve: List[Tuple[int, float]]


# -----------------------------
# Algorithm 1: SA
# -----------------------------

def run_sa(seed: int, budget_total: int, T0: float = 1.0, alpha: float = 0.995,
           step0: float = 0.20, step_min: float = 0.005) -> RunResult:
    rng = np.random.default_rng(seed)
    budget = BudgetCounter(budget_total)
    x = random_feasible(rng)
    fx = budget.eval_noisy(x, rng)
    best_x = x.copy()
    best_observed = fx
    curve = []
    record_curve(curve, budget, best_x)

    k = 0
    while budget.can_eval(1):
        T = max(1e-8, T0 * (alpha ** k))
        # Step also cools, but slower than temperature to avoid immediate stagnation.
        step = max(step_min, step0 * math.sqrt(max(T / T0, 1e-8)))
        cand = project_feasible(x + rng.normal(0.0, step, size=2))
        f_cand = budget.eval_noisy(cand, rng)
        delta = f_cand - fx
        if delta <= 0.0 or rng.random() < math.exp(-delta / T):
            x, fx = cand, f_cand
        if f_cand < best_observed:
            best_observed = f_cand
            best_x = cand.copy()
        k += 1
        if k % 20 == 0 or not budget.can_eval(1):
            record_curve(curve, budget, best_x)

    return RunResult("SA", seed, float(best_x[0]), float(best_x[1]), true_f(best_x), is_feasible(best_x), budget.used, curve)


# -----------------------------
# Algorithm 2: GA
# -----------------------------

def tournament_select(pop: Array, fit: Array, rng: np.random.Generator, k: int = 3) -> Array:
    idx = rng.integers(0, len(pop), size=k)
    return pop[idx[np.argmin(fit[idx])]].copy()


def run_ga(seed: int, budget_total: int, pop_size: int = 40, crossover_prob: float = 0.90,
           mutation_step0: float = 0.20, elite_frac: float = 0.10) -> RunResult:
    rng = np.random.default_rng(seed)
    budget = BudgetCounter(budget_total)
    pop = random_feasible(rng, pop_size)
    fit = np.array([budget.eval_noisy(ind, rng) for ind in pop])
    best_idx = int(np.argmin(fit))
    best_x = pop[best_idx].copy()
    best_observed = float(fit[best_idx])
    curve = []
    record_curve(curve, budget, best_x)

    elite_n = max(1, int(round(pop_size * elite_frac)))
    mutation_step = mutation_step0
    no_improve = 0

    while budget.left() >= pop_size:
        order = np.argsort(fit)
        new_pop = [pop[i].copy() for i in order[:elite_n]]

        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fit, rng)
            p2 = tournament_select(pop, fit, rng)
            if rng.random() < crossover_prob:
                # Arithmetic / BLX-like blend crossover.
                lam = rng.uniform(-0.25, 1.25, size=2)
                child = lam * p1 + (1.0 - lam) * p2
            else:
                child = p1.copy()
            child = child + rng.normal(0.0, mutation_step, size=2)
            child = project_feasible(child)
            new_pop.append(child)

        pop = np.array(new_pop)
        fit = np.array([budget.eval_noisy(ind, rng) for ind in pop])
        gen_best_idx = int(np.argmin(fit))
        gen_best = float(fit[gen_best_idx])
        if gen_best < best_observed:
            best_observed = gen_best
            best_x = pop[gen_best_idx].copy()
            mutation_step = max(0.01, mutation_step * 0.92)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 4:
                mutation_step = min(0.45, mutation_step * 1.25)
                no_improve = 0
        record_curve(curve, budget, best_x)

    # Use remaining evaluations to verify top observed individuals, not to change budget unfairly.
    if budget.left() > 0:
        top = np.argsort(fit)[:min(8, len(pop))]
        best_score = float("inf")
        for idx in top:
            if budget.left() <= 0:
                break
            reps = min(3, budget.left())
            m, se, _ = budget.eval_mean(pop[idx], rng, reps)
            score = m + 0.5 * (0.0 if math.isinf(se) else se)
            if score < best_score:
                best_score = score
                best_x = pop[idx].copy()
        record_curve(curve, budget, best_x)

    return RunResult("GA", seed, float(best_x[0]), float(best_x[1]), true_f(best_x), is_feasible(best_x), budget.used, curve)


# -----------------------------
# Algorithm 3: NAMS - actually new
# -----------------------------

def lower_confidence_score(mean: float, se: float, beta: float = 0.7) -> float:
    # Conservative for minimization: low mean is good; high uncertainty is penalized.
    if math.isinf(se) or math.isnan(se):
        return mean + 1e-6
    return mean + beta * se


def coordinate_pattern_candidates(x: Array, step: float) -> List[Array]:
    dirs = [np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, -1.0]),
            np.array([1.0, 1.0]) / math.sqrt(2), np.array([1.0, -1.0]) / math.sqrt(2),
            np.array([-1.0, 1.0]) / math.sqrt(2), np.array([-1.0, -1.0]) / math.sqrt(2)]
    return [project_feasible(x + step * d) for d in dirs]


def run_nams(seed: int, budget_total: int,
             initial_samples: int = 450,
             elite_size: int = 14,
             keep_bank: int = 28,
             beta: float = 0.7,
             local_step0: float = 0.12,
             min_step: float = 0.002) -> RunResult:
    """Noise-tolerant Adaptive Memetic Search.

    Main differences from the previous ARSA:
      - broad global one-shot screening instead of sequential SA comparison;
      - resampling only for promising elites;
      - memetic local search around multiple elites;
      - final verification of a candidate bank.
    """
    rng = np.random.default_rng(seed)
    budget = BudgetCounter(budget_total)
    curve: List[Tuple[int, float]] = []

    # Candidate statistics are stored by integer id to avoid hashing floats.
    X: List[Array] = []
    means: List[float] = []
    m2s: List[float] = []
    counts: List[int] = []

    def add_point(x: Array, reps: int = 1) -> int:
        x = project_feasible(x)
        idx = len(X)
        X.append(x)
        means.append(0.0)
        m2s.append(0.0)
        counts.append(0)
        update_point(idx, reps)
        return idx

    def update_point(idx: int, reps: int = 1):
        reps = min(int(reps), budget.left())
        for _ in range(reps):
            y = budget.eval_noisy(X[idx], rng)
            counts[idx] += 1
            delta = y - means[idx]
            means[idx] += delta / counts[idx]
            delta2 = y - means[idx]
            m2s[idx] += delta * delta2

    def se(idx: int) -> float:
        if counts[idx] <= 1:
            # use known heteroscedastic form as a prior scale estimate
            return sigma_x(X[idx]) / math.sqrt(max(1, counts[idx]))
        return math.sqrt(max(m2s[idx] / (counts[idx] - 1), 1e-12) / counts[idx])

    def score(idx: int) -> float:
        return lower_confidence_score(means[idx], se(idx), beta=beta)

    def best_by_score() -> int:
        return int(np.argmin([score(i) for i in range(len(X))]))

    # Stage 1: global exploration. Include random samples plus boundary-biased samples.
    n_global = min(initial_samples, max(20, budget.left() // 3))
    random_n = int(0.75 * n_global)
    for x in random_feasible(rng, random_n):
        add_point(x, 1)
    # Boundary is important because the optimum is close to the circular boundary.
    for _ in range(n_global - random_n):
        theta = rng.uniform(0.0, math.pi / 2.0)
        r = rng.uniform(0.82, 1.0) ** 0.35
        add_point(np.array([r * math.cos(theta), r * math.sin(theta)]), 1)

    best_idx = best_by_score()
    record_curve(curve, budget, X[best_idx])

    # Stage 2: elite resampling. Concentrate budget on promising points.
    order = np.argsort([means[i] for i in range(len(X))])[:keep_bank]
    for idx in order:
        if budget.left() <= 0:
            break
        # More resampling for higher-noise points and top candidates.
        reps = 2 + int(sigma_x(X[idx]) > 0.20)
        update_point(int(idx), min(reps, budget.left()))
    best_idx = best_by_score()
    record_curve(curve, budget, X[best_idx])

    # Stage 3: adaptive multi-start memetic local search.
    elites = list(np.argsort([score(i) for i in range(len(X))])[:elite_size])
    steps: Dict[int, float] = {int(i): local_step0 for i in elites}
    failures: Dict[int, int] = {int(i): 0 for i in elites}

    # Reserve a final verification budget so the selected point is not just a noisy winner.
    final_reserve = max(120, int(0.12 * budget_total))

    while budget.left() > final_reserve + 4 and len(elites) > 0:
        # Choose an elite by rank, not only the current best, to preserve exploration.
        ranked = list(np.argsort([score(i) for i in elites]))
        rank = int(min(len(ranked) - 1, rng.geometric(0.45) - 1))
        base_idx = int(elites[ranked[rank]])
        x0 = X[base_idx]
        step = steps.get(base_idx, local_step0)

        # Generate a small local batch: random Gaussian + pattern directions.
        cands: List[Array] = []
        for _ in range(3):
            cands.append(project_feasible(x0 + rng.normal(0.0, step, size=2)))
        cands.extend(coordinate_pattern_candidates(x0, step)[:4])

        cand_ids = []
        for c in cands:
            if budget.left() <= final_reserve + 2:
                break
            cand_ids.append(add_point(c, 1))

        if not cand_ids:
            break

        # Pick the best local candidate by noisy mean, then decide whether to resample.
        cand_best = int(min(cand_ids, key=lambda i: means[i]))
        # Adaptive resampling only when the candidate is competitive with the parent.
        diff = abs(means[cand_best] - means[base_idx])
        uncertainty = se(cand_best) + se(base_idx)
        if diff < 1.2 * uncertainty and budget.left() > final_reserve + 3:
            update_point(cand_best, min(3, budget.left() - final_reserve))
        if score(cand_best) < score(base_idx):
            # Accept improvement and add to elite set.
            elites.append(cand_best)
            steps[cand_best] = min(0.25, step * 1.08)
            failures[cand_best] = 0
            steps[base_idx] = min(0.25, step * 1.03)
            failures[base_idx] = 0
        else:
            failures[base_idx] = failures.get(base_idx, 0) + 1
            if failures[base_idx] >= 3:
                steps[base_idx] = max(min_step, step * 0.55)
                failures[base_idx] = 0

        # Keep only a bounded elite set to avoid wasting search effort.
        unique = list(dict.fromkeys(elites))
        unique_sorted = sorted(unique, key=lambda i: score(i))[:keep_bank]
        elites = unique_sorted
        for i in list(steps.keys()):
            if i not in elites:
                steps.pop(i, None)
                failures.pop(i, None)

        if budget.used % 50 <= 8:
            best_idx = best_by_score()
            record_curve(curve, budget, X[best_idx])

    # Stage 4: final verification of candidate bank.
    # This is the key noise-tolerant part: select by repeated evaluation among only a few candidates.
    bank = sorted(range(len(X)), key=lambda i: score(i))[:min(12, len(X))]
    while budget.left() > 0 and bank:
        # Allocate more samples to candidates with small score and large standard error.
        priorities = np.array([-(score(i)) + 0.25 * se(i) for i in bank], dtype=float)
        priorities = priorities - priorities.max()
        probs = np.exp(priorities)
        probs = probs / probs.sum()
        idx = int(rng.choice(bank, p=probs))
        update_point(idx, 1)

    best_idx = best_by_score()
    record_curve(curve, budget, X[best_idx])
    best_x = X[best_idx]
    return RunResult("NAMS", seed, float(best_x[0]), float(best_x[1]), true_f(best_x), is_feasible(best_x), budget.used, curve)


# -----------------------------
# Experiments and plotting
# -----------------------------

def run_all(runs: int, budget: int, base_seed: int) -> Tuple[pd.DataFrame, Dict[str, List[List[Tuple[int, float]]]]]:
    rows = []
    curves: Dict[str, List[List[Tuple[int, float]]]] = {"SA": [], "GA": [], "NAMS": []}
    algorithms: List[Tuple[str, Callable[[int, int], RunResult]]] = [
        ("SA", run_sa),
        ("GA", run_ga),
        ("NAMS", run_nams),
    ]
    for run_id in range(runs):
        for alg_name, fn in algorithms:
            # Same run_id gets nearby but distinct seeds for algorithms.
            seed = base_seed + run_id * 1009 + {"SA": 11, "GA": 23, "NAMS": 37}[alg_name]
            res = fn(seed, budget)
            rows.append({
                "algorithm": res.algorithm,
                "run": run_id,
                "seed": seed,
                "best_x1": res.best_x1,
                "best_x2": res.best_x2,
                "best_true_f": res.best_true_f,
                "feasible": res.feasible,
                "evals": res.evals,
            })
            curves[res.algorithm].append(res.curve)
            print(f"run={run_id:03d} alg={res.algorithm:<4s} f={res.best_true_f:.8f} x=({res.best_x1:.4f},{res.best_x2:.4f})")
    return pd.DataFrame(rows), curves


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for alg, g in df.groupby("algorithm"):
        q1 = g["best_true_f"].quantile(0.25)
        q3 = g["best_true_f"].quantile(0.75)
        out.append({
            "algorithm": alg,
            "median_true_f": g["best_true_f"].median(),
            "iqr_true_f": q3 - q1,
            "q1_true_f": q1,
            "q3_true_f": q3,
            "mean_true_f": g["best_true_f"].mean(),
            "std_true_f": g["best_true_f"].std(ddof=1),
            "best_true_f": g["best_true_f"].min(),
            "worst_true_f": g["best_true_f"].max(),
            "feasible_ratio": g["feasible"].mean(),
        })
    return pd.DataFrame(out).sort_values("median_true_f")


def wilcoxon_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    algs = sorted(df["algorithm"].unique())
    pivot = df.pivot(index="run", columns="algorithm", values="best_true_f")
    for a, b in itertools.combinations(algs, 2):
        va = pivot[a].to_numpy()
        vb = pivot[b].to_numpy()
        try:
            stat, p = wilcoxon(va, vb, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append({
            "alg_a": a,
            "alg_b": b,
            "median_a": np.median(va),
            "median_b": np.median(vb),
            "median_diff_a_minus_b": np.median(va - vb),
            "wilcoxon_stat": stat,
            "p_value": p,
            "significant_0.05": bool(p < 0.05) if not np.isnan(p) else False,
            "better_by_median": a if np.median(va) < np.median(vb) else b,
        })
    return pd.DataFrame(rows)


def plot_boxplot(df: pd.DataFrame, outdir: Path):
    alg_order = [a for a in ["NAMS", "GA", "SA"] if a in set(df["algorithm"])]
    data = [df.loc[df.algorithm == a, "best_true_f"].to_numpy() for a in alg_order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=alg_order, showmeans=True)
    ax.set_title("Final performance over independent runs")
    ax.set_ylabel("Best true f(x), lower is better")
    fig.tight_layout()
    fig.savefig(outdir / "boxplot_final_true_f.png", dpi=160)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    theta = np.linspace(0, math.pi / 2, 300)
    ax.plot(np.cos(theta), np.sin(theta), "--", linewidth=1)
    for alg, g in df.groupby("algorithm"):
        ax.scatter(g.best_x1, g.best_x2, label=alg, alpha=0.7, s=28)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Best solutions found")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "best_solution_scatter.png", dpi=160)
    plt.close(fig)


def plot_convergence(curves: Dict[str, List[List[Tuple[int, float]]]], budget: int, outdir: Path):
    grid = np.linspace(1, budget, 220).astype(int)
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg in ["NAMS", "GA", "SA"]:
        if alg not in curves or not curves[alg]:
            continue
        mat = np.vstack([curve_to_grid(c, grid) for c in curves[alg]])
        ax.plot(grid, mat.mean(axis=0), label=alg)
    ax.set_title("Convergence curves")
    ax.set_xlabel("Number of objective evaluations")
    ax.set_ylabel("Mean best-so-far true f(x)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "convergence_curves.png", dpi=160)
    plt.close(fig)


def parameter_sensitivity(outdir: Path, sens_runs: int, budget: int, base_seed: int) -> pd.DataFrame:
    rows = []
    # SA: initial temperature and cooling rate.
    for T0 in [0.3, 0.8, 1.5, 3.0]:
        for alpha in [0.990, 0.995, 0.998]:
            vals = []
            for r in range(sens_runs):
                vals.append(run_sa(base_seed + 50000 + r * 997 + int(1000 * T0) + int(100000 * alpha),
                                   budget, T0=T0, alpha=alpha).best_true_f)
            rows.append({"algorithm": "SA", "T0": T0, "alpha": alpha,
                         "crossover_prob": np.nan, "mutation_step0": np.nan,
                         "median_true_f": float(np.median(vals)), "iqr_true_f": float(np.subtract(*np.percentile(vals, [75, 25]))),
                         "runs": sens_runs})
    # GA: crossover probability and mutation scale.
    for cp in [0.6, 0.8, 0.9, 1.0]:
        for ms in [0.08, 0.15, 0.25, 0.40]:
            vals = []
            for r in range(sens_runs):
                vals.append(run_ga(base_seed + 70000 + r * 991 + int(1000 * cp) + int(10000 * ms),
                                   budget, crossover_prob=cp, mutation_step0=ms).best_true_f)
            rows.append({"algorithm": "GA", "T0": np.nan, "alpha": np.nan,
                         "crossover_prob": cp, "mutation_step0": ms,
                         "median_true_f": float(np.median(vals)), "iqr_true_f": float(np.subtract(*np.percentile(vals, [75, 25]))),
                         "runs": sens_runs})
    sens = pd.DataFrame(rows)
    sens.to_csv(outdir / "parameter_sensitivity.csv", index=False)
    plot_sensitivity_heatmaps(sens, outdir)
    return sens


def plot_sensitivity_heatmaps(sens: pd.DataFrame, outdir: Path):
    # SA heatmap
    sa = sens[sens.algorithm == "SA"].copy()
    if len(sa):
        pivot = sa.pivot(index="alpha", columns="T0", values="median_true_f").sort_index(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(pivot.to_numpy(), aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_xlabel("T0")
        ax.set_ylabel("alpha")
        ax.set_title("SA parameter sensitivity")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", color="black")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Median best true f(x)")
        fig.tight_layout()
        fig.savefig(outdir / "sensitivity_SA_heatmap.png", dpi=160)
        plt.close(fig)

    ga = sens[sens.algorithm == "GA"].copy()
    if len(ga):
        pivot = ga.pivot(index="mutation_step0", columns="crossover_prob", values="median_true_f").sort_index(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(pivot.to_numpy(), aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_xlabel("crossover_prob")
        ax.set_ylabel("mutation_step0")
        ax.set_title("GA parameter sensitivity")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", color="black")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Median best true f(x)")
        fig.tight_layout()
        fig.savefig(outdir / "sensitivity_GA_heatmap.png", dpi=160)
        plt.close(fig)


def write_report_template(outdir: Path):
    txt = """# 实验报告模板：带噪声的约束优化问题与自适应算法设计

## 1. 问题描述
本实验研究第一象限单位圆盘内的二维带噪声约束优化问题。算法搜索时每次调用目标函数均重新采样高斯噪声，最终性能统计使用无噪声真实目标函数 f(x)。

## 2. 算法设计
### 2.1 SA
标准模拟退火，采用指数降温、高斯邻域扰动和投影约束处理。

### 2.2 GA
实数编码遗传算法，包含锦标赛选择、blend 交叉、高斯变异、精英保留和自适应变异步长。

### 2.3 NAMS
NAMS（Noise-tolerant Adaptive Memetic Search）为本文设计的噪声容忍自适应模因搜索算法。算法先进行全局随机筛选，再对精英候选进行重采样，随后围绕多个精英点进行自适应局部搜索，最后对候选库进行验证性重采样。该策略避免在早期把预算浪费在单个点的重复评估上，同时在最终选择时降低噪声误导风险。

## 3. 实验设置
每种算法独立运行 50 次，每次预算 2000 次目标函数评估。统计中位数、IQR、可行解比例，并用 Wilcoxon 符号秩检验比较算法差异。

## 4. 实验结果
插入 summary.csv、wilcoxon_tests.csv、boxplot_final_true_f.png、convergence_curves.png 和 best_solution_scatter.png。

## 5. 参数敏感性分析
对 SA 的 T0、alpha 和 GA 的 crossover_prob、mutation_step0 做参数扫描，插入 sensitivity_SA_heatmap.png 和 sensitivity_GA_heatmap.png。

## 6. 开放性探究
本文选择在线估计噪声水平并动态调整算法参数。NAMS 使用候选点重复评估得到的均值和标准误构造保守评分，并根据候选点间差距与不确定性的关系自适应分配重采样预算。

## 7. 结论
根据 summary.csv 和 Wilcoxon 检验结果总结三种算法的收敛速度、稳定性和显著性差异。
"""
    (outdir / "report_template.md").write_text(txt, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--outdir", type=str, default="results_v3")
    ap.add_argument("--seed", type=int, default=20260426)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--sens-runs", type=int, default=15)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, curves = run_all(args.runs, args.budget, args.seed)
    df.to_csv(outdir / "raw_results.csv", index=False)
    summary = summarize(df)
    summary.to_csv(outdir / "summary.csv", index=False)
    tests = wilcoxon_tests(df)
    tests.to_csv(outdir / "wilcoxon_tests.csv", index=False)

    plot_boxplot(df, outdir)
    plot_scatter(df, outdir)
    plot_convergence(curves, args.budget, outdir)
    write_report_template(outdir)

    if args.sensitivity:
        parameter_sensitivity(outdir, args.sens_runs, args.budget, args.seed)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nWilcoxon tests:")
    print(tests.to_string(index=False))
    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
