"""
run_bo_benchmark.py

Budget-matched benchmarking of Bayesian Optimization (GP+EI via BoTorch) vs
non-Bayesian derivative-free baselines (Random Search, Nelder–Mead, Differential Evolution, optional CMA-ES).

Outputs (parseable):
  - logs/evals.jsonl   : one JSON record per evaluation (easy to parse)
  - logs/summary.csv   : one row per (problem, method, seed) run
  - logs/stats.json    : paired Wilcoxon + bootstrap CIs (BO vs each baseline), per problem

Figures:
  - figures/convergence_<problem>.png
  - figures/final_boxplot_<problem>.png

Windows note:
  Torch MUST be imported before SciPy/NumPy/Matplotlib to avoid OpenMP runtime conflicts.
"""

# ----------------------------
# Windows-safe import order
# ----------------------------
import os

# Workarounds for Windows OpenMP/MKL runtime collisions (common cause of WinError 1114)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import json
import time
import math
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch

# Reduce thread init complexity (helps on some Windows setups)
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Remaining scientific stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, minimize
from scipy.stats import wilcoxon

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

# Optional CMA-ES baseline (published, widely used)
try:
    import cma
    HAS_CMA = True
except Exception:
    HAS_CMA = False


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, record: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def auc_best_so_far(best_series: List[float]) -> float:
    """Area under best-so-far curve (lower is better for minimization)."""
    y = np.asarray(best_series, dtype=float)
    return float(np.trapz(y, dx=1.0))


def bootstrap_ci_mean(diffs: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Bootstrap CI for mean(diffs)."""
    rng = np.random.default_rng(seed)
    n = diffs.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boots[i] = float(np.mean(sample))
    boots.sort()
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def clip_to_bounds(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.clip(x, bounds[:, 0], bounds[:, 1])


# ----------------------------
# Benchmark problems (minimize)
# ----------------------------
def branin(x: np.ndarray) -> float:
    # Typical domain: x1 in [-5, 10], x2 in [0, 15]
    x1, x2 = float(x[0]), float(x[1])
    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s


def hartmann6(x: np.ndarray) -> float:
    # Hartmann6 is often defined for maximization; return negative for minimization.
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ])
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    outer = 0.0
    for i in range(4):
        inner = 0.0
        for j in range(6):
            inner += A[i, j] * (x[j] - P[i, j]) ** 2
        outer += alpha[i] * math.exp(-inner)
    return -outer


def ackley(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    d = x.size
    a, b, c = 20.0, 0.2, 2.0 * math.pi
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    term1 = -a * math.exp(-b * math.sqrt(s1 / d))
    term2 = -math.exp(s2 / d)
    return term1 + term2 + a + math.e


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


@dataclass
class Problem:
    name: str
    dim: int
    bounds: np.ndarray          # shape (dim, 2)
    f: Callable[[np.ndarray], float]
    noise_std: float = 0.0      # additive Gaussian observation noise

    def eval(self, x: np.ndarray, rng: np.random.Generator) -> float:
        y = float(self.f(x))
        if self.noise_std > 0:
            y += float(rng.normal(0.0, self.noise_std))
        return y


# ----------------------------
# Methods
# ----------------------------
def run_random_search(problem: Problem, budget: int, seed: int, evals_jsonl: str) -> Dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = float("inf")
    best_series: List[float] = []

    for i in range(1, budget + 1):
        x = rng.uniform(problem.bounds[:, 0], problem.bounds[:, 1])
        y = problem.eval(x, rng)
        best = min(best, y)
        best_series.append(best)

        write_jsonl(evals_jsonl, {
            "problem": problem.name, "method": "random_search", "seed": seed,
            "eval": i, "x": x.tolist(), "y": y, "best_so_far": best,
            "elapsed_s": time.time() - t0
        })

    return {"best": best, "best_series": best_series, "n_evals": budget, "wall_s": time.time() - t0}


def run_nelder_mead(problem: Problem, budget: int, seed: int, evals_jsonl: str) -> Dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = float("inf")
    best_series: List[float] = []
    eval_count = 0

    def wrapped(x):
        nonlocal best, eval_count
        if eval_count >= budget:
            return best  # discourage further calls
        x = clip_to_bounds(np.asarray(x, dtype=float), problem.bounds)
        y = problem.eval(x, rng)
        eval_count += 1
        best = min(best, y)
        best_series.append(best)

        write_jsonl(evals_jsonl, {
            "problem": problem.name, "method": "nelder_mead", "seed": seed,
            "eval": eval_count, "x": x.tolist(), "y": y, "best_so_far": best,
            "elapsed_s": time.time() - t0
        })
        return y

    x0 = rng.uniform(problem.bounds[:, 0], problem.bounds[:, 1])
    minimize(wrapped, x0=x0, method="Nelder-Mead", options={"maxfev": budget, "disp": False})

    if not best_series:
        best_series = [best]
    return {"best": best, "best_series": best_series, "n_evals": eval_count, "wall_s": time.time() - t0}


def run_differential_evolution(problem: Problem, budget: int, seed: int, evals_jsonl: str) -> Dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = float("inf")
    best_series: List[float] = []
    eval_count = 0

    bounds = [(float(lo), float(hi)) for lo, hi in problem.bounds]

    def wrapped(x):
        nonlocal best, eval_count
        if eval_count >= budget:
            return best
        x = clip_to_bounds(np.asarray(x, dtype=float), problem.bounds)
        y = problem.eval(x, rng)
        eval_count += 1
        best = min(best, y)
        best_series.append(best)

        write_jsonl(evals_jsonl, {
            "problem": problem.name, "method": "differential_evolution", "seed": seed,
            "eval": eval_count, "x": x.tolist(), "y": y, "best_so_far": best,
            "elapsed_s": time.time() - t0
        })
        return y

    dim = problem.dim
    popsize = 8
    evals_per_gen = popsize * dim
    maxiter = max(1, int(budget / max(1, evals_per_gen)) - 1)

    differential_evolution(
        wrapped, bounds=bounds, seed=seed,
        popsize=popsize, maxiter=maxiter,
        polish=False, disp=False
    )

    if not best_series:
        best_series = [best]
    return {"best": best, "best_series": best_series, "n_evals": eval_count, "wall_s": time.time() - t0}


def run_cma_es(problem: Problem, budget: int, seed: int, evals_jsonl: str) -> Dict:
    if not HAS_CMA:
        raise RuntimeError("CMA-ES baseline requested but 'cma' is not installed. Run: pip install cma")

    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = float("inf")
    best_series: List[float] = []
    eval_count = 0

    x0 = rng.uniform(problem.bounds[:, 0], problem.bounds[:, 1])
    sigma0 = 0.3 * float(np.mean(problem.bounds[:, 1] - problem.bounds[:, 0]))

    def wrapped(x):
        nonlocal best, eval_count
        if eval_count >= budget:
            return best
        x = clip_to_bounds(np.asarray(x, dtype=float), problem.bounds)
        y = problem.eval(x, rng)
        eval_count += 1
        best = min(best, y)
        best_series.append(best)

        write_jsonl(evals_jsonl, {
            "problem": problem.name, "method": "cma_es", "seed": seed,
            "eval": eval_count, "x": x.tolist(), "y": y, "best_so_far": best,
            "elapsed_s": time.time() - t0
        })
        return y

    es = cma.CMAEvolutionStrategy(
        x0.tolist(), sigma0,
        {"seed": seed, "maxfevals": budget, "verb_log": 0, "verb_disp": 0}
    )

    while not es.stop():
        X = es.ask()
        Y = [wrapped(x) for x in X]
        es.tell(X, Y)

    if not best_series:
        best_series = [best]
    return {"best": best, "best_series": best_series, "n_evals": eval_count, "wall_s": time.time() - t0}


def run_gp_bo_ei(problem: Problem, budget: int, seed: int, n_init: int, evals_jsonl: str) -> Dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    t0 = time.time()

    dim = problem.dim
    bounds_t = torch.tensor(problem.bounds.T, dtype=torch.double)  # (2, d)

    # Sobol initial design in [0,1]^d, then scale
    sobol = draw_sobol_samples(
        bounds=torch.zeros(2, dim, dtype=torch.double),
        n=n_init, q=1
    ).squeeze(1)
    X = bounds_t[0] + (bounds_t[1] - bounds_t[0]) * sobol

    Y_list: List[List[float]] = []
    best = float("inf")
    best_series: List[float] = []
    eval_count = 0

    # Evaluate initial points
    for i in range(n_init):
        x_np = X[i].detach().cpu().numpy()
        y = problem.eval(x_np, rng)
        Y_list.append([y])
        eval_count += 1
        best = min(best, y)
        best_series.append(best)

        write_jsonl(evals_jsonl, {
            "problem": problem.name, "method": "gp_bo_ei", "seed": seed,
            "eval": eval_count, "x": x_np.tolist(), "y": y, "best_so_far": best,
            "elapsed_s": time.time() - t0
        })

    Y = torch.tensor(Y_list, dtype=torch.double)

    # BO loop
    while eval_count < budget:
        # Fit GP surrogate
        gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        best_f = Y.min().item()
        acq = ExpectedImprovement(model=gp, best_f=best_f, maximize=False)

        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds_t,
            q=1,
            num_restarts=8,
            raw_samples=128,
            options={"maxiter": 200}
        )

        x_new = candidate.detach().squeeze(0)
        x_np = x_new.cpu().numpy()

        y_new = problem.eval(x_np, rng)
        eval_count += 1
        best = min(best, y_new)
        best_series.append(best)

        write_jsonl(evals_jsonl, {
            "problem": problem.name, "method": "gp_bo_ei", "seed": seed,
            "eval": eval_count, "x": x_np.tolist(), "y": float(y_new), "best_so_far": best,
            "elapsed_s": time.time() - t0
        })

        X = torch.cat([X, x_new.view(1, -1)], dim=0)
        Y = torch.cat([Y, torch.tensor([[y_new]], dtype=torch.double)], dim=0)

    return {"best": float(best), "best_series": best_series, "n_evals": eval_count, "wall_s": time.time() - t0}


# ----------------------------
# Plotting
# ----------------------------
def plot_convergence(df_eval: pd.DataFrame, outpath: str, problem_name: str) -> None:
    plt.figure()
    for method in sorted(df_eval["method"].unique()):
        sub = df_eval[df_eval["method"] == method]
        grouped = sub.groupby("eval")["best_so_far"]
        mean = grouped.mean()
        sem = grouped.sem()
        plt.plot(mean.index, mean.values, label=method)
        plt.fill_between(mean.index, mean - sem, mean + sem, alpha=0.2)

    plt.xlabel("Function evaluations")
    plt.ylabel("Best-so-far objective (lower is better)")
    plt.title(f"Convergence: {problem_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_box(df_summary: pd.DataFrame, outpath: str, problem_name: str) -> None:
    plt.figure()
    methods = sorted(df_summary["method"].unique())
    data = [df_summary[df_summary["method"] == m]["final_best"].values for m in methods]
    plt.boxplot(data, labels=methods, showfliers=True)
    plt.ylabel("Final best objective (lower is better)")
    plt.title(f"Final performance: {problem_name}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ----------------------------
# Experiment runner
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12, help="Number of random seeds per problem")
    parser.add_argument("--budget", type=int, default=80, help="Max function evaluations per method")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Additive Gaussian noise std-dev")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory")
    parser.add_argument("--include_cma", action="store_true", help="Include CMA-ES (requires pip install cma)")
    args = parser.parse_args()

    print("PYTHON EXE:", sys.executable)
    print("Torch:", torch.__version__)
    print("CMA available:", HAS_CMA)

    outdir = args.outdir
    log_dir = os.path.join(outdir, "logs")
    fig_dir = os.path.join(outdir, "figures")
    ensure_dir(log_dir)
    ensure_dir(fig_dir)

    evals_jsonl = os.path.join(log_dir, "evals.jsonl")
    if os.path.exists(evals_jsonl):
        os.remove(evals_jsonl)

    problems = [
        Problem("branin", 2, np.array([[-5.0, 10.0], [0.0, 15.0]]), branin, args.noise_std),
        Problem("hartmann6", 6, np.array([[0.0, 1.0]] * 6), hartmann6, args.noise_std),
        Problem("ackley10", 10, np.array([[-5.0, 5.0]] * 10), ackley, args.noise_std),
        Problem("rosenbrock5", 5, np.array([[-2.0, 2.0]] * 5), rosenbrock, args.noise_std),
    ]

    methods = ["gp_bo_ei", "random_search", "differential_evolution", "nelder_mead"]
    if args.include_cma and HAS_CMA:
        methods.append("cma_es")

    rows = []

    for prob in problems:
        for s in range(args.seeds):
            seed = 1000 + s

            # BO init points: enough to fit GP, but still budget-conscious
            n_init = max(8, prob.dim * 2)

            # 1) GP BO EI
            res = run_gp_bo_ei(prob, budget=args.budget, seed=seed, n_init=n_init, evals_jsonl=evals_jsonl)
            rows.append({
                "problem": prob.name, "method": "gp_bo_ei", "seed": seed,
                "final_best": res["best"], "auc": auc_best_so_far(res["best_series"]),
                "n_evals": res["n_evals"], "wall_s": res["wall_s"]
            })

            # 2) Random Search
            res = run_random_search(prob, budget=args.budget, seed=seed, evals_jsonl=evals_jsonl)
            rows.append({
                "problem": prob.name, "method": "random_search", "seed": seed,
                "final_best": res["best"], "auc": auc_best_so_far(res["best_series"]),
                "n_evals": res["n_evals"], "wall_s": res["wall_s"]
            })

            # 3) Differential Evolution
            res = run_differential_evolution(prob, budget=args.budget, seed=seed, evals_jsonl=evals_jsonl)
            rows.append({
                "problem": prob.name, "method": "differential_evolution", "seed": seed,
                "final_best": res["best"], "auc": auc_best_so_far(res["best_series"]),
                "n_evals": res["n_evals"], "wall_s": res["wall_s"]
            })

            # 4) Nelder–Mead
            res = run_nelder_mead(prob, budget=args.budget, seed=seed, evals_jsonl=evals_jsonl)
            rows.append({
                "problem": prob.name, "method": "nelder_mead", "seed": seed,
                "final_best": res["best"], "auc": auc_best_so_far(res["best_series"]),
                "n_evals": res["n_evals"], "wall_s": res["wall_s"]
            })

            # 5) CMA-ES (optional)
            if args.include_cma and HAS_CMA:
                res = run_cma_es(prob, budget=args.budget, seed=seed, evals_jsonl=evals_jsonl)
                rows.append({
                    "problem": prob.name, "method": "cma_es", "seed": seed,
                    "final_best": res["best"], "auc": auc_best_so_far(res["best_series"]),
                    "n_evals": res["n_evals"], "wall_s": res["wall_s"]
                })

    summary = pd.DataFrame(rows)
    summary_csv = os.path.join(log_dir, "summary.csv")
    summary.to_csv(summary_csv, index=False)

    # Load evals for plots
    eval_records = []
    with open(evals_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            eval_records.append(json.loads(line))
    df_eval = pd.DataFrame(eval_records)

    # Figures per problem
    for prob in problems:
        sub_eval = df_eval[df_eval["problem"] == prob.name]
        sub_sum = summary[summary["problem"] == prob.name]
        plot_convergence(sub_eval, os.path.join(fig_dir, f"convergence_{prob.name}.png"), prob.name)
        plot_box(sub_sum, os.path.join(fig_dir, f"final_boxplot_{prob.name}.png"), prob.name)

    # Stats: paired Wilcoxon + bootstrap CI on (baseline - BO) improvements
    stats = {
        "meta": {"seeds": args.seeds, "budget": args.budget, "noise_std": args.noise_std, "methods": methods},
        "by_problem": {}
    }

    for prob in problems:
        prob_stats = {}
        bo = summary[(summary.problem == prob.name) & (summary.method == "gp_bo_ei")].sort_values("seed")

        for m in sorted(summary[summary.problem == prob.name]["method"].unique()):
            if m == "gp_bo_ei":
                continue
            base = summary[(summary.problem == prob.name) & (summary.method == m)].sort_values("seed")

            merged = bo.merge(base, on=["problem", "seed"], suffixes=("_bo", "_base"))

            # Improvements defined as (baseline - BO): positive => BO better (lower objective)
            dif_final = merged["final_best_base"].values - merged["final_best_bo"].values
            dif_auc = merged["auc_base"].values - merged["auc_bo"].values

            # Paired Wilcoxon tests (BO vs baseline)
            try:
                p_final = float(wilcoxon(merged["final_best_bo"], merged["final_best_base"]).pvalue)
            except Exception:
                p_final = None
            try:
                p_auc = float(wilcoxon(merged["auc_bo"], merged["auc_base"]).pvalue)
            except Exception:
                p_auc = None

            ci_final = bootstrap_ci_mean(dif_final, seed=123)
            ci_auc = bootstrap_ci_mean(dif_auc, seed=456)

            prob_stats[m] = {
                "n_pairs": int(len(merged)),
                "final_best": {
                    "mean_improvement_base_minus_bo": float(np.mean(dif_final)),
                    "bootstrap_ci_95": [ci_final[0], ci_final[1]],
                    "wilcoxon_pvalue": p_final
                },
                "auc": {
                    "mean_improvement_base_minus_bo": float(np.mean(dif_auc)),
                    "bootstrap_ci_95": [ci_auc[0], ci_auc[1]],
                    "wilcoxon_pvalue": p_auc
                }
            }

        stats["by_problem"][prob.name] = prob_stats

    stats_json = os.path.join(log_dir, "stats.json")
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nDone.")
    print("Parseable outputs:")
    print(" ", evals_jsonl)
    print(" ", summary_csv)
    print(" ", stats_json)
    print("Figures:")
    print(" ", fig_dir)


if __name__ == "__main__":
    main()
