from __future__ import annotations

import csv
import itertools
import json
import os
import sys
import time
from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# Step-size helpers
def arange_param(start: float, stop: float, step: float) -> list[float]:
    """Like np.arange but returns a plain Python list, rounded to 6 dp."""
    return [round(v, 6) for v in np.arange(start, stop, step).tolist()]


def linspace_param(start: float, stop: float, n: int) -> list[float]:
    """Like np.linspace but returns a plain Python list, rounded to 6 dp."""
    return [round(v, 6) for v in np.linspace(start, stop, int(n)).tolist()]


def _expand_param_values(raw: Any) -> list:
    """Expand a param_grid entry from JSON into a flat list of candidates.

    Accepts three formats:
      - Plain list:    [0.05, 0.10, 0.15]
      - arange spec:   {"arange":   [start, stop, step]}
      - linspace spec: {"linspace": [start, stop, n]}
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if "arange" in raw:
            start, stop, step = raw["arange"]
            return arange_param(start, stop, step)
        if "linspace" in raw:
            start, stop, n = raw["linspace"]
            return linspace_param(start, stop, n)
    raise ValueError(
        f"Unrecognised param_grid value: {raw!r}. "
        'Use a list, {"arange": [start, stop, step]}, or {"linspace": [start, stop, n]}.'
    )


# Config loader

def load_sweep_config(path: str = "sweep_config.json") -> dict:
    """Load and validate a sweep config JSON file.

    Returns the parsed dict with all param_grid values expanded into
    flat Python lists ready for ParameterGrid.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    KeyError
        If required top-level keys are missing.
    ValueError
        If a param_grid entry uses an unrecognised format.
    """
    with open(path, "r") as f:
        cfg = json.load(f)

    required = {"param_grid", "base_params"}
    missing  = required - cfg.keys()
    if missing:
        raise KeyError(f"sweep config is missing required keys: {missing}")

    # Expand param_grid values (list / arange / linspace)
    cfg["param_grid"] = {
        k: _expand_param_values(v)
        for k, v in cfg["param_grid"].items()
    }

    return cfg


# ParameterGrid — cart product over a dict of lists
class ParameterGrid:
    """Exhaustive grid of parameter combinations (sklearn-compatible).

    Parameters
    ----------
    param_grid : dict[str, list]
        Maps each hyperparameter name to a list of candidate values.

    Example
    -------
    >>> list(ParameterGrid({"a": [1, 2], "b": [0.1, 0.2]}))
    [{'a': 1, 'b': 0.1}, {'a': 1, 'b': 0.2}, {'a': 2, 'b': 0.1}, {'a': 2, 'b': 0.2}]
    """

    def __init__(self, param_grid: dict[str, list]) -> None:
        self.param_grid = param_grid

    def __iter__(self):
        keys   = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def __len__(self) -> int:
        n = 1
        for v in self.param_grid.values():
            n *= len(v)
        return n


# GeneticGridSearch 
class GeneticGridSearch:
    """Run an exhaustive hyperparameter sweep over genetic().

    Mirrors the sklearn GridSearchCV interface:
        .fit()          — run all candidates
        .best_params_   — dict of best-scoring params
        .best_score_    — best average final fitness
        .cv_results_    — list of dicts (all candidates + scores)
        .save_results() — dump cv_results_ to CSV
        .plot()         — heatmap (2 params) or line plot (1 param)

    Construct directly or use the convenience classmethod:
        gs = GeneticGridSearch.from_config("sweep_config.json")

    Parameters
    ----------
    param_grid : dict[str, list]
        Hyperparameter search space. Any key accepted by genetic() works.
    config_path : str
        Game config JSON, passed through to genetic().
    n_repeats : int
        Run each candidate this many times and average best fitness.
        Reduces noise from stochastic evaluation. Default 1 (fast).
    base_params : dict | None
        Fixed hyperparameters NOT in param_grid. Merged before each call.
    results_dir : str
        Where to write per-run CSVs (internal data, not for analysis).
    output : dict | None
        Controls output paths. Keys: results_csv, plot_path, x_param, y_param.
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        param_grid: dict[str, list],
        config_path: str = "config.json",
        n_repeats: int = 1,
        base_params: dict[str, Any] | None = None,
        results_dir: str = "results/sweep",
        output: dict | None = None,
        verbose: bool = True,
    ) -> None:
        self.param_grid  = param_grid
        self.config_path = config_path
        self.n_repeats   = n_repeats
        self.base_params = base_params or {}
        self.results_dir = results_dir
        self.output      = output or {}
        self.verbose     = verbose

        # Populated after fit()
        self.cv_results_:   list[dict]  = []
        self.best_params_:  dict        = {}
        self.best_score_:   float       = float("-inf")
        self.best_weights_: list[float] = []

    # build from config
    @classmethod
    def from_config(cls, cfg_or_path: dict | str = "sweep_config.json") -> "GeneticGridSearch":
        """Build a GeneticGridSearch from a config dict or JSON file path."""
        if isinstance(cfg_or_path, str):
            cfg = load_sweep_config(cfg_or_path)
        else:
            cfg = cfg_or_path

        return cls(
            param_grid  = cfg["param_grid"],
            config_path = cfg.get("config_path", "config.json"),
            n_repeats   = cfg.get("n_repeats", 1),
            base_params = cfg.get("base_params", {}),
            results_dir = cfg.get("results_dir", "results/sweep"),
            output      = cfg.get("output", {}),
            verbose     = cfg.get("verbose", True),
        )

    def fit(self) -> "GeneticGridSearch":
        """Run all parameter combinations. Populates cv_results_ etc."""
        from genetic import genetic  

        grid  = ParameterGrid(self.param_grid)
        total = len(grid)

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs("weights/sweep", exist_ok=True)

        if self.verbose:
            print(f"\n[GeneticGridSearch] {total} candidates × {self.n_repeats} repeat(s) "
                  f"= {total * self.n_repeats} genetic() calls\n")

        for idx, candidate in enumerate(grid, start=1):
            params = {**self.base_params, **candidate}
            scores: list[float] = []

            for rep in range(self.n_repeats):
                run_name = f"sweep_{idx:04d}_rep{rep}"
                csv_path = os.path.join(self.results_dir, f"{run_name}_results.csv")

                t0          = time.perf_counter()
                best_genome = genetic(
                    config_path=self.config_path,
                    csv_path=csv_path,
                    **params,
                )
                elapsed = time.perf_counter() - t0

                score = self._read_best_fitness(csv_path)
                scores.append(score)

                if self.verbose:
                    print(
                        f"  [{idx}/{total}] rep {rep+1}/{self.n_repeats} "
                        f"score={score:.4f}  ({elapsed:.1f}s)  params={candidate}"
                    )

            mean_score = float(np.mean(scores))
            std_score  = float(np.std(scores))

            result = {
                "candidate_idx": idx,
                "mean_score":    mean_score,
                "std_score":     std_score,
                **candidate,
            }
            self.cv_results_.append(result)

            if mean_score > self.best_score_:
                self.best_score_   = mean_score
                self.best_params_  = deepcopy(candidate)
                self.best_weights_ = best_genome

        if self.verbose:
            print(f"\n[GeneticGridSearch] Done. Best score: {self.best_score_:.4f}")
            print(f"[GeneticGridSearch] Best params: {self.best_params_}")

        return self

    def save_results(self, path: str | None = None) -> None:
        """Write cv_results_ to a CSV file.

        Falls back to output.results_csv in the config, then to
        "results/sweep_results.csv".
        """
        if not self.cv_results_:
            raise RuntimeError("Call fit() before save_results().")

        path = path or self.output.get("results_csv", "results/sweep_results.csv")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        fieldnames = list(self.cv_results_[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.cv_results_)

        print(f"[GeneticGridSearch] Results saved to: {path}")

    def plot(
        self,
        path: str | None = None,
        x_param: str | None = None,
        y_param: str | None = None,
    ) -> None:
        """Plot sweep results.

        - 1 swept param  → line plot (param vs mean_score ± std)
        - 2 swept params → heatmap  (x_param × y_param, mean_score as colour)
        - 3+ swept params → heatmap over x_param × y_param, others fixed at best

        All three arguments fall back to output.plot_path / output.x_param /
        output.y_param from the loaded config if not supplied directly.
        """
        if not self.cv_results_:
            raise RuntimeError("Call fit() before plot().")

        path    = path    or self.output.get("plot_path", "graphs/sweep.png")
        x_param = x_param or self.output.get("x_param")
        y_param = y_param or self.output.get("y_param")

        swept = list(self.param_grid.keys())
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if len(swept) == 1:
            self._plot_line(path, swept[0])
        elif len(swept) == 2:
            self._plot_heatmap(path, x_param or swept[0], y_param or swept[1])
        else:
            self._plot_heatmap(
                path,
                x_param or swept[0],
                y_param or swept[1],
                fix_params={
                    p: self.best_params_[p]
                    for p in swept
                    if p not in (x_param, y_param)
                },
            )

    # Private helpers
    @staticmethod
    def _read_best_fitness(csv_path: str) -> float:
        best = float("-inf")
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    v = float(row["best_fitness"])
                    if v > best:
                        best = v
        except (FileNotFoundError, KeyError):
            pass
        return best

    def _results_as_arrays(self, x_param: str, y_param: str | None, fix_params: dict | None):
        rows = self.cv_results_
        if fix_params:
            rows = [r for r in rows if all(r[k] == v for k, v in fix_params.items())]

        xs = sorted(set(r[x_param] for r in rows))
        if y_param:
            ys   = sorted(set(r[y_param] for r in rows))
            grid = np.full((len(ys), len(xs)), np.nan)
            for r in rows:
                xi = xs.index(r[x_param])
                yi = ys.index(r[y_param])
                grid[yi, xi] = r["mean_score"]
            return xs, ys, grid
        else:
            means = [next(r["mean_score"] for r in rows if r[x_param] == x) for x in xs]
            stds  = [next(r["std_score"]  for r in rows if r[x_param] == x) for x in xs]
            return xs, means, stds

    def _plot_line(self, path: str, x_param: str) -> None:
        xs, means, stds = self._results_as_arrays(x_param, None, None)
        means = np.array(means)
        stds  = np.array(stds)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, means, marker="o", linewidth=2, color="steelblue", label="mean score")
        ax.fill_between(xs, means - stds, means + stds, alpha=0.25, color="steelblue", label="±1 std")
        ax.axvline(xs[int(np.argmax(means))], color="steelblue", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel(x_param, fontsize=12)
        ax.set_ylabel("Best fitness (avg snake length)", fontsize=12)
        ax.set_title(f"Sweep: {x_param}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[GeneticGridSearch] Plot saved to: {path}")

    def _plot_heatmap(
        self,
        path: str,
        x_param: str,
        y_param: str,
        fix_params: dict | None = None,
    ) -> None:
        xs, ys, grid = self._results_as_arrays(x_param, y_param, fix_params)

        fig, ax = plt.subplots(figsize=(max(6, len(xs) * 1.2), max(5, len(ys) * 1.0)))
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="Blues")
        plt.colorbar(im, ax=ax, label="Mean best fitness")

        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(x) for x in xs], rotation=45, ha="right")
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels([str(y) for y in ys])
        ax.set_xlabel(x_param, fontsize=12)
        ax.set_ylabel(y_param, fontsize=12)

        title = f"Sweep: {x_param} x {y_param}"
        if fix_params:
            title += "\n(fixed: " + ", ".join(f"{k}={v}" for k, v in fix_params.items()) + ")"
        ax.set_title(title, fontsize=13, fontweight="bold")

        for yi in range(len(ys)):
            for xi in range(len(xs)):
                val = grid[yi, xi]
                if not np.isnan(val):
                    ax.text(xi, yi, f"{val:.2f}", ha="center", va="center",
                            fontsize=9, color="white" if val > (np.nanmax(grid) * 0.6) else "black")

        plt.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[GeneticGridSearch] Heatmap saved to: {path}")

# Entry point
if __name__ == "__main__":
    cfg_path = "sweep.json"
    if "--config" in sys.argv:
        idx      = sys.argv.index("--config")
        cfg_path = sys.argv[idx + 1]

    print(f"[sweep] Loading config from: {cfg_path}")
    gs = GeneticGridSearch.from_config(cfg_path)
    gs.fit()
    gs.save_results()
    gs.plot()

    print("\n=== Best params ===")
    for k, v in gs.best_params_.items():
        print(f"  {k}: {v}")
    print(f"  best_score: {gs.best_score_:.4f}")