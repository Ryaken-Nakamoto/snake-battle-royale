from __future__ import annotations

import csv
import itertools
import json
import os
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from neural_network import Neural_Network, Basic_Neural_Network, Two_Layer_Neural_Network, Base_algorithm, N_FEATURES

NN_REGISTRY: dict[str, type[Neural_Network]] = {
    "Basic_Neural_Network":     Basic_Neural_Network,
    "Two_Layer_Neural_Network": Two_Layer_Neural_Network,
    "Base_algorithm":           Base_algorithm,
}


def _resolve_nn_class(name: str) -> type[Neural_Network]:
    if name not in NN_REGISTRY:
        raise ValueError(f"Unknown nn_class {name!r}. Valid: {list(NN_REGISTRY)}")
    return NN_REGISTRY[name]


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


# Param abbreviations for readable directory names
_PARAM_ABBREVS: dict[str, str] = {
    "mutation_rate":     "mr",
    "mutation_strength": "ms",
    "crossover_rate":    "xr",
    "population_size":   "pop",
    "elitism_count":     "ec",
    "games_per_genome":  "gpg",
    "num_generations":   "ng",
}


def _make_cand_dir(
    idx: int,
    candidate_params: dict,
    abbrevs: dict[str, str] | None = None,
) -> str:
    """Return a directory name like '0003_mr=0.15,ms=0.25'."""
    effective = {**_PARAM_ABBREVS, **(abbrevs or {})}
    parts = [f"{effective.get(k, k[:4])}={v}" for k, v in candidate_params.items()]
    return f"{idx:04d}_" + ",".join(parts)


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

    # Resolve nn_class string to actual class
    if "nn_class" in cfg:
        cfg["nn_class"] = _resolve_nn_class(cfg["nn_class"])

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
        nn_class: type[Neural_Network] | None = None,
        run_name: str | None = None,
        prev_weights: str | list[str] | None = None,
        abbreviations: dict[str, str] | None = None,
    ) -> None:
        self.param_grid    = param_grid
        self.config_path   = config_path
        self.n_repeats     = n_repeats
        self.base_params   = base_params or {}
        self.results_dir   = results_dir
        self.output        = output or {}
        self.verbose       = verbose
        self.nn_class      = nn_class
        self.run_name      = run_name or f"sweep_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        self.prev_weights  = prev_weights
        self.abbreviations = abbreviations

        # Populated after fit()
        self.cv_results_:   list[dict]  = []
        self.best_params_:  dict        = {}
        self.best_score_:   float       = float("-inf")
        self.best_weights_: list[float] = []

    # build from config
    @classmethod
    def from_config(cls, cfg_or_path: dict | str = "sweep_config.json") -> "GeneticGridSearch":
        """Build a GeneticGridSearch from a config dict or JSON file path."""
        config_file_path: str | None = None
        if isinstance(cfg_or_path, str):
            cfg = load_sweep_config(cfg_or_path)
            config_file_path = cfg_or_path
        else:
            cfg = cfg_or_path

        instance = cls(
            param_grid    = cfg["param_grid"],
            config_path   = cfg.get("config_path", "config.json"),
            n_repeats     = cfg.get("n_repeats", 1),
            base_params   = cfg.get("base_params", {}),
            results_dir   = cfg.get("results_dir", "results/sweep"),
            output        = cfg.get("output", {}),
            verbose       = cfg.get("verbose", True),
            nn_class      = cfg.get("nn_class"),
            run_name      = cfg.get("run_name"),
            prev_weights  = cfg.get("prev_weights"),
            abbreviations = cfg.get("abbreviations"),
        )

        # Snapshot the config file into runs/{run_name}/sweep_config.json
        if config_file_path is not None:
            run_dir = os.path.join("runs", instance.run_name)
            os.makedirs(run_dir, exist_ok=True)
            shutil.copy2(config_file_path, os.path.join(run_dir, "sweep_config.json"))

        return instance

    def fit(self) -> "GeneticGridSearch":
        """Run all parameter combinations. Populates cv_results_ etc."""
        from genetic import genetic, plot_fitness

        grid    = ParameterGrid(self.param_grid)
        total   = len(grid)
        run_dir = os.path.join("runs", self.run_name)
        os.makedirs(run_dir, exist_ok=True)

        if self.verbose:
            print(f"\n[GeneticGridSearch] run_name={self.run_name}")
            print(f"[GeneticGridSearch] {total} candidates × {self.n_repeats} repeat(s) "
                  f"= {total * self.n_repeats} genetic() calls\n")

        for idx, candidate in enumerate(grid, start=1):
            params       = {**self.base_params, **candidate}
            cand_dir_name = _make_cand_dir(idx, candidate, self.abbreviations)
            cand_dir     = os.path.join(run_dir, "candidates", cand_dir_name)
            os.makedirs(cand_dir, exist_ok=True)

            # Write full params once per candidate
            with open(os.path.join(cand_dir, "params.json"), "w") as f:
                json.dump(params, f, indent=2)

            # Resolve seed weight for this candidate
            seed: str | None = None
            if isinstance(self.prev_weights, str):
                seed = self.prev_weights
            elif isinstance(self.prev_weights, list):
                if idx - 1 < len(self.prev_weights):
                    seed = self.prev_weights[idx - 1]
                else:
                    if self.verbose:
                        print(f"  [prev_weights] no entry for candidate {idx}, using random init")

            scores: list[float] = []

            for rep in range(self.n_repeats):
                rep_dir      = os.path.join(cand_dir, f"rep{rep}")
                os.makedirs(rep_dir, exist_ok=True)
                csv_path     = os.path.join(rep_dir, "fitness.csv")
                weights_path = os.path.join(rep_dir, "weights.csv")
                plot_path    = os.path.join(rep_dir, "fitness.png")

                call_params = dict(params)
                if self.nn_class is not None:
                    call_params["nn_class"] = self.nn_class

                t0 = time.perf_counter()
                best_genome = genetic(
                    config_path  = self.config_path,
                    csv_path     = csv_path,
                    weights_path = weights_path,
                    prev_weights = seed,
                    **call_params,
                )
                elapsed = time.perf_counter() - t0

                nn_cls = self.nn_class if self.nn_class is not None else Two_Layer_Neural_Network
                plot_params = {**params, "genome_length": nn_cls.genome_length(N_FEATURES)}
                plot_fitness(csv_path=csv_path, plot_path=plot_path,
                             hyperparams=plot_params, exp_name=cand_dir_name)

                score = self._read_best_fitness(csv_path)
                scores.append(score)

                if self.verbose:
                    seed_note = f"  seed={seed}" if seed else ""
                    print(
                        f"  [{idx}/{total}] rep {rep+1}/{self.n_repeats} "
                        f"score={score:.4f}  ({elapsed:.1f}s)  params={candidate}{seed_note}"
                    )

            mean_score = float(np.mean(scores))
            std_score  = float(np.std(scores))

            result = {
                "candidate_idx": idx,
                "cand_dir":      cand_dir_name,
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
            print(f"[GeneticGridSearch] All outputs in: {run_dir}/")

        return self

    def save_results(self, path: str | None = None) -> None:
        """Write cv_results_ to a CSV file.

        Falls back to output.results_csv in the config, then to
        runs/{run_name}/summary.csv.
        """
        if not self.cv_results_:
            raise RuntimeError("Call fit() before save_results().")

        default = os.path.join("runs", self.run_name, "summary.csv")
        path = path or self.output.get("results_csv", default)
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
        facet_param: str | None = None,
    ) -> None:
        """Plot sweep results.

        - 1 swept param  → line plot (param vs mean_score ± std)
        - 2 swept params → heatmap (x_param × y_param, mean_score as colour)
        - 3 swept params → faceted heatmap (grid of subplots, one per facet_param value)
        - 4+ swept params → all C(n,2) pairwise heatmaps saved as separate files

        Arguments fall back to output.plot_path / output.x_param / output.y_param /
        output.facet_param from the loaded config if not supplied directly.
        """
        if not self.cv_results_:
            raise RuntimeError("Call fit() before plot().")

        default_path = os.path.join("runs", self.run_name, "heatmap.png")
        path        = path        or self.output.get("plot_path", default_path)
        x_param     = x_param     or self.output.get("x_param")
        y_param     = y_param     or self.output.get("y_param")
        facet_param = facet_param or self.output.get("facet_param")

        swept = list(self.param_grid.keys())
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if len(swept) == 1:
            self._plot_line(path, swept[0])
        elif len(swept) == 2:
            self._plot_heatmap(path, x_param or swept[0], y_param or swept[1])
        elif len(swept) == 3:
            xp = x_param or swept[0]
            yp = y_param or swept[1]
            fp = facet_param or next(p for p in swept if p not in (xp, yp))
            self._plot_faceted_heatmap(path, xp, yp, fp)
        else:
            # 4+ params: one heatmap per pair, saved into the run dir
            heatmap_dir = os.path.join("runs", self.run_name)
            self._plot_all_pairwise_heatmaps(heatmap_dir, x_param, y_param)

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

    def _plot_faceted_heatmap(
        self,
        path: str,
        x_param: str,
        y_param: str,
        facet_param: str,
    ) -> None:
        """Grid of subplots: one heatmap per value of facet_param, shared color scale."""
        facet_vals = sorted(set(r[facet_param] for r in self.cv_results_))
        n_panels   = len(facet_vals)
        ncols      = min(n_panels, 4)
        nrows      = (n_panels + ncols - 1) // ncols

        all_scores = [r["mean_score"] for r in self.cv_results_ if not np.isnan(r["mean_score"])]
        vmin, vmax = (min(all_scores), max(all_scores)) if all_scores else (0, 1)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 5, nrows * 4.5),
            squeeze=False,
        )

        for panel_idx, fv in enumerate(facet_vals):
            row, col = divmod(panel_idx, ncols)
            ax       = axes[row][col]
            xs, ys, grid = self._results_as_arrays(x_param, y_param, {facet_param: fv})

            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="Blues",
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([str(x) for x in xs], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(ys)))
            ax.set_yticklabels([str(y) for y in ys], fontsize=8)
            ax.set_xlabel(x_param, fontsize=9)
            ax.set_ylabel(y_param, fontsize=9)
            ax.set_title(f"{facet_param}={fv}", fontsize=10, fontweight="bold")

            for yi in range(len(ys)):
                for xi in range(len(xs)):
                    val = grid[yi, xi]
                    if not np.isnan(val):
                        ax.text(xi, yi, f"{val:.2f}", ha="center", va="center",
                                fontsize=7, color="white" if val > (vmax * 0.6) else "black")

        # Hide unused subplots
        for panel_idx in range(len(facet_vals), nrows * ncols):
            row, col = divmod(panel_idx, ncols)
            axes[row][col].set_visible(False)

        # Shared colorbar via ScalarMappable so it isn't tied to one axes
        sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), label="Mean best fitness", shrink=0.6)
        fig.suptitle(
            f"Sweep: {x_param} × {y_param}  |  panels: {facet_param}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[GeneticGridSearch] Faceted heatmap saved to: {path}")

    def _plot_all_pairwise_heatmaps(
        self,
        base_dir: str,
        x_param: str | None = None,
        y_param: str | None = None,
    ) -> None:
        """Generate all C(n,2) pairwise heatmaps, fixing remaining params at best values."""
        swept = list(self.param_grid.keys())
        os.makedirs(base_dir, exist_ok=True)

        for param_a, param_b in itertools.combinations(swept, 2):
            fix_params = {
                p: self.best_params_[p]
                for p in swept
                if p not in (param_a, param_b)
            }
            a_abbrev = _PARAM_ABBREVS.get(param_a, param_a[:4])
            b_abbrev = _PARAM_ABBREVS.get(param_b, param_b[:4])
            fname    = f"heatmap_{a_abbrev}_x_{b_abbrev}.png"
            path     = os.path.join(base_dir, fname)
            self._plot_heatmap(path, param_a, param_b, fix_params)

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