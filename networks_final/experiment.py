
# experiment.py - run our 100 paired trials, compute statistics with stat library, generate the plots with matplot


from __future__ import annotations

import argparse
import json
import os # to create our results directory
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non interactive backend for batch runs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from tqdm import tqdm  # need this for animated progress bar when running experiment in terminal


from manet_env import MANETEnv # need to run_episode() and get all simulation data

warnings.filterwarnings("ignore", category=UserWarning)

# data containers

@dataclass
class TrialResult:
    seed:               int
    first_death_time:   Optional[int]
    time_30pct_death:   Optional[int]
    time_disconnection: Optional[int]
    pdr:                float
    mean_path_length:   float
    energy_variance:    float          # variance at end of an episode
    alive_history:      List[int]      = field(default_factory=list)
    energy_var_history: List[float]    = field(default_factory=list)


@dataclass
class ExperimentResults:
    algorithm:  str
    n_trials:   int
    max_steps:  int = 1000 
    results:    List[TrialResult] = field(default_factory=list)

    def _arr(self, attr: str, fill_missing: int = 0) -> np.ndarray:
        vals = []
        for r in self.results:
            v = getattr(r, attr)
            vals.append(fill_missing if v is None else v)
        return np.array(vals, dtype=float)


    # fixed bug where the lifetime metrics were using 0 for fill missing intead of max_steps, this was skewing our lifetime data
    def first_death_arr(self)   -> np.ndarray: return self._arr("first_death_time",   fill_missing=self.max_steps)
    def time_30pct_arr(self)    -> np.ndarray: return self._arr("time_30pct_death",    fill_missing=self.max_steps)
    def disconnection_arr(self) -> np.ndarray: return self._arr("time_disconnection",  fill_missing=self.max_steps)
    def pdr_arr(self)           -> np.ndarray: return self._arr("pdr")
    def path_len_arr(self)      -> np.ndarray: return self._arr("mean_path_length")
    def energy_var_arr(self)    -> np.ndarray: return self._arr("energy_variance")


# statistical tests

def compare_metrics(
    #compare the two algorithms ExperimentResults objects
    res_hop: ExperimentResults,
    res_ea:  ExperimentResults,
) -> Dict[str, Dict]:
    
    # paired t-test for normal distribution, and wilcoxon for non normal for each metric.
    # returns a dict with keys = metric names, values = dict of stats 
                                        # hop count                      energy aware
    metrics = {
        "Time to First Node Death":   (res_hop.first_death_arr(),   res_ea.first_death_arr()),
        "Time to 30 % Node Loss":     (res_hop.time_30pct_arr(),    res_ea.time_30pct_arr()),
        "Time to Disconnection":      (res_hop.disconnection_arr(), res_ea.disconnection_arr()),
        "Packet Delivery Ratio":      (res_hop.pdr_arr(),           res_ea.pdr_arr()),
        "Average Path Length (hops)": (res_hop.path_len_arr(),      res_ea.path_len_arr()),
        "Energy Variance":            (res_hop.energy_var_arr(),    res_ea.energy_var_arr()),
    }

    output = {}
    for name, (a_hop, a_ea) in metrics.items():
        diff = a_ea - a_hop      #store difference between energy aware and hop

        # Shapiro-Wilk normality test on the difference from stats library
        # shapiro returns both a test statistic and p value, only concerned with p value here
        _, p_normal = stats.shapiro(diff) 
        normal = p_normal > 0.05

        # if normally distributed then run paired t-test
        if normal:
            t_stat, p_val = stats.ttest_rel(a_ea, a_hop)
            test_name = "paired t-test"
        # no normality assumption needed 
        else:
            t_stat, p_val = stats.wilcoxon(diff)
            test_name = "Wilcoxon signed-rank"

        # Cohen's d, so that we can get an idea of how big the difference actually is in a + or - direction 
        d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))

        output[name] = {
            "hop_mean":  float(np.mean(a_hop)),
            "hop_std":   float(np.std(a_hop, ddof=1)),
            "ea_mean":   float(np.mean(a_ea)),
            "ea_std":    float(np.std(a_ea, ddof=1)),
            "test":      test_name,
            "statistic": float(t_stat),
            "p_value":   float(p_val),
            "cohens_d":  d,
            "sig":       p_val < 0.05,
            "normal":    normal,
        }
    return output


# Plotting

PALETTE = {
    "hop":     "#4C72B0",   # blue
    "energy":  "#DD8452",   # orange
}

def _label(alg: str) -> str:
    return "Hop-Count (AODV)" if alg == "hop" else "Energy-Aware (AODVM-style)"


def plot_boxplots(
    res_hop: ExperimentResults,
    res_ea:  ExperimentResults,
    stats_dict: Dict,
    out_dir: str,
):
    # box plots for all 6 metrics compared side by side
    metric_pairs = [
        ("Time to First Node Death",   res_hop.first_death_arr(),   res_ea.first_death_arr(),   "Timestep"),
        ("Time to 30 % Node Loss",     res_hop.time_30pct_arr(),    res_ea.time_30pct_arr(),    "Timestep"),
        ("Time to Disconnection",      res_hop.disconnection_arr(), res_ea.disconnection_arr(), "Timestep"),
        ("Packet Delivery Ratio",      res_hop.pdr_arr(),           res_ea.pdr_arr(),           "PDR (0–1)"),
        ("Average Path Length (hops)", res_hop.path_len_arr(),      res_ea.path_len_arr(),      "Hops"),
        ("Energy Variance",            res_hop.energy_var_arr(),    res_ea.energy_var_arr(),    "Variance (energy units²)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"MANET Routing Comparison — {res_hop.n_trials} Paired Trials\n"
        "Hop-Count (AODV) vs Energy-Aware (AODVM-style)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, (title, a_hop, a_ea, ylabel) in zip(axes.flat, metric_pairs):
        bp = ax.boxplot(
            [a_hop, a_ea],
            labels=[_label("hop"), _label("ea")],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
            widths=0.4,
        )
        bp["boxes"][0].set_facecolor(PALETTE["hop"])
        bp["boxes"][1].set_facecolor(PALETTE["energy"])
        for box in bp["boxes"]:
            box.set_alpha(0.7)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)

        # significance annotation on plot
        s = stats_dict[title]
        sig_str = "✓ p<0.05" if s["sig"] else "✗ n.s."
        d_str   = f"d={s['cohens_d']:.2f}"
        ax.text(
            0.97, 0.97, f"{sig_str}\n{d_str}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="darkgreen" if s["sig"] else "gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    plt.tight_layout()
    path = os.path.join(out_dir, "boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_alive_over_time(
    res_hop: ExperimentResults,
    res_ea:  ExperimentResults,
    out_dir: str,
):
    # mean and 1 standard deviation of living node count over time 
    # align histories to the same length 
    def _stack(results):
        hists = [r.alive_history for r in results]
        max_len = max(len(h) for h in hists)
        padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in hists])
        return padded

    hop_mat = _stack(res_hop.results)
    ea_mat  = _stack(res_ea.results)

    fig, ax = plt.subplots(figsize=(12, 5))

    for mat, key in [(hop_mat, "hop"), (ea_mat, "energy")]:
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)
        ts   = np.arange(len(mean))
        ax.plot(ts, mean, color=PALETTE[key], label=_label(key), linewidth=2)
        ax.fill_between(ts, mean - std, mean + std, color=PALETTE[key], alpha=0.2)

    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Living Nodes", fontsize=11)
    ax.set_title("Network Survival — Mean + or - 1 SD across 100 Trials", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    path = os.path.join(out_dir, "alive_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_energy_variance_over_time(
    res_hop: ExperimentResults,
    res_ea:  ExperimentResults,
    out_dir: str,
):
    # mean energy variance (fairness) over time with one standard deviation
    def _stack(results):
        hists = [r.energy_var_history for r in results]
        max_len = max(len(h) for h in hists)
        padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in hists])
        return padded

    hop_mat = _stack(res_hop.results)
    ea_mat  = _stack(res_ea.results)

    fig, ax = plt.subplots(figsize=(12, 5))

    for mat, key in [(hop_mat, "hop"), (ea_mat, "energy")]:
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)
        ts   = np.arange(len(mean))
        ax.plot(ts, mean, color=PALETTE[key], label=_label(key), linewidth=2)
        ax.fill_between(ts, mean - std, mean + std, color=PALETTE[key], alpha=0.2)

    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Energy Variance (fairness indicator)", fontsize=11)
    ax.set_title("Energy Variance Over Time — Mean + or - 1 SD (lower = fairer load distribution)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    path = os.path.join(out_dir, "energy_variance_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_pdr_vs_path_length(
    res_hop: ExperimentResults,
    res_ea:  ExperimentResults,
    out_dir: str,
):
    #Scatter: PDR vs average path length, showing efficiency–fairness tradeoff
    fig, ax = plt.subplots(figsize=(8, 6))

    for res, key in [(res_hop, "hop"), (res_ea, "energy")]:
        ax.scatter(
            res.path_len_arr(), res.pdr_arr(),
            color=PALETTE[key], alpha=0.5, s=30, label=_label(key), edgecolors="none",
        )
        # Mean marker
        ax.scatter(
            [np.mean(res.path_len_arr())], [np.mean(res.pdr_arr())],
            color=PALETTE[key], s=150, marker="D", edgecolors="black", linewidths=1.5, zorder=5,
        )

    ax.set_xlabel("Average Path Length (hops)", fontsize=11)
    ax.set_ylabel("Packet Delivery Ratio", fontsize=11)
    ax.set_title("PDR vs Path Length — Efficiency–Fairness Tradeoff\n(diamonds = means)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    path = os.path.join(out_dir, "pdr_vs_path_length.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")




def print_stats_table(stats_dict: Dict):
    #print the statistical comparison table to terminal 
    col_w = 30
    print("\n" + "=" * 100)
    print(f"{'Metric':<{col_w}} {'Hop mean +- SD':<22} {'EA mean +- SD':<22} {'Test':<25} {'p':<10} {'d':<8} {'Sig?'}")
    print("=" * 100)
    for name, s in stats_dict.items():
        hop_str  = f"{s['hop_mean']:7.2f} +- {s['hop_std']:6.2f}"
        ea_str   = f"{s['ea_mean']:7.2f} +- {s['ea_std']:6.2f}"
        sig_str  = "YES" if s["sig"] else "no"
        p_str    = f"{s['p_value']:.4f}"
        d_str    = f"{s['cohens_d']:+.3f}"
        print(f"{name:<{col_w}} {hop_str:<22} {ea_str:<22} {s['test']:<25} {p_str:<10} {d_str:<8} {sig_str}")
    print("=" * 100 + "\n")


# Main experiment runner

def run_experiment(
    n_trials: int = 100,
    run_sensitivity: bool = False,
    out_dir: str = "results",
    env_kwargs: Optional[dict] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    if env_kwargs is None:
        env_kwargs = {}

    seeds = list(range(n_trials))

    print(f"\n{'='*60}")
    print(f"  MANET Routing Experiment — {n_trials} paired trials")
    print(f"{'='*60}\n")

    #  Run hop-count algorithm 
    print("  [1/2] Running hop-count (AODV) trials …")
    env_hop = MANETEnv(**env_kwargs)
    res_hop = ExperimentResults(algorithm="hop", n_trials=n_trials, max_steps=env_hop.max_steps)
    t0 = time.time()
    for seed in tqdm(seeds, desc="  Hop-count", ncols=70):
        info = env_hop.run_episode(action=0, seed=seed)
        res_hop.results.append(TrialResult(
            seed               = seed,
            first_death_time   = info["first_death_time"],
            time_30pct_death   = info["time_30pct_death"],
            time_disconnection = info["time_disconnection"],
            pdr                = info["pdr"],
            mean_path_length   = info["mean_path_length"],
            energy_variance    = info["energy_variance"],
            alive_history      = info["alive_history"],
            energy_var_history = info["energy_var_history"],
        ))
    print(f"  Done in {time.time()-t0:.1f}s\n")

    # Run energy-aware algorithm 
    print("  [2/2] Running energy-aware (AODVM-style) trials …")
    env_ea = MANETEnv(**env_kwargs)
    res_ea = ExperimentResults(algorithm="energy_aware", n_trials=n_trials, max_steps=env_ea.max_steps)
    t0 = time.time()
    for seed in tqdm(seeds, desc="  Energy-aware", ncols=70):
        info = env_ea.run_episode(action=1, seed=seed)
        res_ea.results.append(TrialResult(
            seed               = seed,
            first_death_time   = info["first_death_time"],
            time_30pct_death   = info["time_30pct_death"],
            time_disconnection = info["time_disconnection"],
            pdr                = info["pdr"],
            mean_path_length   = info["mean_path_length"],
            energy_variance    = info["energy_variance"],
            alive_history      = info["alive_history"],
            energy_var_history = info["energy_var_history"],
        ))
    print(f"  Done in {time.time()-t0:.1f}s\n")

    # statistical comparison
    stats_dict = compare_metrics(res_hop, res_ea)
    print_stats_table(stats_dict)

    # save JSON summary (convert numpy types for serialization)
    def _jsonify(obj):
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    json_path = os.path.join(out_dir, "stats_summary.json")
    with open(json_path, "w") as f:
        json.dump(_jsonify(stats_dict), f, indent=2)
    print(f"  Stats saved to: {json_path}\n")

    # generate the plots
    print("  Generating plots …")
    plot_boxplots(res_hop, res_ea, stats_dict, out_dir)
    plot_alive_over_time(res_hop, res_ea, out_dir)
    plot_energy_variance_over_time(res_hop, res_ea, out_dir)
    plot_pdr_vs_path_length(res_hop, res_ea, out_dir)

    
    # the TA suggested saving detailed logs for one sample trial (seed=0) 
    # this produces per-step energy snapshots and per-packet route logs for both algorithms, saved to JSON
    print("  Generating sample trial logs (seed=0) …")
    for alg_name, alg_action in [("hop_count", 0), ("energy_aware", 1)]:
        sample_env = MANETEnv(**env_kwargs)
        sample_info = sample_env.run_episode(action=alg_action, seed=0)

        sample_log = {
            "algorithm":      alg_name,
            "seed":           0,
            "n_nodes":        sample_env.n_nodes,
            "max_steps":      sample_env.max_steps,
            "alpha":          sample_env.alpha,
            "beta":           sample_env.beta,
            "final_metrics": {
                "pdr":                sample_info["pdr"],
                "mean_path_length":   sample_info["mean_path_length"],
                "energy_variance":    sample_info["energy_variance"],
                "first_death_time":   sample_info["first_death_time"],
                "time_30pct_death":   sample_info["time_30pct_death"],
                "time_disconnection": sample_info["time_disconnection"],
            },
            "energy_history": sample_info["energy_history"],
            "route_log":      sample_info["route_log"],
        }

        log_path = os.path.join(out_dir, f"sample_log_{alg_name}.json")
        with open(log_path, "w") as f:
            json.dump(sample_log, f, indent=2)
        print(f"  Saved: {log_path}  "
              f"({len(sample_info['route_log'])} route entries, "
              f"{len(sample_info['energy_history'])} energy snapshots)")

    print(f"\n  All outputs saved to: {out_dir}/\n")


# CLI entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MANET routing experiment")
    parser.add_argument("--trials",      type=int,   default=100,    help="Number of paired trials")
    parser.add_argument("--out-dir",     type=str,   default="results", help="Output directory for plots")
    parser.add_argument("--n-nodes",     type=int,   default=30)
    parser.add_argument("--max-steps",   type=int,   default=800)
    parser.add_argument("--alpha",       type=float, default=1.0)
    parser.add_argument("--beta",        type=float, default=50.0)
    args = parser.parse_args()

    env_kwargs = dict(
        n_nodes   = args.n_nodes,
        max_steps = args.max_steps,
        alpha     = args.alpha,
        beta      = args.beta,
    )

    run_experiment(
        n_trials        = args.trials,
        out_dir         = args.out_dir,
        env_kwargs      = env_kwargs,
    )
