"""Parts 4-6: plot learning curves and compare runs.

Your group's runs (W&B curves + your observer logs):
    python scripts/plot_results.py --runs noHIL HIL
    python scripts/plot_results.py --runs HIL expB --xlim 20

Whole class (Part 6):
    python scripts/plot_results.py --class

Other options:
    --list-metrics         print the metric names logged in W&B (if a plot is empty)
    --offline              only use the observer logs (no W&B)
    --reward-key / --intervention-key   substrings used to find the W&B metrics
    --threshold 0.8        rolling-mean reward counted as "solved"

Figures and tables are saved in runs/plots/.
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import LOGS, PLOTS, REPO, WANDB_PROJECT, add_common_args, require_group  # noqa: E402


# --------------------------------------------------------------------------- W&B
def wandb_api():
    import wandb
    return wandb.Api()


def project_path(api, entity):
    return f"{entity or api.default_entity}/{WANDB_PROJECT}"


def history(api, entity, tag):
    runs = [r for r in api.runs(project_path(api, entity)) if (r.name or "") == tag]
    if not runs:
        print(f"  (no W&B run named {tag} in {project_path(api, entity)})")
        return None
    run = sorted(runs, key=lambda r: r.created_at)[-1]
    return run.history(samples=100_000, pandas=True)


def find_col(df, key):
    cols = [c for c in df.columns if key.lower() in c.lower() and not c.startswith("_")]
    return cols[0] if cols else None


def time_to_threshold(df, key, thr, window):
    col = find_col(df, key) if df is not None else None
    if col is None or "_runtime" not in df:
        return np.nan
    s = df[["_runtime", col]].dropna()
    r = s[col].rolling(window, min_periods=window).mean()
    hit = s.loc[r >= thr, "_runtime"]
    return round(hit.iloc[0] / 60, 1) if len(hit) else np.nan


# --------------------------------------------------------------------------- observer logs
def observer(group, run):
    path = LOGS / f"{group}_{run}_observer.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def observer_summary(df):
    if df is None:
        return {}
    fs = df.loc[df["kind"] == "first_success", "minutes"]
    iv = df.loc[df["kind"] == "intervention", "value"]
    return {"first_success_min (observer)": fs.iloc[0] if len(fs) else np.nan,
            "n_interventions": int(len(iv)), "human_effort_s": float(iv.sum())}


# --------------------------------------------------------------------------- modes
def plot_group(args, group):
    api = None if args.offline else wandb_api()
    hist, obs, rows = {}, {}, []
    for run in args.runs:
        tag = f"{group}_{run}"
        hist[run] = None if api is None else history(api, args.entity, tag)
        obs[run] = observer(group, run)
        rows.append({"run": tag,
                     f"min_to_reward>={args.threshold}": time_to_threshold(hist[run], args.reward_key, args.threshold, args.window),
                     **observer_summary(obs[run])})

    if args.list_metrics:
        for run, df in hist.items():
            if df is not None:
                print(f"\n{run}: " + ", ".join(sorted(c for c in df.columns if not c.startswith("_"))))
        return

    n_panels = 1 if args.offline else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5), squeeze=False)
    axes = axes[0]
    obs_ax = axes[-1]
    if not args.offline:
        axes[0].set_title(f"Episodic reward (W&B, rolling mean over {args.window})")
        axes[1].set_title(f"Intervention rate (W&B, rolling mean over {args.window})")
    for run in args.runs:
        df = hist[run]
        if not args.offline:
            for ax, key in ((axes[0], args.reward_key), (axes[1], args.intervention_key)):
                col = find_col(df, key) if df is not None else None
                if col is None:
                    continue
                s = df[["_runtime", col]].dropna()
                ax.plot(s["_runtime"] / 60, s[col].rolling(args.window, min_periods=1).mean(), label=run)
                ax.set_ylabel(col)
        o = obs[run]
        if o is not None:
            prog = o[o["kind"] == "successes_last10"]
            obs_ax.plot(prog["minutes"], prog["value"], marker="o", label=run)
    obs_ax.set_title("Observer: successes in last 10 episodes")
    obs_ax.set_ylim(-0.5, 10.5)
    for ax in axes:
        ax.set_xlabel("wall-clock time (min)")
        ax.grid(alpha=0.3)
        if args.xlim:
            ax.set_xlim(0, args.xlim)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    name = f"{group}_{'_vs_'.join(args.runs)}"
    fig.tight_layout()
    fig.savefig(PLOTS / f"{name}.png", dpi=120)
    table = pd.DataFrame(rows)
    table.to_csv(PLOTS / f"{name}.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nSaved {(PLOTS / (name + '.png')).relative_to(REPO)} and .csv")


def plot_class(args):
    api = wandb_api()
    rows = []
    for run in api.runs(project_path(api, args.entity)):
        name = run.name or ""
        suffix = name.rsplit("_", 1)[-1] if "_" in name else ""
        if suffix not in ("noHIL", "HIL") and not suffix.startswith("exp"):
            continue
        h = run.history(samples=100_000, pandas=True)
        rows.append({"run": name, "condition": suffix,
                     "minutes_to_threshold": time_to_threshold(h, args.reward_key, args.threshold, args.window)})
    if not rows:
        sys.exit(f"No runs found in {project_path(api, args.entity)}")
    df = pd.DataFrame(rows).sort_values(["condition", "minutes_to_threshold"])
    df.to_csv(PLOTS / "class_results.csv", index=False)
    print(df.to_string(index=False))
    print("\n", df.groupby("condition")["minutes_to_threshold"].describe().round(1).to_string())

    conds = [c for c in ("noHIL", "HIL") if c in set(df["condition"])] + sorted(
        c for c in set(df["condition"]) if c.startswith("exp"))
    rng = np.random.default_rng(0)
    plt.figure(figsize=(1.6 * len(conds) + 3, 4.5))
    labels = []
    for i, c in enumerate(conds):
        v = df.loc[df["condition"] == c, "minutes_to_threshold"]
        plt.scatter(i + rng.uniform(-0.1, 0.1, len(v)), v, alpha=0.7)
        labels.append(f"{c}\n{int(v.notna().sum())}/{len(v)} solved")
    plt.xticks(range(len(conds)), labels)
    plt.ylabel(f"minutes to rolling reward ≥ {args.threshold}")
    plt.title("Class results (lower is faster)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "class_results.png", dpi=120)
    print(f"\nSaved {(PLOTS / 'class_results.png').relative_to(REPO)} and class_results.csv")


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--runs", nargs="+", default=["noHIL", "HIL"], help="Run suffixes, e.g. noHIL HIL expB")
    p.add_argument("--class", dest="class_mode", action="store_true", help="Compare all runs of the class")
    p.add_argument("--offline", action="store_true", help="Observer logs only, no W&B")
    p.add_argument("--list-metrics", action="store_true")
    p.add_argument("--entity", default=None, help="W&B entity (team) holding the class project")
    p.add_argument("--reward-key", default="reward")
    p.add_argument("--intervention-key", default="intervention")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--xlim", type=float, default=None, help="Limit the time axis (minutes)")
    args = p.parse_args()

    if args.class_mode:
        plot_class(args)
    else:
        plot_group(args, require_group(args))


if __name__ == "__main__":
    main()
