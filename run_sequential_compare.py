import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithms import run_onebit_shared_top_two, run_full_feedback_top_two
from oracles import solve_shared_lp, full_feedback_beta_oracle


def run_many(args):
    mu = np.asarray(args.mu, dtype=float)
    true_best = int(np.argmax(mu))

    rows = []
    trajectories = {}

    for delta in args.deltas:
        for algo in ["onebit", "full"]:
            for rep in range(args.reps):
                seed = args.seed + 100000*rep + int(1e6*delta)
                if algo == "onebit":
                    out = run_onebit_shared_top_two(
                        mu=mu, delta=delta, beta=args.beta, sigma=args.sigma,
                        param_bounds=(args.imin, args.imax),
                        q_bounds=(args.qmin, args.qmax),
                        q_grid_size=args.Mq, x_grid_size=args.Mx,
                        challenger=args.challenger,
                        max_rounds=args.max_rounds,
                        oracle_period=args.oracle_period,
                        reg_scale=args.reg_scale,
                        reg_power=args.reg_power,
                        boundary=args.onebit_boundary,
                        seed=seed,
                        record_every=args.record_every,
                    )
                else:
                    out = run_full_feedback_top_two(
                        mu=mu, delta=delta, beta=args.beta, sigma=args.sigma,
                        challenger=args.challenger,
                        max_rounds=args.max_rounds,
                        boundary="common",
                        seed=seed,
                        record_every=args.record_every,
                    )

                rows.append({
                    "algorithm": out["algorithm"],
                    "delta": delta,
                    "rep": rep,
                    "tau": out["tau"],
                    "correct": out["correct"],
                    "stopped": out["stopped"],
                    "runtime_sec": out["runtime_sec"],
                })

                if rep == 0 and delta == args.deltas[0]:
                    trajectories[algo] = out["history"]

                print(algo, "delta=", delta, "rep=", rep,
                      "tau=", out["tau"], "correct=", out["correct"],
                      "runtime=", round(out["runtime_sec"], 3))

    return pd.DataFrame(rows), trajectories


def plot_summary(df, out_prefix):
    summary = (
        df.groupby(["algorithm", "delta"])
          .agg(mean_tau=("tau", "mean"),
               se_tau=("tau", lambda x: x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0.0),
               error_rate=("correct", lambda x: 1.0 - np.mean(x)),
               mean_runtime=("runtime_sec", "mean"))
          .reset_index()
    )
    summary.to_csv(out_prefix + "_summary.csv", index=False)

    plt.figure()
    for algo, g in summary.groupby("algorithm"):
        g = g.sort_values("delta", ascending=False)
        plt.plot(np.log(1.0/g["delta"]), g["mean_tau"], marker="o", label=algo)
    plt.xlabel(r"$\log(1/\delta)$")
    plt.ylabel(r"mean stopping time $\mathbb{E}[\tau_\delta]$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_mean_tau.png", dpi=180)
    plt.close()

    plt.figure()
    for algo, g in summary.groupby("algorithm"):
        g = g.sort_values("delta", ascending=False)
        plt.plot(g["delta"], g["mean_tau"]/np.log(1.0/g["delta"]),
                 marker="o", label=algo)
    plt.xscale("log")
    plt.xlabel(r"$\delta$")
    plt.ylabel(r"$\mathbb{E}[\tau_\delta]/\log(1/\delta)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_normalized_tau.png", dpi=180)
    plt.close()

    plt.figure()
    for algo, g in summary.groupby("algorithm"):
        plt.plot(g["delta"], g["mean_runtime"], marker="o", label=algo)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\delta$")
    plt.ylabel("mean wall-clock time per run [s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_runtime.png", dpi=180)
    plt.close()

    return summary


def plot_trajectory(hist, label, out_file):
    if not hist or len(hist["n"]) == 0:
        return
    n = np.asarray(hist["n"])
    props = np.asarray(hist["counts_prop"])
    plt.figure()
    for k in range(props.shape[1]):
        plt.plot(n, props[:, k], label=f"arm {k+1}")
    plt.xlabel("n")
    plt.ylabel(r"$N_{n,k}/n$")
    plt.title(label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", nargs="+", type=float, default=[0.0, -0.1, -0.3])
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--deltas", nargs="+", type=float,
                        default=[0.1, 0.05, 0.02, 0.01])
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--challenger", choices=["TC", "TCI"], default="TCI")
    parser.add_argument("--imin", type=float, default=-2.0)
    parser.add_argument("--imax", type=float, default=1.0)
    parser.add_argument("--qmin", type=float, default=-2.0)
    parser.add_argument("--qmax", type=float, default=1.0)
    parser.add_argument("--Mq", type=int, default=31)
    parser.add_argument("--Mx", type=int, default=61)
    parser.add_argument("--oracle-period", type=int, default=20)
    parser.add_argument("--reg-scale", type=float, default=0.02)
    parser.add_argument("--reg-power", type=float, default=0.10)
    parser.add_argument("--onebit-boundary", choices=["theorem", "common"], default="common")
    parser.add_argument("--max-rounds", type=int, default=20000)
    parser.add_argument("--record-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-prefix", type=str, default="seq_compare")
    args = parser.parse_args()

    df, trajectories = run_many(args)
    df.to_csv(args.out_prefix + "_raw.csv", index=False)
    summary = plot_summary(df, args.out_prefix)

    if "onebit" in trajectories:
        plot_trajectory(trajectories["onebit"], "One-bit shared Top-Two",
                        args.out_prefix + "_traj_onebit.png")
    if "full" in trajectories:
        plot_trajectory(trajectories["full"], "Full-feedback Top-Two",
                        args.out_prefix + "_traj_full.png")

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nSaved files with prefix:", args.out_prefix)


if __name__ == "__main__":
    main()
