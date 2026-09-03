import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithms_fast import run_onebit_shared_top_two_fast
from algorithms import run_full_feedback_top_two


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mu", nargs="+", type=float, default=[0.0, -0.3, -0.7])
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.1, 0.05])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--challenger", choices=["TC", "TCI"], default="TCI")

    parser.add_argument("--imin", type=float, default=-2.0)
    parser.add_argument("--imax", type=float, default=1.0)
    parser.add_argument("--qmin", type=float, default=-2.0)
    parser.add_argument("--qmax", type=float, default=1.0)
    parser.add_argument("--Mq", type=int, default=21)
    parser.add_argument("--Mx", type=int, default=41)

    parser.add_argument("--oracle-period", type=int, default=50)
    parser.add_argument(
        "--oracle-mode",
        choices=["lp", "regularized"],
        default="lp",
        help="'lp' is much faster; 'regularized' is closer to the theorem.",
    )
    parser.add_argument("--reg-scale", type=float, default=0.02)
    parser.add_argument("--reg-power", type=float, default=0.10)

    parser.add_argument(
        "--onebit-boundary",
        choices=["theorem", "common"],
        default="common",
    )
    parser.add_argument("--max-rounds", type=int, default=30000)
    parser.add_argument("--record-every", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-prefix", type=str, default="gaussian_fast")
    args = parser.parse_args()

    rows = []
    total_jobs = len(args.deltas) * args.reps * 2
    job = 0

    for delta in args.deltas:
        for rep in range(args.reps):
            seed = args.seed + 100000 * rep + int(1e6 * delta)

            job += 1
            print(
                f"\n=== Job {job}/{total_jobs} | onebit | "
                f"delta={delta} | rep={rep+1}/{args.reps} ==="
            )
            one = run_onebit_shared_top_two_fast(
                mu=np.asarray(args.mu),
                delta=delta,
                beta=args.beta,
                sigma=args.sigma,
                param_bounds=(args.imin, args.imax),
                q_bounds=(args.qmin, args.qmax),
                q_grid_size=args.Mq,
                x_grid_size=args.Mx,
                challenger=args.challenger,
                max_rounds=args.max_rounds,
                oracle_period=args.oracle_period,
                oracle_mode=args.oracle_mode,
                reg_scale=args.reg_scale,
                reg_power=args.reg_power,
                boundary=args.onebit_boundary,
                seed=seed,
                record_every=args.record_every,
                progress_every=args.progress_every,
            )
            rows.append({
                "algorithm": one["algorithm"],
                "delta": delta,
                "rep": rep,
                "tau": one["tau"],
                "correct": one["correct"],
                "stopped": one["stopped"],
                "runtime_sec": one["runtime_sec"],
            })
            print(
                f"onebit done: tau={one['tau']}, "
                f"stopped={one['stopped']}, correct={one['correct']}, "
                f"runtime={one['runtime_sec']:.3f}s"
            )

            job += 1
            print(
                f"\n=== Job {job}/{total_jobs} | full | "
                f"delta={delta} | rep={rep+1}/{args.reps} ==="
            )
            full = run_full_feedback_top_two(
                mu=np.asarray(args.mu),
                delta=delta,
                beta=args.beta,
                sigma=args.sigma,
                challenger=args.challenger,
                max_rounds=args.max_rounds,
                boundary="common",
                seed=seed,
                record_every=args.record_every,
            )
            rows.append({
                "algorithm": full["algorithm"],
                "delta": delta,
                "rep": rep,
                "tau": full["tau"],
                "correct": full["correct"],
                "stopped": full["stopped"],
                "runtime_sec": full["runtime_sec"],
            })
            print(
                f"full done: tau={full['tau']}, "
                f"stopped={full['stopped']}, correct={full['correct']}, "
                f"runtime={full['runtime_sec']:.3f}s"
            )

    df = pd.DataFrame(rows)
    raw_file = args.out_prefix + "_raw.csv"
    summary_file = args.out_prefix + "_summary.csv"
    df.to_csv(raw_file, index=False)

    summary = (
        df.groupby(["algorithm", "delta"])
          .agg(
              mean_tau=("tau", "mean"),
              se_tau=("tau", lambda x: x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0.0),
              stop_rate=("stopped", "mean"),
              error_rate=("correct", lambda x: 1.0 - np.mean(x)),
              mean_runtime=("runtime_sec", "mean"),
          )
          .reset_index()
    )
    summary.to_csv(summary_file, index=False)

    plt.figure()
    for algo, g in summary.groupby("algorithm"):
        g = g.sort_values("delta", ascending=False)
        plt.plot(np.log(1.0/g["delta"]), g["mean_tau"], marker="o", label=algo)
    plt.xlabel(r"$\log(1/\delta)$")
    plt.ylabel(r"mean stopping time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_prefix + "_mean_tau.png", dpi=180)
    plt.close()

    plt.figure()
    for algo, g in summary.groupby("algorithm"):
        g = g.sort_values("delta", ascending=False)
        plt.plot(g["delta"], g["mean_runtime"], marker="o", label=algo)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\delta$")
    plt.ylabel("mean runtime [s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_prefix + "_runtime.png", dpi=180)
    plt.close()

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(raw_file)
    print(summary_file)
    print(args.out_prefix + "_mean_tau.png")
    print(args.out_prefix + "_runtime.png")


if __name__ == "__main__":
    main()
