import argparse
import numpy as np
import pandas as pd

from oracles import (
    solve_shared_lp,
    solve_pairwise_relaxation_lp,
    solve_fixed_threshold_lp,
    full_feedback_beta_oracle,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", nargs="+", type=float, default=[0.0, -0.1, -0.3])
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--qmin", type=float, default=-2.0)
    parser.add_argument("--qmax", type=float, default=1.0)
    parser.add_argument("--imin", type=float, default=-2.0)
    parser.add_argument("--imax", type=float, default=1.0)
    parser.add_argument("--Mq", type=int, default=61)
    parser.add_argument("--Mx", type=int, default=121)
    parser.add_argument("--q0", type=float, default=0.0)
    parser.add_argument("--out", type=str, default="oracle_results.csv")
    args = parser.parse_args()

    mu = np.asarray(args.mu, dtype=float)
    q_grid = np.linspace(args.qmin, args.qmax, args.Mq)
    x_grid = np.linspace(args.imin, args.imax, args.Mx)

    shared = solve_shared_lp(mu, args.beta, q_grid, x_grid, args.sigma)
    pair = solve_pairwise_relaxation_lp(mu, args.beta, q_grid, x_grid, args.sigma)
    fixed = solve_fixed_threshold_lp(mu, args.beta, args.q0, x_grid, args.sigma)
    full = full_feedback_beta_oracle(mu, args.beta, args.sigma)

    rows = [
        {"method": "onebit_fixed_q", "gamma": fixed["gamma"],
         "T_beta": 1.0/fixed["gamma"]},
        {"method": "onebit_shared", "gamma": shared["gamma"],
         "T_beta": 1.0/shared["gamma"]},
        {"method": "onebit_pairwise_relaxation", "gamma": pair["gamma"],
         "T_beta": 1.0/pair["gamma"]},
        {"method": "full_feedback_gaussian", "gamma": full["gamma"],
         "T_beta": 1.0/full["gamma"]},
    ]
    df = pd.DataFrame(rows)
    df["gamma_over_full"] = df["gamma"] / full["gamma"]
    df["samples_vs_full"] = full["gamma"] / df["gamma"]
    df.to_csv(args.out, index=False)

    print(df.to_string(index=False))
    print("\nShared allocation:", np.round(shared["w"], 6))
    print("Full-feedback allocation:", np.round(full["w"], 6))
    print("Shared / pairwise =", shared["gamma"] / pair["gamma"])
    print("Shared / full     =", shared["gamma"] / full["gamma"])
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
