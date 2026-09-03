import time
import numpy as np
from scipy.optimize import minimize_scalar

from gaussian_models import (
    normal_tail_prob,
    pair_glr_full_gaussian,
    onebit_theorem_boundary,
    common_experimental_boundary,
)
from oracles import solve_shared_lp, solve_regularized_shared_qp
from streams import GaussianArmStreams


class OneBitGridState:
    """
    Exact sufficient statistics for a finite threshold grid.

    n1[k,m] = # {Y=1 from arm k at threshold q_m}
    n0[k,m] = # {Y=0 from arm k at threshold q_m}

    For a finite q_grid this is exactly equivalent, for likelihood/MLE/GLR
    purposes, to storing the entire (q_t,Y_t) history.
    """

    def __init__(self, K, q_grid, param_bounds, sigma=1.0):
        self.K = int(K)
        self.q_grid = np.asarray(q_grid, dtype=float)
        self.M = len(self.q_grid)
        self.lo, self.hi = map(float, param_bounds)
        self.sigma = float(sigma)

        self.n1 = np.zeros((self.K, self.M), dtype=np.int64)
        self.n0 = np.zeros((self.K, self.M), dtype=np.int64)
        self.N = np.zeros(self.K, dtype=np.int64)
        self.mle = np.full(
            self.K, 0.5 * (self.lo + self.hi), dtype=float
        )

    def add(self, arm, q_index, y):
        if int(y) == 1:
            self.n1[arm, q_index] += 1
        else:
            self.n0[arm, q_index] += 1
        self.N[arm] += 1

    def loglik(self, arm, u):
        p = normal_tail_prob(u, self.q_grid, self.sigma)
        return float(
            self.n1[arm] @ np.log(p)
            + self.n0[arm] @ np.log(1.0 - p)
        )

    def update_mle(self, arm):
        res = minimize_scalar(
            lambda u: -self.loglik(arm, u),
            bounds=(self.lo, self.hi),
            method="bounded",
            options={"xatol": 1e-7, "maxiter": 120},
        )
        self.mle[arm] = float(res.x)
        return self.mle[arm]

    def pair_glr(self, i, j):
        # If the unconstrained MLE already satisfies u_i <= u_j, GLR = 0.
        if self.mle[i] <= self.mle[j]:
            return 0.0

        unconstrained = (
            self.loglik(i, self.mle[i])
            + self.loglik(j, self.mle[j])
        )

        # By the scalar tie reduction, the constrained optimum is on u_i=u_j=x.
        res = minimize_scalar(
            lambda x: -(self.loglik(i, x) + self.loglik(j, x)),
            bounds=(self.lo, self.hi),
            method="bounded",
            options={"xatol": 1e-7, "maxiter": 120},
        )
        tied_max = -float(res.fun)
        return max(0.0, unconstrained - tied_max)


def _draw_grid_index(rng, weights):
    weights = np.asarray(weights, dtype=float)
    s = float(weights.sum())
    if s <= 1e-15:
        return int(rng.integers(len(weights)))
    return int(rng.choice(len(weights), p=weights / s))


def _solve_shared_oracle(
    mle,
    beta,
    q_grid,
    x_grid,
    sigma,
    best,
    oracle_mode,
    reg_weight,
    warm_start=None,
):
    if oracle_mode == "lp":
        out = solve_shared_lp(
            mle, beta, q_grid, x_grid, sigma=sigma, best=best
        )
        out["z"] = None
        return out

    if oracle_mode == "regularized":
        return solve_regularized_shared_qp(
            mle,
            beta,
            q_grid,
            x_grid,
            reg_weight,
            sigma=sigma,
            best=best,
            x0=warm_start,
        )

    raise ValueError("oracle_mode must be 'lp' or 'regularized'")


def run_onebit_shared_top_two(
    mu,
    delta=0.05,
    beta=0.5,
    sigma=1.0,
    param_bounds=(-2.0, 1.0),
    q_bounds=(-2.0, 1.0),
    q_grid_size=21,
    x_grid_size=41,
    challenger="TCI",
    max_rounds=30000,
    oracle_period=50,
    oracle_mode="lp",
    reg_scale=0.02,
    reg_power=0.10,
    boundary="common",
    seed=0,
    record_every=50,
    progress_every=1000,
):
    """
    Fast finite-grid Shared Top-Two implementation.

    Exact speedup:
      * Replace the entire (q_t,Y_t) history by finite-grid sufficient counts.

    Experimental speedups:
      * Recompute the shared oracle every `oracle_period` rounds.
      * oracle_mode='lp' uses the unregularized discretized shared LP.

    For theorem-faithful experiments, use:
      oracle_mode='regularized', oracle_period=1.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed + 987654321)
    mu = np.asarray(mu, dtype=float)
    K = len(mu)
    true_best = int(np.argmax(mu))

    q_grid = np.linspace(q_bounds[0], q_bounds[1], q_grid_size)
    x_grid = np.linspace(param_bounds[0], param_bounds[1], x_grid_size)

    state = OneBitGridState(K, q_grid, param_bounds, sigma)
    streams = GaussianArmStreams(mu, sigma=sigma, seed=seed)

    # Initialization: one pull per arm at the middle grid threshold.
    m0 = q_grid_size // 2
    for k in range(K):
        x = streams.sample(k)
        y = int(x >= q_grid[m0])
        state.add(k, m0, y)
        state.update_mle(k)

    n = K
    oracle = None
    oracle_warm = None
    last_oracle_n = -10**9
    last_oracle_best = None

    history = {
        "n": [],
        "leader": [],
        "counts_prop": [],
        "mle": [],
        "min_glr_over_n": [],
        "oracle_w": [],
    }

    stopped = False
    recommendation = int(np.argmax(state.mle))
    stopping_stat = 0.0

    while n < max_rounds:
        leader = int(np.argmax(state.mle))
        glrs = {
            j: state.pair_glr(leader, j)
            for j in range(K) if j != leader
        }
        stopping_stat = min(glrs.values())

        if boundary == "theorem":
            c = onebit_theorem_boundary(
                n, delta, K, param_bounds, q_bounds, sigma
            )
        else:
            c = common_experimental_boundary(n, delta, K)

        if stopping_stat >= c:
            stopped = True
            recommendation = leader
            break

        # TC / TCI challenger
        scores = {}
        for j, w in glrs.items():
            if challenger.upper() == "TCI":
                scores[j] = w + np.log(max(int(state.N[j]), 1))
            else:
                scores[j] = w
        challenger_arm = min(scores, key=scores.get)

        # Shared threshold oracle.
        if (
            oracle is None
            or n - last_oracle_n >= oracle_period
            or leader != last_oracle_best
        ):
            reg_weight = reg_scale * max(n, 1) ** (-reg_power)
            warm = oracle_warm if leader == last_oracle_best else None
            oracle = _solve_shared_oracle(
                state.mle,
                beta,
                q_grid,
                x_grid,
                sigma,
                leader,
                oracle_mode,
                reg_weight,
                warm_start=warm,
            )
            oracle_warm = oracle.get("z", None)
            last_oracle_n = n
            last_oracle_best = leader

        # Top-Two arm choice.
        arm = leader if rng.random() < beta else challenger_arm

        # Conditional threshold design of the selected arm.
        q_index = _draw_grid_index(rng, oracle["rho"][arm])
        x = streams.sample(arm)
        y = int(x >= q_grid[q_index])

        state.add(arm, q_index, y)
        state.update_mle(arm)
        n += 1

        if progress_every and n % progress_every == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"[onebit] n={n}/{max_rounds} "
                f"({100*n/max_rounds:.1f}%) | "
                f"elapsed={elapsed:.1f}s | leader={leader}",
                flush=True,
            )

        if record_every and n % record_every == 0:
            curr_leader = int(np.argmax(state.mle))
            curr_glrs = {
                j: state.pair_glr(curr_leader, j)
                for j in range(K) if j != curr_leader
            }
            history["n"].append(n)
            history["leader"].append(curr_leader)
            history["counts_prop"].append((state.N / n).copy())
            history["mle"].append(state.mle.copy())
            history["min_glr_over_n"].append(min(curr_glrs.values()) / n)
            history["oracle_w"].append(oracle["w"].copy())

    runtime = time.perf_counter() - t0
    return {
        "algorithm": f"onebit-shared-{challenger.upper()}-{oracle_mode}",
        "stopped": stopped,
        "tau": int(n),
        "recommendation": int(recommendation),
        "correct": bool(recommendation == true_best),
        "runtime_sec": float(runtime),
        "counts": state.N.copy(),
        "mle": state.mle.copy(),
        "stopping_stat": float(stopping_stat),
        "history": history,
    }


def run_full_feedback_top_two(
    mu,
    delta=0.05,
    beta=0.5,
    sigma=1.0,
    challenger="TCI",
    max_rounds=30000,
    boundary="common",
    seed=0,
    record_every=50,
    progress_every=0,
):
    """
    Full-feedback Gaussian Top-Two baseline with known common variance.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed + 987654321)
    mu = np.asarray(mu, dtype=float)
    streams = GaussianArmStreams(mu, sigma=sigma, seed=seed)
    K = len(mu)
    true_best = int(np.argmax(mu))

    counts = np.zeros(K, dtype=int)
    sums = np.zeros(K, dtype=float)
    means = np.zeros(K, dtype=float)

    n = 0
    for k in range(K):
        x = streams.sample(k)
        counts[k] += 1
        sums[k] += x
        means[k] = sums[k] / counts[k]
        n += 1

    history = {
        "n": [],
        "leader": [],
        "counts_prop": [],
        "means": [],
        "min_glr_over_n": [],
    }

    stopped = False
    recommendation = int(np.argmax(means))
    stopping_stat = 0.0

    while n < max_rounds:
        leader = int(np.argmax(means))
        pair_glr = {
            j: pair_glr_full_gaussian(
                leader, j, counts, means, sigma
            )
            for j in range(K) if j != leader
        }
        stopping_stat = min(pair_glr.values())

        c = common_experimental_boundary(n, delta, K)
        if stopping_stat >= c:
            stopped = True
            recommendation = leader
            break

        scores = {}
        for j, w in pair_glr.items():
            if challenger.upper() == "TCI":
                scores[j] = w + np.log(max(counts[j], 1))
            else:
                scores[j] = w
        challenger_arm = min(scores, key=scores.get)

        arm = leader if rng.random() < beta else challenger_arm
        x = streams.sample(arm)
        counts[arm] += 1
        sums[arm] += x
        means[arm] = sums[arm] / counts[arm]
        n += 1

        if progress_every and n % progress_every == 0:
            print(
                f"[full] n={n}/{max_rounds} "
                f"({100*n/max_rounds:.1f}%)",
                flush=True,
            )

        if record_every and n % record_every == 0:
            curr_leader = int(np.argmax(means))
            glrs = {
                j: pair_glr_full_gaussian(
                    curr_leader, j, counts, means, sigma
                )
                for j in range(K) if j != curr_leader
            }
            history["n"].append(n)
            history["leader"].append(curr_leader)
            history["counts_prop"].append((counts / n).copy())
            history["means"].append(means.copy())
            history["min_glr_over_n"].append(min(glrs.values()) / n)

    runtime = time.perf_counter() - t0
    return {
        "algorithm": f"full-feedback-{challenger.upper()}",
        "stopped": stopped,
        "tau": int(n),
        "recommendation": int(recommendation),
        "correct": bool(recommendation == true_best),
        "runtime_sec": float(runtime),
        "counts": counts.copy(),
        "means": means.copy(),
        "stopping_stat": float(stopping_stat),
        "history": history,
    }
