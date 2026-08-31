import numpy as np
from scipy.optimize import linprog, minimize, Bounds, LinearConstraint
from gaussian_models import onebit_kl


def _info_tables(mu, q_grid, x_grid, sigma):
    """D[k, r, m] = d_{q_m}(mu_k, x_r)."""
    mu = np.asarray(mu, dtype=float)
    q_grid = np.asarray(q_grid, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    K, M, X = len(mu), len(q_grid), len(x_grid)
    D = np.empty((K, X, M), dtype=float)
    for k in range(K):
        # Broadcasting: x_grid[:,None] with q_grid[None,:]
        D[k] = onebit_kl(q_grid[None, :], mu[k], x_grid[:, None], sigma)
    return D


def solve_shared_lp(mu, beta, q_grid, x_grid, sigma=1.0, best=None):
    """
    Discretized shared one-bit oracle.

    Variables: rho[k,m] and gamma.
    Maximize gamma subject to
      sum rho = 1,
      sum_m rho[best,m] = beta,
      gamma <= sum_m rho[best,m] d_q(mu_best,x)
               + sum_m rho[j,m] d_q(mu_j,x)
      for every challenger j and every x on x_grid.

    This is an LP after discretizing q and x.
    """
    mu = np.asarray(mu, dtype=float)
    K = len(mu)
    M = len(q_grid)
    best = int(np.argmax(mu) if best is None else best)
    D = _info_tables(mu, q_grid, x_grid, sigma)

    nvar = K * M + 1
    gamma_idx = nvar - 1
    c = np.zeros(nvar)
    c[gamma_idx] = -1.0  # maximize gamma

    Aeq = []
    beq = []

    row = np.zeros(nvar)
    row[:K*M] = 1.0
    Aeq.append(row)
    beq.append(1.0)

    row = np.zeros(nvar)
    row[best*M:(best+1)*M] = 1.0
    Aeq.append(row)
    beq.append(beta)

    Aub = []
    bub = []
    for j in range(K):
        if j == best:
            continue
        for r in range(len(x_grid)):
            row = np.zeros(nvar)
            row[best*M:(best+1)*M] = -D[best, r]
            row[j*M:(j+1)*M] = -D[j, r]
            row[gamma_idx] = 1.0
            Aub.append(row)
            bub.append(0.0)

    bounds = [(0.0, None)] * (K*M) + [(0.0, None)]
    res = linprog(
        c,
        A_ub=np.asarray(Aub), b_ub=np.asarray(bub),
        A_eq=np.asarray(Aeq), b_eq=np.asarray(beq),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Shared LP failed: {res.message}")

    rho = res.x[:K*M].reshape(K, M)
    return {
        "gamma": float(res.x[gamma_idx]),
        "rho": rho,
        "w": rho.sum(axis=1),
        "best": best,
        "success": True,
    }


def solve_pairwise_relaxation_lp(mu, beta, q_grid, x_grid, sigma=1.0, best=None):
    """
    Discretized pairwise relaxation.
    Each challenger j gets its own independent best-arm threshold measure of mass beta.
    Challenger masses are coupled only by sum_j w_j = 1-beta.
    """
    mu = np.asarray(mu, dtype=float)
    K = len(mu)
    M = len(q_grid)
    best = int(np.argmax(mu) if best is None else best)
    challengers = [j for j in range(K) if j != best]
    J = len(challengers)
    D = _info_tables(mu, q_grid, x_grid, sigma)

    # For each challenger r:
    # block 2*r: pair-specific best measure
    # block 2*r+1: challenger measure
    n_measure = 2 * J * M
    gamma_idx = n_measure
    nvar = n_measure + 1
    c = np.zeros(nvar)
    c[gamma_idx] = -1.0

    Aeq, beq = [], []

    # Every pair-specific best measure has mass beta.
    for r in range(J):
        row = np.zeros(nvar)
        s = (2*r)*M
        row[s:s+M] = 1.0
        Aeq.append(row)
        beq.append(beta)

    # Total challenger mass is 1-beta.
    row = np.zeros(nvar)
    for r in range(J):
        s = (2*r+1)*M
        row[s:s+M] = 1.0
    Aeq.append(row)
    beq.append(1.0 - beta)

    Aub, bub = [], []
    for r, j in enumerate(challengers):
        sb = (2*r)*M
        sj = (2*r+1)*M
        for xidx in range(len(x_grid)):
            row = np.zeros(nvar)
            row[sb:sb+M] = -D[best, xidx]
            row[sj:sj+M] = -D[j, xidx]
            row[gamma_idx] = 1.0
            Aub.append(row)
            bub.append(0.0)

    bounds = [(0.0, None)] * n_measure + [(0.0, None)]
    res = linprog(
        c,
        A_ub=np.asarray(Aub), b_ub=np.asarray(bub),
        A_eq=np.asarray(Aeq), b_eq=np.asarray(beq),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Pairwise relaxation LP failed: {res.message}")

    pair_best = {}
    challenger_measure = {}
    w = np.zeros(K)
    w[best] = beta
    for r, j in enumerate(challengers):
        sb = (2*r)*M
        sj = (2*r+1)*M
        pair_best[j] = res.x[sb:sb+M].copy()
        challenger_measure[j] = res.x[sj:sj+M].copy()
        w[j] = challenger_measure[j].sum()

    return {
        "gamma": float(res.x[gamma_idx]),
        "pair_best_rho": pair_best,
        "challenger_rho": challenger_measure,
        "w": w,
        "best": best,
        "success": True,
    }


def solve_fixed_threshold_lp(mu, beta, q0, x_grid, sigma=1.0, best=None):
    """Best arm allocation if every sample uses the same fixed threshold q0."""
    mu = np.asarray(mu, dtype=float)
    K = len(mu)
    best = int(np.argmax(mu) if best is None else best)
    challengers = [j for j in range(K) if j != best]
    J = len(challengers)

    nvar = J + 1
    gamma_idx = J
    c = np.zeros(nvar)
    c[gamma_idx] = -1.0

    Aeq = np.zeros((1, nvar))
    Aeq[0, :J] = 1.0
    beq = np.array([1.0 - beta])

    Aub, bub = [], []
    for r, j in enumerate(challengers):
        db = onebit_kl(q0, mu[best], np.asarray(x_grid), sigma)
        dj = onebit_kl(q0, mu[j], np.asarray(x_grid), sigma)
        for xidx in range(len(x_grid)):
            row = np.zeros(nvar)
            row[r] = -dj[xidx]
            row[gamma_idx] = 1.0
            Aub.append(row)
            bub.append(beta * db[xidx])

    bounds = [(0.0, None)] * J + [(0.0, None)]
    res = linprog(
        c,
        A_ub=np.asarray(Aub), b_ub=np.asarray(bub),
        A_eq=Aeq, b_eq=beq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Fixed-threshold LP failed: {res.message}")

    w = np.zeros(K)
    w[best] = beta
    for r, j in enumerate(challengers):
        w[j] = res.x[r]

    return {"gamma": float(res.x[gamma_idx]), "w": w, "best": best}


def full_feedback_beta_oracle(mu, beta, sigma=1.0, best=None, tol=1e-12):
    """
    Exact beta-characteristic value for Gaussian arms with known common variance.

    Pair rate:
        H_j(beta,w) = beta*w/(beta+w) * Delta_j^2/(2 sigma^2).

    Equalization is solved by scalar bisection in gamma.
    """
    mu = np.asarray(mu, dtype=float)
    K = len(mu)
    best = int(np.argmax(mu) if best is None else best)
    challengers = [j for j in range(K) if j != best]
    c = np.array([(mu[best] - mu[j])**2 / (2.0 * sigma**2) for j in challengers])

    def mass_for_gamma(g):
        # H_j = c_j beta w/(beta+w) = g
        return g * beta / (c * beta - g)

    lo = 0.0
    hi = float(np.min(c * beta)) * (1.0 - 1e-12)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        s = np.sum(mass_for_gamma(mid))
        if s > 1.0 - beta:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    gamma = 0.5 * (lo + hi)
    w_ch = mass_for_gamma(gamma)

    w = np.zeros(K)
    w[best] = beta
    for val, j in zip(w_ch, challengers):
        w[j] = val
    return {"gamma": float(gamma), "w": w, "best": best}


def gaussian_kernel_matrix(q_grid, bandwidth=None):
    q = np.asarray(q_grid, dtype=float)
    if bandwidth is None:
        bandwidth = max(np.ptp(q) / 5.0, 1e-3)
    dist2 = (q[:, None] - q[None, :])**2
    return np.exp(-dist2 / bandwidth**2)


def solve_regularized_shared_qp(mu, beta, q_grid, x_grid, reg_weight,
                                sigma=1.0, best=None, bandwidth=None,
                                x0=None, maxiter=500):
    """
    Practical discretized version of the regularized shared oracle:
       maximize gamma - reg_weight * R(rho).

    The continuous-measure theorem is approximated by finite q/x grids.
    Uses scipy SLSQP, so no CVXPY dependency is required.
    """
    mu = np.asarray(mu, dtype=float)
    K = len(mu)
    M = len(q_grid)
    best = int(np.argmax(mu) if best is None else best)
    D = _info_tables(mu, q_grid, x_grid, sigma)
    Kq = gaussian_kernel_matrix(q_grid, bandwidth)

    nvar = K*M + 1
    gamma_idx = nvar - 1

    if x0 is None or len(x0) != nvar:
        rho0 = np.zeros((K, M))
        rho0[best, :] = beta / M
        for j in range(K):
            if j != best:
                rho0[j, :] = (1.0 - beta) / ((K - 1) * M)
        x0 = np.r_[rho0.ravel(), 0.0]
    else:
        x0 = np.asarray(x0, dtype=float).copy()
        x0[gamma_idx] = max(0.0, x0[gamma_idx])

    def objective(z):
        rho = z[:K*M].reshape(K, M)
        R = 0.0
        for k in range(K):
            R += rho[k] @ Kq @ rho[k]
        return -z[gamma_idx] + reg_weight * R

    def gradient(z):
        rho = z[:K*M].reshape(K, M)
        g = np.zeros_like(z)
        for k in range(K):
            g[k*M:(k+1)*M] = 2.0 * reg_weight * (Kq @ rho[k])
        g[gamma_idx] = -1.0
        return g

    Aeq = np.zeros((2, nvar))
    Aeq[0, :K*M] = 1.0
    Aeq[1, best*M:(best+1)*M] = 1.0
    eq = LinearConstraint(Aeq, [1.0, beta], [1.0, beta])

    rows = []
    for j in range(K):
        if j == best:
            continue
        for r in range(len(x_grid)):
            row = np.zeros(nvar)
            # info - gamma >= 0
            row[best*M:(best+1)*M] = D[best, r]
            row[j*M:(j+1)*M] = D[j, r]
            row[gamma_idx] = -1.0
            rows.append(row)
    A = np.asarray(rows)
    ineq = LinearConstraint(A, np.zeros(A.shape[0]), np.full(A.shape[0], np.inf))

    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    bounds = Bounds(lb, ub)

    res = minimize(
        objective, x0, jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=[eq, ineq],
        options={"ftol": 1e-9, "maxiter": maxiter, "disp": False},
    )

    if not res.success:
        # Robust fallback: unregularized LP.
        lp = solve_shared_lp(mu, beta, q_grid, x_grid, sigma=sigma, best=best)
        rho = lp["rho"]
        z = np.r_[rho.ravel(), lp["gamma"]]
        return {
            "gamma": lp["gamma"], "rho": rho, "w": rho.sum(axis=1),
            "best": best, "z": z, "success": False,
            "message": res.message,
        }

    rho = res.x[:K*M].reshape(K, M)
    rho[rho < 1e-14] = 0.0
    return {
        "gamma": float(res.x[gamma_idx]),
        "rho": rho,
        "w": rho.sum(axis=1),
        "best": best,
        "z": res.x.copy(),
        "success": True,
        "message": res.message,
    }
