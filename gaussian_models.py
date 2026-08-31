import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize_scalar

EPS = 1e-12


def normal_tail_prob(mu, q, sigma=1.0):
    """P_mu[X >= q] for X ~ N(mu, sigma^2)."""
    z = (np.asarray(mu) - np.asarray(q)) / sigma
    return np.clip(ndtr(z), EPS, 1.0 - EPS)


def bernoulli_kl(p, r):
    """Elementwise Bernoulli KL."""
    p = np.clip(np.asarray(p), EPS, 1.0 - EPS)
    r = np.clip(np.asarray(r), EPS, 1.0 - EPS)
    return p * np.log(p / r) + (1.0 - p) * np.log((1.0 - p) / (1.0 - r))


def onebit_kl(q, mu, alt, sigma=1.0):
    """d_q(mu, alt). Broadcasting is supported."""
    return bernoulli_kl(
        normal_tail_prob(mu, q, sigma),
        normal_tail_prob(alt, q, sigma),
    )


def onebit_loglik(mu, qs, ys, sigma=1.0):
    """Log-likelihood for one arm from threshold/bit observations."""
    if len(qs) == 0:
        return 0.0
    qs = np.asarray(qs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    p = normal_tail_prob(mu, qs, sigma)
    return float(np.sum(ys * np.log(p) + (1.0 - ys) * np.log(1.0 - p)))


def mle_onebit(qs, ys, param_bounds, sigma=1.0):
    """Scalar MLE for one Gaussian arm observed only through threshold bits."""
    lo, hi = map(float, param_bounds)
    if len(qs) == 0:
        return 0.5 * (lo + hi)

    obj = lambda u: -onebit_loglik(u, qs, ys, sigma)
    res = minimize_scalar(obj, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-7, "maxiter": 300})
    if not res.success:
        # Fallback to a dense deterministic grid.
        grid = np.linspace(lo, hi, 501)
        vals = np.array([onebit_loglik(x, qs, ys, sigma) for x in grid])
        return float(grid[np.argmax(vals)])
    return float(res.x)


def pair_glr_onebit(i, j, mle, qs_by_arm, ys_by_arm, param_bounds, sigma=1.0):
    """
    Pairwise GLR W_n(i,j) for H0: u_i <= u_j.
    If the unconstrained MLE already satisfies u_i <= u_j, W=0.
    Otherwise concavity puts the constrained optimum on the boundary u_i=u_j=x.
    """
    mi, mj = float(mle[i]), float(mle[j])
    if mi <= mj:
        return 0.0

    li_hat = onebit_loglik(mi, qs_by_arm[i], ys_by_arm[i], sigma)
    lj_hat = onebit_loglik(mj, qs_by_arm[j], ys_by_arm[j], sigma)

    lo, hi = map(float, param_bounds)
    def neg_tied_ll(x):
        return -(onebit_loglik(x, qs_by_arm[i], ys_by_arm[i], sigma)
                 + onebit_loglik(x, qs_by_arm[j], ys_by_arm[j], sigma))

    res = minimize_scalar(neg_tied_ll, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-7, "maxiter": 300})
    tied_max = -float(res.fun)
    return max(0.0, li_hat + lj_hat - tied_max)


def pair_glr_full_gaussian(i, j, counts, means, sigma=1.0):
    """
    Closed-form Gaussian known-variance transportation cost:
    W = N_i N_j/(N_i+N_j) * (mu_hat_i-mu_hat_j)^2/(2 sigma^2),
    when empirical mean i > empirical mean j; otherwise W=0.
    """
    ni, nj = int(counts[i]), int(counts[j])
    if ni <= 0 or nj <= 0 or means[i] <= means[j]:
        return 0.0
    return (ni * nj / (ni + nj)) * (means[i] - means[j])**2 / (2.0 * sigma**2)


def onebit_theorem_boundary(n, delta, K, param_bounds, q_bounds, sigma=1.0):
    """
    Boundary used in the current K-arm proof:
      1 + log(pi^2/(6 delta)) + K log n + K log(1 + G R n).
    G is computed exactly for Gaussian log p/log(1-p) over compact I x Q.
    """
    lo, hi = map(float, param_bounds)
    qlo, qhi = map(float, q_bounds)
    R = hi - lo

    # z=(mu-q)/sigma ranges over [zmin,zmax].
    zmin = (lo - qhi) / sigma
    zmax = (hi - qlo) / sigma

    # phi(z)
    phi_min = np.exp(-0.5 * zmin*zmin) / np.sqrt(2*np.pi)
    phi_max = np.exp(-0.5 * zmax*zmax) / np.sqrt(2*np.pi)

    # d/dmu log Phi(z) = phi(z)/(sigma Phi(z))
    g1 = phi_min / (sigma * max(ndtr(zmin), EPS))
    # |d/dmu log(1-Phi(z))| = phi(z)/(sigma Phi(-z))
    g2 = phi_max / (sigma * max(ndtr(-zmax), EPS))
    G = max(g1, g2)

    n = max(int(n), 1)
    return (1.0 + np.log(np.pi**2 / (6.0 * delta))
            + K * np.log(n)
            + K * np.log(1.0 + G * R * n))


def common_experimental_boundary(n, delta, K):
    """
    A common boundary useful for apples-to-apples plots.
    It is intentionally labeled 'experimental'; use the theorem-calibrated
    boundary when reporting rigorous one-bit delta-correctness.
    """
    n = max(int(n), 1)
    return np.log((K - 1) * np.pi**2 * n**2 / (6.0 * delta))
