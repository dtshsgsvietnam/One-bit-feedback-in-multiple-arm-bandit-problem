import numpy as np


class GaussianArmStreams:
    """
    Independent reproducible Gaussian stream for each arm.

    With the same base seed, two algorithms see the same m-th latent reward
    whenever they pull arm k for the m-th time.  This is useful for paired
    Monte-Carlo comparisons even when the two algorithms choose arms at
    different calendar times.
    """
    def __init__(self, mu, sigma=1.0, seed=0):
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = float(sigma)
        ss = np.random.SeedSequence(seed)
        children = ss.spawn(len(self.mu))
        self.rng = [np.random.default_rng(s) for s in children]

    def sample(self, arm):
        return float(self.rng[arm].normal(self.mu[arm], self.sigma))
