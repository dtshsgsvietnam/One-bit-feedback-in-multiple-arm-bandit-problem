# Gaussian one-bit shared Top-Two experiments

This folder is a first experimental implementation for the K-arm proof note.

## Where does the data come from?

No external dataset is required for the Gaussian experiments.

For a chosen instance

\[
X_{t,k}\sim \mathcal N(\mu_k,\sigma^2),
\]

the simulator generates fresh Gaussian rewards.

* **One-bit algorithm:** it immediately converts the generated reward to
  `Y = 1{X >= q}` and the algorithm stores only `(q,Y)`.
* **Full-feedback Top-Two baseline:** it observes the Gaussian reward `X` directly.

Thus the "dataset" is synthetic Monte-Carlo data generated from the statistical model itself.
This is standard for controlled BAI experiments because the true best arm and true
characteristic times are known.

## Files

- `gaussian_models.py`: Gaussian probabilities, one-bit KL, MLE, GLRs, stopping boundaries.
- `oracles.py`: discretized shared oracle, pairwise relaxation, fixed-q oracle,
  exact Gaussian full-feedback beta oracle, regularized shared QP.
- `algorithms.py`: sequential one-bit shared Top-Two and full-feedback Gaussian Top-Two.
- `streams.py`: paired per-arm Gaussian reward streams for fair comparisons.
- `run_oracle_study.py`: offline information-rate comparison.
- `run_sequential_compare.py`: Monte-Carlo stopping-time/runtime comparison.

Only NumPy/SciPy/Pandas/Matplotlib are required.

## Step 1: offline oracle study

Start here before doing expensive sequential simulation.

```bash
python run_oracle_study.py \
  --mu 0 -0.1 -0.3 \
  --beta 0.5 \
  --qmin -2 --qmax 1 \
  --imin -2 --imax 1 \
  --Mq 61 --Mx 121 \
  --q0 0
```

It reports

- fixed-threshold one-bit rate,
- shared adaptive one-bit rate,
- pairwise-relaxed one-bit rate,
- exact full-feedback Gaussian rate,

and the corresponding characteristic times.

The main ratios to inspect are

```text
shared / pairwise
shared / full
```

## Step 2: quick sequential smoke test

```bash
python run_sequential_compare.py \
  --mu 0 -0.2 -0.5 \
  --deltas 0.1 0.05 \
  --reps 5 \
  --Mq 21 --Mx 41 \
  --oracle-period 25 \
  --max-rounds 5000
```

For paper-quality experiments, increase `reps`, `Mq`, and `Mx`.

## Step 3: paper-scale runs

Example:

```bash
python run_sequential_compare.py \
  --mu 0 -0.1 -0.3 \
  --deltas 0.1 0.05 0.02 0.01 0.005 \
  --reps 500 \
  --Mq 41 --Mx 81 \
  --oracle-period 20 \
  --challenger TCI \
  --max-rounds 50000 \
  --out-prefix gaussian_K3
```

## Important numerical notes

1. The continuous threshold measure is discretized on `q_grid`.
2. The least-favorable alternative `x` is discretized in the oracle on `x_grid`.
3. The sequential regularized oracle uses SciPy SLSQP.
4. For speed, it is recomputed only once every `oracle_period` rounds.
5. The practical regularization schedule in the simulation is
   `lambda_n = reg_scale * n^{-reg_power}`.  It is a numerical surrogate for the
   vanishing regularization used in the proof.
6. `--onebit-boundary theorem` uses the K-arm theorem boundary from the proof note.
   `--onebit-boundary common` uses the same experimental boundary as the full-feedback
   baseline and is better for an apples-to-apples finite-time plot, but should not be
   advertised as the rigorous theorem calibration.

## Suggested Gaussian instances

```text
(0, -0.1, -0.12)             multiple hard challengers
(0, -0.1, -1.0)              one hard, one easy challenger
(0, -0.2, -0.4, -0.6)        regular K=4 instance
(0, -0.1, -0.11, -0.12)      many near-best arms
```

## What should be plotted in the paper?

1. `Gamma_shared / Gamma_pairwise` across instances.
2. `Gamma_shared / Gamma_full`.
3. Mean stopping time versus `log(1/delta)`.
4. `E[tau]/log(1/delta)` versus delta.
5. Allocation trajectories `N_{n,k}/n`.
6. Wall-clock runtime per Monte-Carlo run.


For sequential comparison, both algorithms use the same arm-specific latent Gaussian streams when given the same seed.
