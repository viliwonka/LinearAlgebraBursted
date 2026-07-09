# Spec: Frisch–Newton exact LAD (and quantile) solver

Goal: the production exact-L1 engine for `LP.lad` — a structure-exploiting primal-dual interior
point on the LAD **dual**, working directly on the original m×n data. No LP reformulation, no
2n+2m blow-up, no m×m matrices: every Newton step is an **n×n weighted normal solve** built by
one streaming pass over A. Reference: Portnoy & Koenker 1997 ("The Gaussian Hare and the
Laplacian Tortoise"); reference implementations: R quantreg `rqfnb.f` (Fortran) and the widely
reproduced MATLAB port `lp_fnm`/`rq_fnm`. **Verify the iteration against one of those sources
before coding — do not reconstruct from memory** (both are fetchable; the MATLAB `lp_fnm` is
short and complete).

## Problem and dual

Quantile regression with parameter τ ∈ (0,1) (τ = 0.5 is LAD up to a factor 2):

    min_x  Σ_i ρ_τ(b_i − a_iᵀ x),     ρ_τ(u) = u·(τ − 1[u<0])

Its dual is a box-constrained LP over dual weights, solved in the shifted form used by rqfnb:

    max_a  bᵀa    s.t.  Aᵀa = (1−τ)·Aᵀ1,    a ∈ [0, 1]^m

The equality multipliers y ∈ ℝⁿ of the dual ARE the regression coefficients x (sign conventions
per the reference port — get them from the source, this is where hand-derivation goes wrong).

## Algorithm (lp_fnm: Mehrotra predictor-corrector on the bounded dual)

State: a ∈ (0,1)^m (interior), s = 1 − a, multipliers y ∈ ℝⁿ, reduced-cost splits z, w ≥ 0.
Init: y from a LEAST-SQUARES fit (we have QR/CHO — use the library), z/w from the residual
split with a small floor; a = (1−τ)·1.

Per iteration:
1. Diagonal weights q_i = 1 / (z_i/a_i + w_i/s_i)   (all control math per library idiom).
2. **The kernel**: solve  (Aᵀ Q A) Δy = rhs  — an n×n SPD system, Q diagonal. Build AᵀQA by one
   pass over A's rows (accumulate q_i·a_i a_iᵀ), factor with the library Cholesky (CHO), solve.
   Predictor and corrector REUSE the same factorization (two solves, one factor per iteration).
3. Mehrotra steps: affine predictor → step-length ratio test toward the a∈[0,1], z,w≥0
   boundaries (0.9995 factor) → centering parameter from the affine gap (µ_aff/µ)³ → corrected
   direction → separate primal/dual step lengths.
4. Converge on duality gap ≤ eps (default 1e-8 double-equivalent; per-dtype via the usual
   Consts-derived tolerances). Iteration cap default 50; return best iterate on cap.

Cost: O(mn²) per iteration for AᵀQA + O(n³) Cholesky — with n small this is IRLS-priced per
iteration, ~10–50 iterations, EXACT result.

## API

- Core: `ladFrischNewtonCore(in fProxyMxN A, in fProxyN b, double tau, ref fProxyN x,
  out double objective, int maxIter)` — τ-parameterized from day one (quantile regression is
  the τ≠0.5 case; a public Stats/ML quantile surface comes later, per memory note).
- Public now: `LP.ladFN(in fProxyMxN A, in fProxyN b, ref fProxyN x, out double objective,
  int maxIter = 0)` — exact LAD via FN (τ=0.5), objective = ‖Ax−b‖₁ recomputed from the
  returned x (honest, not the internal gap). Returns `LPInfo` (Optimal/MaxIter; FN on a
  bounded-dual is never Infeasible/Unbounded for finite data).
- `LP.lad`'s default routing is NOT changed in this round — benchmark first, flip after.

## Files / repo constraints

`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.FrischNewton.fProxy.cs` (new). Fully
fProxy-templated (NO double-only kernel — standing user rule; double allowed only for pure
local scalar accumulators like the gap and objective). Reuse library CHO for the n×n solves
and QR/CHO for the LS init. Allocator.Temp scratch, disposed on all paths; CS1750 forwarding
overloads; no fProxy-returning parameterless helpers (CS0111).

## Tests (LPTests.fProxy.cs pattern)

1. FN vs the LP.lad oracles: random overdetermined instances (m ∈ {48, 96, 192}, n=4, gross
   outliers — reuse the benchmark construction), FN L1 residual matches LPMethod.Simplex /
   RevisedSimplex within 1e-6 rel (double) / 1e-3 (float).
2. Stackloss known-answer (the existing LadStackloss literature vector) via FN.
3. τ sanity: τ=0.5 equals LAD; and a τ=0.25 instance where the known property holds —
   approximately 25% of residuals negative (count-based assert with slack).
4. Degenerate: exact-fit data (b = A·x_true, no noise) → residual ~0, no division blowups.

## Benchmark

Add an `LP.ladFN` row to LAD Section 2 at all existing sizes (it should sit near IRLS's
timings while being exact); do not extend sizes in this round.
