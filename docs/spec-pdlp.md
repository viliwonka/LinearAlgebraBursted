# Spec: PDLP — matrix-free first-order LP solver (restarted PDHG)

Status: ✅ BUILT (2026-07-08) — all incremental stages 1a→1e landed and tested (9/9 PDLP tests: dense +
BSR Wyndor + dense-vs-BSR agreement, ×float/double). `LP.pdlp(dense)` and `LP.pdlp(BSR)` share one generic
`pdlpCore<TOp>` (restart + primal weight + adaptive step) and one `pdlpScaledSolve<TOp>` (equilibration
glue); preconditioning is applied matrix-free via the new `fProxyRowColScaledOperator<TInner>`. Remaining:
the large-sparse benchmark (§6.1e) — kept small pending timing validation.

Original proposal (2026-07-08) follows. A faithful port of **PDLP** (Applegate et al., "PDLP: A Practical
First-Order Method for Large-Scale Linear Programming", arXiv:2501.07018; original NeurIPS 2021
arXiv:2106.04756; reference impl `FirstOrderLp.jl`, Apache-2.0). Motivation: our bespoke Mehrotra
interior point has proven numerically fragile (float dense-LAD residual 449 vs 270; sparse needs a
preconditioner we don't have). PDLP is **matrix-free by construction** — only `A·v` / `Aᵀ·v` +
elementwise projections — so it maps 1:1 onto our BSR operator, sidesteps normal equations / Cholesky /
condition-squaring entirely, is robust in `float`, and is deterministic under Burst.

Not a replacement for the dense simplex/IPM (those stay for small dense + exact vertices); PDLP is the
**large/sparse LP engine** and the foundation for an OSQP-style QP port later.

## 1. Problem form (Eq. 1 of the paper)

    minimize    cᵀx
    subject to  ℓ_c ≤ A x ≤ u_c        (two-sided rows; ℓ_c=u_c ⇒ equality, one side = ±∞ ⇒ one-sided)
                ℓ_v ≤ x  ≤ u_v          (variable box; ℓ_v=0,u_v=∞ ⇒ our x≥0)

This subsumes our current `LP.solve` form (`Ax {≤,=,≥} b, x≥0`): map `≤`→`(−∞,b]`, `=`→`[b,b]`,
`≥`→`[b,∞)`; `x≥0`→`[0,∞)`. Facade stays `LPInfo`.

## 2. Core PDHG iteration (Eq. 4)

Saddle form `min_{x∈[ℓ_v,u_v]} max_y  cᵀx + yᵀAx − p(y; −u_c,−ℓ_c)`. One iteration, step sizes
`τ = η/ω` (primal), `σ = η·ω` (dual):

    x⁺ = proj_[ℓ_v,u_v]( x − τ (c − Aᵀ y) )                                   // 1 spMVT
    ỹ  = y + σ ( A(2x⁺ − x) )                                                 // 1 spMV on the extrapolated primal
    y⁺ = ỹ − σ · proj_[ℓ_c,u_c]( ỹ/σ )       (equivalently the two-sided-constraint prox)

`proj_[l,u](v)` is an elementwise clamp. Stability requires `η ‖A‖ < 1` (η is what the adaptive rule
tunes). Both matvecs are BSR `spMV`/`spMVT`; everything else is length-n / length-m vector ops.

## 3. The three practical layers (port faithfully, add incrementally)

### 3.1 Adaptive step size (Alg. 2)
Take a tentative step with `η`. Accept iff
    η ≤ ‖(x⁺−x, y⁺−y)‖²_ω / ( 2 (y⁺−y)ᵀ A (x⁺−x) )
where `‖(a,b)‖²_ω = ω‖a‖² + ‖b‖²/ω`. If violated, shrink `η` and retry (the step is recomputed). Next
tentative:
    η' = min( (1 − (k+1)^−0.3)·η̄ , (1 + (k+1)^−0.6)·η )
with `η̄` the acceptance bound just computed. The `Aᵀy` / `A x⁺` products are reused across the test
(no extra matvec beyond `A(x⁺−x)` which is a byproduct).

### 3.2 Adaptive restart (Alg. in §3) — restart to the running average
Maintain the running average `z̄ = (x̄, ȳ)` over the current inner loop. Using a normalized duality-gap
surrogate `μ(z, z_ref)` (a localized primal-dual gap; the KKT residual norm is the practical
implementation), restart the inner loop — reset `z^{n,0} ← z̄` — when ANY of:
  (i)  `μ(z̄, z^{n,0}) ≤ 0.1 · μ(z^{n,0}, z^{n−1,0})`      (sufficient decay)
  (ii) `μ(z̄, z^{n,0}) ≤ 0.9 · μ(z^{n,0}, z^{n−1,0})` AND the gap started increasing (necessary + stall)
  (iii) inner iteration count `t ≥ 0.5 · k`                (long-loop safeguard)
Restart is what gives PDHG its practical (near-linear) convergence; without it PDHG crawls.

### 3.3 Primal weight `ω` (Alg. 3) — updated only at restarts
    Δx = ‖x^{n,0} − x^{n−1,0}‖₂,  Δy = ‖y^{n,0} − y^{n−1,0}‖₂
    ω  = exp( 0.5·log(Δy/Δx) + 0.5·log(ω_prev) )         (guard Δx,Δy > 0)

### 3.4 Diagonal preconditioning (once, up front)
Ruiz equilibration + Pock–Chambolle diagonal scaling of `A` (row scales `D_r`, col scales `D_c`) to
flatten `‖A‖`. Scale `A,b,c,bounds`, solve, unscale `x = D_c x̂`. Reuses the Ruiz idea already noted in
`research-lp-preconditioners.md`; here it is essential (it sets the geometry PDHG lives in).

## 4. Termination (checked every ~64 iters, on the average iterate)

Relative KKT, all below `ε` (default `1e-8` feasibility, `1e-4` gap — expose as options):
    ‖Ax − proj_[ℓ_c,u_c](Ax)‖ / (1+‖b‖)          primal infeasibility
    ‖c − Aᵀy − r‖ / (1+‖c‖)                        dual infeasibility  (r = reduced cost from box duals)
    |cᵀx − dual_obj| / (1 + |cᵀx| + |dual_obj|)    duality gap
Report `Optimal` / `MaxIterations`. (PDHG also gives approximate infeasibility certificates from the
iterate *differences* — a later add; for now Optimal/MaxIterations like our IPM.)

## 5. Mapping to the codebase

- New operator use: `IfProxyLinearOperator` (dense `fProxyDenseOperator` or BSR `fProxyBSROperator`) for
  `A·x` and `Aᵀ·y`. NO new operator type needed — PDLP only needs the two matvecs we already have.
- New file `TemplateSource/OP/PDLP.fProxy.cs`: `LP.pdlp(A, ℓ_c,u_c, ℓ_v,u_v, c, ref x, out info, opts)`
  dense + a BSR overload; both call one generic `pdlpCore<TOp>(in TOp A, …)`.
- `fProxyPDLPOptions` struct: `maxIter`, `epsOpt`, `epsFeas`, initial `η`,`ω`, restart constants
  (defaulted to the paper's 0.1/0.9/0.5). Job-safe: all scratch `Allocator.Temp`.
- Reuse `LPInfo` (add nothing) — objective + iterations + status.

## 6. Incremental build plan (each stage testable — avoid a big-bang that hides subtle bugs)

- **1a. Vanilla PDHG**, fixed `η = 0.9/‖A‖` (power-iteration estimate of `‖A‖`), no restart/adaptivity,
  equality+box only. Verify it converges (slowly) to the same optimum as `LP.solve` on the small LP
  test vectors. This nails the update equations + projections in isolation.
- **1b. Adaptive restart + running average** (§3.2). Expect a large iteration-count drop. Re-verify.
- **1c. Primal weight (§3.3) + adaptive step size (§3.1).** Re-verify; tune against Wyndor / a few LPs.
- **1d. Ruiz + Pock–Chambolle preconditioning (§3.4).** Re-verify; now benchmark vs sparse IPM.
- **1e. BSR overload + large-sparse benchmark.** This is where PDLP should shine (matrix-free, no
  preconditioner problem).

## 7. Test plan

- Same literature vectors as `LPTests` (Wyndor Glass optimum (2,6) Z=36; the small ≤/=/≥ LPs), asserting
  PDLP agrees with simplex/IPM within `epsOpt`.
- Sparse: agree with `LP.solve(BSR)` / dense on a mixed-sense LP; robustness on an ill-conditioned case
  where our IPM struggled (the m=512 LAD-as-LP that gave float 449 — PDLP should not degrade in float).
- Determinism: identical result across runs (single-thread, fixed reduction order).

## 8. Non-goals / later

- QP: the same PDHG machinery extends to `min ½xᵀQx+cᵀx` (OSQP-style, or PDQP) — a **follow-on** once
  PDLP lands.
- Exact infeasibility/unboundedness certificates (iterate-difference based) — later.
- Presolve (row/col elimination) — orthogonal; can front any solver, separate work item.
