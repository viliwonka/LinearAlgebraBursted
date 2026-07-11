# Research: preconditioning / conditioning for the LP interior-point linear solves

Status: RESEARCH NOTES (2026-07-08). Motivated by the sparse LP benchmark: `LP.lad(BSR)` at
`m = 8192` spends most of its time with PCG pinned at its 500-iteration cap, because the only
preconditioner today is **diagonal Jacobi** and the normal matrix is badly conditioned near the
optimum. This surveys the options, grounded in the IPM linear-algebra literature, and maps each to
what this library already has.

## 0. The problem, precisely

Every Mehrotra iteration solves the **normal equations**

    M Δy = rhs,   M = Aₛ D Aₛᵀ + δI,   D = Z S⁻¹  (diagonal, changes every iteration).

Two independent sources of ill-conditioning:

1. **Normal-equations squaring.** `M = (Aₛ√D)(Aₛ√D)ᵀ`, so `κ(M) = κ(Aₛ√D)²`. Forming `A D Aᵀ` and
   running CG on it squares the condition number of the underlying operator — the classic "don't solve
   the normal equations" penalty.
2. **IPM boundary blow-up.** As the interior point approaches the optimal vertex, complementarity
   drives `D = Z S⁻¹` to `0` on inactive components and `∞` on active ones. `κ(Aₛ√D) → ∞`
   *independently* of `A`. This is why a preconditioner that is fine early **stalls in the last few
   iterations** — exactly our symptom.

For **LAD** specifically `Aₛ = [A|−A|−I|I]`, so `M = A(d⁺+d⁻)Aᵀ + diag(dᵤ+dᵥ) + δI`. The `diag(·)`
term (from the `±I` blocks) is an always-positive Tikhonov floor — it is *why* diagonal Jacobi is not
hopeless here — but it does not stop the `A(d⁺+d⁻)Aᵀ` part from becoming rank-deficient-ill near the
optimum.

**Note on "dense vs sparse".** The dense backend (`interiorCore`) solves `M` with an *exact* Cholesky
(`CHO`), so "preconditioner" does not apply there — a direct factor is already the best conditioning
you can get for that `M`. The dense analogue of this research is **problem formulation and numerical
stability** (§6). Preconditioners proper live on the sparse/iterative path (§1–§5).

---

## 1. Escape the squaring: LSQR/LSMR on the rectangular operator  ★ best effort/payoff here

The normal-equations solve `Aₛ D Aₛᵀ Δy = rhs` is mathematically the weighted least-squares problem
whose operator is the **rectangular** `√D Aₛᵀ`. Running **LSQR** or **LSMR** on that operator solves
the same system while working with `κ(Aₛ√D)`, **not its square** — the standard fix for the
normal-equations penalty (§0.1).

Why this is the cheapest big win *for this codebase specifically*:

- The library **already ships** matrix-free `Krylov.lsqr` / `Krylov.lsmr` / `Krylov.cgls` over any
  `IfProxyLinearOperator`.
- It **already ships** `fProxyColScaledOperator<TInner>` — a right-diagonal-scaling wrapper. Composing
  `fProxyColScaledOperator<fProxyLadOperator>` with `d = √D` gives exactly the `Aₛ√D` operator LSQR
  needs, with no new kernel.
- Same matrix-free per-iteration cost as the current PCG (one `Apply` + one `ApplyT`), but each
  iteration is worth far more because the effective condition number is square-rooted.

Cost: re-express the IPM inner solve as an LS problem and swap `pcg` → `lsmr` (LSMR is the more robust
of the two on ill-conditioned problems). Keep diagonal Jacobi as the LS column preconditioner. This is
a moderate, self-contained change that reuses three existing pieces and directly attacks source (1),
and partially (2). **Recommended first.**

## 2. Cheap scaling up front: Ruiz equilibration  ★ low effort

Before IPM starts, two-sided **equilibration** of `A` (iteratively rescale rows and columns to equal
ℓ₂/ℓ∞ norms) lowers `κ(A)` once, helping *every* subsequent solve regardless of which Krylov method /
preconditioner is used. **Ruiz scaling** is the standard choice — O(nnz) per sweep, a handful of
sweeps, and it is the default preconditioner in modern QP/LP solvers (PIQP, OSQP) with reported ~20%
speedups. Fits this library trivially: a `BSR.ruizScale` pass returning row/col scale vectors, applied
to `A`, `b`, `c` and unwound on the solution. Complements §1 and §3, competes with neither.

## 3. Block Jacobi  ★ moderate effort, moderate payoff

Invert small **dense diagonal blocks** of `M` instead of just its diagonal. Natural fit for BSR (the
block structure is already there) and strictly stronger than point-Jacobi when variables couple within
a block. Still cheap to build matrix-free (`BR×BR` block inverses). A sensible "one notch up from
diagonal Jacobi" before reaching for the heavy machinery in §4.

## 4. Splitting preconditioner (Oliveira–Sorensen) + hybrid  ★ the literature's real fix, high effort

This is the state-of-the-art answer to **our exact symptom** — PCG stalling in the *last* IPM
iterations. A **splitting preconditioner** picks `m` linearly independent columns of `Aₛ√D` as a basis
`B` (via a rectangular LU) and preconditions with `(BBᵀ)⁻¹`. Key theoretical result (Oliveira &
Sorensen; Velazco/Oliveira successors): with a suitable column ordering the preconditioned condition
number is **bounded by a quantity that depends only on the problem data, not on the IPM iteration** —
i.e. it does *not* blow up at the boundary.

Because it is expensive early (and unnecessary when `M` is still well-conditioned), the practical form
is the **Bocanegra–Campos–Oliveira hybrid**: a controlled/incomplete Cholesky of `M` in the early
iterations, then **switch to the splitting preconditioner for the final ill-conditioned iterations**.
This is the principled cure but it needs a rectangular LU / basis-finding step that is hard to keep
fully matrix-free, so it is a real project, not a tweak.

## 5. Augmented (KKT) system + constraint preconditioner  ★ high effort, different route

Instead of the normal equations, solve the symmetric **indefinite** saddle-point system

    [ -(D⁻¹+δI)   Aₛᵀ ] [ Δx ]   [ · ]
    [   Aₛ         γI  ] [ Δy ] = [ · ]

with **MINRES** (the library has it) or SQMR, preconditioned by a **constraint preconditioner** (Keller–
Gould–Wathen). This avoids squaring `κ` entirely and gives tight eigenvalue clustering → few Krylov
iterations. Modern refinements build the seed constraint preconditioner once and **update it across IPM
iterations by low-rank Schur corrections** (Bellavia–De Simone–di Serafino–Morini), and there is recent
Schur-complement-based work aimed squarely at IPM. Downsides: assembling/factoring the constraint
preconditioner is costly and less obviously matrix-free; this is the most invasive option.

## 6. Dense path (direct solve): formulation & stability, not preconditioning

For `interiorCore` (dense Cholesky) the levers are about *robustness*, since there is no Krylov loop to
precondition:

- **Augmented-system LDLᵀ instead of normal-equations Cholesky.** Factoring the symmetric indefinite
  KKT system with Bunch–Kaufman LDLᵀ avoids squaring `κ(A)` and is markedly more stable when columns of
  `A` are nearly dependent or `D` is extreme. More expensive per factorization; more accurate.
- **Primal-dual regularization + iterative refinement.** We already add `δI`; making it a proper
  primal-dual regularization (Friedlander–Orban) keeps the factor well-defined even for rank-deficient
  `A`, and one or two steps of iterative refinement recover digits lost to conditioning — cheap.
- **Mixed precision.** Factor in the working type, refine in `double`. Relevant to the `float` path.

---

## 7. Free wins that are NOT preconditioners (do these regardless)

The benchmark symptom is partly self-inflicted and fixable without any new preconditioner:

- **Adaptive inner tolerance.** We solve every PCG to `√eps` even in the first outer iterations, where a
  loose direction is fine (Mehrotra tolerates inexact steps). Use `tol_pcg = clamp(0.1·μ/μ₀, 1e-8,
  0.1)` — loose early, tight late. Typically a large cut in total inner iterations.
- **Warm-start** each PCG from the previous outer iteration's `Δy` (IPM steps change slowly). We
  already keep the buffers; just don't zero them.
- **Cap + record.** Keep the inner-iteration cap but log when it is hit, so "stalled" is visible rather
  than silently expensive.

These three alone likely explain a good fraction of the benchmark cost and are nearly free.

---

## 8. Recommendation (prioritized for this library)

1. **§7 adaptive tolerance + warm start** — trivial, attacks the benchmark symptom directly.
2. **§2 Ruiz equilibration** — cheap, global, complements everything.
3. **§1 LSQR/LSMR reformulation** — moderate, reuses `lsmr` + `fProxyColScaledOperator`, removes the
   `κ²` penalty. The highest-value *structural* change.
4. **§3 block Jacobi** — moderate, one notch up from diagonal Jacobi.
5. **§4 hybrid IC→splitting** (and/or §5 constraint-preconditioned MINRES) — the heavy, "proper" fixes
   for boundary ill-conditioning; only worth it if §1–§3 prove insufficient at target scale.
6. **§6** applies only if we ever want the dense path more robust on near-degenerate problems.

Non-goals unchanged: a sparse direct Cholesky, sparse revised simplex.

## Sources

- [Preconditioning indefinite systems in IPM for large-scale LP (Bergamaschi, Gondzio, Zilli)](https://www.researchgate.net/publication/228622276_Preconditioning_indefinite_systems_in_interior_point_methods_for_large_scale_linear_optimisation)
- [A new approach for finding a basis for the splitting preconditioner (Comput. Optim. Appl.)](https://link.springer.com/article/10.1007/s10589-016-9887-0)
- [Computing a hybrid preconditioner (controlled Cholesky + splitting) for IPM-CG](https://www.researchgate.net/publication/314866499_Computing_a_hybrid_preconditioner_approach_to_solve_the_linear_systems_arising_from_interior_point_methods_for_linear_programming_using_the_conjugate_gradient_method)
- [General-purpose preconditioning for regularized interior point methods (Comput. Optim. Appl. 2022)](https://link.springer.com/article/10.1007/s10589-022-00424-5)
- [Iterative Solution of Augmented Systems Arising in Interior Methods (SIAM J. Optim.)](https://epubs.siam.org/doi/10.1137/060650210)
- [Updating constraint preconditioners for KKT systems via low-rank corrections (arXiv:1312.0047)](https://arxiv.org/abs/1312.0047)
- [Constraint-Preconditioned Krylov Solvers for Regularized Saddle-Point Systems (arXiv:1910.02552)](https://arxiv.org/pdf/1910.02552)
- [Efficient Preconditioners for IPM via a new Schur Complement-Based Strategy (arXiv:2104.12916)](https://arxiv.org/pdf/2104.12916)
- [Influence of matrix reordering on iterative methods for IPM linear systems (Math. Meth. Oper. Res.)](https://link.springer.com/article/10.1007/s00186-017-0571-7)
- [PIQP: A Proximal Interior-Point QP Solver — Ruiz equilibration default preconditioner (arXiv:2304.00290)](https://arxiv.org/pdf/2304.00290)
- [Epperly, "Don't Solve the Normal Equations" — the κ² penalty and LSQR/CGNE alternative](https://www.ethanepperly.com/index.php/2022/07/26/dont-solve-the-normal-equations/)
