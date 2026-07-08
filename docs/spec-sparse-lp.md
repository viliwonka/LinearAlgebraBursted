# Spec: Sparse-system linear programs (matrix-free interior point)

Status: PROPOSED (2026-07-08). Depends on: the dense `LP` facade (`LP.solve` / `LP.lad`,
`LP.InteriorPoint.fProxy.cs`), the sparse BSR stack (`fProxyBSR`, `BSR.spMV`/`spMVT`,
`fProxyBSROperator`), the generic Krylov solvers (`Krylov.pcg<TOp,TPre>`), and the operator/
preconditioner interfaces (`IfProxyLinearOperator`, `IfProxyPreconditioner`).

## 1. Goal

Solve linear programs and L1 / least-absolute-deviation problems whose constraint matrix `A` is
**large and sparse** (BSR), at sizes where the dense `LP` backends are not viable (a dense `A` of
10⁴×10⁴ is hundreds of MB and its per-iteration `O(m³)` factor is minutes). Reuse the library's
existing strength — **matrix-free preconditioned CG over BSR** — rather than adding a sparse direct
factorization.

Deliverables:
- Sparse `LP.solve` (interior point) and sparse `LP.lad` taking a `fProxyBSR` (or any
  `IfProxyLinearOperator`) constraint matrix, returning the existing `LPInfo`.
- The reusable operator/preconditioner pieces that make the interior-point normal solve matrix-free.

## 2. Why interior point (not simplex)

The dense two-phase simplex (`LP.fProxy.cs`) is a **dense tableau** algorithm — it fills in during
pivoting and cannot exploit sparsity without becoming a different algorithm (revised simplex + sparse
LU basis factorization + Forrest–Tomlin updates + sparse pricing). That is a large, separate build and
is explicitly **out of scope** here.

The Mehrotra interior point (`LP.InteriorPoint.fProxy.cs`) is the natural fit: its only heavy step per
iteration is one **SPD solve** of the normal equations

    M Δy = rhs,   M = Aₛ D Aₛᵀ,   D = Z S⁻¹  (diagonal, changes each iteration),

and everything else is cheap sparse mat-vecs `Aₛ·v`, `Aₛᵀ·v`. Because `M` is SPD, we solve it with the
existing **`Krylov.pcg`** — never forming `M` — which turns the whole interior point matrix-free.

## 3. New pieces

### 3.1 Normal-equations operator (symmetric, matrix-free) — `fProxyNormalOperator<TInner>`

A readonly `IfProxyLinearOperator` that presents `M = Aₛ D Aₛᵀ` (with `Aₛ.Rows == m`) WITHOUT forming
it, composing over any inner `Aₛ`:

    Apply(v):                          // v length m  ->  y length m
        w = Aₛᵀ v                      // Inner.ApplyT, w length Cols(Aₛ)=nvar  (owned scratch)
        w ∘= D                         // elementwise, D length nvar (the IPM diagonal, updated per iter)
        y = Aₛ w                       // Inner.Apply

`Rows == Cols == Aₛ.Rows == m`. Symmetric and (for `D > 0`) SPD, so it is a valid `TOp` for `pcg`.
`ApplyT == Apply` (symmetric). `D` is a mutable `fProxyN` the interior point rewrites each outer
iteration (the operator holds the handle, not a copy). Holds one owned scratch vector of length
`nvar`. Mirrors `fProxyColScaledOperator<TInner>` exactly in shape (composes over an arbitrary inner
operator, owns a scratch buffer, readonly struct). This is the single most reusable new type — it is
also a general "AᵀDA normal operator" usable outside LP.

### 3.2 Standard-form constraint operator

The interior point works on standard form `min cᵀz s.t. Aₛ z = b, z ≥ 0`. Two shapes are needed:

- **Equality already** (this is what LAD produces — see §5): `Aₛ` is the caller's operator directly.
- **Slack-augmented** (general LP with `≤`/`≥`): `Aₛ = [A | S]` where `S` is one `±1` column per
  inequality. Provide `fProxySlackAugmentedOperator<TInner>`: `Apply([z_struct; z_slack]) = A z_struct
  ± z_slack`; `ApplyT(v) = [Aᵀ v ; (±v at the inequality rows)]`. Matrix-free — never materializes
  `[A | ±I]` (whose mixed 1×1 identity vs BR×BC blocks would be awkward in a single BSR).

For **LAD** specifically, `Aₛ = [A | −A | −I | I]` (§5). Provide `fProxyLadOperator<TInner>`:
`Apply([xp; xn; u; v]) = A xp − A xn − u + v`; `ApplyT(r) = [Aᵀr; −Aᵀr; −r; r]`. Two `Inner.Apply`/
`ApplyT` calls plus copies — fully matrix-free over a sparse `A`.

### 3.3 Preconditioner — `fProxyNormalJacobi` (diagonal Jacobi on `M`)

PCG on the normal equations needs preconditioning — `M` grows ill-conditioned as the interior point
approaches the boundary (`D` entries → 0/∞). The cheap, matrix-free-computable choice is the
**diagonal Jacobi** preconditioner `M⁻¹ ≈ diag(M)⁻¹`, with

    diag(M)_i = Σ_k Aₛ[i,k]² D_k.

`diag(M)` is computed once per interior-point iteration directly from the sparse blocks (one `O(nnz)`
pass: for each stored block accumulate `A[i,k]²·D_k` into row `i`; slack/identity columns contribute
`D_k` to their single row). Add the augmented columns' contributions (`±1`² = 1 → `+D_slack`). Provide
`BSR.normalDiagonal(in A, in D, ref diag)` (and augmented variants) as the builder;
`fProxyNormalJacobi` stores `1/diag` and `Apply(r,z): z = r ∘ invdiag`.

Phase B may add a **block-Jacobi** (small dense diagonal blocks of `M`) or a **partial-Cholesky /
Schur** preconditioner for problems where diagonal Jacobi stalls; start with diagonal Jacobi.

## 4. Algorithm (sparse interior point)

Same Mehrotra predictor-corrector as the dense `interiorCore`, with ONE change: replace the
`CHO.decomp` + two `CHO.decompSolve` calls with two **`Krylov.pcg`** solves against
`fProxyNormalOperator` + `fProxyNormalJacobi`:

1. Build/refresh `D_k = z_k/s_k`; refresh the Jacobi diagonal (`BSR.normalDiagonal`).
2. Predictor rhs `= b − Aₛ(D rc)`; **pcg** solve `M Δy_aff = rhs` (warm-start from previous `Δy`).
3. Affine step lengths, `σ = (μ_aff/μ)³` (identical to dense).
4. Corrector rhs `= b − Aₛ(D rc) − Aₛ g`; **pcg** solve `M Δy = rhs` (warm-start from `Δy_aff`).
5. Recover `Δs = −rc − Aₛᵀ Δy`, `Δz = −D Δs − z + g`; ratio-test step lengths; update `z,y,s`.
6. Converge on relative primal/dual residual + duality gap `μ` (same tests as dense).

**Inexact-IPM controls** (the delicate part — call out in the doc comment):
- Adaptive inner tolerance: loose early, tight late, e.g. `tol_pcg = clamp(0.1·μ/μ₀, 1e-8, 0.1)` —
  no point solving the normal system to 1e-10 while still far from optimum.
- Warm-start each pcg from the previous `Δy` (interior-point steps change slowly).
- Cap inner pcg iterations (e.g. `min(m, 200)`); on non-convergence, take the inexact step (Mehrotra
  tolerates inexact directions) but record it.
- Optional **primal-dual regularization** (add `δ·I` to `M`, small): keeps `M` PD/conditioned near the
  boundary at the cost of a slightly perturbed step. Cheap lever; include as a parameter with a small
  default.

Job-safe: all scratch from `Allocator.Temp`, disposed before return (same discipline as the dense
cores). The operators own their own Temp scratch.

## 5. Sparse LAD (comes almost for free)

LAD reformulates to an **all-equality** standard-form LP (`LP.lad`): variables `[x⁺|x⁻|u|v]`,
constraints `A(x⁺−x⁻) − u + v = b`. With `A` sparse, `Aₛ = [A|−A|−I|I]` is sparse and expressed by
`fProxyLadOperator<TInner>` (§3.2) — no slack machinery needed. So sparse `LP.lad` is: build the LAD
operator over the caller's BSR `A`, run the §4 sparse interior point, recover `x = x⁺ − x⁻`. The
objective is `‖A x − b‖₁` exactly as in the dense path.

Note: the LAD normal matrix `M = Aₛ D Aₛᵀ` is `m×m` and, because of the `−I/I` blocks, has a diagonal
that is always populated — diagonal Jacobi is a genuinely useful preconditioner here.

## 6. API surface

Add sparse overloads to the `LP` facade (interior point is implied — simplex is n/a for sparse):

    // general sparse LP:  min cᵀx  s.t.  A x {≤,=,≥} b,  x ≥ 0
    LPInfo LP.solve(in fProxyBSR A, in fProxyN b, in fProxyN c,
                    in NativeArray<ConstraintSense> senses, ref fProxyN x, out double objective,
                    fProxySparseLPOptions opts = default);

    // sparse least absolute deviation:  min ‖A x − b‖₁
    LPInfo LP.lad(in fProxyBSR A, in fProxyN b, ref fProxyN x, out double objective,
                  fProxySparseLPOptions opts = default);

`fProxySparseLPOptions` (a small struct, defaults sensible): outer `maxIter`, inner pcg cap + base
tolerance, regularization `δ`. Returns the existing `LPInfo` (interior point already reports
Optimal/MaxIterations only — no exact infeasible/unbounded certificate; document, same caveat as dense
interior point).

Optional generic entry (`in TOp A` over `IfProxyLinearOperator`) if a caller has a non-BSR operator —
but the Jacobi diagonal builder needs the concrete sparse entries, so the BSR overload is primary;
a generic overload would require the caller to supply the diagonal (or fall back to identity
preconditioning).

## 7. Test plan

1. **Sparse == dense agreement**: build a small dense LP, materialize its `A` as BSR, solve both the
   dense interior point and the sparse interior point; assert the objectives agree (loose interior
   tolerance) and KKT residuals are small. Do the same for LAD.
2. **Literature vectors over BSR**: Wyndor Glass (§ dense tests) and stack-loss LAD, with `A` as BSR —
   assert the same published optima. (Reuses the vectors already in `LPTests`.)
3. **Genuinely sparse LP**: a min-cost-flow / network LP (node-arc incidence `A`, very sparse) with a
   known optimum, or a large sparse random feasible LP checked by KKT-residual optimality.
4. **Sparse LAD robustness**: sparse design + gross outliers → recovers the majority fit (mirror the
   dense outlier test).
5. **Scale**: a sparse LP/LAD at `N` in the thousands that the dense path cannot hold — assert it
   solves (status + KKT), and benchmark it in `LargeSparseBenchmark` or `LPBenchmark`.

Acceptance: objective matches the dense reference within interior-point tolerance; primal/dual/gap
KKT residuals below tolerance on convergence; runs inside a `[BurstCompile]` job; templated
float/double; suite green.

## 8. Phasing

- **Phase A** (this spec's core): `fProxyNormalOperator<TInner>`, `fProxyNormalJacobi` +
  `BSR.normalDiagonal`, `fProxyLadOperator<TInner>`, the sparse interior-point loop reusing
  `Krylov.pcg`, sparse `LP.lad` (equality — no slack machinery), tests 1/2/4. This already ships
  **sparse LAD**, the highest-value piece, because LAD is equality-constrained.
- **Phase B**: `fProxySlackAugmentedOperator<TInner>` → general sparse `LP.solve` with `≤/≥`; block-
  Jacobi / Schur preconditioner; primal-dual regularization tuning; tests 3/5 + benchmark.

## 9. Non-goals

Sparse revised simplex (different algorithm, separate large effort). A sparse direct Cholesky
(valuable on its own — would be its own feature, and a better preconditioner/solver for the normal
equations when Jacobi-PCG stalls). Exact infeasibility/unboundedness certificates (interior point does
not provide them; use the dense simplex for small problems needing them).
