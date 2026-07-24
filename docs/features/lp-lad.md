# Linear programming & LAD (`LP`)

Solves linear programs in canonical primal form

```
minimize    cᵀx
subject to  Aᵢ·x {≤, =, ≥} bᵢ   (per-row sense)
            x ≥ 0
```

and, on the same machinery, exact least-absolute-deviation (L1) regression and quantile regression.
Every entry point is job-safe — scratch is `Allocator.Temp`, disposed before return, so the whole
thing runs inside a `[BurstCompile] IJob`.

## `LP.solve` — general linear programs

```csharp
var A = new floatMxN(2, 2, Allocator.Temp);
A[0, 0] = 1; A[0, 1] = 1;   // x0 + x1 <= 4
A[1, 0] = 1; A[1, 1] = -1;  // x0 - x1 <= 2
var b = new floatN(2, Allocator.Temp); b[0] = 4; b[1] = 2;
var c = new floatN(2, Allocator.Temp); c[0] = -1; c[1] = -1;   // maximize x0+x1 == minimize -x0-x1
var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.LessEqual;

var x = new floatN(2, Allocator.Temp);
LPInfo info = LP.solve(in A, in b, in c, senses, ref x, out double objective);
if (info) Print.Log(info);   // implicit bool -> "reached an optimum?"
```

`x` (length `A.N_Cols`) is overwritten with the optimal vertex; `objective` is `cᵀx`. Variables are
non-negative by construction — model a free variable by splitting it into a `+`/`−` pair yourself,
or use `LP.lad` below, which already does that for L1 regression.

## Backends (`LPMethod`)

- **`RevisedSimplex`** (default) — bounded-variable primal revised simplex over an LU-factored
  basis (FTRAN/BTRAN + product-form-of-the-inverse eta file). Fastest exact backend at every
  benchmarked size on cold solves, and the fastest infeasibility certifier.
- **`DualSimplex`** — bounded-variable dual revised simplex (dual steepest-edge pricing, long-step
  Harris ratio test). The only backend that takes a warm-started basis (see below); pick it
  explicitly for cold solves only when you're already re-solving from a near-dual-feasible state.
- **`InteriorPoint`** — Mehrotra primal-dual predictor-corrector. Scales to larger/denser LPs and
  handles very ill-conditioned vertices; converges to an interior point rounded onto a vertex rather
  than an exact one, and only reports `Optimal`/`MaxIterations` (no infeasibility/unboundedness
  certificate — that needs a homogeneous self-dual embedding).

All three reach the same optimal vertex on a bounded, feasible problem; `maxIter <= 0` picks a
size-based default for every backend.

## Warm-started re-solve

Re-solving the same problem shape after a small perturbation (a tightened bound, a changed RHS) is
much cheaper than a cold solve if the old basis is reused as the starting point — the dual simplex
only needs a handful of pivots to repair feasibility instead of rebuilding a vertex from scratch.
Two escalating overloads of `LP.solve`, both routed through `DualSimplex`:

- **`LP.solve(..., ref LPBasis basis)`** — seeds (and returns) the terminal basis through `basis`.
  Pass `default(LPBasis)` the first time (managed-thread only — it self-allocates
  `Allocator.Persistent`), or a job-safe `new LPBasis(n, m, Allocator.Temp)` seeded on first use.
  Re-solve by passing the same `basis` back in after mutating `A`/`b`/`c`.
- **`LP.solve(..., ref LPBasis basis, ref floatLPCache cache)`** — additionally persists the
  computational form and the basis factorization (LU + eta file) and DSE pricing weights across
  calls via the generated per-dtype `LPCache` struct (`floatLPCache`/`doubleLPCache`), skipping both
  the O(mN) form rebuild and the O(m³) refactorization a warm re-solve otherwise pays even with zero
  or few pivots. `cache.matrixVersion` must be bumped by the caller whenever `A`'s coefficients/
  senses or `c` change (an RHS/bound-only change needs no bump); under
  `ENABLE_UNITY_COLLECTIONS_CHECKS` a missed bump is caught and throws rather than silently solving
  the wrong problem.

```csharp
LPBasis basis = default;
floatLPCache cache = default;   // or new floatLPCache(n, m, Allocator.Persistent) to reuse it
for (int step = 0; step < steps; step++)
{
    UpdateRhs(ref b);   // structure/cost unchanged -> no matrixVersion bump needed
    LP.solve(in A, in b, in c, senses, ref x, out double obj, ref basis, ref cache);
}
basis.Dispose(); cache.Dispose();
```

## `LP.lad` — least absolute deviation (L1 regression)

Minimizes `‖Ax − b‖₁` over a **free** `x` — robust to outliers where ordinary least squares (L2) is
not. Two reformulation-free exact engines, both working directly on the original `m×n` design (no
`2n+2m`-variable LP blow-up):

- **`LP.ladBR`** — Barrodale-Roberts specialized simplex (a Koenker-d'Orey `rqbr` port). Converges to
  an exact vertex — at the optimum, `n` of the `m` residuals are exactly zero. Near-constant,
  few-microsecond latency at small-to-moderate `m`.
- **`LP.ladFN`** — Frisch-Newton primal-dual interior point (Portnoy & Koenker 1997). Each iteration
  is one `n×n` weighted normal solve (pivoted Cholesky); wins once `m` grows large enough that BR's
  per-pivot sweep over `m` rows dominates.

```csharp
LPInfo info = LP.lad(in A, in b, ref x, out double l1Residual);
```

**`LP.lad(in A, in b, ref x, out objective[, maxIter])`** dispatches between `ladBR` and `ladFN` by `A.M_Rows`
(the measured crossover; see Performance below). Call `ladBR`/`ladFN` directly to force one engine.

An explicit-backend overload, **`LP.lad(in A, in b, ref x, out objective, LPMethod method[, maxIter])`**,
reformulates LAD as a general LP (`x = x⁺ − x⁻`) and routes it through any `LPMethod` — exact but slower;
kept mainly as an independent cross-check.

### Quantile regression

Both exact engines take an optional `tau` in `(0, 1)`: `ladBR(in A, in b, tau, ref x, out objective)`
/ `ladFN(in A, in b, tau, ref x, out objective)` fit the conditional τ-quantile of `b` given `A` by
minimizing the check loss `Σᵢ ρτ(bᵢ − Aᵢ·x)`. `tau = 0.5` is median regression, identical to the
tau-less overload; `tau = 0.9` fits the 90th conditional percentile, and so on.

For a fast *approximate* alternative (iteratively-reweighted least squares, no LP at all), see
`Optimize.ladIRLS`.

## Sparse (matrix-free over BSR)

`LP.solve`/`LP.lad` accept `floatBSR`/`doubleBSR` matrices — a matrix-free Mehrotra interior point over
[block-sparse](sparse-bsr.md) constraints. Each normal-equation solve runs through `Krylov.cg` against
a matrix-free operator (Jacobi-preconditioned), so cost does not scale with `N²`. Interior point only
(no simplex), reporting `Optimal`/`MaxIterations`; use dense simplex backends for exact infeasibility/unboundedness certificates.

## Diagnostics

Every entry point returns `LPInfo` by value: `objective` (`cᵀx`, or the L1 residual for `lad`),
`iterations`, and `status : LPStatus` (`Optimal`, `Infeasible`, `Unbounded`, `MaxIterations`).
`LPInfo` has an implicit `bool` conversion (`== Optimal`) and a `.Solved` property, so
`if (LP.solve(...))` reads naturally.

## Performance

`RevisedSimplex` is fastest on cold solves and fastest at infeasibility (1-2 pivots); `DualSimplex` wins on
warm re-solves.

`LP.lad` hybrid default routes on `A.M_Rows` per dtype (`Benchmarks/LPBenchmark.cs` Section 2b):
- **double**: `ladBR` wins through `m=4096` (2.49ms vs 2.71ms), default threshold 4096
- **float**: `ladBR` wins through `m=384`, `ladFN` through `m=1024` (0.47ms vs 0.62ms), default threshold 512
