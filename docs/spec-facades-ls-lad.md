# Spec: problem-domain facades `LS` (least squares) and `LAD` (least absolute deviation)

DESIGN DOCUMENT for user review — nothing here is implemented yet.

Goal: the library organizes solvers by METHOD (`QR`, `QRCP`, `LQRP`, `SVD`, `CHO`, `Krylov`, …) but
users arrive with PROBLEMS ("fit this line", "solve this overdetermined system", "ignore these
outliers"). `LP` is the proven precedent: a problem-first static class (`LP.solve` with an
`LPMethod` enum, `LP.lad` with measured hybrid routing) over method-level backends. This spec
proposes the same shape for the two remaining big problem domains:

- **`LS`** — minimize ‖Ax − b‖₂. Today scattered across `QR.solveInPlace`/`decompSolve`,
  `QRCP.solveInPlace`/`minNormSolveInPlace`, `LQRP.solveInPlace`/`minNormSolveInPlace`,
  `SVD.pinvSolve`, and `Krylov.cgls/lsqr/lsmr`.
- **`LAD`** — minimize ‖Ax − b‖₁. Today living awkwardly on `LP` (`lad` hybrid, `ladBR`, `ladFN`,
  tau overloads, sparse `lad(BSR)`) plus `Optimize.ladIRLS`.

Total least squares (`tls`) and the future `Optimize.irls<TWeight>` generic also get their surfaced
homes here.

---

## OPEN QUESTIONS FOR USER

Each needs a decision before implementation; one-line recommendation attached.

1. **Approve the two facades at all?** — RECOMMEND yes: `LS` + `LAD` static classes in namespace
   `LinearAlgebra`, exactly the `LP` pattern.
2. **LAD migration: move or forward?** `LP.lad`/`ladBR`/`ladFN`/`lad(BSR)`/`Optimize.ladIRLS` —
   RECOMMEND MOVE (delete the old names, no `[Obsolete]` forwarders): pre-release, breaking is
   cheap now and forwarders are permanent API debt. (Presented explicitly because the M_Rows/N_Cols
   rename was cancelled before — this is a smaller, additive-feeling rename, but it is your call.)
3. **LAD method names**: `fit` / `fitBR` / `fitFN` / `fitIRLS` / `fitLP` / `quantile` (PCA
   `fitCov`/`fitSvd` precedent: `fit` verb + route suffix) — RECOMMEND as listed; alternative is
   the terser `LAD.br`/`LAD.fn` (rejected: reads as noun soup, no verb).
4. **LS.solve default method** — RECOMMEND `LSMethod.QR` (fastest robust dense route, the
   universal default in LAPACK/NumPy `lstsq`-adjacent APIs; `Normal` is faster but κ²-fragile).
5. **Include a new `LSMethod.Normal` route** (AᵀA + `CHO`, the one genuinely new numeric path in
   this spec, ~30 lines from existing kernels) — RECOMMEND yes: it is the fastest m≫n route and
   the facade is where users would look for it.
6. **Facade semantics: inputs preserved (internal Temp copy) or destructive?** — RECOMMEND
   preserved (`in A, in b`), matching `LP.solve`: facades are the convenience tier, the
   method-level `*InPlace` grid remains the zero-copy tier. Cost: one m×n Temp copy per call.
7. **Iterative LS (cgls/lsqr/lsmr) through the facade?** — RECOMMEND no: they stay Krylov-only
   (operator/workspace/damp/preconditioner ladder does not fit a one-shot facade; `LstsqInfo` ≠
   `RankInfo`); `LS` docs point sparse/huge users at `Krylov` explicitly.
8. **`LS.minNorm` routing** — RECOMMEND shape-auto COD (QRCP for m ≥ n, LQRP for m < n), no method
   enum on it; `LSMethod.SVD` users call `LS.solve(..., LSMethod.SVD)` (pinv IS min-norm) or
   `SVD.pinvSolve` directly.
9. **Info-struct reuse**: `LS` returns `RankInfo`, `LAD` keeps returning `LPInfo` — RECOMMEND yes
   to both, no new structs (per the solver-diag policy: reuse OK, don't force; LAD genuinely IS an
   LP, its statuses are LP statuses).
10. **TLS scope**: `LS.tls` (regression, augmented-[A|b] SVD) in this round; orthogonal
    plane/subspace FIT (centroid + smallest PC) deferred to the planned small-dim fit-helper
    feature — RECOMMEND yes: plane fitting is a geometry surface (points in, plane out), not an
    Ax≈b surface, and its natural home is the future `float3`-facing helpers.
11. **Multi-RHS overloads** (`LS.solve(A, B, X)` etc.) — RECOMMEND include (all backends already
    have matrix-RHS paths; the facade is thin forwarding).
12. **Keep `LP.lad`'s hybrid crossover data** — the measured per-dtype `/*+choose[512|4096]*/`
    threshold and its benchmark-provenance comment move VERBATIM into `LAD.fit` — RECOMMEND yes
    (no re-measurement; the engines don't change).

---

## Survey: what exists today (the scattered surfaces)

| Problem | Surface today | Return | Notes |
|---|---|---|---|
| L2, full-rank, tall | `QR.solveInPlace` / `decomp`+`decompSolve` | `DirectSolveInfo` | destructive one-shot; multi-RHS; caches |
| L2, rank-revealing basic solution | `QRCP.solveInPlace(..., relativeTolerance)` | `RankInfo` | destructive |
| L2, min-norm (x = A⁺b), tall | `QRCP.minNormSolveInPlace` (COD/xGELSY) | `RankInfo` | destructive; multi-RHS |
| L2, min-norm, wide | `LQRP.minNormSolveInPlace` | `RankInfo` | destructive; multi-RHS |
| L2, most robust | `SVD.pinvSolve(ref A, ...)` | `RankInfo` | destroys A; tolerance + sweep ladder |
| L2, sparse/matrix-free | `Krylov.cgls/lsqr/lsmr` (+`damp`, +`Jacobi`, `<TOp>`, BSR) | `LstsqInfo` | big deliberate overload ladder |
| L1 hybrid (default) | `LP.lad` (BR/FN by measured per-dtype size crossover) | `LPInfo` | `/*+choose[512\|4096]*/` dispatch |
| L1 exact engines | `LP.ladBR`, `LP.ladFN` (+ `double tau` overloads each) | `LPInfo` | Barrodale-Roberts / Frisch-Newton |
| L1 via LP reformulation | `LP.lad(..., LPMethod)` | `LPInfo` | split-variable; slow; cross-check value |
| L1 sparse | `LP.lad(in fProxyBSR, ...)` | `LPInfo` | matrix-free interior point |
| L1 approximate | `Optimize.ladIRLS(A, b, ref x[, delta, xTol, maxIter])` | `LPInfo` | IRLS, the future `irls<TWeight>` skeleton |

No existing type is named `LS`, `LAD`, `LSMethod`, or `LADInfo` — no conflicts (checked 2026-07-09).

---

## 1. `LS` facade

### Class and semantics

`public static partial class LS`, namespace `LinearAlgebra`, template
`TemplateSource/OP/LS.fProxy.cs` (bare name → merged partial across dtypes; safe: every method has
a precision-bearing `fProxyMxN`/`fProxyN` param). Job-safe like `LP`: scratch from
`Allocator.Temp`, disposed on all paths, runs inside `[BurstCompile]` jobs with no arena.

**Facade contract (differs from the method-level grid on purpose): inputs are PRESERVED.**
`in A, in b`; the facade copies A (and b where the backend consumes it) into Temp scratch and
calls the destructive backend. This is `LP.solve`'s exact semantics and the reason the facade can
be a clean one-liner at call sites. The doc comment on every `LS` method points at the
corresponding `*InPlace` method for the zero-copy path. (Open question 6.)

### `LS.solve` — the main entry

```csharp
/// minimize ‖Ax − b‖₂, m ≥ n (square or tall). x overwritten (length n).
public static RankInfo solve(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                             LSMethod method = LSMethod.QR);
public static RankInfo solve(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                             LSMethod method, fProxy relativeTolerance);   // CS1750 forwarder pair
// multi-RHS (open question 11):
public static RankInfo solve(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X,
                             LSMethod method = LSMethod.QR);
```

Wide input (m < n) throws `ArgumentException` ("LS.solve: A must be square or tall; use
LS.minNorm for wide/rank-deficient systems") — same guard `QR.solveInPlace` has today.

Return type `RankInfo` for every route (open question 9): QRCP/SVD fill it natively; the QR and
Normal routes wrap their `DirectSolveInfo` status and report `rank = N_Cols` (documented as
"assumed, not revealed — these routes are not rank-revealing; a rank-deficient A gives Success
with a garbage x, use QRCP/SVD if rank is in doubt").

### `LSMethod` enum

Non-templated singular file `TemplateSource/OP/LS.Info.cs` (`//singularFile//`, CS0102 — same
reasoning and file shape as `LP.Info.cs`).

| Member | Backend | Character |
|---|---|---|
| `QR = 0` | `QR.solveInPlace` on the copy | **Default.** Fast, backward-stable, full-rank tall/square |
| `QRCP = 1` | `QRCP.solveInPlace` | Rank-revealing basic solution; honest `RankInfo.rank`; `relativeTolerance` honored |
| `SVD = 2` | `SVD.pinvSolve` | Most robust; min-norm x = A⁺b; slowest; tolerance honored |
| `Normal = 3` | AᵀA via `matMatDotTransA` + `CHO.solveInPlace` | NEW ROUTE (open question 5). Fastest at m ≫ n; condition number squared — documented loudly; `NotPositiveDefinite` status = "numerically rank-deficient, use QRCP/SVD" |

`relativeTolerance` default sentinel `(fProxy)(-1)` = backend's own default, matching QRCP/SVD
house style; ignored (documented) by `QR`/`Normal`.

### `LS.minNorm` — rank-deficient / wide / min-norm entry

```csharp
/// x = A⁺b (the minimum-L2-norm least-squares solution), any shape, any rank.
public static RankInfo minNorm(in fProxyMxN A, in fProxyN b, ref fProxyN x);
public static RankInfo minNorm(in fProxyMxN A, in fProxyN b, ref fProxyN x, fProxy relativeTolerance);
public static RankInfo minNorm(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X /*, tol */);
```

Routing (open question 8): shape-auto COD — `QRCP.minNormSolveInPlace` when m ≥ n,
`LQRP.minNormSolveInPlace` when m < n — on the internal copy. No method enum: COD is the right
default at every shape, and the SVD alternative is already reachable as
`LS.solve(..., LSMethod.SVD)` (tall) or `SVD.pinvSolve` (anything). Alternative considered:
`minNorm(..., LSMethod)` accepting only `{QRCP, SVD}` and throwing on the rest — rejected as an
enum that lies about its domain.

### `LS.tls` — total least squares

Errors-in-variables regression (all coordinates noisy, not just b): minimize ‖[E|r]‖_F subject to
(A+E)x = b+r. Classic augmented-matrix SVD solution (Golub & Van Loan §6.3; fetch before coding,
per house rule): thin-SVD of C = [A | b] (m×(n+1)), take the right singular vector v of the
smallest singular value; if |v[n]| > tol, x = −v[0..n)/v[n].

```csharp
public static RankInfo tls(in fProxyMxN A, in fProxyN b, ref fProxyN x);
public static RankInfo tls(in fProxyMxN A, in fProxyN b, ref fProxyN x, out double sigma);
                       // sigma = σ_{n+1}(C): the orthogonal residual, the natural TLS diagnostic
```

Nongeneric-TLS failure (v[n] ≈ 0, i.e. b nearly ⟂ range interaction) returns
`DirectSolveStatus.Singular` — no new status needed. Built on `SVD.thin` of the Temp-assembled
augmented matrix; no PCA/Arena dependency (facades must stay arena-free/job-safe).

The **orthogonal plane/subspace fit** (centroid + smallest principal component, one `PCA.fitSvd`
call) is deliberately NOT in `LS` (open question 10): it takes a point cloud and returns geometry
(centroid + normal), needs the Arena (PCA does), and belongs with the approved small-dim
`float2/3/4` fit-helper feature (regressions TODO item 4). `LS.tls`'s doc comment cross-references
it once it exists.

### Iterative solvers: Krylov-only (open question 7)

`cgls`/`lsqr`/`lsmr` do NOT get `LSMethod` members. Reasons: (a) their value is the operator/BSR/
workspace/damp/preconditioner ladder, none of which fits a one-shot dense facade; (b) they return
`LstsqInfo` (iteration counts, rnorm/arnorm) which would force a lossy squeeze into `RankInfo`;
(c) the dense convenience overloads (`Krylov.cgls(A, b, ref x)`) already ARE the facade for that
world. `LS`'s class doc gets a routing paragraph: "sparse, matrix-free, or very large → `Krylov`
(`cgls`/`lsqr`/`lsmr`); L1/robust → `LAD`."

---

## 2. `LAD` facade

### Class and surface

`public static partial class LAD`, namespace `LinearAlgebra`. All of this is a MOVE of existing,
tested code — signatures, doc comments, guards, and the measured hybrid dispatch carry over
unchanged except for the class/method renames:

| New name | Today | Notes |
|---|---|---|
| `LAD.fit(in A, in b, ref x, out objective, maxIter = 0)` | `LP.lad` (hybrid) | the `/*+choose[512\|4096]*/` BR/FN crossover + its provenance comment move verbatim |
| `LAD.fitBR(...)` (+ `tau` overload) | `LP.ladBR` | Barrodale-Roberts explicit engine |
| `LAD.fitFN(...)` (+ `tau` overload) | `LP.ladFN` | Frisch-Newton explicit engine |
| `LAD.quantile(in A, in b, double tau, ref x, out objective, maxIter = 0)` | — (thin new hybrid) | routes to `fitBR`/`fitFN` tau overloads by the SAME size crossover (measured at τ=0.5; documented as such) |
| `LAD.fitIRLS(in A, in b, ref x[, delta, xTol, maxIter])` | `Optimize.ladIRLS` | approximate; fast; doc keeps the "approximate, not exact" warning |
| `LAD.fitLP(in A, in b, ref x, out objective, LPMethod method, maxIter = 0)` | `LP.lad(..., LPMethod)` | split-variable reformulation; retained as independent exact cross-check |
| `LAD.fit(in fProxyBSR A, ...)` | `LP.lad(BSR)` | sparse matrix-free interior point |

Naming rationale (open question 3): the PCA precedent is exactly this shape — model-fitting
methods take the `fit` verb plus a route suffix (`fitCov`/`fitSvd` ↔ `fitBR`/`fitFN`/`fitIRLS`/
`fitLP`). `quantile` is its own verb-adjacent name because τ≠0.5 is a different estimator
(quantile regression), not a routing variant — `LAD.quantile(A, b, 0.9, ...)` reads as the
literature does. Alternative `fit(A, b, tau, ...)` overload rejected: an overload that silently
changes the statistical model on an extra double is a trap.

Return type stays `LPInfo` everywhere (open question 9): LAD is an LP, the statuses
(`Optimal`/`MaxIterations`; FN/BR never `Infeasible`/`Unbounded` on finite data) and the
implicit-bool idiom carry over. `LPInfo`'s doc comments that say "the L1 residual for
`LP.lad`" get updated to cite `LAD.fit`.

### Relation to the future `Optimize.irls<TWeight>` (approved, regressions TODO item 1)

Division of labor, decided now so neither feature blocks the other:

- **`Optimize` keeps the ENGINE**: the generic `Optimize.irls<TWeight>(A, b, ref x, ...) where
  TWeight : struct, IfProxyIrlsWeight` struct-functor loop (Huber/Tukey/Lp/L1 weight structs),
  built later from today's `ladIRLS` body. It is an optimization primitive, method-tier, like
  `bisection`/`newtonRoot`.
- **`LAD` keeps the PROBLEM surface**: `LAD.fitIRLS` is the L1 instance. In THIS round it owns the
  moved `ladIRLS` body (private core in `LAD`); when `irls<TWeight>` ships, `fitIRLS` becomes a
  thin forwarder `Optimize.irls(A, b, ref x, new L1Weight(delta), ...)` and the body moves to
  `Optimize` — zero public-surface change at that point.
- Future robust fits (Huber regression etc.) surface as their own problem entries when wanted
  (e.g. a later `Robust.fitHuber` or `LAD`-sibling — out of scope here), all over the one engine.

### What stays on `LP`

`LP.solve` (dense + BSR), `LPMethod`, `LPStatus`, `LPInfo`, `ConstraintSense`, the four LP
backends, revised-simplex kernel constants. `LP` becomes a pure linear-programming facade — its
class doc drops the "L1 regression is the flagship application" paragraph in favor of one line:
"for least-absolute-deviation regression see `LAD` (which reduces to an LP internally)."

---

## 3. Migration policy (open question 2)

RECOMMENDED: **hard move, no forwarders.**

- Delete `LP.lad` (both overloads), `LP.ladBR`, `LP.ladFN` (+tau), `LP.lad(BSR)`,
  `Optimize.ladIRLS` (both overloads) — bodies move to `LAD.*` per the table above.
- Pre-release (public-release goal): breaking is cheap exactly once, and a `LP.lad` forwarder
  kept "temporarily" ships in v1.0 and is permanent. The library has done this repeatedly
  (Solvers split, _OP purge) without forwarders.
- Internal cores (`ladBarrodaleRobertsCore`, `ladFrischNewtonCore`, the IRLS loop, the sparse LAD
  operator plumbing) move file-for-file; `BR_CAND_SORT_THRESHOLD` moves from `LP.Info.cs`'s
  non-templated `partial class LP` block into the new `LAD.Info.cs` equivalent.

ALTERNATIVE (if the user prefers continuity): keep `LP.lad*`/`Optimize.ladIRLS` as one-line
forwarders marked `[Obsolete("use LAD.fit / LAD.fitIRLS")]` for one release. Costs: 10 extra
public methods frozen into the v1.0 API surface, doc/test duplication, and the README/examples
still have to move. Not recommended.

Either way the call-site sweep is mechanical: `grep -r "LP\.lad\|ladIRLS"` over TemplateSource,
TemplateSourceTests, TemplateSourceBenchmarks, hand-written Benchmarks, and docs.

---

## 4. Naming summary (per docs/dev/naming-style-guide.md)

- **Classes**: `LS`, `LAD` — all-caps literature initialisms, same family as `LP`/`SVD`/`QR`.
  Both merge-safe (every public method carries a concrete `fProxy*` param; no arg-less factories,
  no bare-generic-only methods).
- **Enum**: `LSMethod` (`QR`, `QRCP`, `SVD`, `Normal`) — mirrors `LPMethod`; lives in singular
  `LS.Info.cs`.
- **Methods**: `solve`, `minNorm`, `tls` on `LS`; `fit`, `fitBR`, `fitFN`, `fitIRLS`, `fitLP`,
  `quantile` on `LAD`. camelCase, no class-name echo, `fit`-verb for model-fitting per PCA/KMeans
  precedent. `minNorm` matches the established `minNormSolveInPlace`/`minNormDecompSolve` token.
- **No new info structs, no new statuses**: `RankInfo` (+`DirectSolveStatus`) for `LS`, `LPInfo`
  (+`LPStatus`) for `LAD`.
- **Conflicts**: none — `LS`, `LAD`, `LSMethod`, `LADInfo` are all unused today. `LS.solve` vs
  `LP.solve` coexist fine (different classes). Exception messages: static literals,
  `"LS.solve: ..."` prefix style.
- CS1750: `relativeTolerance` (fProxy-typed) defaults via forwarding overloads, never `= -1` in a
  template signature.

---

## 5. File layout, tests, benchmarks

### New / moved template files

| File | Content |
|---|---|
| `TemplateSource/OP/LS.fProxy.cs` | NEW — `LS.solve` (all routes incl. Normal), `LS.minNorm`, `LS.tls`, multi-RHS |
| `TemplateSource/OP/LS.Info.cs` | NEW, `//singularFile//` — `LSMethod` |
| `TemplateSource/OP/LAD.fProxy.cs` | NEW — `fit` hybrid (moved dispatch), `quantile`, `fitLP` (moved reformulation), `fitIRLS` (moved IRLS body) |
| `TemplateSource/OP/LAD.BarrodaleRoberts.fProxy.cs` | RENAMED from `LP.BarrodaleRoberts.fProxy.cs`; class token `LP` → `LAD`; `ladBR` → `fitBR` |
| `TemplateSource/OP/LAD.FrischNewton.fProxy.cs` | RENAMED from `LP.FrischNewton.fProxy.cs`; `ladFN` → `fitFN` |
| `TemplateSource/OP/LAD.Sparse.fProxy.cs` | SPLIT out of `LP.Sparse.fProxy.cs` (the `lad(BSR)` half); `LP.Sparse.fProxy.cs` keeps `solve(BSR)` |
| `TemplateSource/OP/LAD.Info.cs` | NEW, `//singularFile//` — non-templated `partial class LAD` consts (`BR_CAND_SORT_THRESHOLD` moves here) |
| `TemplateSource/OP/LP.fProxy.cs`, `LP.Info.cs`, `Optimize.fProxy.cs` | EDITED — lad/ladIRLS surfaces removed, class docs updated |

Codegen note: file renames/moves of generated types = the headless bootstrap path
(`Tools/CodegenBootstrap`, regen.ps1) — known deadlock trap if done through Unity.

### Tests

- `LADTests.fProxy.cs` NEW — receives the moved LAD tests from `LPTests.fProxy.cs` (Stackloss
  literature vector, BR/FN/hybrid/LP-reformulation cross-checks, tau/quantile property test,
  exact-fit degenerate) and `OptimizeTests.fProxy.cs` (ladIRLS); call sites renamed. `LPTests`
  keeps pure-LP tests only.
- `LSTests.fProxy.cs` NEW — (a) cross-route equivalence: QR/QRCP/SVD/Normal agree on random
  well-conditioned tall systems (tolerance per dtype); (b) facade-vs-backend equivalence: each
  route bit-matches its underlying `*InPlace` call on the same copy; (c) input-preservation
  assert (A, b untouched); (d) rank-deficient: `solve(QRCP)` rank matches `SVD.pinvSolve`,
  `minNorm` x matches pinv x both tall and wide (the inconsistent-b COD trap case from the
  minNorm test suite gets a facade-level twin); (e) `tls`: a literature known-answer (Golub-Van
  Loan worked example or the classic Van Huffel test set — fetch and cite), plus the
  exact-consistent case (tls ≈ ls when residual ~0) and the v[n]≈0 Singular path; (f) throw
  guards (wide `solve`, shape mismatches).
- Existing `QRLeastSquaresResidualTests`, `SVDSolverTests`, `MultiRHSSolveTests`,
  `SparseSolverTests` are untouched (method tier unchanged).

### Benchmarks

- `LPBenchmark` LAD sections (2/2b) — call sites renamed to `LAD.*`, labels updated; no
  re-measurement (engines unchanged), the Section-2b crossover comment keeps pointing at the
  moved `/*+choose*/` literal.
- NO new LS benchmark file: the facade adds one Temp copy over already-benchmarked kernels. If
  desired, one facade-overhead row (LS.solve QR vs QR.solveInPlace at 512×128) can ride in an
  existing solver benchmark section; budget ≈ seconds.

### Docs

`LPInfo`/`LP` class docs (lad references), README solver table (problem-first row: LS/LAD/LP),
`docs/dev/naming-style-guide.md` gains the facade precedent line ("problem-domain facades: `LP`,
`LS`, `LAD` — problem-first classes over method backends; facades preserve inputs").

---

## 6. Staged implementation plan (coder-agent rounds)

Sized so each round is one committable, suite-green unit. Round 1 is pure mechanics; the single
genuinely new numeric code is in rounds 2–3.

1. **Round 1 — LAD move (mechanical, no new math).** Create `LAD.*` files by moving/renaming;
   strip lad surfaces from `LP`/`Optimize`; add `LAD.quantile` (thin dispatch over the moved tau
   overloads); sweep call sites in tests/benchmarks; split `LPTests` → `LADTests`; regen; full
   suite. No behavior change anywhere — every moved test must pass with only name edits.
2. **Round 2 — `LS.solve` + `LS.minNorm` + `LSMethod`.** New facade file + `LS.Info.cs`; the
   `Normal` route (matMatDotTransA + CHO) if approved; `LSTests` groups (a)–(d), (f); regen; suite.
3. **Round 3 — `LS.tls`.** Fetch the reference (Golub & Van Loan §6.3 / Van Huffel), implement
   over `SVD.thin` on the augmented Temp matrix, `LSTests` group (e); suite.
4. **Round 4 (separate future feature, unblocked by this design)** — `Optimize.irls<TWeight>` +
   weight structs; `LAD.fitIRLS` becomes the forwarder; Huber/Tukey surfaces decided then.
   Likewise the small-dim orthogonal-fit helpers (plane/axis) consume `LS.tls`/PCA later.

Definition of done per `docs/spec-shipped-feature.md` (naming approval = this spec's open
questions; literature vectors cited; Burst-executed tests both dtypes; benchmark budget stated;
no managed allocations).
