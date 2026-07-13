# W6 — Naming, semantics & method names (templates only)

Scanner: W6, release-readiness scan 2026-07-13.
Canon: `docs/dev/naming-style-guide.md` + settled decisions (M_Rows/N_Cols KEPT — not flagged;
solver overload arity settled — not flagged; LP/QP/MIP `maxIter` family DELIBERATELY kept per the
doc-jargon-cleanup T5 ruling — not flagged).

Scope swept: all of `TemplateSource/**`, `TemplateSourceTests/**`, `TemplateSourceBenchmarks/**`.
Method: full type/method inventory extraction, per-suffix (`InPlace`/`Inpl`/`Into`) census,
purged-token greps (`Elem`, `Linear_OP`, `BSM`, `_OP`, `Inplace`, `symmetric(`/`valuesSymmetric(`),
param-name census (`maxIter*`, `tol`/`tolerance`/`eps*`/`relTol`), Pascal-vs-camel census on public
statics, solver-grid conformance pass, ref-param direction audit, stale-class-name cross-reference
scan, plus DEVLOG/coherence-audit checks before flagging anything as accidental.

---

## HIGH

### H1 — `Eigen.valuesQR` destroys A but its name lacks the `InPlace` suffix (missed twin of the executed symmetric renames)
- **Template:** `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Eigen.fProxy.cs:1606` (primary) and `:1892` (default-budget overload).
- **Defect:** the method's own doc states "On input A must be square; A is DESTROYED (overwritten during reduction/iteration)", and it takes `ref fProxyMxN A` — yet the name carries no `InPlace`. The library's naming contract is "InPlace suffix appears exactly when the method destroys its input", and the T5 breaking-rename pass fixed exactly this defect on the same class's siblings (`symmetric`→`symmetricInPlace`, `valuesSymmetric`→`valuesSymmetricInPlace`, per coherence-audit §2.1: "its name silently lies about whether A survives the call"). Every other destructive member of `Eigen` is suffixed (`symmetricInPlace`, `valuesSymmetricInPlace`, `decompInPlace`); `valuesQR` is the one remaining destructive method without it. Non-destructive `ref A` methods exist in the same family (`SVD.pinvSolve` docs "A is NOT modified"), so users cannot infer destruction from `ref` — the suffix IS the signal, and here it signals "preserved". Concrete failure: a user calls `Eigen.valuesQR(ref A, ref re, ref im)` trusting the convention, then reads A → garbage (Hessenberg/Schur scraps).
- **Checked for a ruling:** no DEVLOG entry; coherence-audit only tabulates `valuesQR`'s `maxIterPerRoot` param (§1 table); §2.1's rename ruling covered only `symmetric`/`valuesSymmetric`. This is an unruled miss, not a decision.
- **Fix direction:** rename to `valuesQRInPlace` (both overloads), matching the executed T5 rename family; pre-release breaking rename.

---

## MEDIUM

### M1 — Public XML docs reference `MatrixMetrics.rank`, a class that does not exist (11 sites)
- **Templates:** `OP/QRCP.fProxy.cs:758, 810, 978, 1021, 1136, 1209` and `OP/LQRP.fProxy.cs:425, 472, 535, 661, 729` (mixed `///` docs and `//` comments).
- **Defect:** e.g. QRCP.fProxy.cs:758 "max(m,n) * Consts.fProxyZeroThreshold (matching SVD.pinvSolve / MatrixMetrics.rank)". The rank metric lives in `class Analysis` (`Analysis/Analysis.Metrics.fProxy.cs:11,53`); no `MatrixMetrics` type exists anywhere. Stale pre-rename class name shipping in generated public docs.
- **Fix direction:** `MatrixMetrics.rank` → `Analysis.rank` at all 11 sites.

### M2 — Exception messages name the retired class `StatsOP` (12 sites) + 2 doc references
- **Templates:** `Statistics/StatsCore.fProxy.cs:266, 287, 335, 358, 381, 403, 427, 462, 524, 545, 566, 586`; also doc/comment references `ML/PCA.fProxy.cs:208, 254` ("StatsOP.correlation").
- **Defect:** e.g. `throw new System.ArgumentException("StatsOP.rowSum: dest.N must equal A.M_Rows")` — the public facade class is `Stats` (`Statistics/StatsOP.fProxy.cs`, class `Stats`); the thrower is internal `fProxyStatsCore`. Users see an exception naming a class that does not exist in the package. Same defect class as the already-fixed "QueryOP. in exception text vs class Query" coherence finding — the Stats copy was missed.
- **Fix direction:** message prefixes → `Stats.rowSum: ...` etc.; PCA docs → `Stats.correlation`.

### M3 — `tol` param in Arena Query wrappers vs T5-renamed `tolerance` in the wrapped methods
- **Templates:** `Arena/ArenaExtensions.Query.fProxy.cs:49` and `Arena/ArenaExtensions.Query.iProxy.cs:47` — `fProxyNonzeroIndices(this ref Arena arena, in T x, fProxy tol)`.
- **Defect:** the wrapper forwards to `Query.countNonzero(in x, tol)` whose parameter was T5-renamed to `tolerance` (`OP/QueryOP.fProxy.cs:917, 953`). Same value, two public names one call apart; T5's unification covered Query but missed its Arena convenience wrappers.
- **Fix direction:** rename param `tol` → `tolerance` in both wrapper templates.

### M4 — `relTol` (Analysis.rank) vs `relativeTolerance` (QRCP/LQRP/SVD) for the identical concept
- **Templates:** `Analysis/Analysis.Metrics.fProxy.cs:53` (`rank(in fProxyMxN A, fProxy relTol)`) vs `OP/QRCP.fProxy.cs`, `OP/LQRP.fProxy.cs`, `OP/SVD.Solvers.fProxy.cs`, `OP/SVD.Subspace.fProxy.cs` (all `relativeTolerance`).
- **Defect:** same semantic (relative rank-detection threshold), same auto-default formula (`max(m,n) * Consts.fProxyZeroThreshold`), same negative-sentinel convention — and the two API families' docs explicitly cross-reference each other as "matching" — yet the public parameter name differs. T5 kept `relativeTolerance` as a blessed name; `relTol` is the outlier.
- **Fix direction:** `relTol` → `relativeTolerance` in Analysis.rank.

### M5 — PCA uses `maxIter` inside the ML family that was T5-unified to `maxIterations`
- **Templates:** `ML/PCA.fProxy.cs:316, 344, 401, 431, 493, 523` (`fitSvd`/`fitRandomized`/`fitSvdTruncated`, param `int maxIter`).
- **Defect:** the T5 ruling unified `maxIterations` across Eigen/SVD/LOBPCG/Query/Control/KMeans and deliberately exempted only the LP/QP/MIP optimization family. PCA is in neither camp but sits in ML next to `KMeans.fit(..., maxIterations)` and forwards its budget into `SVD` (which takes `maxIterations`) — so within one user workflow the same knob has two names. Looks like a T5 sweep miss, not a family decision.
- **Fix direction:** `maxIter` → `maxIterations` in the PCA fit surface (or record a deliberate carve-out in the ML DEVLOG).
- **Related (no action, boundary note):** post-T5 features consistently chose sides — `Kalman`/`Control` use `maxIterations`; `NLS`/`MPC`/`Optimize.ladIRLS` use `maxIter` with the LP/QP/MIP family they wrap. Internally consistent; only PCA straddles.

### M6 — `fitSvd` / `fitSvdTruncated` violate the trailing-acronym-stays-LOUD casing rule
- **Templates:** `ML/PCA.fProxy.cs:316–397` (`fitSvd` ×8 overloads), `:400–470` (`fitSvdTruncated` ×7 overloads).
- **Defect:** canon: "a trailing/mid acronym stays LOUD (`valuesQR`, `normalizeL2`)"; precedents in this codebase: `valuesQR`, `ladIRLS`, `ladBR`, `ladFN`, `spMV`, `normalizeL2InPlace`. `fitSvd`/`fitSvdTruncated` lowercase the SVD acronym mid-name — the only such methods in the public surface. (`fitCov` and `fitRandomized` are fine — Cov/Randomized are words, not acronyms.)
- **Fix direction:** `fitSVD` / `fitSVDTruncated` (breaking, pre-release), or record an explicit exception in the style guide.

### M7 — Pascal-case public statics inside otherwise camelCase op classes
Canon: methods are camelCase; the recorded Pascal-predicates open question (coherence-audit §3, user
has not ruled) covers part of this, but one pair is internally inconsistent regardless of any ruling:
- **`whichTrue` vs `WhichTrue` in the SAME file:** `Analysis/BoolAnalysis.cs:85, 99` (`Analysis.whichTrue`, camel) vs `:142, 156` (`ArenaExtensions.WhichTrue`, Pascal). Same operation, two casings, one file. Whatever the Pascal ruling ends up being, these two must agree.
- Pascal members in camel classes (all locations, for the pending ruling):
  - `Analysis.IsAllSame` / `IsAllEqualTo` / `IsAnyEqualTo` — `Analysis/BoolAnalysis.cs:24, 34, 47` (same class also has camel `any`/`all`/`isZero`/`isDiagonal`).
  - `Analysis.MaxZeroError` — `Analysis/Analysis.fProxy.cs:68, 77` (not even a predicate; siblings `rank`, `determinant` are camel).
  - `Rand.UniformICDF`/`ExponentialICDF`/`RayleighICDF`/`WeibullICDF`/`CauchyICDF`/`LogisticICDF`/`ParetoICDF`/`TriangularICDF` — `OP/RandomOP.fProxy.cs:211, 239, 270, 305, 343, 378, 415, 461` (same class: camel `nextUniformInPlace`, `shuffleInPlace`).
  - `Swap.Rows` / `Swap.Columns` — `OP/SwapOP.cs:35, 57` (op class, contrast `Blas.dot`, `Query.nonzero`).
- **Not flagged:** Pascal members on DATA types (`fProxyMxN.Copy/CopyTo/Dispose/IsCreated`, `Pivot.ApplyRow/Swap/Copy`, `Indices`) — those follow the Unity-container convention consistently; and the managed Debug/Export facade (`Print.Log`, `Export.ToCsv/SaveCsv`, `Spy`) which is uniformly Pascal as a family.
- **Fix direction:** needs the user's Pascal-predicate ruling; at minimum align `WhichTrue`→`whichTrue` (or vice versa) within BoolAnalysis.cs.

### M8 — bool `Analysis.isDiagonal` does not mean "diagonal": it tests the identity mask, unlike its float sibling
- **Template:** `Analysis/BoolAnalysis.cs:9–22` — `isDiagonal(in boolMxN bm, bool compare = true)`.
- **Defect:** the float family's `isDiagonal` (`Analysis/Analysis.fProxy.cs:165`) checks off-diagonal entries only (diagonal values are free). The bool version returns true only when `bm[i,j] == (i==j)` everywhere — i.e. the diagonal must be ALL TRUE (it is `isIdentity` semantics; the bool `diag(true,false,true)` mask, a perfectly diagonal mask, returns false). With `compare:false` it tests the complement-of-identity pattern — an undocumented, unnamed behavior; the method has no XML doc and no sibling anywhere uses a `compare` param. Same name, two meanings across the type family.
- **Fix direction:** either rename to match its real contract (`isIdentity` on the bool family, dropping `compare`), or reimplement to the family's off-diagonal-only meaning; document whichever wins.

---

## LOW

### L1 — Purged token "BSM" + roadmap dev-speak resurfaced in a doc comment
- **Template:** `OP/LOBPCG.fProxy.cs:622–623` — "This is the preconditioned entry point the sparse-BSM eigensolver roadmap calls out".
- BSM was renamed to BSR; "the roadmap calls out" is dev-speak (belongs in DEVLOG per comment policy).
- **Fix direction:** trim to the contract sentence; DEVLOG entry if the provenance matters.

### L2 — `Consts` member-name drift: `Epsilon` vs `Eps`, and `Chol` vs the CHO class family
- **Template:** `Consts.cs:15–16` (`fProxyEpsilon` vs `fProxySqrtEps` — two spellings of the same word in adjacent members) and `:20–21` (`fProxyCholBlockMinN`/`fProxyCholPivotBlockMinN` — "Chol" where the library's blessed shorthand is CHO/CHOP; siblings use the class-token style `fProxyQrBlockMinN`/`fProxyLuBlockMinN`).
- **Fix direction:** pick one spelling per concept; public consts, so decide pre-release.

### L3 — Pivot exception messages skip the canon "MethodName: what went wrong" format
- **Template:** `Pivot/Pivot.Operations.cs:84, 98, 111, 124, 137, 150` — e.g. `throw new System.ArgumentException("Matrix rows and pivot must have same dimension")`.
- No method-name prefix (canon format), and fully-qualified `System.ArgumentException` where the rest of the library uses bare `ArgumentException`. Messages are static literals, so Burst-safe — format only.
- **Fix direction:** `"Pivot.ApplyRow: A.M_Rows must equal pivot.N"` style.

### L4 — The canon document itself is stale in four places (naming-style-guide.md vs shipped reality)
Not a template defect, but W6's canon contradicts the code it governs; a reviewer using the guide will file false findings:
1. Guide mandates **`Inpl`** for elementwise/arithmetic in-place ops ("mulInpl, not mulInplace") — zero `Inpl` methods exist; the library uniformly ships `addInPlace`/`mulInPlace`/`divInPlace`/`modInPlace`/`subInPlace` (`OP/OP.Component.fProxy.cs`), and coherence-audit already blessed the long form ("in-place ops are consistently *InPlace").
2. Guide's echo-exception example **`LOBPCG.lobpcg`** — there is no LOBPCG class; the method is `Eigen.lobpcg` (`OP/LOBPCG.fProxy.cs:51`).
3. Guide's Internal-namespace list names **`Unsafe_OP`, `UnsafeBool_OP`, `UnsafeSelect_OP`** — shipped classes are `UnsafeOP`, `UnsafeBoolOP`, `UnsafeSelectOP` (no underscore).
4. Guide presents **`Stats_OP`/`Norms_OP`/`Elem_OP`/`Linear_OP`** as current class names in the `_OP`-suffix section — all retired (`Stats`, `Norms`, `fProxyComp`, `Blas`); the `_OP` suffix convention it describes no longer exists in the public surface.
- **Fix direction:** one editing pass over docs/dev/naming-style-guide.md (internal doc, non-breaking).

### L5 — Output param named `outv` in `ApplyT` vs `y` in sibling `Apply` within the same operator structs
- **Template:** `Sparse/fProxySparseLP.fProxy.cs:159, 248` (`ApplyT(in fProxyN r, ref fProxyN outv)`) vs `:149, 240` (`Apply(in fProxyN z, ref fProxyN y)`); every other operator (`fProxyBSROperator:61–63`) uses `y` for both.
- **Fix direction:** `outv` → `y`.

---

## Areas confirmed clean

- **Purged tokens:** no `Elem`/`Linear_OP`/`BSM`/`Inplace` in any API name (templates, tests, benchmarks); `_OP` survives only in template FILE names and Internal-namespace kernel classes (`UnsafeOP` family — settled), not on public op classes. `fProxyComp`/`boolComp` are the settled Comp classes. One comment-level BSM leak (L1).
- **T5 renames hold:** no stale `Eigen.symmetric(`/`valuesSymmetric(` call or declaration anywhere; `tolerance`/`maxIterations` consistent across Eigen/SVD/LOBPCG/Krylov/KMeans/Control/Kalman; test/benchmark named-arguments (`maxIter:` in LPTests/MIPTests) all target the deliberately-kept LP/QP/MIP surface.
- **Four-token solver grid:** LU/CHO/CHOP/QR/QRCP/LQ/LQRP/Bidiag/SVD/Eigen conform (`decomp`/`decompInPlace`/`decompSolve`/`solveInPlace` + `minNormSolveInPlace`, `decompSolveTransA` modifiers); `b_to_x`/`B_to_X`/`A_to_LU` transformation names used consistently; QR.decompSolve's separate `(ref b, ref x)` is dimensionally forced (m≠n) and documents b as preserved; destructive one-shots document what they destroy (H1 is the single naming exception found). Sparse IC0/ILU0 expose only ctor+`Apply` (their private `FactorizeInPlace` helpers don't violate the grid's forward guidance).
- **Info/status surface:** `SolveInfo`/`LstsqInfo`/`DirectSolveInfo`/`RankInfo`/`SVDInfo`/`EigenInfo`/`EigenSolveInfo`/`LanczosInfo` + `*Status` enums + `*StatusExtensions.Name()` — uniform shape (`status`, `Solved`, implicit bool, `ToFixedString`), no same-name-two-meanings.
- **Class casing:** all-caps only on literature acronyms (QR/LU/LQ/SVD/BSR/FFT/CHO/CHOP/QRCP/LQRP/LP/QP/MIP/MPC/NLS/PCA); truncated words Pascal (`Rand`, `Bidiag`, `Blas`, `Comp`); split-vs-merge prefix usage consistent; `Cache`-suffix workspace structs consistent (19 types); Arena `Vec`/`Mat` abbreviations consistent, no stray `Vector`/`Matrix` method names.
- **No misdescribing verbs found beyond H1/M8:** no `get*` that allocates (zero `get*` methods); predicates return bool or a documented mask (`ispow2` elementwise map mirrors `math.ispow2`); `powerIteration`/`inversePowerIteration` document `v` in-out and `w` scratch explicitly; `SVD.pinvSolve(ref A)` explicitly contracts "A is NOT modified".
- **Visibility:** internal kept internal (`fProxy*Core`, `Assume`, `Helpers`, `fProxyChooseMarkerDemo`, private sparse factorize helpers); `LinearAlgebra.Internal` kernels public-by-namespace-signal per canon; no member-visibility drift between siblings found.

## Summary

| Severity | Count | IDs |
|---|---|---|
| HIGH | 1 | H1 (valuesQR destructive without InPlace suffix) |
| MEDIUM | 8 | M1–M8 |
| LOW | 5 | L1–L5 |

Plus one no-action boundary note (maxIter/maxIterations family split, under M5).
