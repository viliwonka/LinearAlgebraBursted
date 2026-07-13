# W8 — Public docs scan (2026-07-13)

Scope: README.md, CHANGELOG.md, docs/features/*.md (22 files), Assets/LinearAlgebra/Source/Third Party Notices.md.
Verified against: Assets/LinearAlgebra/CodeGen/TemplateSource/** (production templates, source of truth).
Read-only scan. No prose was rewritten — findings only, per instructions.

Every public doc file in scope was read in full. Every class/method/enum/struct-field claim was
cross-checked against the actual template source (not the generated `Source/` output) by direct
grep/read and by four parallel verification passes covering: (1) LP/QP/MIP/Control, (2)
Eigen/SVD/Sparse/solvers/least-squares/decompositions, (3) FFT/Stats/Random/Query/ML, (4)
dense-types/Comp/Select/Hash/Generators/Print/README-quickstart/CHANGELOG.

---

## HIGH

### H1 — `docs/features/realtime.md:16-18` falsely claims the Kalman filter is unimplemented
> "This is deliberately the only piece of the "realtime" design surface that's built. Frame-amortized
> solvers, resumable iterative state (CG/PCG stepping across frames), online covariance/PCA, and a
> Kalman filter are still unsettled design, not implemented."

This is false. A complete, tested, benchmarked Kalman filter (linear predict/update, steady-state
gain via `Control.SDACore` duality, EKF via user Jacobian functors) and a UKF (Van der Merwe scaled
sigma points) are fully implemented production templates:
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Kalman.fProxy.cs` (`Kalman.predict`, `Kalman.update`,
`Kalman.steadyStateGain`, `Kalman.ekfPredict<TModel>`, `Kalman.ekfUpdate<TMeas>`),
`OP/Kalman.UKF.fProxy.cs` (`Kalman.ukfPredict<TModel>`, `Kalman.ukfUpdate<TMeas>`), plus
`OP/Kalman.State.fProxy.cs`, `OP/Kalman.UKFCache.fProxy.cs`, `OP/Kalman.Info.cs`. Backed by
`TemplateSourceTests/fProxy/KalmanTests.fProxy.cs`, `UKFTests.fProxy.cs`, and
`TemplateSourceBenchmarks/KalmanBenchmark.fProxy.cs` — matching recent shipped commits ("Kalman: UKF
(Van der Merwe scaled sigma points)"). Fix direction: rewrite realtime.md's closing paragraph, and add
a proper feature doc for Kalman/UKF (see H4).

### H2 — `docs/features/random.md` documents five method names that don't exist
Doc text (lines ~26-29):
> `randomOrthogonalInPlace` (Haar-uniform, Mezzadri sign-fixed QR), `randomSpdInPlace(..., minEig,
> maxEig)`, `randomMatrixWithConditionInPlace(..., cond)`, `randomMatrixWithRankInPlace(..., rank)`,
> `randomStochasticInPlace`.

None of these identifiers exist anywhere in the codebase. The real API
(`TemplateSource/OP/RandomMatrixOP.fProxy.cs`) is:
- `Rand.orthogonalInPlace(ref Random rng, ref fProxyMxN dest)` — line 146
- `Rand.spdInPlace(ref Random rng, ref fProxyMxN dest, fProxy minEig, fProxy maxEig)` — line 205
- `Rand.conditionedInPlace(ref Random rng, ref fProxyMxN dest, fProxy cond)` — line 277
- `Rand.withRankInPlace(ref Random rng, ref fProxyMxN dest, int rank)` — line 348
- `Rand.stochasticInPlace(ref Random rng, ref fProxyMxN dest)` — line 392

Any code written against the documented names fails to compile. Fix direction: rename the five
identifiers in random.md to match the shipped API (drop the `random`/`Matrix` prefix).

### H3 — README.md's License section omits the GPL-pending redistribution hold documented in the same package's Third Party Notices
README.md's entire License section reads:
```
## License
[MIT](LICENSE).
```
`Assets/LinearAlgebra/Source/Third Party Notices.md` (in scope for this scan, ships inside the same
UPM package) states, for `LP.ladBR`/`LP.ladFN`:
> "Status: permission to distribute these two derived implementations under this package's MIT
> license has been requested from the authors. Until that is resolved, **this package must not be
> redistributed**."

README's unqualified "[MIT](LICENSE)" gives no indication that the shipped package currently carries
this GPL-derived-code redistribution hold — a reader/redistributor relying only on README would
reasonably but wrongly conclude the whole package is free to redistribute under MIT today. Fix
direction: README's License section should at minimum point to Third Party Notices.md and flag the
current hold (this is also flagged as an open release blocker in the maintainer's own release
tracking).

### H4 — Coverage gap: MPC and NLS (Levenberg-Marquardt) are complete, shipped features with zero mention in any public doc
Companion to H1. Two more fully-implemented, tested, benchmarked features have **no** mention
anywhere in README.md's Features list, CHANGELOG.md's `[Unreleased]` section, or any
`docs/features/*.md` page (grepped all three locations for `MPC|NLS|Levenberg|Marquardt|curveFit`:
zero hits outside templates):

- **`MPC`** (`TemplateSource/OP/MPC.fProxy.cs`, `MPC.Info.cs`, `MPC.State.fProxy.cs`) — condensed
  linear MPC, `MPC.solve(ref fProxyMPCState, in x0, in reference, ref u0out, maxIter)`, warm-started
  over the active-set QP, matching the shipped commit "MPC: condensed linear MPC over the
  warm-started active-set QP." Tested (`MPCTests.fProxy.cs`) and benchmarked (`MPCBenchmark.fProxy.cs`).
- **`Optimize.nlsSolve<TF>` / `Optimize.curveFit<TModel>`** (`TemplateSource/OP/NLS.fProxy.cs`,
  `NLS.Info.cs`) — Levenberg-Marquardt nonlinear least squares (Nielsen damping), with optional robust
  loss functors `fProxyL2Loss`/`fProxyHuberLoss`/`fProxyCauchyLoss`/`fProxyTukeyLoss`
  (`Interfaces/ResidualFunction.fProxy.cs`). Tested (`NLSTests.fProxy.cs`) and benchmarked (part of
  the same commit as Kalman/MPC benchmarks).

The brief's own checklist flagged that CHANGELOG is missing "at least 2 known pending" behavior
changes — this is very likely exactly it (3 features: Kalman/UKF, MPC, NLS). Fix direction: add
CHANGELOG `[Unreleased]` entries, README Features bullets, and dedicated
`docs/features/kalman.md`/`mpc.md`/`nls.md` (or a combined `control-advanced.md`) pages.

### H5 — README.md Quick-start example: `floatComp.mulInPlace(sum, matRand)` does not mutate the argument the surrounding comment implies
Quick-start code (README.md ~lines 51-52):
```csharp
floatComp.addInPlace(sum, 1f);
floatComp.mulInPlace(sum, matRand);
```
The `addInPlace` call genuinely mutates `sum` (its receiver). The very next line, by the same
call-shape and the "in place, allocates nothing" framing, implies `mulInPlace(sum, matRand)` also
mutates `sum` (i.e. `sum *= matRand`). It does not — `mulInPlace`'s buffer-pairwise overload mutates
its **second** argument instead:
- `OP.Component.fProxy.cs:122-126` — `mulInPlace<T>(this T from, T to)` calls
  `UnsafeOP.compMul(from.Data.Ptr, to.Data.Ptr, ...)`.
- `UnsafeOP.fProxy.cs:983-987` — `compMul(fProxy* from, fProxy* target, int n) { target[i] *=
  from[i]; }` — mutates `target`, its second parameter.

So `floatComp.mulInPlace(sum, matRand)` actually computes `matRand *= sum`, leaving `sum` untouched.
This directly contradicts sibling methods `divInPlace`/`modInPlace` (`OP.Component.fProxy.cs:131-146`,
via `compDiv`/`compMod`'s `(targetDividend, fromDivisor)` convention), which correctly mutate the
receiver — and contradicts `mulInPlace`'s own doc comment, which claims to "match
addInPlace/subInPlace's existing pattern" (it does not). In the README example itself this is
harmless (neither buffer is read again), but a caller copying this exact pattern with either buffer
read afterward gets a silently wrong result. This is a real template bug surfaced through the public
docs, not a doc-only issue — reported here because it was found via the README audit; also worth a
W5/logic-track fix at `OP.Component.fProxy.cs:122-126`.

### H6 — `docs/features/decompositions.md` says CHOP is "unblocked by design" — the template has a full blocked path, and CHANGELOG.md already documents it
decompositions.md states:
> "**`CHOP.decomp(in A, ref L, ref Pivot P, ref ws)`** — rank-revealing (xPSTRF-style) pivoted
> Cholesky, upper-triangle-only working storage, **unblocked by design**; returns `RankInfo`."

`TemplateSource/OP/CHOP.fProxy.cs:102-196` has an explicit "blocked (level-3) path — LAPACK-style
right-looking PSTRF" gated by `CHOLP_BLOCK_MIN_N = Consts.fProxyCholPivotBlockMinN` (→
`floatCholPivotBlockMinN = 512` / `doubleCholPivotBlockMinN = 512`, `Consts.cs:50-51`), with a
32-wide panel (`CHOLP_BLOCK = 32`). CHANGELOG.md's own `[Unreleased]/Changed` section already
documents this: "LU gains a blocked level-3 `decompInPlace` path; pivoted Cholesky (`CHOP`) gains a
blocked level-3 factorization." decompositions.md directly contradicts both the code and the
project's own CHANGELOG — it is stale, describing CHOP's pre-blocking behavior. Fix direction: update
decompositions.md's CHOP bullet to match LU/CHO's phrasing ("blocked above n ≥ ...").

---

## MEDIUM

### M1 — `docs/features/solvers.md:89` lists an incomplete `DirectSolveStatus` enum
Doc: `` `DirectSolveStatus`: `Success, Singular, NotPositiveDefinite, Indefinite, RankDeficient`. ``
Actual enum (`TemplateSource/OP/SolveStatus.cs:38-71`) has **six** members — the doc omits
`NotConverged = 5` (returned when an SVD-backed rank-revealing call fails to converge). This is
inconsistent within the same doc set: `docs/features/svd.md:76` correctly references
`DirectSolveStatus.NotConverged` for exactly this case, so a reader who trusts solvers.md's "complete"
list will be surprised by a status value svd.md already told them about. Fix direction: add
`NotConverged` to the solvers.md enum listing, with a one-line note on which callers report it.

### M2 — `docs/features/solvers.md` `RankInfo` "Used by" column is incomplete
The diagnostics table lists `RankInfo | status, rank | QRCP (solveInPlace), CHOP`. Per
`SolveStatus.cs:64-69`'s own doc comment and the actual code
(`SVD.Solvers.fProxy.cs` lines 65,112,247,288,424,467; `SVD.Subspace.fProxy.cs` lines 56,143),
`RankInfo` is also returned by `SVD.pinvSolve`, `SVD.pseudoInverse`, `SVD.nullspaceBasis`, and
`SVD.rangeBasis` (with `status = DirectSolveStatus.NotConverged` on SVD non-convergence). Fix
direction: extend the "Used by" cell.

### M3 — `docs/features/decompositions.md` blocking-threshold numbers don't match the shipped per-dtype constants
- LU: doc says "Blocked ... above `n ≥ 256`". Real: `floatLuBlockMinN = 256` (matches) but
  `doubleLuBlockMinN = 128` (double actually blocks *earlier*, not at 256) — `Consts.cs:46-47`.
- CHO: doc says "Blocked ... above `n ≥ 256`". Real: `floatCholBlockMinN = 1024`,
  `doubleCholBlockMinN = 512` — matches **neither** dtype — `Consts.cs:44-45`.
- QR: doc says "blocked (compact-WY) above `N_Cols ≥ 64`". Real: `floatQrBlockMinN = 128`,
  `doubleQrBlockMinN = 512` — matches **neither** dtype — `Consts.cs:40-41`.

All three cited numbers (256, 256, 64) match the *template-only* `fProxy*BlockMinN` placeholder
defaults (`Consts.cs:17-22`, explicitly stripped from generated output via `//+deleteThis` and used
only when the raw template compiles as its own pre-substitution assembly), not the real per-dtype
constants that ship. Contrast with the LQ bullet in the same doc, which correctly calls out the
float/double split explicitly. Fix direction: cite the real `float*/double*` pair for each family (as
already done for LQ), or drop specific numbers in favor of "see Consts.cs" wording.

### M4 — `docs/features/eigen.md` diagnostics table omits `EigenInfo`
The "Diagnostics structs" table lists `EigenSolveInfo`, `LanczosInfo`, `LOBPCGInfo` but never mentions
`EigenInfo` (`Eigen.Info.cs:103-131`, fields `status`/`sweeps`/`converged`), which is the actual return
type of `symmetricInPlace`, `valuesSymmetricInPlace`, `valuesQR`, and `decompInPlace` — all four
covered earlier in the same doc. Fix direction: add a row for `EigenInfo`.

### M5 — `docs/features/eigen.md` buckling recipe points to a worked example that no longer exists
Doc: "The buckling recipe (documented with a worked sample in the class doc) ...". The worked
numeric example was removed from `LOBPCG.fProxy.cs`'s class doc comment on 2026-07-11 "for length"
(per `TemplateSource/OP/DEVLOG.md:477`) and now survives only in the DEVLOG; the live code
(`LOBPCG.fProxy.cs:452-453`) still has a dangling in-code cross-reference to the same now-missing
note. eigen.md's own inline recipe text is accurate — only the "documented ... in the class doc"
pointer is stale. Fix direction: drop the "in the class doc" claim or restore a short worked example.

### M6 — README.md benchmark table mislabels the LU/CHO rows
README's table cites `` `LU.solveInPlace` LU | 1024×1024, float | 15.3 ms `` and
`` `CHO.solveInPlace` Cholesky | 1024×1024, float | 12.1 ms ``. But `docs/features/solvers.md`'s own
benchmark section (describing the identical `Benchmarks/DirectSolveBenchmark.cs`, N=1024 case)
states explicitly: "LU and CHO time the explicit `decomp`+`decompSolve` composition (A preserved,
distinct from L/U); QR times the fused `solveInPlace`" — and its table rows are literally labeled
`` `LU.decomp` + `LU.decompSolve` `` (15.28/15.33 ms) and `` `CHO.decomp` + `CHO.decompSolve` ``
(12.00/12.08 ms), never `solveInPlace`. README's headline table attributes these numbers to the wrong
method name for two of its three solver rows (QR's row is correctly labeled `QR.solveInPlace` in both
docs). Fix direction: relabel the README rows to `LU.decomp+decompSolve` / `CHO.decomp+decompSolve`,
or re-benchmark the actual `solveInPlace` path if that's the number intended.

### M7 — Three `docs/features/*.md` files link directly into `docs/dev/*.md` (forbidden internal-spec references in public docs)
Per CLAUDE.md ("docs/features/*.md are user-facing ... no internal spec/ticket references" and
"docs/dev/ is internal and exempt") and this scan's own forbidden-content list, public docs should not
reference `docs/dev/*.md`. Four such links exist (all resolve to real files, so not "broken links" —
the issue is the reference itself):
- `docs/features/solvers.md:6` → `../dev/naming-style-guide.md`
- `docs/features/solvers.md:7` → `../dev/spec-solver-api-rework.md`
- `docs/features/sparse-bsr.md:4` → `../dev/spec-sparse-bsm.md`
- `docs/features/stats.md:43` → `../dev/spec-histogram-resample.md`

Fix direction: drop the links (or the parenthetical they live in) from the public doc; keep any
useful pointer in that folder's DEVLOG.md instead.

---

## LOW

### L1 — Minor benchmark-number drift between README and sparse-bsr.md
README's table cites the CG-dense case (N=1024, double, 7% fill) at 15.05 ms; `sparse-bsr.md`'s own
table for what reads as the identical case cites 15.02 ms. The sparse figure (0.37 ms) matches
exactly in both. Likely just separate benchmark runs (median-of-9 noise), not a logic error, but the
two public docs don't agree bit-for-bit on "the same" number.

### L2 — Borderline benchmark-methodology narration
Two Performance sections read as explanatory narration about *why* a benchmark shows what it shows,
which borders on the brief's "no benchmark methodology narration" rule (though it's arguably useful
reader context, consistent with this doc set's established style of a one-line caveat under every
table):
- `docs/features/eigen.md`'s LOBPCG benchmark note: "...the point of this benchmark is the
  per-iteration cost, not a convergence demonstration; a real caller would set a reachable
  `tolerance` instead."
- `docs/features/sparse-bsr.md`'s block-Jacobi note: "...expected, since Jacobi preconditioning's
  real win shows up on ill-conditioned systems ... not on a benchmark's synthetic well-conditioned
  one."

Flagged for the maintainer's judgment call rather than as a clear violation.

### L3 — `docs/features/generators.md` window section reads as standalone functions but is one enum + one method
Doc lists "Window: `Box`,`Hann`,`Hamming`,`Blackman`" in the same bullet style as the easing/wave
struct functors (which genuinely are individually-named callables). In the template these are members
of a single `enum WindowType` (`TemplateSource/OP/WindowType.cs:7-13`) consumed by one method
`Generate.window(ref dest, WindowType type)` (`GenOP.fProxy.cs:163`) — not standalone
`Generate.Hann(...)`-style functions. Not factually wrong, but shaped to mislead about the call
pattern.

### L4 — `Optimize`'s pre-existing 1D optimizers have no dedicated feature doc or README bullet
Root-finding/minimization/gradient-descent optimizers on `Optimize` were already shipped and
mentioned once in CHANGELOG's `[0.1.0]` entry, but have no `docs/features/*.md` page and no README
Features bullet — the class is cross-referenced exactly once, in passing, from
`docs/features/lp-lad.md` ("see `Optimize.ladIRLS`"). Lower-priority, pre-existing companion to H4.

---

## Summary

| Severity | Count |
|---|---|
| HIGH | 6 |
| MEDIUM | 7 |
| LOW | 4 |
| **Total** | **17** |

**Areas confirmed clean** (zero mismatches across ~90+ individual claims checked against templates):
- `docs/features/lp-lad.md`, `qp-mip.md`, `control.md` — every signature, enum, struct field,
  destructive/non-destructive contract, and cross-doc consistency claim confirmed exact.
- `docs/features/fft.md`, `query.md`, `ml.md` — fully confirmed, including the FFT forward/inverse
  scaling convention and the `LinearAlgebra.ML` namespace claim.
- `docs/features/stats.md` — fully confirmed including the subtle "uint deliberately excluded"
  design claim (verified absent from generated `Source/Statistics/` output).
- `docs/features/dense-types.md`, `comp-elementwise.md`, `select-bits.md`, `hash.md`,
  `print-export.md` — fully confirmed, including the `Select.select(a,b,c,dest)` argument-order
  semantics (an easy place for an off-by-one bug — verified correct) and `Print`'s
  `FixedString4096Bytes`/G7/G9 claims.
- `Third Party Notices.md` — HiGHS attribution (RevisedSimplex, DualSimplex, QP, MIP) and
  Koenker/quantreg attribution (`LP.ladBR`/`LP.ladFN`) both verified against template header comments;
  accurate as written (the *omission* of this file's caveat from README is H3, not an error in the
  notices file itself).
- `package.json` version (`0.1.0`) is consistent with CHANGELOG's `[0.1.0]`/`[Unreleased]` structure.
- README.md's Quick-start example compiles against the current API in full (every symbol/signature
  checked resolves) except for the behavioral issue in H5; the Determinism section's `FloatMode`
  claims are confirmed against actual `[BurstCompile(...)]` attributes on benchmark templates.
- All README ↔ docs/features cross-links and internal docs/features cross-links resolve to existing
  files (no broken links found other than the internal-reference concern in M7).
