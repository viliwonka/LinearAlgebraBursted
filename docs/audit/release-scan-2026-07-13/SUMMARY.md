# Release-readiness scan 2026-07-13 — consolidated summary

22 scans over the template trees only (`TemplateSource`, `TemplateSourceTests`,
`TemplateSourceBenchmarks`) plus public markdown: 8 wide (one dimension across everything)
and 14 narrow (every line of one partition each, including a gap scan for two files
orphaned by a sort-collation mismatch between chunk agents). Briefs: `docs/dev/release-scan-2026-07-13-briefs.md`.

Totals: **wide 8 HIGH / 35 MEDIUM / 34 LOW · narrow 6 HIGH / 80 MEDIUM / 140 LOW**
(narrow HIGHs include 2 re-confirmations of wide HIGHs; unique defect count below).

| Report | Scope | H | M | L |
|---|---|---|---|---|
| W1 | comments/XML docs | 0 | 10 | 5 |
| W2 | error handling | 0 | 1 | 3 |
| W3 | numerics | 0 | 1 | 8 |
| W4 | type-split/constants | 0 | 1 | 3 |
| W5 | logic | 1 | 2 | 2 |
| W6 | naming/semantics | 1 | 8 | 5 |
| W7 | style | 0 | 5 | 7 |
| W8 | public docs | 6 | 7 | 4 |
| N1 | OP Bidiag–Kalman | 1 | 0 | 4 |
| N2 | OP Krylov–MIP.Info | 0 | 2 | 5 |
| N3 | OP MIP–QueryCore.Metric | 1 | 2 | 4 |
| N4 | OP QueryCore.Predicate–SVD.Metrics | 0 | 4 | 11 |
| N5 | OP SVD.Randomized–WindowType | 1 | 5 | 12 |
| N6 | Sparse | 1 | 4 | 13 |
| N7 | Arena + root | 1 | 7 | 18 |
| N8 | fProxy/iProxy/bool types | 0 | 7 | 13 |
| N9 | Debug/Interfaces/Hash/Pivot/Indices/Realtime | 0 | 8 | 10 |
| N10 | ML/Statistics/Analysis | 1 | 13 | 12 |
| N11 | tests 1–68 | 0 | 14 | 10 |
| N12 | tests 69–136 | 0 | 9 | 18 |
| N13 | benchmarks | 0 | 4 | 9 |
| N14 | Krylov gap files | 0 | 1 | 1 |

## Unique HIGH defects — code (7)

1. **`mulInPlace(T,T)` mutates the wrong operand** — root cause `UnsafeOP compMul(from, target)`
   reversed parameter order (`OP.Component.fProxy.cs:122`, iProxy twin; W5/N5). The four
   `operator *` sites pass args pre-swapped to compensate (`fProxyMxN.Operators.cs:141`,
   `fProxyN.Operators.cs:146`, iProxy twins) — **any fix must flip those in the same commit**
   (N8 M1). README quick-start example exercises the bug (W8 H5).
2. **`Eigen.valuesQR` destroys A, name lacks `InPlace`** — `Eigen.fProxy.cs:1606,1892`;
   the one twin missed by the T5 rename pass. Cross-refs to update: `FFT.fProxy.cs:11`,
   `Eigen.Info.cs:81`. (W6/N1)
3. **`Blas.dotRows` missing `rows` bounds validation** — `OP.Dot.fProxy.cs:194-209`; OOB read
   via `matMatDot` and OOB `MemClear` write. (W5→N3)
4. **`fProxyILU0.Apply` stackalloc inside per-block-row loop** — `fProxyILU0.cs:313`; stack
   grows `nb*16*sizeof(fProxy)` per Apply, ~1.3 MB at nb=10240 double, every pbiCGStab
   iteration; hoist above the loop. (N6)
5. **Long-variant bitwise shifts computed in 32-bit int** — `UnsafeOP.iProxy.cs:329-339` +
   wrapper `OP.Component.iProxy.cs:213-230`; shift count masked mod 32, result truncated. (N5)
6. **`iProxyLinVec` interpolates via float `math.lerp`** — `ArenaExtensions.iProxy.cs:77-79`;
   long interior values off by up to ~2^38, int/uint by ~128 past 2^24. Interpolate in double. (N7)
7. **bool `Analysis.isDiagonal` tests identity-pattern, not diagonality** —
   `BoolAnalysis.cs:9-22`; wrong result for diagonal-with-false-diagonal, no squareness check,
   undocumented `compare` inverter. (N10, escalating W6)

## Unique HIGH defects — public docs (5, W8; report-only, prose is maintainer's)

8. `docs/features/realtime.md` claims Kalman "not implemented" — KF/EKF/UKF shipped.
9. `docs/features/random.md` documents five nonexistent method names (real: `Rand.orthogonalInPlace` etc.).
10. README License omits the GPL-pending redistribution hold declared in Third Party Notices.md.
11. MPC and NLS/Levenberg-Marquardt absent from README/CHANGELOG/features docs entirely.
12. `docs/features/decompositions.md` "CHOP unblocked by design" contradicts CHANGELOG's blocked level-3 entry.
(W8 H5 = the README `mulInPlace` example, folded into #1.)

## MEDIUM clusters (details in the per-scan reports)

- **Memory-safety validation gaps vs siblings**: `BSR.spMM` rows (N6), `NormalJacobi.Apply` (N6),
  static `Pivot.Apply*InPlace` (N9), `Stats.covarianceInto` C-shape (N10), unguarded
  linear/`System.Index` indexers (N8), `RollingWindow.GetSample` (N9), Query fill-pass
  unchecked `idx[written++]` copy-paste seam (N7).
- **Rename stragglers**: `maxIter`→`maxIterations` in 13 files + PCA + SVD `maxSweeps` (W7/W6/N5/N10);
  `tol`/`relTol`→`tolerance`/`relativeTolerance` (W6/N7/N10); `StatsOP.` in 12 exception
  messages (N10); `MatrixMetrics` in 11 QRCP/LQRP crefs (W6); `BSM` in LOBPCG doc (W7);
  `SolversTests`/`MatrixMetricsTests` class names (W7/N11); `intQuery_OP`/`shortQuery_OP`
  phantom types in iProxy test headers (N12).
- **Discarded convergence/status**: `SVD.Metrics.singularValues`/`cond`/`rank` ignore SVDInfo
  (N4/N10); `BlockJacobi` discards LU status and preconditioners throw `ArgumentException`
  for numerical breakdown where dense siblings return `DirectSolveStatus` (W2 — needs ruling).
- **`[NoAlias]` self-aliasing calls**: `signFlipInPlace` (W5), SelectOP (N4), UnsafeBoolOP
  in-place wrappers (N5), `matMatDotTransA` from `isOrthogonal`/`covarianceInto` (N10),
  `Blas.dot(A,A,true)` (N13) — needs a single policy ruling.
- **Codegen artifacts in shipped output**: `SimdMath.cs` wrong branch → mangled comment emitted
  3× (N4, fix `//singularFile//`); `SolveInfo.cs` proxy-typed crefs unresolvable in package (N4);
  `const float` in double Gallery (W4).
- **Test-template comment debt** (W1/N11/N12): R6a (7×), FM2 (6×), STAGE/Stage-3/Stage E,
  commit hash `de74c48`, "commit 2"/"2.5" tags, agent-speak ("fable-caught trap", "coder's
  smoke tests", memory-file pointer in LiteratureTests), "per the spec" (4×), measured
  baselines — all with proposed DEVLOG relocations in the reports.
- **Silent test gaps**: `CHOTests.NotSPDStatus` has no [Test] driver — never runs (N11);
  dead `AssemblyTestJob` structs in 5 files (N11); `DotOperationTests` MatVec length-only
  asserts, both halves (N11/N12).
- **Benchmark validity**: PCG "SPD" generator not symmetric (N13); dense LOBPCG job skips
  cold-start reset its BSR siblings perform (N13).
- **Guard-tool blind spot**: `check-doc-leaks.ps1` never scans `TemplateSource*` or generated
  test/benchmark folders; dev-history regexes too narrow (W1 #15).
- Misc: Eigen `[Obsolete]` "~30x" claims (W1), UKF ctor doc float32 benchmark narrative (W1),
  `LQ.minNormSolve` `ref`-where-`in` (N2), `LOBPCGInfo.ToFixedString` vs its doc example (N2),
  pbiCGStab default `2*M_Rows` vs family `M_Rows` (N14), `==`/`!=` without `Equals`/
  `GetHashCode` → CS0660/661 ×28 for consumers (N8), `Print.Log` missing for 7 newer info
  structs (N9), `fProxyFullStats.count` typed fProxy (N10), `rowMean`/`colMean` NaN-fill on
  empty where siblings throw (W5/N10), `BSR.samePattern` ignores `Symmetric` flag (N6),
  Arena `AllocationsCount` omissions + public bool temp factories (N7), "UNITARY"→"unary"
  banners (W7), `Svd`-cased test classes (W7).

## Open questions needing a maintainer ruling

1. Preconditioner numerical breakdown: keep `ArgumentException` or move to status returns? (W2)
2. `[NoAlias]` policy: annotate kernels truthfully vs forbid aliased calls at wrappers?
3. Stats in-place transforms mutating through `in` params — rename, re-sign, or document? (N10)
4. pbiCGStab `2*M_Rows` default: intentional (preconditioned restarts) or drift? (N14)
5. Pascal-case predicates + Stats namespace — pre-existing open ruling (coherence-audit §3-4).
6. No inherently-ℝ ops leak into int/uint — the deletion question is moot (W4).

## Verified-clean headlines

Numerics (per-type constants, NaN-safe pivots, squared-relative convergence tests), error
handling (validation uniform and Burst-legal, no swallowed statuses), all `//+choose` branch
counts and `//+skipFor` uint gating, GEMM/WY/Francis/SYRK/TRSM index math, Pivot apply
directions, Kalman KF/EKF/UKF, MPC condensing, LM, both Gallery files entry-exact vs
literature, xxHash32 kernel, benchmark timing methodology (setup outside timed regions,
results consumed), iProxy exact-oracle tests.
