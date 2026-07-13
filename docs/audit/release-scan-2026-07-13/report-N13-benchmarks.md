# N13 — Narrow scan: TemplateSourceBenchmarks (all 26 .cs)

Scanner: N13 (narrow pass, all dimensions). Date: 2026-07-13.

## Files covered (every line read)

All 26 templates under `Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/`:
CholeskyBenchmark, DirectSolveBenchmark, EigenSvdBenchmark, FFTBenchmark, GemmBenchmark,
IterativeBenchmark, KalmanBenchmark, KernelBenchmark, KMeansBenchmark, LOBPCGBenchmark,
LPBenchmark, LQRBenchmark, LUBenchmark, LargeSparseBenchmark, MIPBenchmark, MPCBenchmark,
PCGBenchmark, QPBenchmark, QRBenchmark, QRVariantsBenchmark, SmallSizeBenchmark,
SparseSolverBenchmark, SvdComparisonBenchmark, SvdSolversBenchmark, TallWideSolveBenchmark,
TriangularSolveBenchmark (all `.fProxy.cs`). Also read: folder DEVLOG.md, TemplateConverter.cs,
and cross-checked hand-written harness halves (`Assets/LinearAlgebra/Benchmarks/Bench.cs`,
`PCGBenchmark.cs`, `LOBPCGBenchmark.cs`, `KalmanBenchmark.cs`) plus production signatures
(`OP/QP.fProxy.cs`, `OP/MIP.fProxy.cs`, `OP/OP.Dot.fProxy.cs`, `OP/UnsafeOP.fProxy.cs`,
`Sparse/SparseOP.fProxy.cs`) for label/contract verification only — no findings reported
against non-template files except as trace-back context.

## Benchmark-specific duties — overall verdict

- **Timed-kernel integrity**: verified per job. Setup (matrix builds, RNG fills, workspace/
  preconditioner construction, LU factor for the triangular-solve rows, transposed-A
  materialization, MPC pre-warm frames, LQR cold seed solve) is consistently outside
  `Bench.Time`. The one deliberate, documented exception class is re-copying pristine
  Src into the working buffer inside `Execute()` for destructive kernels (QR/QRCP/CHO/
  LU-in-place, Eigen, LQRP, TriSolve rhs) — an O(n^2) copy vs an O(n^3) kernel, consistent
  convention, explicitly commented. Acceptable.
- **DCE resistance**: every timed job stores results into NativeArray-backed containers or
  length-1 out arrays consumed by the report; `ReduceJobFProxy` has an explicit sink plus an
  `acc*1e-30` feedback write; matvec microbenches ping-pong x and y; `MpcConstructJobFProxy`
  accumulates a checksum. No dead-code-elimination hazards found.
- **Sizes/labels**: spot-verified row formatters against the printed numbers (dense LOBPCG
  header `min(ms) med(ms)` matches `stat.Min, stat.Median`; face-off `med,min` matches
  `Median, Min`; PCG rows `med,min`; Bench.Row GFLOP/s uses median). All consistent.
- **Bench.Time contract** (`Benchmarks/Bench.cs:32`): Warmup runs + Runs timed, median
  reported — matches the "1 warmup + 4 timed" claims in the LP/MIP/QP/Kalman comments.

## Findings

### HIGH

None.

### MEDIUM

**M1. PCGBenchmark.fProxy.cs:164-186 — `BuildTridiagBlockSPDFProxy` builds a NON-symmetric
matrix but is named/labeled SPD and fed to CG/PCG (SPD-contract solvers).**
`Di[r, c] = (r == c ? BR * 8f : 0f) + rng.NextFloat(-0.1f, 0.1f);` — every entry of each
diagonal block, including off-diagonal entries, gets independent noise, so Di[r,c] != Di[c,r]
(the off-diagonal blocks ARE mirrored via `offT`; only the diagonal blocks break symmetry).
The harness header (`Benchmarks/PCGBenchmark.cs:29`) prints "block-tridiagonal SPD BSR".
Asymmetry is small (0.1 vs diagonal BR*8) and rows run a fixed K with tol=0, so timings are
unaffected, but the matrix violates the CG/PCG symmetry contract and the printed residual
column reflects CG-on-nonsymmetric behavior. Fix direction: symmetrize the diagonal-block
noise (fill j>=i, mirror), like the SmallSizeBenchmark/CholeskyBenchmark builders do.

**M2. LOBPCGBenchmark.fProxy.cs:19-28 (+ BenchFProxy:159-185) — dense `LobpcgJobFProxy` does
not cold-start `ws.X`, unlike all four BSR sibling jobs in the same file.**
Each BSR job zeroes `ws.X` at the top of `Execute()` with the comment "otherwise the reused
workspace warm-starts already-converged and times a no-op"; the dense job
(`public void Execute() => infoOut[0] = Eigen.lobpcg(in A, ref ws, k, tol, maxIter);`) reuses
the same workspace across Bench.Time's warmup + 4 timed calls with no reset. tol=(fProxy)1e-20
is unattainable so full maxIter is nominally forced, but samples 2+ start from the previous
sample's converged block — any per-vector locking/stagnation/breakdown path inside lobpcg makes
warm reps do different (less) work, and the reported iters/converged describe a warm-started
run, not the cold solve the row implies. Fix direction: zero `ws.X` per Execute exactly like
the BSR siblings (or add a comment stating why warm reuse is intended for the dense row).

**M3. LPBenchmark.fProxy.cs:370-372 — measured perf number in a code comment (policy).**
"its O(m*nCols) per-pivot tableau update makes it the slow tail of this section past there
(measured ~101ms already at m=192, double)". Measured numbers belong in DEVLOG.md.
Proposed DEVLOG entry: `## LPBenchmark` / `- 2026-07-13 | Tableau-simplex LAD row capped at
LadSimplexCap because its O(m*nCols) per-pivot update was measured ~101ms at m=192 double —
the section's slow tail. (was LPBenchmark.fProxy.cs:370)`. Keep only the contract ("capped at
LadSimplexCap; slowest backend of this section") in-source.

**M4. MIPBenchmark.fProxy.cs:84-85 — measured baseline in a code comment (policy).**
"proven optimum 9 (double: ~275 nodes / ~4307 LP iterations per the test file's own measured
baseline)". The node/iteration counts are measured results, not a contract of the builder.
Proposed DEVLOG entry: `## MIPBenchmark` / `- 2026-07-13 | stein15 measured baseline ~275
nodes / ~4307 LP iterations (double), matching MIPTests' own run. (was
MIPBenchmark.fProxy.cs:84)`. Keep "proven optimum 9" (that IS instance contract).

### LOW

**L1. LPBenchmark.fProxy.cs:426-432 — budget-estimate narration in a comment (policy).**
"Budget estimate: ... ~3.3M flops per solve -- sub-millisecond. Times 5 runs ... total added
wall-clock is estimated at well under 10s (most rows sub-ms, expected sum in the low hundreds
of ms)." — benchmark-budget bookkeeping; belongs in DEVLOG. Proposed entry: `## LPBenchmark` /
`- 2026-07-13 | Section 2b budget estimate: dominated by dispatch/alloc overhead, total well
under 10s. (was LPBenchmark.fProxy.cs:426)`. Keep the crossover rationale (Portnoy & Koenker
1997) and the size-choice contract in-source.

**L2. LQRBenchmark.fProxy.cs:24 — internal-spec reference in comment.** "...the 'naive'
baseline the spec wants SDA/warm compared against, reached without touching Control's internal
RiccatiIterate directly." "the spec wants" is dev-workflow speak; the contract is simply "the
plain-fixed-point-recursion baseline". Fix: drop the clause; DEVLOG already covers LQR history.

**L3. LargeSparseBenchmark.fProxy.cs:28-88 — all 11 jobs in this file omit
`FloatPrecision = FloatPrecision.High` from `[BurstCompile]`, unlike every other benchmark
template (all use `FloatPrecision.High, FloatMode.Default`).** Probably harmless for these
kernels, but it is the lone compile-option outlier in the suite, so cross-file comparisons
(e.g. vs SparseSolverBenchmark's identical CG/PCG jobs) are not apples-to-apples in principle.
Fix direction: add `FloatPrecision = FloatPrecision.High` for uniformity, or note the reason.

**L4. MPCBenchmark.fProxy.cs:41/69 + 179/215 — `objOut` is populated by both the warm and cold
jobs but never printed** (`MPCBenchmarkFmt.Row` takes iters/changes/status only); it is
allocated, written, and disposed unused. Not a DCE concern (u0/state writes anchor the solve).
Fix direction: drop the field or add the objective column.

**L5. Same pointer passed to two `[NoAlias]` kernel parameters (addendum pattern 4) via
`Blas.dot(A, A, transposeA:true)` — benign here, but a contract violation at the production
seam.** Benchmark call sites: IterativeBenchmark.fProxy.cs:53, LOBPCGBenchmark.fProxy.cs:169,
SparseSolverBenchmark.fProxy.cs:364, 424, 481, 565, 906 (all build MtM). These route to
`UnsafeOP.matMatDotTransA([NoAlias] matA, [NoAlias] matB, [NoAlias] matC)`
(TemplateSource/OP/UnsafeOP.fProxy.cs:365) with matA == matB. Both aliased pointers are
read-only in the kernel, so no store/load reordering hazard exists in practice — recording as
an OPEN QUESTION for the maintainer: either the `Blas.dot` facade should document that a and b
may alias when only read, or `[NoAlias]` should come off the two input params. (Production
template concern; benchmarks merely exercise the pattern users will also use.)

**L6. Test-file name references in comments and in PRINTED report strings.**
MIPBenchmark.fProxy.cs:56, 61, 83, 110, 165 (comments: "SAME literal instance data as
MIPTests.fProxy.cs's ...") are arguably data-provenance contracts; but the runtime section
headers at :208-209, :299-302, :338-339, :382 print "see MIPTests.fProxy.cs's Stein15/P0033"
/ "reproduces MIPTests.fProxy.cs's Branchy12" into the benchmark report output itself, and
KalmanBenchmark.fProxy.cs:87, 371 references "KalmanTests.fProxy.cs". Internal test-file names
leaking into benchmark report text is the sore thumb; the comments are borderline-acceptable.
Fix direction: shorten printed labels ("stein15 — double only; float cannot prove optimality
within a sane node budget"), keep provenance detail in comments or DEVLOG.

**L7. FFTBenchmark.fProxy.cs:74-81 — `FftBuildRunJobFProxy.Execute()` constructs
`new Arena(Allocator.Persistent)` inside a Burst job.** Intent (clock the table build in
Burst) is documented and it compiles/runs, but it deviates from both the arena memory model
(arena = main-thread authoring; Temp = job-safe) and the file's own siblings (Pivot/local
buffers inside jobs use `Allocator.Temp`). Fix direction: use `Allocator.Temp` for the
job-local arena, matching house convention.

**L8. QPBenchmark.fProxy.cs:43-87 — the timed `Execute()` includes the KKT-residual recompute
(two dense matvecs + O(n+m) loops) after `QP.solve`.** Documented in the comment and negligible
vs the active-set solve at n<=192, but strictly the row's time is "solve + KKT check", not
solve alone. Fix direction: none required; optionally note in the section header that times
include the residual check.

**L9. LPBenchmark.fProxy.cs — setup-matvec convention inconsistency (harmless).**
Sections 1/5 use the Burst `LpRhsMatVecJobFProxy` for b = A*x0 with the rationale that a
managed matvec is Mono-interpreted, while Sections 2/2b/3 build comparable-size RHS via managed
`Blas.dot(A, xt)` (up to m=16384 x n=4) and QPBenchmark.fProxy.cs:117-121 documents the
opposite choice. All are untimed setup, so no measurement impact; just an internal
inconsistency in which recipe gets the Burst job. No action needed beyond awareness.

## Addendum-pattern sweep results

1. Role-swapped InPlace wrappers: none — the only InPlace call on benchmark-owned data is
   `xNext.addInPlace(Bu)` (MPCBenchmark.fProxy.cs:53); the receiver is correctly the mutated
   operand.
2. Rename stragglers: none in benchmark-owned names. Job fields `maxIter`/`tol`/`K` mirror the
   production API's own parameter names (`QP.solve`/`MIP.solve` take `maxIter` —
   TemplateSource/OP/QP.fProxy.cs:97, a production-surface question outside this partition).
   No `MatrixMetrics`/`StatsOP`/`BSM`/`Solvers`/`_OP` occurrences.
3. Missing InPlace suffix: benchmarks correctly treat `Eigen.valuesQR`
   (EigenSvdBenchmark.fProxy.cs:74) as destructive (re-copy Src each Execute) — corroborates
   the wide HIGH; no new instance found.
4. `[NoAlias]` violations: L5 above (read-only aliasing; open question).
5. Sibling-validation gaps: N/A (benchmarks call public entry points only).
6. Literal type keywords surviving substitution: none found — dtype-sensitive literals are
   consistently `(fProxy)`-cast; the deliberate `float` literals (PCG/SparseSolver density
   params and NextFloat noise streams) are documented as intentional cross-dtype-identical
   seeding, and LOBPCG's `tol=(fProxy)1e-20` is a valid normal float. `Consts.fProxySqrtEps`
   is used for all real tolerances (correctly per-type).
7. Comment-policy debt: M3, M4, L1, L2, L6 above; everything else in the folder is
   contract-only, and the folder DEVLOG.md is present, current, and correctly formatted.

## Areas confirmed clean

- Gemm/LU/QR/Iterative/SvdSolvers/Kernel/SmallSize/QRVariants/SvdComparison/TallWide/
  DirectSolve/Cholesky/EigenSvd/KMeans: clean on all dimensions (setup outside timing,
  results consumed, seeds deterministic, diagonal-dominance/SPD builders genuinely symmetric
  where claimed, XML/inline contracts match code).
- TriangularSolveBenchmark: exemplary — factor-once-outside-timing with an explicit contract
  comment; kernel-isolation rows correctly labeled.
- MPC/Kalman/LQR warm-vs-cold methodology: state-carry semantics, cold-reset fields, pre-warm
  frames, and drift-safety bounds all verified against the code; comments are accurate
  (the previously-wrong EKF drift comment is fixed exactly as DEVLOG records).
- MIP/LP/QP self-reporting outputs (length-1 arrays written inside Execute) verified: no
  second solve, statuses cross as int, honest-residual recomputes are correct L1 sums.
- `/*+choose[0|1]*/` double-only gating in MIPBenchmark (sections 1/3/4) verified against the
  float/double type order (float -> 0 cases, double -> 1) — correct for both variants.
- All `NextFProxy`/caps-proxy token uses expand to real `Unity.Mathematics.Random` API
  (NextFloat/NextDouble); no token misuse.

## Summary

| Severity | Count |
|----------|-------|
| HIGH     | 0     |
| MEDIUM   | 4     |
| LOW      | 9     |

MEDIUM: M1 PCG "SPD" builder asymmetry; M2 LOBPCG dense job missing cold-start reset;
M3/M4 measured numbers in comments (LP ~101ms, MIP ~275 nodes/~4307 iters) -> DEVLOG.
LOW: L1 budget narration, L2 spec-speak, L3 BurstCompile-option outlier, L4 dead objOut,
L5 [NoAlias] read-only aliasing (open question), L6 test-file names in printed output,
L7 Persistent Arena inside a job, L8 QP timed KKT check, L9 setup-matvec inconsistency.
