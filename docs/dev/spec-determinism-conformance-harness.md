# Spec: cross-CPU determinism conformance harness

Status: ready for implementation (coder agent).
Origin: user design 2026-07-15 (memory: determinism-conformance-harness). Internal doc — `docs/dev/` is exempt from the public-docs prose bar.

## 1. Goal

A headless, test-like harness that exercises the library's op families as ~25 GROUPS, hashes every
op's output bytes with a bit-exact integer hash, folds op hashes into one hash per group and group
hashes into one ROOT hash — a two-level Merkle-like tree (op → group → root, non-binary). Run it on
two machines (x86 vs ARM, different microarch), `diff` the two text reports: a root mismatch says
"something diverged", the group line localizes the capability, the op lines pinpoint the culprit op.
Re-run on the same machine after a kernel change = regression guard (report must be byte-identical).

This is NOT tolerance-based testing. Determinism is bit identity of output buffers.

## 2. Where the code lives (decision + justification)

The harness is a benchmark-style tool, not a shipped feature and not an NUnit test:

- **Generated per-dtype halves** (the Burst jobs + input builders + case runners, one float + one
  double instantiation each): templates in
  `Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/Determinism*.fProxy.cs`.
  Rationale: the codegen pipeline has exactly three fixed template→output mappings
  (`TemplateSource → Source`, `TemplateSourceTests → SourceTests/Generated`,
  `TemplateSourceBenchmarks → Benchmarks/Generated` — see `Tools/regen.ps1` and
  `Tools/CodegenBootstrap/`). `TemplateSource` would ship the harness in the UPM package (package
  root is `Assets/LinearAlgebra/Source`) — wrong. `TemplateSourceTests` makes it an NUnit suite —
  wrong shape: we need an `-executeMethod` report writer, and the Benchmarks assembly
  (`BurstLinearAlgebra.Benchmarks`, Editor-only asmdef) already has exactly that wiring
  (`AllBenchmarks.Run` + `Tools/benchmark.ps1`). So: benchmarks tree.
- **Hand-written dtype-agnostic half** (report writer, group registry, root folding, entry point):
  `Assets/LinearAlgebra/Benchmarks/DeterminismReport.cs` (new, hand-authored, same split as e.g.
  `CholeskyBenchmark.cs` hand half + `Benchmarks/Generated/CholeskyBenchmark.{float,double}.cs`
  generated half).
- **PS wrapper**: `Tools/determinism-report.ps1` (new, hand-authored, modeled on
  `Tools/benchmark.ps1` minus CPU-affinity pinning — hashes are timing-independent).
- Never hand-edit anything under `Benchmarks/Generated/` — edit the template, run `Tools/regen.ps1`.
- Comment policy: code comments state contracts only. Design rationale, divergence postmortems,
  observed cross-arch results go to `Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/DEVLOG.md`
  (create the entry under a `## Determinism harness` heading), not into code comments.

Suggested template file split (mirrors benchmark granularity; exact split is the coder's call, keep
each file reviewable):

- `DeterminismDirect.fProxy.cs` — BLAS, norms, QR/QRCP/LQ/LQRP, LU, CHO/CHOP, triangular
- `DeterminismEigenSvd.fProxy.cs` — eigen sym/nonsym, SVD family, LOBPCG
- `DeterminismIterativeSparse.fProxy.cs` — dense Krylov, BSR ops, preconditioned Krylov
- `DeterminismOptimize.fProxy.cs` — LP/LAD, QP, MIP, NLS/curveFit, control (LQR/Riccati/Kalman/MPC)
- `DeterminismStatsMl.fProxy.cs` — stats, histogram/resample, query, k-means, PCA, gallery, FFT
- `DeterminismNativeSensitive.fProxy.cs` — section B (DetMath, transcendental elementwise, samplers,
  softmax, dft/signal)
- `DeterminismInt.iProxy.cs` — integer-family group (iProxy)

## 3. Hashing scheme

**Reuse the library's own `LinearAlgebra.Hash` (xxHash32)** — do not add a new hash:

- `Hash.hash(byte* data, int byteLength, uint seed = 0)` — core kernel,
  `Assets/LinearAlgebra/CodeGen/TemplateSource/Hash/Hash.Shared.cs`. Word assembly is an explicit
  little-endian byte combine (`ReadLE32`), i.e. the algorithm is fixed regardless of host
  endianness; all Unity/Burst targets are little-endian anyway, so hashing raw buffer bytes is
  well-defined and identical across x86/ARM **iff the bytes are identical** — which is exactly the
  property under test.
- Typed wrappers exist per element type: `Hash.hash(in floatN)`, `Hash.hash(in doubleMxN)`,
  `Hash.hash(in intN)`, … (generated from `Hash/Hash.fProxy.cs`, `Hash.iProxy.cs`, `Hash.bool.cs`).
- `Hash.combine(uint a, uint b)` — order-sensitive fold; this is the tree combiner.

Definitions (fix these exactly; they are the frozen report contract):

- **op hash** = the op's output buffers hashed in a fixed, documented order. Multiple outputs fold
  left-to-right: `h = Hash.hash(buf0); h = Hash.combine(h, Hash.hash(buf1)); …`. Scalar outputs
  (a returned float/int, `Info.iterations`, a status enum) are folded in as their raw bits:
  `Hash.combine(h, math.asuint(scalar))` / `Hash.combine(h, (uint)info.iterations)`. Which buffers
  and scalars participate is part of each case's contract — hash everything the op promises to
  produce (including Pivot/Indices integer buffers), nothing it doesn't.
- **group hash** = fold of the group's op hashes in registration order, starting from the seed
  `0x9E3779B9u`: `g = seed; foreach op: g = Hash.combine(g, opHash)`.
- **root hash** = same fold over group hashes in registration order (section A groups only).
  Section B (below) folds into its own separate `ROOT-B` and never touches the main root.

Hashing happens **inside the Burst job** (the job writes the final `uint` into a
`NativeArray<uint>` result slot), so the hashed bytes are the native-code results, never a managed
copy. `Hash` is part of the library, so calling it inside the job is natural.

Bit-hash caveats are features here: `-0.0` vs `+0.0` hash differently, distinct NaN payloads hash
differently — that is the sensitivity we want.

## 4. Registration structure

Keep it as dumb as the benchmarks — no reflection, no attributes, no registry framework:

- Each generated case is a method `static uint Case_<name>FProxy()` in the generated half:
  builds fixed inputs (arena, seeded), runs one `[BurstCompile]` IJob whose `Execute()` calls the
  library op(s) and computes the op hash into `NativeArray<uint>`, disposes, returns the hash.
- The hand-written `DeterminismReport.cs` half holds the group list as plain code, in fixed order:

```csharp
// Sketch — hand-written half (dtype-agnostic).
static void Group(StringBuilder sb, ref uint root, string name,
                  params (string id, Func<uint> run)[] cases)
{
    uint g = 0x9E3779B9u;
    var lines = new List<string>();
    foreach (var (id, run) in cases)
    {
        uint h = run();
        g = Hash.combine(g, h);
        lines.Add($"OP {name}/{id} {h:x8}");
    }
    root = Hash.combine(root, g);
    sb.AppendLine($"GROUP {name} {g:x8}");
    foreach (var l in lines) sb.AppendLine(l);
}
```

- Case ids are stable strings encoding op + dtype + shape, e.g. `qr.decompSolve.f32.53x37`,
  `cg.laplacian2d.f64.n1024`. Never reuse an id for different work.
- `public const int HarnessRev = 1;` in `DeterminismReport` — bump on ANY change to the case list,
  case inputs, or fold order (all of these legitimately change hashes). Printed in the header so a
  cross-machine diff of different revisions is recognized as meaningless.

Job attribute — on every harness job, exactly:

```csharp
[BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
```

FloatMode on the job governs the whole inlined call graph: library methods carry no FloatMode of
their own (the few `[BurstCompile]` attributes inside `Source/` are bare, on data-movement helpers
like `UnsafeOP.swapRows`, and on Info structs). `FloatMode.Strict` is the library's cross-arch
determinism contract (`+ - * / sqrt` only; kernels' fixed summation trees are Strict-safe by
construction — see `Source/OP/UnsafeOP.float.cs` header comment).

## 5. Binding determinism rules (the traps)

1. **Fixed seeds, no wall clock.** All randomized inputs use `Unity.Mathematics.Random` with
   literal seeds (benchmark convention: `new Random(2654435761u ^ (uint)n)` — fine; any fixed
   literal is fine). Never `DateTime`, `Environment.TickCount`, `GetHashCode`, iteration timing.
2. **Pre-fill every hashed output buffer** with a fixed pattern (e.g. `fillInPlace` with a constant,
   or zero) BEFORE the op runs. Ops that leave part of a buffer unwritten (unused triangles,
   truncated columns, early-exit iterative solvers) must not leak uninitialized arena bytes into a
   hash — that would be nondeterministic on ONE machine and poison the whole tree.
3. **Hash only defined output.** If an op documents that a region is scratch/undefined, either
   pre-zero it (rule 2 makes its contribution constant) or exclude it from the hashed byte range.
4. **Fixed sizes, awkward on purpose.** Use non-power-of-two dims to exercise SIMD tails (e.g.
   53×37, n=48, n=1000) plus one power-of-two FFT size (n=256) and one mixed size (n=192 = 2^6·3).
   Sizes small — the whole run must stay in the low minutes; this measures bits, not speed.
5. **Iterative solvers**: fixed `maxIter`/`tol` arguments; fold `info.iterations` and status into
   the op hash (an arch that converges in a different number of iterations is itself a divergence
   signal).
6. **Struct-functor inputs defined in the harness** (NLS model, predicate/sampler functors) must be
   transcendental-free unless the case is in section B — use polynomial models for NLS/curveFit
   (the library's built-in `ResidualFunction` models use `DetMath`, i.e. are native-sensitive;
   define a local polynomial functor instead).
7. **The report contains no timestamps, timings, hostnames, or paths in the hashed sections** —
   the acceptance criterion "re-run ⇒ byte-identical file" forbids them anywhere in the file.
   Environment facts (arch, Unity/Burst versions, Burst on/off, DetMath mode) go in `#`-prefixed
   header lines; they legitimately differ across machines, and `diff` shows them as context. No
   date/time line at all.
8. Run all jobs with `.Run()` on the main thread (benchmark convention) — no scheduling
   nondeterminism, and it dodges the job-struct-copy trap for ping-pong caches (LOBPCG-class bug).

## 6. Group list

Verified against the templates (class names and member names below all exist; the coder should
re-verify exact overload arity when writing calls — templates under
`Assets/LinearAlgebra/CodeGen/TemplateSource/`). Every group runs each case for BOTH `float` and
`double` (the two generated halves) except where noted. Target ≈ 22 groups in section A, 6 in
section B — inside the 20–30 budget; merging or splitting adjacent groups is allowed if a group
gets unwieldy, but bump `HarnessRev` accordingly.

### Section A — deterministic contract (folds into ROOT; any cross-arch diff is a BUG)

| # | group id | ops (namespace `LinearAlgebra`) | input recipe |
|---|----------|--------------------------------|--------------|
| 1 | `hash-selftest` | `Hash.hash(byte*)` over a fixed 0..255 pattern, `Hash.hash(in fProxyN)`, `Hash.rowHashes`/`colHashes`, `Hash.combine` chain | fixed byte pattern; ALSO compare against hardcoded expected constants — if the hash itself is broken on a platform, the report must say so (`OP hash-selftest/known-answer` emits `FAIL` sentinel hash `00000000` on mismatch) |
| 2 | `blas-dense` | `Blas.dot(vec,vec)`, `dot(A,x)` matvec, `dot(y,A)` vecmat, `dot(A,B,ref C, transposeA, transposeB)` GEMM (all 4 transpose combos), `dotSym`, `outerDot`, `trans` | seeded uniform fills, 53×37 / 37×29 |
| 3 | `elementwise-core` | `Comp` family (`fProxyComp` per-dtype: `floatComp`/`doubleComp`): abs, sqrt, clamp, lerp, min, max, sign, mad, floor/ceil/round, saturate — the `+ - * / sqrt`-only subset | seeded vector n=1000 incl. negatives, denormals, ±0 |
| 4 | `norms` | `Norms.L1/L2/LInf/matrixL1/matrixL2/matrixLInf`, `normalize`, `normalizeRows/Columns` | seeded 53×37 |
| 5 | `stats-core` | `Stats.sum/mean/variance/stdDev/min/max/median/covariance/correlation`, row/col reductions (`rowMean`, `colStdDev`, …), `standardize`, `center`, `rescale` | seeded 200×7 (sqrt-only ⇒ section A) |
| 6 | `qr-family` | `QR.decomp`, `QR.decompSolve`; `QRCP.decompSolve`, `QRCP.minNormSolveInPlace`; `LQ.decomp`, `LQ.minNormSolve`; `LQRP.minNormDecompSolve` | seeded tall 53×37 and wide 37×53; hash Q/R/L factors, solution, and pivot Indices |
| 7 | `lu` | `LU.decomp`, `LU.decompInPlace(+Pivot)`, `LU.decompSolve`, `LU.decompSolveTransA` | seeded 48×48 diagonally-dominated; hash factors + Pivot buffer + x |
| 8 | `cholesky` | `CHO.decomp`, `CHO.decompSolve`; `CHOP.decomp(+Pivot,+ws)`, `CHOP.solveInPlace` | gallery SPD (seeded symmetric + diagonal dominance, benchmark recipe), n=48 |
| 9 | `eigen-sym` | `Eigen.symmetricInPlace`, `Eigen.valuesSymmetricInPlace`, `Eigen.lanczos` | `Arena.fProxyWilkinsonPlus` / seeded SPD, n=48; hash eigenvalues + eigenvectors |
| 10 | `eigen-nonsym` | `Eigen.decompInPlace`, `Eigen.valuesQRInPlace`, `Eigen.powerIteration`, `Eigen.inversePowerIteration` | `fProxyFrank` / `fProxyGrcar` gallery, n=32 |
| 11 | `svd` | `SVD.thin`, `SVD.values`, `SVD.truncated` (GKL), `SVD.randomized` (seeded), `SVD.pinvSolve`, `SVD.pseudoInverse`, `SVD.nullspaceBasis` | seeded 53×37; `fProxyLauchli` for the rank-deficient case |
| 12 | `fft` | `FFT.fft/ifft/rfft/irfft` with `fProxyFFTCache` workspace; `FFT.magnitude`, `FFT.powerSpectrum` | seeded signals n=256 and n=192 (mixed radix); workspace build is `+ - * sqrt`-only ⇒ section A. `FFT.phase` is NOT here (atan2 ⇒ section B) |
| 13 | `krylov-dense` | `Krylov.cg`, `minres`, `biCGStab`, `lsqr`, `lsmr`, `cgls` (dense overloads) | seeded SPD n=64 (cg/minres), seeded 96×48 (LS solvers); fixed maxIter/tol; fold `SolveInfo.iterations` + status |
| 14 | `sparse-bsr` | `BSR.spMV`, `spMVT`, `spMM`, `sweepLower/sweepUpper`; BSR assembly itself (hash the assembled block/index buffers) | `Arena.fProxyLaplacian2D` (32×32 grid, N=1024) and seeded `fProxyRandomSparseSPD` |
| 15 | `krylov-sparse-precond` | `Krylov.pcg` with `fProxyBlockJacobi`, `fProxySSOR`, `fProxyIC0`; `fProxyILU0` via its shipped solver path; hash preconditioner factor buffers AND solution | same Laplacian2D; fold iterations |
| 16 | `lobpcg` | `LOBPCG.lobpcg` k=4 | Laplacian1D (well-separated spectrum — avoid the known clustered-spectrum breakdown); hash eigenvalues + eigenvectors + info |
| 17 | `lp-lad` | `LP.solve` (default revised simplex; plus the IPM path via its options route), `LP.lad` (both BR small-m and FN large-m sides: one small, one large instance) | small seeded LP (m=20,n=30), seeded LAD regression 120×5 |
| 18 | `qp` | `QP.solve` (model via `QP.Create`) | small seeded strictly-convex QP n=24, few constraints |
| 19 | `mip` | `MIP.solve` | tiny fixed instance (stein9-class or hand-fixed 10-var knapsack); hash incumbent x, objective bits, node count. Double-only if float parity is not provided by existing MIP surface — mirror what MIPBenchmark instantiates |
| 20 | `control` | `LQR.lqr`, `LQR.lqrSchedule`, `Riccati.dare`; `Kalman.predict/update/steadyStateGain`; `MPC.solve` | fixed small state-space (4-state, 2-input, literal constants); MPC: short horizon, cart-pole-like fixed matrices |
| 21 | `nls-optimize` | `Optimize.nlsSolve` (harness-local polynomial residual functor), `Optimize.curveFit` (polynomial model), `Optimize.ladIRLS` | fixed xdata/ydata arrays (literal or seeded); fixed iteration caps |
| 22 | `ml` | `KMeans.fit` (k-means++, seeded), `PCA.fitCov/fitSvd/fitSvdTruncated/fitRandomized(seeded)`, `PCA.transform` | seeded 200×8; hash centroids/assignments, PCA model buffers |
| 23 | `histogram-resample-query` | `Histogram.histogramInto/cdfInto`, `Resample.resampleInto` (nearest/linear/Catmull-Rom), `Query.nearestRow/kNearestRows/rowArgMax/rowsWithinRadius`, `Select.select` | seeded inputs; int outputs hashed via int wrappers |
| 24 | `gallery-analysis` | transcendental-free gallery generators hashed directly: `fProxyHilbert`, `fProxyLehmer`, `fProxyKMS`, `fProxyPascal`, `fProxyMinIJ`, `fProxyLaplacian1D/2D`; `Analysis.isSymmetric/isOrthogonal/isDiagonal` (bool→hash) | n=32. Coder: grep which `Gallery.Special` generators call `DetMath` — those go to group 30, not here |
| 25 | `int-family` (iProxy + uint; no float) | int/uint `dot`, `Norms` (iProxy), `Stats` (iProxy), Pivot/Indices round-trip, `Rand.nextUniformInPlace` + `weightedPick` (integer-state xorshift + `* /` only ⇒ exact) | seeded; single dtype set (int/uint as generated by iProxy templates) |

### Section B — native-math-sensitive (folds into ROOT-B only; EXPECTED to diverge across arch when built with `LINALG_NATIVE_MATH`; expected to MATCH in the default DetMath build)

Background: as of the DetMath routing (2026-07-15/16), every transcendental call site in the library
routes through `DetMath` (verified: `math.exp/log/sin/cos/tan/atan/pow` appear ONLY inside
`OP/DetMath.fProxy.cs`; DetMath users: UnsafeMathOP, StatsCore, RandomOP, RandomMatrixOP, GenOP,
Wave, Easing, FFT(dft), Gallery.Special, Analysis.Metrics, ResidualFunction, ArenaExtensions,
UnsafeOP). In the default build these are cross-arch deterministic like section A. Under
`LINALG_NATIVE_MATH` (`DetMath.UseNative == true`, a `public const bool`) they flip to `math.*` and
WILL diverge. Keeping them in a separate subtree means a native-math build perturbs only `ROOT-B`
and the main ROOT stays comparable. The report header records the mode.

| # | group id | ops | notes |
|---|----------|-----|-------|
| 26 | `detmath` | `DetMath.Exp/Exp2/Exp10/Log/Log2/Log10/Pow/Sin/Cos/SinCos/Tan/Atan/Atan2/Asin/Acos/Sinh/Cosh/Tanh/Acosh` | fixed grid of ~64 inputs per fn spanning domains (incl. edge values); hash the output vector |
| 27 | `elementwise-transcendental` | `Comp` exp/log/sin/cos/tan/atan2/tanh/pow family; ALSO `Comp.rsqrt` and `Comp.fmod` | rsqrt/fmod are raw `math.rsqrt`/`math.fmod` (the only non-DetMath math.* left, in `UnsafeMathOP`) — keep them here conservatively until verified exact cross-arch; note result in DEVLOG |
| 28 | `random-samplers` | `Rand` ICDF samplers (`WeibullICDF`, `CauchyICDF`, `LogisticICDF`, `ParetoICDF`, `ExponentialICDF`, `RayleighICDF`, `TriangularICDF`, `UniformICDF`), sampler struct-functors, `Rand.multivariateNormalInPlace`, `orthogonalInPlace`, `spdInPlace`, `conditionedInPlace`, `withRankInPlace` | seeded `Unity.Mathematics.Random`; the ICDFs use DetMath.Log etc. |
| 29 | `softmax` | `Stats.softmax/softmaxRows/softmaxColumns` | seeded 53×37 |
| 30 | `dft-signal` | `FFT.dft/idft`, `FFT.phase`, `Generate.window` (each `WindowType`), `Generate.gaussianKernel/gaussianKernel2D`, `fProxyWave` + `fProxyEasing` sampled over a fixed grid, DetMath-dependent `Gallery.Special` generators (from group-24 triage) | fixed signals |
| 31 | `ukf` | `Kalman.ukfPredict/ukfUpdate` with a harness-local model | put in B only if the sigma-point math or the chosen model needs DetMath; if a linear model keeps it `+ - * / sqrt`-only, move to group 20 and record the decision in DEVLOG |

## 7. Report format

Written to `TestResults/determinism-report.txt`, UTF-8 no BOM (codegen/benchmark convention), LF or
CRLF — pick one explicitly (`\n`) and always use it, so byte-identity holds across writes.

```
=== LinearAlgebra determinism conformance report ===
rev 1
# host: Windows 11 / X64 / Unity 6000.0.32f1 / Burst 1.8.18        (informational; differs across machines)
# burst-enabled: True
# detmath-native: False
# dtypes: float double

ROOT 4f1d9c2a
ROOT-B 913bb0e7

GROUP hash-selftest 7be21c04
OP hash-selftest/known-answer.bytes 89abcdef
OP hash-selftest/vec.f32.n1000 0134f00d
...
GROUP blas-dense 5a77d001
OP blas-dense/dot.vv.f32.n1000 c0ffee12
OP blas-dense/gemm.nn.f64.53x37x29 deadbeef
...
=== section B: native-math-sensitive (expected to differ across arch under LINALG_NATIVE_MATH) ===
GROUP detmath 22aa8844
OP detmath/exp.f32 12345678
...
```

Rules:

- `rev` line: `HarnessRev`. Two reports with different revs must not be diffed; the compare script
  refuses.
- `#` lines are informational context, never hashed, never used by the comparer.
- Hash lines are exactly `ROOT <hex8>`, `ROOT-B <hex8>`, `GROUP <id> <hex8>`,
  `OP <group>/<case> <hex8>` — one token of whitespace, lowercase hex, fixed width 8. All op hashes
  are always emitted (a single machine cannot know what will mismatch; the cross-machine diff does
  the localization for free).
- No timings, no dates, no absolute paths anywhere in the file.
- Section B is visually fenced with the `=== section B ... ===` line so a human reading a diff can't
  mistake an expected divergence for a bug.
- The entry method also `Debug.Log`s the report and logs
  `Determinism report written to <path>` on its own line (the wrapper parses it — same contract as
  `Bench.WriteReport`, see `Assets/LinearAlgebra/Benchmarks/Bench.cs`). Do NOT reuse
  `Bench.WriteReport` itself (its preamble includes warmup/runs prose); write a dedicated writer in
  `DeterminismReport.cs`.

## 8. Entry point + wrapper

- Entry: `LinearAlgebra.Benchmarks.DeterminismReport.Run` — `public static void Run()`, invoked via
  `-executeMethod`. If `BurstCompiler.Options.EnableBurstCompilation` is false, still write the
  report (header records it) but ALSO log a `Determinism report FAILED: Burst disabled` line — a
  Mono run does not validate Burst codegen; the wrapper turns that into exit 1.
- Optional editor menu item (nice-to-have, tiny): a `[MenuItem("Tools/LinearAlgebra/Determinism Report")]`
  wrapper calling `Run()` — put it in the Benchmarks assembly (it is Editor-only already).
- `Tools/determinism-report.ps1` — copy the structure of `Tools/benchmark.ps1`:
  - dot-source `Tools/_unity-common.ps1`; run `Tools/regen.ps1` first (skippable via `-NoRegen`);
  - `Invoke-Unity -Arguments @("-nographics","-quit","-executeMethod","LinearAlgebra.Benchmarks.DeterminismReport.Run") -LogFile TestResults\determinism.log` (no affinity mask);
  - fail on compile errors (`Get-CompileErrors`), on "could not find method", and on the
    `Determinism report FAILED` marker;
  - parse `Determinism report written to <path>`, echo the file via `[System.IO.File]::ReadAllText`
    (PS 5.1 BOM-less UTF-8 trap — do not `Get-Content`);
  - `-Compare <pathA> <pathB>` mode (no Unity run): refuse on differing `rev` lines, then diff only
    `ROOT*/GROUP/OP` lines and print the first diverging GROUP and its diverging OP lines,
    section-B divergences labeled `expected under native-math builds`. Exit 1 on any section-A
    divergence, 0 otherwise.

## 9. Acceptance criteria

1. `Tools/regen.ps1 -Check` clean after committing templates + regenerated files (generated files
   are committed, benchmark-tree convention).
2. `Tools/determinism-report.ps1` runs headlessly (Editor closed) and writes
   `TestResults/determinism-report.txt` containing: `rev`, `ROOT`, `ROOT-B`, every section-A and
   section-B `GROUP` line, and every `OP` line.
3. **Byte-identity guard**: running the wrapper twice in a row on the same machine yields
   byte-identical report files (`fc.exe /b` or hash compare — the wrapper's own `-Compare` must also
   report zero divergences). This is the regression-guard property and catches any uninitialized-
   memory leak into a hash (rule 5.2).
4. `hash-selftest` known-answer case passes (non-sentinel hash) on the reference machine.
5. Whole run completes in low minutes (sizes are small; this is not a benchmark).
6. Full test suite still green (`Tools/run-tests.ps1`) — the harness must not touch library code.
7. No comment-policy violations: no dev history/benchmarks in code comments; DEVLOG entry added
   under `TemplateSourceBenchmarks/DEVLOG.md`.

## 10. Verification steps (coder runs these)

```powershell
Tools\regen.ps1                       # templates -> Benchmarks/Generated, drift check
Tools\determinism-report.ps1          # run 1
Copy-Item TestResults\determinism-report.txt TestResults\determinism-report.run1.txt
Tools\determinism-report.ps1 -NoRegen # run 2
Tools\determinism-report.ps1 -Compare TestResults\determinism-report.run1.txt TestResults\determinism-report.txt
Tools\run-tests.ps1                   # suite unaffected
```

Cross-arch validation (x86 vs ARM editor) is a user step, out of scope for the coder; the deliverable
is the mechanism.

## 11. Open questions / decisions left to the user

- **FloatMode.Strict vs Default section**: the harness runs Strict per the design. Benchmarks and
  tests compile the library under `FloatMode.Default`. If Burst's Default is not bit-equivalent to
  Strict on some arch, a customer running Default could diverge where the harness says "conformant".
  Option (cheap): duplicate 2–3 sentinel groups (blas-dense, lu, fft) as `*-defaultmode` jobs under
  `FloatMode.Default` inside section A. Not specced as mandatory — user call.
- **Player-build runner**: the Benchmarks asmdef is Editor-only, so ARM coverage means an
  Apple-Silicon (or Windows-on-ARM) editor. A standalone-player runner for Android/iOS devices is
  out of scope v1.
- **`Comp.rsqrt`/`Comp.fmod`** placement (section B conservatively) pending a cross-arch check —
  promote to section A if they prove exact; record in DEVLOG.
- **MIP float instantiation**: if the MIP surface is effectively double-only for exact instances
  (MIPLIB history), group 19 may be double-only; mirror what `MIPBenchmark` instantiates.
- **Native-math conformance run**: generating a second report under `LINALG_NATIVE_MATH` requires a
  scripting-define toggle in the wrapper (`-NativeMath` flag adding the define via a temp
  `csc.rsp`/ProjectSettings edit). Deferred — v1 records the mode in the header only.
