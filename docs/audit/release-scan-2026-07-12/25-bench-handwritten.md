# Release scan 2026-07-12 — area: bench-handwritten (non-template)

Scanned 27 files (bench), counts: {"total":3,"confirmed":3,"uncertain":0,"unverified":0,"refuted":0,"high":0,"medium":0,"low":3}

## Scope

- Assets/LinearAlgebra/Benchmarks/AllBenchmarks.cs
- Assets/LinearAlgebra/Benchmarks/Bench.cs
- Assets/LinearAlgebra/Benchmarks/CholeskyBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/DirectSolveBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/EigenSvdBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/FFTBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/GemmBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/IterativeBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/KMeansBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/KernelBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/LOBPCGBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/LPBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/LQRBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/LUBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/LargeSparseBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/MIPBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/MultiRhsSolveBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/PCGBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/QPBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/QRBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/QRVariantsBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/SmallSizeBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/SparseSolverBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/SvdComparisonBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/SvdSolversBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/TallWideSolveBenchmark.cs
- Assets/LinearAlgebra/Benchmarks/TriangularSolveBenchmark.cs

## Findings

### 1. [low/naming/CONFIRMED] Assets/LinearAlgebra/Benchmarks/LPBenchmark.cs:105 — Class-header section enumeration is stale: omits Section 3 (sparse LAD) and Section 6 (warm re-solve) that the code actually runs.

**Evidence**

> The block comment at lines ~105-148 enumerates only '=== Section 1 (LP.solve)', 'Section 2 (LAD)', 'Section 2 (LAD, fast routes only)' [2b], 'Section 4 (dense covering LP)', 'Section 5 (infeasibility detection)'. There is no Section 3 or Section 6 heading in that enumeration, yet Section() calls SectionSparseLadFloat/Double (Section 3, described in the runtime prose at line 173) and SectionWarmResolveFloat/Double (Section 6, described at line 184). The enumerated doc block skips straight from 2b to 4 and stops at 5.

**Verifier**

Verified against the file at Assets/LinearAlgebra/Benchmarks/LPBenchmark.cs. The class-header block comment spanning lines 105-154 enumerates only Section 1 (lines 108-112), Section 2 (lines 114-127), Section 2b (lines 129-133), Section 4 (lines 135-140), and Section 5 (lines 142-148). Section 3 and Section 6 are absent — the enumeration jumps from 2b to 4 and stops at 5. Yet the same class actually ships those sections: (a) the runtime prose block inside Section(StringBuilder sb) explicitly describes "Section 3: SPARSE LAD" (line 173) and "Section 6: warm re-solve chain" (line 184); (b) Section(sb) calls SectionSparseLadFloat/Double (lines 194-195) and SectionWarmResolveFloat/Double (lines 200-201); (c) LPBenchmarkFmt itself references "Section 6 (warm re-solve chain)" at line 92 and Section 3 concepts (SparseLadDenseCap etc.) at lines 44-51 with the comment "exactly like SparseLadDenseCap stops the dense interior-point baseline in Section 3 below" at line 22. The doc block therefore is provably stale relative to both the runtime prose it's supposed to summarize and the code it heads. Severity is appropriately low (documentation-only; no runtime behavior affected), but the defect is real. Suggested fix direction from the claim (insert Section 3 and Section 6 entries between 2b/4 and after 5 respectively) is correct.

**Suggested fix**

Add the missing 'Section 3 (sparse LAD ...)' and 'Section 6 (warm re-solve chain ...)' entries to the class-header enumeration so it matches the runtime prose and the SectionSparseLad*/SectionWarmResolve* calls.

### 2. [low/logical/CONFIRMED] Assets/LinearAlgebra/Benchmarks/AllBenchmarks.cs:3 — Doc claims 'every kernel section' is aggregated, but SvdComparisonBenchmark.Section is not called in the combined report.

**Evidence**

> Line 3 states 'Runs every kernel section into one combined report (TestResults/benchmark-all.txt).' The Run() body (lines 17-40) invokes Section on every other benchmark class (Kernel, Gemm, LU, ... SvdSolvers, ... LQR) but never calls SvdComparisonBenchmark.Section, even though that class exposes a public Section(StringBuilder) (SvdComparisonBenchmark.cs line 60). So the SVD method comparison never appears in benchmark-all.txt.

**Verifier**

Verified against source. AllBenchmarks.cs:3 declares "Runs every kernel section into one combined report (TestResults/benchmark-all.txt)." The Run() body (lines 17-40) explicitly invokes .Section(sb) on 24 benchmark classes (KernelBenchmark ... LQRBenchmark) but omits SvdComparisonBenchmark, whose SvdComparisonBenchmark.cs:60 defines a matching public Section(StringBuilder sb) with the same shape as every aggregated sibling. A repo-wide grep for SvdComparisonBenchmark.Section returns only the standalone SvdComparisonBenchmark.Run() call (line 58, writing benchmark-svd-compare.txt) and the audit doc itself — nothing wires it into AllBenchmarks or any generated partial. Considered defenses: (a) "kernel"-means-throughput is not stated, and sibling SVD sections EigenSvdBenchmark/SvdSolversBenchmark are included; (b) the Section overload is a plain public method mirroring the others (dispatches to BenchSizeFloat/BenchSizeDouble internally); (c) the codegen template SvdComparisonBenchmark.fProxy.cs only extends the partial with per-dtype measure methods, does not touch AllBenchmarks. Doc claim overstates coverage. Low severity, documentation/UX only, no numerical or memory impact.

**Suggested fix**

Either add SvdComparisonBenchmark.Section(sb) to the combined Run(), or narrow the doc wording (e.g. 'runs the throughput kernel sections; the SVD accuracy comparison is standalone via SvdComparisonBenchmark.Run').

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/Benchmarks/DirectSolveBenchmark.cs:6 — Class-header solver list is incomplete relative to the rows actually emitted (omits the TransA rows).

**Evidence**

> The header (lines 6-14) lists the benched entry points as 'LU.decompSolve, CHO.decomp+decompSolve, QR.solveInPlace (square) ... The QR-cache variant', but Section() also emits LuSolveTransAFloat/Double rows (lines 33-34), i.e. LU.decompInPlace + LU.decompSolveTransA, which are only mentioned in the inline comment at lines 29-32, not the top-of-class summary.

**Verifier**

Verified against C:\Users\viliv\Documents\LinearAlgebraBursted\Assets\LinearAlgebra\Benchmarks\DirectSolveBenchmark.cs. The class-header comment (lines 5-10) enumerates entry points as "LU.decompSolve, CHO.decomp+decompSolve, QR.solveInPlace (square) ... The QR-cache variant." Section() (lines 23-42), however, appends ten rows including LuSolveTransAFloat(N) and LuSolveTransADouble(N) at lines 33-34, which correspond to LU.decompInPlace + LU.decompSolveTransA. That TransA path is only mentioned in the inline block comment at lines 29-32, never in the top-of-class summary. Result: a reader relying on the header summary to know what the benchmark section covers will not learn that a TransA LU row is emitted. Severity is appropriately low (documentation/naming completeness only; no runtime or numerical impact), matching the claim. Suggested fix (extend the class-header enumeration to include LU.decompSolveTransA) is consistent with the actual gap.

**Suggested fix**

Add 'LU.decompSolveTransA' to the class-header solver enumeration so the summary lists all rows the section produces.

## Scanner notes

Verified against generated output where a header/row shape mismatch was plausible: (1) LOBPCGBenchmark.cs's two tables use opposite min/med column order, but each matches its generated row (BenchFloat prints stat.Min,stat.Median; BenchSparsePrecondFloat prints sN.Median,sN.Min) - not a defect. (2) TallWideSolve/QRVariants main sections use Bench.Header+Bench.Row (7 cols) while only the rank-deficient sub-sections use HeaderKernel/RowKernel (8 cols) - consistent. (3) MultiRhsSolveBenchmark QRCP loop uses the default-tol overload which forwards to tol=-1 (QRCP.float.cs:988), the same value the block overload passes explicitly - loop and block are comparable, not a defect. Bench.cs median/mean/min/max and GFLOP/s (flops/(median/1000)/1e9) are correct; Runs=4 even-count median averages the two central samples correctly. All 8 hand-written jobs in MultiRhsSolveBenchmark carry [BurstCompile(CompileSynchronously=true)]; every Run* disposes its Arena and any Pivot; setup (FillGen/FillSpd/FillRhs) is outside Bench.Time; per-run copies inside the jobs are intentional and symmetric across loop/block. Flop formulas (GEMM 2N^3, LU 2/3 N^3, Chol 1/3 N^3, QR 4/3 N^3, QrFlops(2k,k)=10/3 k^3, TallFlops 2n^2(m-n/3)) all match their comments and the AllBenchmarks summary line. SparseSolverFmt block choosers are index-correct with injective dedup keys and terminating loops. No high/medium correctness, leak, or measurement defects found in the hand-written halves.
