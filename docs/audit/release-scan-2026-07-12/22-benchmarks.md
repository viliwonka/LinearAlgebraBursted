# Release scan 2026-07-12 — area: benchmarks

Scanned 24 template files (bench). Findings: total 8 — confirmed 7, uncertain 1, unverified 0, refuted 0; severity: high 0, medium 0, low 8.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/CholeskyBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/DirectSolveBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/EigenSvdBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/FFTBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/GemmBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/IterativeBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/KMeansBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/KernelBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LOBPCGBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LPBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LQRBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LUBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LargeSparseBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/MIPBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/PCGBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/QPBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/QRBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/QRVariantsBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/SmallSizeBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/SparseSolverBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/SvdComparisonBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/SvdSolversBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/TallWideSolveBenchmark.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/TriangularSolveBenchmark.fProxy.cs

## Findings

### 1. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LPBenchmark.fProxy.cs:64 — Code comment records a benchmark bug-postmortem with a specific observed measurement, which the contracts-only comment policy requires to live in DEVLOG.md, not in code.

**Evidence**

```
"...report an internal objective BELOW the true residual -- impossible for a real sum-of-absolute-values -- which silently misled the benchmark table (observed: m=192 float revised printed 4.37 vs a true residual of 104.08)."
```

The comment embeds a bug postmortem and a concrete observed measurement in code, where the comment policy allows contracts only.

**Verifier**: Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LPBenchmark.fProxy.cs lines 59-65 contain the comment cited verbatim, including the concrete measurement "observed: m=192 float revised printed 4.37 vs a true residual of 104.08" and the narrative "silently misled the benchmark table". CLAUDE.md's comment policy is explicit: code comments state contracts only, while "bug postmortems and debugging narration" and "benchmark results, perf verdicts" belong in the folder's DEVLOG.md. The contract part (recompute residual because LPInfo.objective is only valid at optimum) is fine; the postmortem clause with the numeric observation is a policy contradiction. Fix direction is exactly the one proposed: trim to the contract, relocate the postmortem+numbers to TemplateSourceBenchmarks/DEVLOG.md.

**Suggested fix**: Reduce the comment to the contract (recompute the L1 residual from returned x because reported objective is only valid at a converged optimum); move the observed-values postmortem to the folder DEVLOG.md.

### 2. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LPBenchmark.fProxy.cs:178 — Comment references a reviewer/workflow event ('coordinator's sanity-scan'), a note-to-reviewers/agent-workflow reference the comment policy forbids in code.

**Evidence**

```
"...which the coordinator's sanity-scan explicitly called out (\"managed matvecs for residual columns\"). Not part of the timed measurement..."
```

The comment references an agent/reviewer workflow event, which the comment policy explicitly excludes from code.

**Verifier**: Line 178 literally contains the phrase "which the coordinator's sanity-scan explicitly called out (\"managed matvecs for residual columns\")" — a direct reference to a reviewer/workflow event. CLAUDE.md's strict comment policy explicitly forbids "notes to reviewers or references to agents/workflow ('coder report', 'third-review finding')" in code comments, and requires such narrative to live in DEVLOG.md. The surrounding lines 174-179 also contain development history ("Moved out of a plain managed Blas.dot(A, x0) call...") which is likewise policy-forbidden. Naming-severity is appropriate; the fix (state contract only, relocate the narrative to DEVLOG.md) matches the policy exactly.

**Suggested fix**: State only the contract (this matvec builds the RHS via a Burst job and is not timed); relocate the workflow/reviewer narrative to DEVLOG.md.

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LPBenchmark.fProxy.cs:609 — Comment references 'the review that requested this section', a reviewer/workflow reference disallowed by the contracts-only policy.

**Evidence**

```
"...infeasible by construction with no subtler failure mode to get wrong (the same robust recipe the review that requested this section specified)."
```

The parenthetical attributes the construction to a review request — reviewer provenance the policy keeps out of code comments.

**Verifier**: Lines 608-610 of LPBenchmark.fProxy.cs contain the parenthetical "(the same robust recipe the review that requested this section specified)". CLAUDE.md's strict comment policy explicitly forbids "notes to reviewers or references to agents/workflow" in code comments — such content belongs in a per-folder DEVLOG.md. The clause adds no contract information (the sentence already describes the infeasibility construction) and can be dropped without loss.

**Suggested fix**: Drop the reviewer-provenance clause; keep only the description of the infeasibility construction.

### 4. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LQRBenchmark.fProxy.cs:98 — Comment carries development history (an earlier value and why it was changed), which belongs in DEVLOG.md per the comment policy.

**Evidence**

```
"...the former fixed +-0.05 was only stable to n~12; at n=128 it produced unstable, likely unstabilizable instances). At n=4 this reproduces the original +-0.05 exactly."
```

The comment documents a former value and the reason for changing it — development history the policy routes to DEVLOG.md.

**Verifier**: Lines 95-99 of Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LQRBenchmark.fProxy.cs contain the sentence "the former fixed +-0.05 was only stable to n~12; at n=128 it produced unstable, likely unstabilizable instances). At n=4 this reproduces the original +-0.05 exactly." This literally references a former/original value and why it was changed — the exact pattern CLAUDE.md's strict comment policy names as development history that must live in DEVLOG.md, not in code comments. The current-contract portion (diagonal range, off-diagonal scaled 0.2/n, Gershgorin bound) is legitimate; only the historical tail violates policy. Fix direction (retain contract, move history to TemplateSourceBenchmarks DEVLOG.md) is correct.

**Suggested fix**: State only the current construction contract (diagonal range and off-diagonal scaling that keep the Gershgorin bound stable at all n); move the 'former fixed +-0.05' history to DEVLOG.md.

### 5. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LargeSparseBenchmark.fProxy.cs:140 — Comments throughout this file carry internal ticket/spec references and change-rationale ('Krylov R3', 'Q7 budget ruling', 'spec §3b', 'DELETED to pay for...'), all of which the policy routes to DEVLOG.md.

**Evidence**

```
Line 139-140: "...was DELETED to pay for the new PCG-SSOR row (Q7 budget ruling: cut redundancy, don't grow the report unboundedly)."  Also line 25 "Krylov R3:", line 322 "Krylov R3b budget trade (spec §3b, disclosed)".
```

Multiple comments in the file carry internal ticket tags, spec references, and change-rationale/budget bookkeeping in code.

**Verifier**: The file at lines 25, 42, 74, 90, 131-140, 238-242, and 322 carries exactly the categories CLAUDE.md forbids in code comments: internal ticket tags ("Krylov R3", "Krylov R3b", "Q7 budget ruling", "spec §3b"), development history ("grew a `tol` field replacing the hardcoded 0f", "was DELETED to pay for the new PCG-SSOR row", "PAID FOR by dropping N=5120", "gained PCG-SSOR"), and rejected-alternative/budget bookkeeping. A DEVLOG.md already exists next to the template, matching the policy's stated destination. Nothing about codegen expansion, contract documentation, or hidden guards makes these lines contractual — they are pure change-rationale, and the reviewer's suggested fix (strip and relocate) is aligned with the policy verbatim.

**Suggested fix**: Strip ticket tags (Krylov R3/R3b, Q7, spec §n) and row-budget bookkeeping from code comments; keep only what each section measures; move the rest to DEVLOG.md.

### 6. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/DirectSolveBenchmark.fProxy.cs:40 — Comment states a benchmark verdict/expectation ('any large gap would mean ... isn't vectorising as intended'), a perf-verdict note the policy keeps out of code.

**Evidence**

```
"...so this should run at roughly the same speed as the forward LU row -- any large gap would mean the right-looking TransA formulation isn't vectorising as intended."
```

The trailing clause is a performance expectation/verdict rather than a contract.

**Verifier**: Lines 39-41 of Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/DirectSolveBenchmark.fProxy.cs contain the exact prose the reviewer quoted: "so this should run at roughly the same speed as the forward LU row -- any large gap would mean the right-looking TransA formulation isn't vectorising as intended." CLAUDE.md explicitly forbids this class of content in code comments ("benchmark results, perf verdicts, rejected alternatives" belong in DEVLOG.md; code comments state contracts only). The preceding sentences that identify the job as the compact-form transposed solve counterpart are contract-legitimate, but the trailing speed expectation is a verdict. The claim is a genuine text-vs-policy contradiction.

**Suggested fix**: Keep the contract (this job times the compact-form transposed solve LU.decompInPlace + decompSolveTransA); move the speed-expectation verdict to DEVLOG.md.

### 7. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/SparseSolverBenchmark.fProxy.cs:843 — Comments tag work with an internal milestone reference ('Milestone B'), an internal ticket/phase reference the comment policy excludes from code.

**Evidence**

```
Line 843: "// Milestone B: transpose-optimized variants -- Aᵀ materialized ONCE (outside timing)..."  Also line 184 "(Milestone B): use a materialized Aᵀ...".
```

Internal milestone/phase tags appear in code comments.

**Verifier**: CLAUDE.md forbids internal spec/ticket/phase references in code comments (list explicitly includes "STAGE n"-style tags) and requires such history to go in DEVLOG.md. The file contains phase-tag comments at line 184 ("Milestone B"), line 647 ("Milestone-A"), and line 843 ("Milestone B") -- none of which add contract information beyond the neighboring text about Aᵀ materialization. Reviewer's claim is a genuine contradiction with the documented comment policy; suggested fix (keep only the contract that Aᵀ is materialized once outside timing) is correct, and the same cleanup should extend to line 647.

**Suggested fix**: Remove the 'Milestone B' phase tag; keep only the contract that Aᵀ is materialized once outside the timed region so ApplyT runs as a forward spMV.

### 8. [low/logical/UNCERTAIN] Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/KMeansBenchmark.fProxy.cs:26 — The cluster count k is hardcoded to 16 inside Execute() while the centroids and workspace buffers are sized to the parameter K, so any future change to K in the harness silently produces a buffer overflow instead of a compile error.

**Evidence**

```
KMeans.fit(in X, 16, 12345u, 10, KMeansInit.Uniform, ref centroids, ref assignment, ...) hardcodes k=16, but BenchFProxy sizes `centroids = arena.fProxyMat(K, D)` and `ws = arena.fProxyKMeansCache(n, D, K)` from the passed K. Currently harmless because the harness const K=16 (KMeansBenchmark.cs:16), but the two are uncoupled: if K != 16, fit writes 16 centroid rows into a K-row buffer.
```

The literal 16 in the fit call and the parameter K sizing the buffers are uncoupled; divergence would overflow silently.

**Verifier**: The mechanical claim is accurate: line 26 hardcodes 16 while lines 36/38 size buffers from parameter K. However there is no current failure — the sole caller (Assets/LinearAlgebra/Benchmarks/KMeansBenchmark.cs, const int K = 16) always passes 16, and the benchmark file's own header documents "Fixed... k = 16 clusters" as an intentional non-parameter, with the printed report string "k=16 clusters" also hardcoded (line 22). The claim admits it's currently harmless and requires a hypothetical future edit contradicting the fixed-k contract, so it's a real code smell / latent hazard but not a provable defect today.

**Suggested fix**: Pass K into the job as a field and use it in the fit call instead of the literal 16, so the requested cluster count and the buffer sizes are driven by one value.

## Scanner notes

Overall the benchmark templates are sound on the things that matter for a benchmark: setup/allocation is kept out of Bench.Time (only job.Run() is timed), destructive/in-place jobs re-copy their pristine source into the working buffer at the top of every Execute so repeated timed reps measure identical work, DCE is defeated (ping-pong x<->y in matvec/axpy jobs, sink[0]=acc in the reduction job, and decomposition outputs are external NativeArray stores Burst cannot prove dead), all timed jobs carry [BurstCompile(CompileSynchronously=true)] so there is no silent Mono fallback, and NativeArray/Pivot/Arena allocations are disposed on every path. I found no wrong-result, leak, overflow, or aliasing defect in normal use.

The comment-policy violations are pervasive, not limited to the eight lines cited above; MIPBenchmark, QPBenchmark, and the LP/LargeSparse headers similarly embed change-rationale, budget bookkeeping, and test-class name references in code comments. I reported the clearest representatives rather than every instance. These are all low severity (documentation hygiene against the project's contracts-only rule), not functional bugs.

One non-finding I considered and rejected: LOBPCGBenchmark.BenchFProxy uses tol=(fProxy)1e-20, which is unreachable in both float and double, so it always runs to maxIter. This is consistent with the fixed-iteration timing convention the other dense iterative benchmarks use (e.g. IterativeBenchmark's Krylov.cg(...,100,0f)), so it is a deliberate fixed-work measurement rather than a defect.
