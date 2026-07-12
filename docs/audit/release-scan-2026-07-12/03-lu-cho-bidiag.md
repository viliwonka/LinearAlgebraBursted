# Release scan 2026-07-12 — area: lu-cho-bidiag

Scanned 6 template files (core). Findings: total 4 — 4 confirmed, 0 uncertain, 0 unverified, 0 refuted; severity: 0 high, 2 medium, 2 low.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LU.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/CHO.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/CHOP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/CHOP.Workspace.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Bidiag.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Bidiag.Workspace.fProxy.cs

## Findings

### 1. [medium/pointer/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Bidiag.fProxy.cs:237 — Bidiag.decomp non-workspace overload allocates 5 Allocator.Temp buffers BEFORE the inner overload validates dimensions, leaking them on any validation throw.

**Evidence**

```
Lines 241-249: `var ws = new fProxyBidiagCache { W = new fProxyMxN(...Temp...), leftU=..., uVec=..., vVec=..., wScratch=... }; decomp(in A, ref U, ref B, ref V, ref ws);`
```

The inner decomp does its checks (`m < n`, U/B/V size, lines 140-147) and throws AFTER these Temp allocations, so the five buffers are never disposed (the Dispose block at 250-254 is skipped). CHOP's non-ws overload (CHOP.fProxy.cs:365-367) explicitly replicates the checks before allocating precisely to avoid this leak; Bidiag does not.

**Verifier**

Bidiag.decomp(in A, ref U, ref B, ref V) at Bidiag.fProxy.cs:237-255 allocates 5 Allocator.Temp buffers into fProxyBidiagCache BEFORE delegating to the ref-workspace overload, whose first action (lines 140-147) is dimension validation that throws ArgumentException on m<n or wrong-sized U/B/V. There is no try/finally, so the Dispose block at 250-254 is unreachable on such throws and all five Temp buffers leak. CHOP.fProxy.cs:364-375 demonstrates the intentional counter-pattern with an inline rationale ("a caller error can't leak the Temp allocation"), which Bidiag fails to replicate.

**Suggested fix**

Replicate the m<n and U/B/V dimension checks at the top of this overload (before the `new fProxyBidiagCache` allocation), mirroring CHOP.decomp(in A, ref L, ref P).

### 2. [medium/pointer/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Bidiag.fProxy.cs:326 — Bidiag.values non-workspace overload allocates 4 Allocator.Temp buffers before the inner overload validates dimensions, leaking them on any validation throw.

**Evidence**

```
Lines 330-337: `var ws = new fProxyBidiagCache { W = new fProxyMxN(...Temp...), uVec=..., vVec=..., wScratch=... }; values(in A, ref d, ref e, ref ws);`
```

The inner values() throws on `m < n` (277-278) or `d.N/e.N != n` (279-282) after allocation, skipping the Dispose block at 338-341. Same allocate-before-validate leak as the decomp overload.

**Verifier**

Outer values() at 326-342 allocates 4 Allocator.Temp buffers (W/uVec/vVec/wScratch) at lines 330-336, then calls the inner workspace overload which validates m<n and d.N/e.N at lines 277-282. On any throw, the Dispose block at 338-341 is skipped and all four buffers leak. The outer overload does not pre-check d.N/e.N so those two throws are reachable in practice. The same allocate-before-validate pattern is present in the decomp non-workspace overload (237-255) whose inner U/B/V shape checks (142-147) will similarly leak 5 buffers. Fix: hoist the checks above the allocations, or wrap the inner call in try/finally.

**Suggested fix**

Move the m<n / d.N / e.N checks above the workspace allocation in this overload.

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LU.fProxy.cs:116 — Benchmark-verdict narration embedded in code comments violates the contracts-only comment policy (belongs in DEVLOG.md).

**Evidence**

```
Line 116-118: `// Size gate: measured crossover, not the naive 4*LU_BLOCK — the panel/TRSM/GEMM bookkeeping // isn't amortised until ~8 panels wide.`
```

This is a measured/benchmark rationale, not a contract. Same pattern recurs at CHO.fProxy.cs:68 ("measured crossover, not the naive 2*CHOL_BLOCK") and CHOP.fProxy.cs:107-108 ("Size gate: measured crossover, higher than plain CHO's gate").

**Verifier**

LU.fProxy.cs:116-118 literally says "measured crossover, not the naive 4*LU_BLOCK — the panel/TRSM/GEMM bookkeeping isn't amortised until ~8 panels wide". CLAUDE.md's comment policy explicitly forbids "benchmark results, perf verdicts, rejected alternatives" in code comments and directs them to DEVLOG.md. The comment states measurement rationale plus a rejected alternative, not a contract. Verified identical pattern at CHO.fProxy.cs:68-70 and CHOP.fProxy.cs:107-108. Low-severity policy finding, not a runtime bug — matching the reviewer's own severity tag.

**Suggested fix**

Reduce to a contract statement ("blocked path used when n >= Consts.fProxyLuBlockMinN") and move the crossover-measurement rationale to the folder DEVLOG.md.

### 4. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/CHOP.fProxy.cs:197 — Port/history narration in code comments (reference-source provenance and 'deviations from a literal port') violates the contracts-only comment policy.

**Evidence**

```
Lines 198-215: `// Port of Lucas/Higham dpstrf.f (upper-triangular branch).` ... `// Two deviations from a literal port: Ukk is read straight from the pivot search's maxDiag ... this port always searches for a pivot rather than reusing LAPACK's precomputed first-column pivot.`
```

Port provenance and deviation history are DEVLOG material, not member contracts.

**Verifier**

Lines 198 and 209-213 of CHOP.fProxy.cs contain port provenance ("Port of Lucas/Higham dpstrf.f"), explicit deviation-from-literal-port narration ("Two deviations from a literal port: Ukk is read straight from the pivot search's maxDiag ... this port always searches for a pivot rather than reusing LAPACK's precomputed first-column pivot"), and an internal LAPACK comparison ("beyond LAPACK's single INFO=1"). CLAUDE.md's strict comment policy states code comments carry contracts only, with development history, rejected alternatives, and internal reference comparisons routed to the folder DEVLOG.md. The (a)/(b) algorithmic contract lines can stay; the port banner and deviation paragraph are the genuine policy violations the reviewer flagged.

**Suggested fix**

Keep only the behavioral contract in the comment; relocate the port provenance and deviation notes to the folder DEVLOG.md.

## Scanner notes

Scanned all 6 files in full. The numerical/logical cores are notably careful and I found no correctness defects: LU blocked GETRF and compact/pivot-indirected paths preserve the unblocked pivot sequence and handle column m-1 and singular early-returns correctly (Ubuf disposed on the singular branch); CHO right-looking POTRF lower-triangle updates and aliasing (L aliases A) are sound; CHOP pivoted PSTRF's deferred dot[] bookkeeping, pivot-swap of dot[k]/dot[q], indefinite-vs-rank-deficient tolerance (scale-relative n*eps*max|W|, NaN-safe via !(maxDiag>stopTol)), and the rank-deficient min-norm pseudoinverse A+ = M(MtM)^-2 Mt (with Tikhonov-ridge retry) all check out for both vector and multi-RHS forms; Bidiag's forward Householder sweep, backward thin-U reconstruction (skipping already-final columns 0..k-1 is valid), and NR (d,e) extraction are correct. These OP files expand only to float/double (fProxy), so no int/uint division/overflow concerns apply here. The two medium findings are Temp leaks confined to caller-error (throw) paths; the two low findings are comment-policy (contracts-only) violations, with additional instances noted inline in the LU finding.
