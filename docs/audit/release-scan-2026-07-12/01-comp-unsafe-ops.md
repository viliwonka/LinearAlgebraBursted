# Release scan 2026-07-12 — area: comp-unsafe-ops

Scanned 17 template files (core). Findings: total 3 — confirmed 3, uncertain 0, unverified 0, refuted 0; severity: high 0, medium 1, low 2.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/OP.Component.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/OP.Component.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeOP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeOP.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeOP.bool.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeBoolOP.bool.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeBoolOP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeBoolOP.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeBitsOP.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeMathOP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeMathOP.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/BoolOP.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SwapOP.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SelectOP.bool.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SelectOP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SelectOP.iProxy.cs

## Findings

### 1. [medium/performance/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeOP.iProxy.cs:115 — iProxy vecMatDot uses a cache-hostile column-strided inner loop, unlike its optimized fProxy sibling.

**Evidence**

```csharp
for (int c = 0; c < n; c++)
{
    for (int r = 0; r < m; r++)
    {
        x[c] += (iProxy)(mat[r * n + c] * y[r]);
    }
}
```

The inner loop is over r, so mat[r*n+c] strides by n each step (column traversal of a row-major matrix) - defeating cache locality and auto-vectorization. The fProxy vecMatDot (UnsafeOP.fProxy.cs:192-200) was explicitly restructured to r-outer/c-inner unit-stride, with a comment 'so mat[baseIdx + c] is unit-stride in c'. The int-family expansion never got that fix.

**Verifier**

UnsafeOP.iProxy.cs:115-121 has c-outer / r-inner with mat[r*n+c] striding by n on a row-major matrix — column traversal, cache-hostile, unvectorizable. The paired UnsafeOP.fProxy.cs:184-201 was deliberately rewritten to r-outer/c-inner with MemClear(x) preamble and unit-stride mat[baseIdx+c] (comment: "unit-stride in c"). The two files are independent templates (fProxy expands to float/double, iProxy to int/uint), so no codegen mechanism unifies them; the int-family expansion genuinely missed the fix. Note: iProxy contract requires caller-zeroed x while fProxy MemClears internally, so any fix must preserve iProxy's calling convention (or migrate its callers) rather than blindly copying fProxy.

**Suggested fix**

Mirror the fProxy kernel: MemClear x, then loop r outer / c inner (x[c] += y[r]*mat[r*n+c]) so the inner access is unit-stride.

### 2. [low/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeOP.fProxy.cs:191 — vecMatDot has divergent zeroing contracts across type expansions: fProxy self-clears x, iProxy requires the caller to pre-zero, causing a redundant double-clear in the shared caller.

**Evidence**

```csharp
fProxy: UnsafeUtility.MemClear(x, (long)n * UnsafeUtility.SizeOf<fProxy>()); ... x[c] += yr * mat[...]   // self-clears
iProxy (UnsafeOP.iProxy.cs:110-122): no clear, comment 'needs to be initialized to zero'
```

The shared caller OP.Dot.*.cs pre-clears result for BOTH types ('vecMatDot accumulates (+=), so the destination must start zeroed' + MemClear), so the fProxy path clears the buffer twice, and a generic caller relying on the iProxy 'accumulate' contract would be silently zeroed on the fProxy path.

**Verifier**

Traced both expansions: UnsafeOP.fProxy.cs:191 does MemClear + accumulate (self-clear contract); UnsafeOP.iProxy.cs:110-122 has no clear and comments "needs to be initialized to zero" (caller-clears contract). The shared caller OP.Dot pre-clears the destination in both fProxy.cs:130 and iProxy.cs:97, so the fProxy path pays for a redundant double MemClear. Additionally, fProxy-only callers (SVD.LowRank:265, LP.InteriorPoint:286, LP.RevisedSimplex:115) rely on the self-clear and do NOT pre-zero — meaning the two expansions genuinely implement different contracts, which is a real codegen invariant violation even though no current caller mixes both contracts across expansions.

**Suggested fix**

Make both expansions agree: either both self-clear (and drop the caller MemClear) or both accumulate (and keep the caller MemClear). Update the kernel comments to match.

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeOP.fProxy.cs:721 — Code comment states a benchmark verdict / rejected alternative, which the project comment policy requires to live in DEVLOG.md, not in code.

**Evidence**

```csharp
//      Vld between consecutive t) does NOT vectorise and was measured far slower.
```

CLAUDE.md: 'Everything else goes in the folder's DEVLOG.md ... benchmark results, perf verdicts, rejected alternatives'. Similar perf-verdict prose appears in the matMatDot header ('re-streams matB ... bandwidth-bound', ~line 220) and sincos ('more cache efficient than calling sin&cos at same time', UnsafeMathOP.fProxy.cs:397).

**Verifier**

Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeOP.fProxy.cs:720-721 contains the sentence "The naive per-(k,i) dot form ... does NOT vectorise and was measured far slower." That is simultaneously a rejected alternative and a benchmark verdict — two categories CLAUDE.md's comment policy explicitly relegates to DEVLOG.md and forbids in code. The OP folder already has a DEVLOG.md, so this is a straightforward policy violation. Severity remains low (documentation-only), and only the perf-verdict clause needs to move; the loop-order contract itself can stay.

**Suggested fix**

Move the benchmark verdict and rejected-alternative rationale to the OP folder's DEVLOG.md; keep only the contract (the loop order chosen) in the comment.

## Scanner notes

Verified clean (no defect) after close inspection: the 2x fProxy4 SIMD reductions and their odd-block tail handling; register-tiled matMatDot / matMatDotTransA remainder-row/col routing and long-cast indexing; syrkLowerSub/syrkUpperSub aliasing and the axpy(Wrow+i, Up+ip, -temp, n-i) index math; trsmLowerPanel forward-substitution; formT / wyTriMul / wyTriTransMul iteration directions; the short-width //+choose branches for countbits/tzcnt/lzcnt/reversebits/rol/ror/ceilpow2 (operator precedence '+' > '&' is correct, and '+' equals '|' there because the shifted halves occupy disjoint bit ranges); jacobiRotate old-value capture; refract/reflect/project formulas; acosh domain; remap argument order; float-literal (0f/1f/2f) usage in the double expansion (all exactly representable, no precision loss). Note: several elementwise divide/normalize kernels (normalizeL1/L2/LMax/LP, project) have no divide-by-zero guard, but that is a pre-existing, consistent, internal-kernel convention, so not reported as a new defect. Comment-policy (benchmark/history) prose is pervasive across these files; only one representative instance is filed as a finding rather than flooding the list.
