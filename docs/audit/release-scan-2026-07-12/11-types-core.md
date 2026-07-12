# Release scan 2026-07-12 — area: types-core

Scanned 27 template files (core). Findings: total 2 — confirmed 2, uncertain 0, unverified 0, refuted 0; high 0, medium 0, low 2.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/AssemblyInfo.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Assume.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Assume.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Assume.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/ChooseMarkerDemo.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/ChooseMarkerDemo.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Consts.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/markers.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/proxyShims.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/proxyStructs.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/proxyStructs.math.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/fProxy/fProxyMxN.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/fProxy/fProxyMxN.Indexing.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/fProxy/fProxyMxN.Operators.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/fProxy/fProxyMxN.Shortcuts.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/fProxy/fProxyMxN.Comparators.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/fProxy/fProxyN.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/fProxy/fProxyN.Indexing.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/fProxy/fProxyN.Operators.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/fProxy/fProxyN.Shortcuts.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/fProxy/fProxyN.Comparators.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Interfaces/Interfaces.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Interfaces/LinearOperator.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Interfaces/PredicateQuery.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Interfaces/PredicateQuery.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Interfaces/Sampler.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Interfaces/ScalarFunction.fProxy.cs

## Findings

### 1. [low/pointer/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Interfaces/LinearOperator.fProxy.cs:210 — fProxyColScaledOperator.ApplyBlock sizes the per-row output scratch to Inner.Cols instead of Inner.Rows, a latent buffer overflow for any rectangular inner operator.

**Evidence**

```
int cols = Vrows.N_Cols;
var rin = new fProxyN(cols, ...);
var rout = new fProxyN(cols, ...);
...
Apply(in rin, ref rout);  // Apply -> Inner.Apply writes y of length Inner.Rows, but rout is length cols(=Inner.Cols)
```

Both scratch vectors are sized from `Vrows.N_Cols` (== Inner.Cols), but the operator's `Apply` produces an output of length Inner.Rows. For any rectangular inner operator the output scratch has the wrong length, and the copy-out loop bounded by `cols` targets the wrong span of `AVrows`.

**Verifier**

fProxyColScaledOperator's ApplyBlock sizes rout to Vrows.N_Cols (== Inner.Cols) and copies out `cols` entries per row, but Apply/Inner.Apply writes an Inner.Rows-length output. For any rectangular Inner (the operator's actual production role in CGLS/LSQR/LSMR — Krylov.fProxy.cs:1881,1901,1921,...), the very first Apply call hits Blas.dot's guard `if (result.N != A.M_Rows) throw` (OP.Dot.fProxy.cs:71-72) and faults. The write loop bounded by `cols` also targets the wrong span of AVrows. The claim mislabels the failure mode ("buffer overflow") — the dimension guard catches it before any pointer write — but the method is genuinely broken for rectangular inners and the suggested fix (rin=Cols, rout=Rows, copy-out bound=Rows) is exactly right. Severity is correctly rated low because ApplyBlock's only caller today is LOBPCG, which never wraps this operator.

**Suggested fix**

Size rout to Rows (Inner.Rows) and AVrows accordingly; rin stays Cols. Only safe today because documented callers pass square/symmetric operators, but the public method overflows if Rows != Cols.

### 2. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Consts.cs:33 — Code comments carry benchmark/measurement narration (measured per dtype, blocked-vs-unblocked sweep, 'err high') that the contracts-only comment policy directs to DEVLOG.

**Evidence**

```
// Cache-dependent; measured per dtype, pinned conservatively (err high) since a too-low gate can regress on a weaker cache.
... // measured per dtype from a blocked-vs-unblocked sweep ... float/double ordering is not universal, so each is measured independently
```

These comments narrate measurement methodology and perf rationale rather than stating the member's contract, which the project's comment policy (CLAUDE.md) directs to the folder's DEVLOG.md.

**Verifier**

Lines 32-33, 37-41, and 51 of Consts.cs mix contract statements with measurement/methodology narration ("measured per dtype", "pinned conservatively (err high)", "blocked-vs-unblocked sweep", "float/double ordering is not universal, so each is measured independently rather than derived"). CLAUDE.md is explicit that "benchmark results, perf verdicts, rejected alternatives" and similar rationale belong in the folder's DEVLOG.md, never in code comments — only the contract (what the gate selects, which dimension it triggers on) should remain in-code. This is a genuine, low-severity policy contradiction.

**Suggested fix**

Keep the contract (what the gate selects) in-code; relocate the measurement methodology / rationale to the folder DEVLOG.md per CLAUDE.md.

## Scanner notes

The types-core template area is clean of high/medium-severity defects. Operator sign/order, row-major indexing, epsilon constants, choose-marker literal counts (2 for float/double, 3 for int/short/long), constructor field-initialization on all paths (arena / standalone / view), Allocator.None view Dispose, and MemCpy element/byte counts were all verified correct. The two findings are low-severity: one latent (usage-guarded) overflow in a rarely-reachable ApplyBlock, and one comment-policy violation. No leaks, aliasing, or numerical errors found in the scanned files.
