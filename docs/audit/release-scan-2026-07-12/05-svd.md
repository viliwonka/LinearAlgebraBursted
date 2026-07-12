# Release scan 2026-07-12 — area: svd

Scanned 12 template files (core). Findings: total 3 — confirmed 3, uncertain 0, unverified 0, refuted 0; high 0, medium 1, low 2.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.FullWorkspace.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.LowRank.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.Metrics.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.Randomized.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.RandomizedWorkspace.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.Solvers.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.Subspace.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.ThinWorkspace.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.TruncatedWorkspace.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.ValuesWorkspace.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.Workspace.fProxy.cs

## Findings

### 1. [medium/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.fProxy.cs:37 — The `tolerance` parameter of SVD.values and SVD.thin (and their workspace overloads) is validated but never used; the bidiagonal-QR deflation threshold is hardcoded to Consts.fProxyEpsilon*anorm, so the user-supplied tolerance knob silently does nothing.

**Evidence**

```
values/thin do `if (tolerance <= (fProxy)0) throw ...` (lines 37-38, 98-99, 182-183, 291-292) but then call `bidiagonalQR(ref Ut, ref dVec, ref eVec, ref Vt, m, n, maxIterations, out sweeps, out convergedCount)` (line 216) and `bidiagonalQRValues(ref dVec, ref eVec, n, maxIterations, ...)` (line 47) with NO tolerance argument. Inside those routines the only threshold is `fProxy thresh = Consts.fProxyEpsilon * anorm;` (line 396 / 581).
```

A caller passing a custom tolerance gets identical behavior to the default — the parameter is dead. The default-overload docs even advertise it ("and tolerance (Consts.fProxyZeroThreshold)", lines 78/261).

**Verifier**

Traced tolerance through all four public entry points. In values (lines 26-76) and thin (lines 166-259), and their workspace overloads (86-136, 275-361), tolerance is only referenced in the guard `if (tolerance <= (fProxy)0) throw ...` — it is never assigned to any local, never captured into the workspace, and never forwarded to `bidiagonalQR` (call sites 216, 322, 534) or `bidiagonalQRValues` (call site 47). Neither internal routine accepts a tolerance parameter; both hardcode `fProxy thresh = Consts.fProxyEpsilon * anorm;` (lines 396 and 581) as the deflation cutoff used at lines 411/412/424 and 596/597/608. The parameter is functionally dead — a caller passing a custom tolerance gets exactly the same numerical behavior as the default, contradicting the default overload's XML docs at lines 78, 138, 261, 266, 363, 368 which advertise it as a knob.

**Suggested fix**

Either thread `tolerance` into bidiagonalQR/bidiagonalQRValues (e.g. `thresh = tolerance * anorm`) so the knob is honored, or remove the parameter and its validation from the public surface if a fixed eps threshold is intended.

### 2. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.RandomizedWorkspace.fProxy.cs:118 — Stale hardcoded 'maxIterations 75' in the default randomized-cache doc contradicts the actual convenience overloads, which use Consts.sweepBudget(l).

**Evidence**

```
Line 116-118 doc: "Allocates a randomized-SVD workspace with the default oversample (10) — matches the randomized convenience overloads (oversample 10, powerIters 2, maxIterations 75)."
```

The real convenience overloads pass `Consts.sweepBudget(math.min(k + 10, A.N_Cols))` (SVD.Randomized.fProxy.cs lines 103/108/144/148), not the constant 75.

**Verifier**

SVD.RandomizedWorkspace.fProxy.cs:118 says the convenience overloads use "maxIterations 75", but all four convenience overloads in SVD.Randomized.fProxy.cs (lines 103, 108, 144, 148) actually pass Consts.sweepBudget(math.min(k+10, A.N_Cols)). Consts.sweepBudget (Consts.cs:67-71) returns max(75, 6*n), so the default matches 75 only when l is tiny (l<=12) and is 6*l otherwise — a real contradiction between the XML contract and the shipped behavior, and inconsistent with every neighboring default doc which spells it as Consts.sweepBudget(...).

**Suggested fix**

Change '75' to 'Consts.sweepBudget(l)' (or drop the specific number) to match the actual default.

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.LowRank.fProxy.cs:218 — Code comment contains a FUTURE/dev-planning note (windowing strategy) which the contracts-only comment policy says belongs in DEVLOG, not in source.

**Evidence**

```
Line 218: "// FUTURE: windowing (compute_int strategy 0) to reorth vs subset only."
```

This is future-work/planning narration inside a generated-from template code comment; CLAUDE.md restricts code comments to contracts and routes such notes to the folder DEVLOG.md.

**Verifier**

Line 218 of Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.LowRank.fProxy.cs literally reads `// FUTURE: windowing (compute_int strategy 0) to reorth vs subset only.` — a future-work planning note pointing at an internal identifier ("compute_int strategy 0"). CLAUDE.md's strict comment policy restricts code comments/XML docs to contracts only and routes planning notes, rejected alternatives, and internal references to the folder DEVLOG.md. The note is not a contract of the surrounding reorth loop, so it violates the stated policy. Low severity, cosmetic — no functional impact — but the claim is factually correct.

**Suggested fix**

Move the windowing note to the folder DEVLOG.md and delete it from the template.

## Scanner notes

Reviewed all 12 SVD template files in full. The core numerics (Golub-Reinsch implicit-shift bidiagonal QR, GKL Lanczos with lanbpro-style partial reorthogonalization mu/nu recurrences, randomized HMT path, pinvSolve/pseudoInverse tall+wide branches, nullspace/range basis) were traced index-by-index and appear correct: dimension checks, transpose conventions, descending sorts carrying U/V columns, rank/tolerance handling, and pointer-arithmetic strides (utp+nm*m, vtp+k*n) all check out. Memory: allocating overloads use arena temp pool by convention; the explicit-Temp path in values/thin disposes dVec/eVec/B/Ut/Vt on all normal paths, and early n==0 returns occur before allocation, so no leaks found. Coefficient-scratch bounds in the reorth loops (vBuf len n used for j<=p-1 coeffs; uBuf len m used for j+1<=p coeffs) are within bounds. The truncated residual index ws.BsvdWs.U[pDone-1, t] is guarded because kOut=min(k,pDone)=0 when pDone==0. No high-severity defects found.
