# Release scan 2026-07-12 — area: sparse

Scanned 20 template files (core). Findings: total 5 — confirmed 5, uncertain 0, unverified 0, refuted 0; high 0, medium 2, low 3.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/Analysis.Sparse.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/Arena.Sparse.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/Comp.Sparse.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/Debug.Sparse.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/Export.Sparse.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/Gallery.Sparse.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/Norms.Sparse.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/SparseOP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/SparseOP.Transpose.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/UnsafeOP.Sparse.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyBSR.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyBSRAssembly.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyBSRBuilder.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyBSROperator.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyBSRRecords.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyBlockJacobi.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyIC0.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyILU0.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxySSOR.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxySparseLP.fProxy.cs

## Findings

### 1. [medium/numerical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/Norms.Sparse.fProxy.cs:12 — L1 under-counts a Symmetric-storage BSR yet its doc claims it equals the dense expanded L1.

**Evidence**

```csharp
/// Implicit (absent) blocks contribute 0, so this equals the dense entrywise L1 of the expanded matrix.
public static fProxy L1(in fProxyBSR A) { unsafe { return UnsafeOP.sumAbs(A.Values.Ptr, A.Values.Length); } }
```

Sums only STORED entries; for Symmetric (lower-block) storage the implicit upper off-diagonal blocks are NOT absent (they equal stored lower blocks) but are never counted, so L1 returns roughly half the true dense value on off-diagonals.

**Verifier**

Norms.L1 sums only stored entries via UnsafeOP.sumAbs(A.Values.Ptr, A.Values.Length), yet its XML doc promises equality with the dense entrywise L1 of the expanded matrix. Per fProxyBSR.cs:38, Symmetric=true stores only the lower block-triangle, so the "implicit" upper off-diagonal blocks are the transposed lower blocks (non-zero), not absent zeros. The result therefore under-counts off-diagonals by ~2x versus the documented dense expansion. Sibling ops (columnNormsSquared, rowSquaredWeighted) already throw for Symmetric storage citing the same reason; L1 has no guard and no doubling.

**Suggested fix**

Either guard against A.Symmetric and throw (as columnNormsSquared/rowSquaredWeighted already do), or double the contribution of strictly-lower stored blocks when A.Symmetric so the result matches the dense expansion.

### 2. [medium/numerical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/Norms.Sparse.fProxy.cs:33 — L2 (Frobenius) is claimed 'exact' but under-counts Symmetric-storage off-diagonal entries.

**Evidence**

```csharp
/// Frobenius (entrywise L2) norm ... over the STORED entries — exact, since implicit zeros contribute nothing.
... return math.sqrt(UnsafeOP.vecDot(vals.Ptr, vals.Ptr, vals.Length));
```

For Symmetric storage the implicit upper blocks are NOT zeros; each strictly-lower off-diagonal entry represents TWO dense entries, so sqrt(Sum stored^2) is smaller than the true dense Frobenius norm.

**Verifier**

fProxyBSR.Symmetric (fProxyBSR.cs:38) is a real flag that stores only the lower block-triangle. Norms.L2 (Norms.Sparse.fProxy.cs:33) does sqrt(vecDot(vals,vals,len)) with NO A.Symmetric guard; strictly-lower off-diagonal entries each represent two dense entries but contribute only once, so the returned value is smaller than the true dense Frobenius norm. The docstring "exact, since implicit zeros contribute nothing" is false for Symmetric storage — implicit upper blocks are mirrors, not zeros. The established codebase pattern (SparseOP.fProxy.cs:179–180, 295–296; fProxySparseLP.fProxy.cs:305–306) throws on A.Symmetric for exactly this class of undercount; L2 (and L1) miss that guard. Fix: guard on A.Symmetric (throw), or double-count the strictly-lower off-diagonal squared contributions before the sqrt.

**Suggested fix**

Guard on A.Symmetric (throw) or, when Symmetric, count strictly-lower off-diagonal squared entries twice before the sqrt.

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/Debug.Sparse.fProxy.cs:77 — Spy doc/comments say absent blocks print '.', but the code prints a space.

**Evidence**

```csharp
str.Append(present ? 'X' : ' ');
```

While the XML doc (line 93-96) and the header comment (line 38) both state "'X' stored / '.' absent".

**Verifier**

Line 77 in Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/Debug.Sparse.fProxy.cs unconditionally emits ' ' (space) for absent blocks: `str.Append(present ? 'X' : ' ');`. Both the header comment (line 35: "'X' stored / '.' absent") and the XML `<summary>` on public `Spy` (lines 92-97: "'X' stored, '.' absent") document the MATLAB-style '.' character. No codegen marker or branch changes this literal, and `Log` shares the same helper. Genuine contract/behavior mismatch — low severity, but the reviewer's claim is correct as stated.

**Suggested fix**

Either emit '.' for absent blocks to match the documented MATLAB-style spy grid, or correct the comments/XML docs to say a space is used.

### 4. [low/numerical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyIC0.cs:117 — Diagonal-shift start uses a float literal (1e-3f) that truncates to float precision in the double expansion.

**Evidence**

```csharp
shift = shift == (fProxy)0 ? (fProxy)1e-3f * diagMax : shift * (fProxy)10;
```

`(double)1e-3f == 0.001000000047...` not 0.001; same pattern in fProxyILU0.cs:78.

**Verifier**

Mechanically accurate: `(fProxy)1e-3f` expands to `(double)1e-3f` in the double variant, which is 0.001000000047497451... instead of the exact double-precision 0.001 you would get from `(fProxy)1e-3`. Same pattern is confirmed at fProxyILU0.cs:78. The relative error is ~5e-11, so while the truncation is real, it is a shift-schedule heuristic that is escalated by 10x on retry — no observable failure follows. Confirmed as a genuine (very minor) precision leak; fix is a one-character edit (`1e-3f` -> `1e-3` or `0.001`).

**Suggested fix**

Use a precision-neutral literal, e.g. (fProxy)1e-3 (double literal) or 0.001, so the shift schedule is exact in both float and double expansions. (Impact is minor — a shift heuristic — but it is an accidental float-in-double truncation.)

### 5. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/SparseOP.fProxy.cs:153 — Code comments carry benchmark verdicts and a DEVLOG reference, violating the contracts-only comment policy.

**Evidence**

```csharp
// one Blas.dot(x,y) pass (a fused kernel was tried and measured slower -- see the Sparse
// DEVLOG). ...
```

Also fProxyBSROperator.cs:71-72 ("a genuinely-fused kernel was tried and measured slower") and UnsafeOP.Sparse.fProxy.cs:1083-1085 ("tried and measured slower ... see the Sparse DEVLOG").

**Verifier**

All three cited comments carry rejected-alternative/benchmark-verdict text that CLAUDE.md explicitly bans from code comments ("benchmark results, perf verdicts, rejected alternatives" -> DEVLOG.md). Two of them ("see the Sparse DEVLOG", "see... the Sparse DEVLOG") self-identify the violation by pointing at the DEVLOG. The DEVLOG already contains the corresponding 2026-07-11 entry for the spMVDot fused-kernel regression, so the code comments duplicate history that has already been moved. Genuine text/policy contradiction, not a false alarm.

**Suggested fix**

Move the benchmark verdict / rejected-alternative history to the folder DEVLOG.md and keep the code comment to the contract (composes spMV then dot).

## Scanner notes

Scanned all 20 sparse template files in full. The compute kernels (bsrMatVec/T/Sym, bsrMatMat*, sweepLower/Upper, blockJacobiApply, IC0/ILU0 factor+apply, transpose, builder sort/compress, assembly cache, LP operators) were cross-checked index-by-index against their row-major/transpose math and found correct and mutually consistent; alias guards, MemClear sizes, and dispose paths in the arena-tracked structs are all in order. The two Norms findings are the only wrong-result defects: L1/L2 silently under-count Symmetric (lower-block) storage while their docs promise the dense-expanded norm, and unlike columnNormsSquared/rowSquaredWeighted they do not guard against Symmetric. No memory leaks or use-after-dispose were found (early-return paths in fProxyBlockJacobi's ctor correctly Dispose scratch + dinv before throwing). fProxyILU0.InvertBlockInPlace allocates a `perm` stackalloc that is written but never read (dead, harmless). fProxyILU0.BlockMulRight does a stackalloc inside a per-row loop (BR<=16, negligible) — not reported as a defect.
