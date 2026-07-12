# Release scan 2026-07-12 — area: types-int-bool

Scanned 24 template files (core). Findings: total 2 — confirmed 1, uncertain 0, unverified 0, refuted 1; severity: high 0, medium 1, low 0.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/iProxy/iProxyMxN.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/iProxy/iProxyMxN.Indexing.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/iProxy/iProxyMxN.Operators.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/iProxy/iProxyMxN.Shortcuts.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/iProxy/iProxyMxN.Comparators.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/iProxy/iProxyN.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/iProxy/iProxyN.Indexing.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/iProxy/iProxyN.Operators.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/iProxy/iProxyN.Shortcuts.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/iProxy/iProxyN.Comparators.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/bool/boolMxN.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/bool/boolMxN.Indexing.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/bool/boolMxN.Operators.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/bool/boolMxN.Shortcuts.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/bool/boolN.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/bool/boolN.Indexing.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/bool/boolN.Operators.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/bool/boolN.Shortcuts.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Indices/Indices.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Pivot/Pivot.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Pivot/Pivot.Operations.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Hash/Hash.Shared.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Hash/Hash.bool.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Hash/Hash.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Hash/Hash.iProxy.cs

## Findings

### 1. [medium/pointer/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Pivot/Pivot.Operations.cs:121 — The inverse-apply methods (ApplyInverseVec/ApplyInverseRow/ApplyInverseColumn) omit the dimension guard that their forward counterparts have, so a size-mismatched operand overflows the buffer.

**Evidence**

ApplyVec guards `if(v.N != this.N) throw ...`, ApplyRow guards `if (A.M_Rows != this.N) throw ...`, but ApplyInverseVec/Row/Column have no such check:

```csharp
public void ApplyInverseVec(ref fProxyN v) {
    Pivot tempPivot = InverseCopy();
    ApplyVecInPlace(ref v, ref tempPivot);
```

InverseCopy() builds a pivot of N=this.N and ApplyVecInPlace loops fromR over pivot.N indexing v[fromR]/v[toR]; if v.N (or A.M_Rows/A.N_Cols) < this.N the loop reads/writes past the operand. In a player build (no collections checks) swapRows/swapColumns/v[] use raw pointers, so this is silent memory corruption.

**Verifier**

ApplyInverseVec/Row/Column (Pivot.Operations.cs lines 121, 131, 141) skip the dimension check that their forward siblings ApplyVec/ApplyRow/ApplyColumn (lines 108, 81, 95) all perform against this.N. InverseCopy() at Pivot.cs:88 returns a pivot of size this.indices.Length, so the *InPlace loops iterate 0..this.N-1 and index v[fromR]/v[toR] or call UnsafeOP.swapRows/swapColumns via raw pointers. An operand with v.N/A.M_Rows/A.N_Cols smaller than this.N therefore over-reads or over-writes silently in player builds without collections checks. The block is inside //+copyReplaceAll so every generated fProxy expansion inherits the gap.

**Suggested fix**

Add the same dimension validation used by ApplyVec/ApplyRow/ApplyColumn to each of the three inverse variants (compare v.N / A.M_Rows / A.N_Cols against this.N and throw ArgumentException on mismatch).

## Refuted

| file:line | claim | why refuted |
|---|---|---|
| Assets/LinearAlgebra/CodeGen/TemplateSource/Pivot/Pivot.Operations.cs:19 | The XML contract says these in-place methods 'reset pivot to [0,1,2,...]', but only the index array is restored to identity; swapCount is left non-zero, so the pivot's Sign is inconsistent with an actual identity permutation afterward. | The claim "Sign can be -1 after ApplyVecInPlace" is mathematically false for any pivot mutated only through the public API. Swap() maintains the invariant swapCount ≡ parity(indices) (mod 2), and reducing indices to identity via transpositions requires exactly parity(indices) swaps mod 2, so swapCount ends up even and Sign = +1 — consistent with identity. The private swapCount not being literally 0 has no observable effect; Sign (the only exposed parity signal) correctly reads +1 for all reachable states. The XML contract "resets pivot to [0,1,2,...]" is honored in every observable way. |

## Scanner notes

Scanned all 24 listed template files in full. The iProxy/bool data-type family (MxN + N: ctors, indexing, operators, comparators, shortcuts), Indices, and the Hash.* kernels are clean: allocation-zeroing patterns are consistent and correct (UnsafeList ClearMemory clears full capacity before the Uninitialized Resize), the xxHash32 kernel in Hash.Shared.cs faithfully matches the reference (block loop `do{...4 rounds...}while(p<=bEnd-16)`, tail loops `p+4<=bEnd` then `p<bEnd`, correct avalanche), the `//+skipFor[u]` guards correctly exclude unary-minus from uint, and the reverse scalar operators (`s - M`, `s / M`, `s % M`) pass arguments in the correct non-commutative order with documented integer div/mod-by-zero behavior. The `s == 0f` divisor guards in the integer operator overloads are exact for every int/uint/short/long expansion (no nonzero integral value rounds to 0.0f), so not a defect. Cross-type/codegen `//+choose` markers in Hash resolve dest buffers to uint as documented. Only the two Pivot findings above are actionable; both are edge-case/contract issues, not normal-path failures.
