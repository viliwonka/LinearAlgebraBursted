# Release scan 2026-07-12 — area: dot-blas-simd

Scanned 11 template files (core). Findings: total 4 — 4 confirmed, 0 uncertain, 0 unverified, 0 refuted; severity: 1 high, 0 medium, 3 low.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/OP.Dot.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/OP.Dot.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/NormsOP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/NormsOP.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/OpHelpers.Shared.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/OpHelpers.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SimdMath.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/GenOP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Blas.ColumnScaling.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Blas.Fused.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Blas.Triangular.fProxy.cs

## Findings

### 1. [high/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/OP.Dot.fProxy.cs:278 — householderInPlace does not apply the Householder reflection H·matrix; it subtracts a constant outer product (2/uᵀu)·u·uᵀ that ignores matrix's contents, so results are wrong for any matrix that is not the identity.

**Evidence**

Doc (line 249-251): "matrix -= (2 / uᵀu) · u·uᵀ · matrix". Code:

```csharp
    fProxy scaleFactor = 2 / vTv;
    ...
        fProxy vvT_element = scaleFactor * u[i] * u[j];
        matrix[i, j] -= vvT_element;
```

This computes matrix - (2/uᵀu)·u·uᵀ (the `· matrix` factor is missing). For a general M the result is M + H - I, which equals the intended H·M only when M is the identity. The public docs (docs/features/la-primitives.md:16) advertise it as "apply a Householder reflection directly", so any external caller passing a non-identity matrix gets silently wrong numbers.

**Verifier**

The template lines 274-281 iterate (i,j) and subtract `scaleFactor * u[i] * u[j]` from `matrix[i,j]`, i.e. `matrix -= (2/uᵀu)·u·uᵀ`. The `·matrix` factor from the documented formula (line 249-250) is missing — there is no per-column dot product uᵀ·M[:,c]. Concrete counterexample: u=[1,0], M=2I yields [[0,0],[0,2]] instead of correct H·M = diag(-2,2). No tests exercise `householderInPlace` in TemplateSourceTests, so both generated float/double variants ship the same wrong result. Docs/features/la-primitives.md:16 exposes it publicly as "apply a Householder reflection directly", so external callers get silently wrong numbers for any non-identity matrix.

**Suggested fix**

Apply the reflection column-by-column: for each column c compute proj = Σ_i u[i]·matrix[i,c], then matrix[i,c] -= scaleFactor·u[i]·proj. (Or, if the intent really is just to subtract the outer product, rename the method and correct the doc.)

### 2. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/OP.Dot.fProxy.cs:257 — householderInPlace's second guard and its 'square or tall' message are unreachable/contradictory: the method already rejects non-square input, so M_Rows < N_Cols can never be true and the 'or tall' wording is misleading.

**Evidence**

```text
Line 254: if(matrix.IsSquare == false) throw ... "Matrix must be square";
then line 257: if(matrix.M_Rows < matrix.N_Cols) throw ... "Matrix must be square or tall (more or equal rows than cols)"
```

Since IsSquare == (M_Rows == N_Cols), the second condition is dead code and the maxDim = math.max(M_Rows, N_Cols) logic (line 260) is vestigial from a tall-matrix design; the messages claim tall matrices are accepted while line 254 rejects them.

**Verifier**

In OP.Dot.fProxy.cs the first guard (line 254) throws unless matrix.IsSquare, meaning M_Rows == N_Cols is guaranteed at line 257. The second guard `if(matrix.M_Rows < matrix.N_Cols)` is therefore unreachable dead code, and its message "Matrix must be square or tall" directly contradicts the preceding "Matrix must be square". The subsequent `math.max(M_Rows, N_Cols)` and the loop's dual `u[i]`/`u[j]` indexing are vestigial from a tall-matrix design that no longer exists — genuine contradiction between text and behavior, as claimed.

**Suggested fix**

Drop the dead tall-matrix guard and the 'or tall' wording, or (if tall support is actually wanted) remove the square requirement and make the kernel loop over min dims correctly.

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/NormsOP.fProxy.cs:38 — L2Range's exception messages are copy-pasted from L2 and name the wrong method, misleading callers who hit the range guard.

**Evidence**

```csharp
public static fProxy L2Range(...) {
    if (start >= end) throw new ArgumentException("Norms.L2: start must be less than end");
    if (start < 0 || end > a.Data.Length) throw new ArgumentOutOfRangeException("Norms.L2: start and end must be within bounds of vector");
}
```

The method is L2Range, not L2.

**Verifier**

Assets/LinearAlgebra/CodeGen/TemplateSource/OP/NormsOP.fProxy.cs lines 35-41: method L2Range throws with prefixes "Norms.L2:" in both guards, not "Norms.L2Range:". Sibling range-guarded methods in the same file (NormalizeL1 line 276, NormalizeL2 line 258, NormalizeLMax line 294, NormalizeLP line 312) all self-name correctly, establishing the convention this method violates. The messages are misleading rather than harmful; low severity is appropriate.

**Suggested fix**

Change the message prefixes to "Norms.L2Range:".

### 4. [low/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/OP.Dot.fProxy.cs:111 — dotSelf has an undocumented square-A precondition; for rectangular A the trailing dot(x,y) throws a confusing 'Vector must have same dimension' error because x has length A.N_Cols and y has length A.M_Rows.

**Evidence**

```text
y = A x is computed with x.N == A.N_Cols and y.N == A.M_Rows (guarded at line 100-101),
but the method ends `return dot(x, y);` which requires x.N == y.N
(line 18-19: throws "dot: Vector must have same dimension").
```

This only holds when A is square; the XML/comment states no such requirement.

**Verifier**

Traced dotSelf at OP.Dot.fProxy.cs:96-112: after `y = A x` (with x.N == A.N_Cols and y.N == A.M_Rows guaranteed by lines 98/100-101), line 111 returns `dot(x, y)` which at line 18-19 throws "dot: Vector must have same dimension" unless x.N == y.N — implicitly requiring A.M_Rows == A.N_Cols. There is no XML doc, no explicit square-A guard, and the code comment at line 93-94 only describes composition ("y = A x, PLUS dot(x, y)") without stating the square precondition; the DEVLOG at line 79 does mention "for square A" but only for the rejected fused variant, not as an API contract. In-repo callers (LinearOperator.ApplyDot for Krylov SPD) never hit it, but the public Blas API can surface a misleading error message for a rectangular A. Severity remains low, as noted by the reviewer.

**Suggested fix**

Either document/guard that A must be square (it is, for its SPD Krylov callers) with a clear message, or restrict the returned dot to the min length if a rectangular quadratic form was intended.

## Scanner notes

Scanned all 11 listed template files in full. The integer NormsOP (L1/LInf/L2 widening, long.MinValue wraparound asymmetry) is heavily and accurately documented and pinned by tests — intentional, not a defect. The Blas.Triangular forward/back and transposed compact-LU solves (vector and multi-RHS) were checked against the Uᵀ/Lᵀ right-looking recurrences and the pivot indirection is consistent throughout. Blas.Fused, Blas.ColumnScaling, GenOP, OpHelpers, and SimdMath are clean. The householderInPlace defect (finding 1) is the only high-severity item; it has no internal callers (QR uses its own reflector pipeline) and no test coverage, so it would only bite external users of the public API. The same defect is present in the generated float/double copies (Source/OP/OP.Dot.{float,double}.cs) — fix belongs in the template only.
