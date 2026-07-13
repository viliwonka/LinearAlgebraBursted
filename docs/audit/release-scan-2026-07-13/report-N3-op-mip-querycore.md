# Narrow Scan N3 -- OP: MIP.Pseudocost.fProxy.cs through QueryCore.Metric.iProxy.cs

Scanner: N3 (narrow, all dimensions)
Partition: 25 files in `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/`, alphabetically sorted from `MIP.Pseudocost.fProxy.cs` through `QueryCore.Metric.iProxy.cs` inclusive.

---

## Findings

### 1. HIGH -- Missing bounds validation on `rows` in `Blas.dotRows`

**File:** `OP.Dot.fProxy.cs:194-209`

The method `dotRows(in fProxyMxN a, in fProxyMxN b, ref fProxyMxN c, int rows)` does not validate that `rows <= a.M_Rows` or `rows <= c.M_Rows`. Its sibling `dot(in fProxyMxN a, in fProxyMxN b, ref fProxyMxN c, ...)` validates `c.M_Rows != m` (line 170). Passing `rows > a.M_Rows` causes `UnsafeOP.matMatDot` to read past `a` buffer; passing `rows > c.M_Rows` causes `UnsafeUtility.MemClear` to write past `c` buffer. Both are unsafe out-of-bounds memory access on a public API.

```csharp
// No check that rows <= a.M_Rows or rows <= c.M_Rows:
UnsafeUtility.MemClear(c.Data.Ptr, (long)rows * kk * UnsafeUtility.SizeOf<fProxy>());
UnsafeOP.matMatDot(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, rows, nn, kk);
```

Concrete scenario: `a` is 4x3, `b` is 3x2, `c` is 6x2, caller passes `rows = 8` -- reads 8 rows from a 4-row matrix, writes 8 rows into a 6-row matrix.

**Fix direction:** Add `if (rows < 0 || rows > a.M_Rows || rows > c.M_Rows) throw new ArgumentException(...)` before the unsafe block. (Confirmed by wide scan W5 as a known gap.)

---

### 2. MEDIUM -- Float literal suffixes `0f` / `1f` survive into double-generated variant

**File:** `NormsOP.fProxy.cs` -- lines 114, 121, 129, 135, 137, 155, 161, 168, 175, 177

In `normalizeRows` and `normalizeColumns`, accumulator initializers use bare `0f` and the inverse uses `1f`:

```csharp
fProxy s = 0f;                          // line 114, 121, 129, 155, 161, 168
if (!(rowNorm > 0f)) continue;          // line 135, 175
fProxy inv = (fProxy)1f / rowNorm;      // line 137, 177
```

When codegen substitutes `fProxy -> double`, these become `double s = 0f;`, `double inv = (double)1f / rowNorm;`, etc. While numerically correct (C# auto-widens `0f` to `0.0`), the generated double code carries a float literal suffix -- inconsistent with every other template in the partition, which uniformly writes `(fProxy)0` or `(fProxy)1`.

**Fix direction:** Replace `0f` with `(fProxy)0` and `(fProxy)1f` with `(fProxy)1` in both methods.


---

### 3. MEDIUM -- `maxIter` parameter name vs canonical `maxIterations`

**Files:** `Optimize.fProxy.cs:191`, `NLS.fProxy.cs:220,363,508,517,524,533,544,551,558,605,625`, `QP.fProxy.cs:97,139,697,793`, `MPC.fProxy.cs:49`, `MIP.fProxy.cs:104`

The memory notes record a T5 breaking rename: `maxIter -> maxIterations`. Every public solver entry point in this partition still uses the short form `maxIter`. If the rename has been applied to sibling solvers outside this partition (e.g. Krylov, LOBPCG, LP), the surface is inconsistent; if it has not been applied anywhere, this is a tracking note for the rename pass.

**Fix direction:** Audit the full public surface; if `maxIterations` is the canonical name, rename all occurrences in one pass.

---

### 4. LOW -- Debugging narration in code comments (CLAUDE.md violation)

**File:** `NLS.fProxy.cs:258-264` and duplicate at `394-403`

```
// found empirically on a NIST StRD
// case whose two parameters sensitivities differ by ~1e6x
```

Per CLAUDE.md, debugging narration ("found empirically on...") belongs in `DEVLOG.md`, not in code comments. The first sentence of each block ("flatThresh classifies a column as flat/negligible") is a valid contract statement; the empirical-finding narration is not.

**Fix direction:** Move the history/narration portion to `OP/DEVLOG.md` under `## NLS`, retaining only the contract statement.

---

### 5. LOW -- Empty XML summary on `fProxyComp` / `iProxyComp` classes

**Files:** `OP.Component.fProxy.cs:11-12`, `OP.Component.iProxy.cs:15-16`

```xml
/// <summary>
/// </summary>
public static partial class fProxyComp {
```

The class-level XML doc is present but empty. Generated code will carry an empty `<summary>` tag. Siblings like `Norms` and `Blas` have at least a one-line class summary.

**Fix direction:** Add a one-line summary, e.g. "Componentwise scalar and buffer arithmetic."

---

### 6. LOW -- MPC.State.fProxy.cs header comment references DEVLOG fix inline

**File:** `MPC.State.fProxy.cs:43`

```
// ... using block k instead (x_{k+1}) is the off-by-one this
// file DEVLOG documents fixing.
```

This meta-commentary about the DEVLOG itself is history/narration. The contract-relevant statement (use block k-1) is already clear without it.

**Fix direction:** Delete the sentence referencing the DEVLOG; the contract statement on the preceding lines suffices.

---

### 7. LOW -- Misleading scratch variable name `RbarM`

**File:** `MPC.State.fProxy.cs:482`

```csharp
var RbarM = new fProxyMxN(nu, nu, Allocator.Temp);   // zero-initialized
for (int k = 0; k < N; k++)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            RbarM[k * m + i, k * m + j] = R[i, j];
```

The variable is named `RbarM` but it holds Rbar (block-diagonal R), not Rbar*M. The next line `Blas.dot(in M, in RbarM, ref MtRbar, transposeA: true)` produces M^T @ Rbar. The name `Rbar` or `RbarBlock` would match what it actually contains.

**Fix direction:** Rename `RbarM` to `Rbar` or `RbarBlock`.


---

## Areas confirmed clean

- **MIP.Pseudocost.fProxy.cs** -- Correct double-precision accumulation for pseudocost statistics, proper division guards (`delta == 0` early return, `count > 0` checks).
- **MIP.fProxy.cs** -- Heap operations correct (sift-up/down, proper swap). SearchCore properly disposes all heap-node snapshots (both on pop and on drain). UnshiftToX handles all three variable kinds. All Allocator.Temp allocations matched by Dispose calls.
- **MPC.Info.cs** -- Singular file, enum + info struct, Burst-safe ToFixedString.
- **MPC.State.fProxy.cs** -- Constructor validates every input, disposes on LQR failure, correct Phi/Gamma assembly (verified k-1 indexing for prestabilization), soft-row slack assembly correct, prestab exclusion with deltaU throws.
- **MPC.fProxy.cs** -- Warm-start guess correctly clips to physical bounds, forward-simulates with real (A,B), converts u-to-v for prestabilization. Fallback captures u0out before QP.
- **NLS.Info.cs** -- Singular file, enum + info struct, Burst-safe.
- **NLS.fProxy.cs** -- Numeric and analytic cores structurally parallel, convergence bookkeeping in double, all scratch disposed symmetrically, robust-loss rescaling correct (verified against scipy formula). No division-by-zero in nlsScaledGradNorm (d is always non-zero when called).
- **NormsOP.iProxy.cs** -- Correct widening (long for L1/LInf, double for L2), documented long.MinValue wraparound.
- **OP.Component.fProxy.cs / iProxy.cs** -- Thin inlined forwarders, skipFor[u] correctly gates signed-only ops (signFlip, abs, relu).
- **OP.Dot.fProxy.cs / iProxy.cs** -- Alias guards on every ref-dest path, iProxy variant correctly excludes dotSelf/dotRows/householderInPlace (float-only ops).
- **OpHelpers.Shared.cs** -- FFT/Resample type-agnostic helpers, correct base-4 digit reversal, mirror mode handles n=1.
- **OpHelpers.fProxy.cs** -- copysign/signOrOne/pythag correct (pythag uses the NR overflow-safe form).
- **Optimize.fProxy.cs** -- bisection/newtonRoot/goldenSection all correctly typed with fProxy casts, ladIRLS accumulates convergence in double.
- **QP.Info.cs** -- Singular file, clean.
- **QP.fProxy.cs** -- Extensive and complex but structurally sound. Null-space machinery, working-set assembly, ratio test, anti-cycling perturbation, warm-start repair all consistent. Cleanup Newton step after perturbation is a correct design.
- **QR.Workspace.fProxy.cs / QR.fProxy.cs** -- Blocked and unblocked factorization paths, cache overloads, solveInPlace fused kernel. Correct blocked-WY T/T-transpose direction.
- **QRCP.Workspace.fProxy.cs / QRCP.fProxy.cs** -- Downdating guard, cache validation.
- **Query.Shared.cs** -- Correct integer decode.
- **QueryCore.Metric.fProxy.cs / iProxy.cs** -- Correct similarity/distance direction split, integer variant correctly forbids Euclidean/Cosine.

---

## Summary table

| Severity | Count |
|----------|-------|
| HIGH     | 1     |
| MEDIUM   | 2     |
| LOW      | 4     |

| Area | Status |
|------|--------|
| MIP (Pseudocost + main) | Clean |
| MPC (Info + State + solve) | Clean (2 LOW) |
| NLS (Info + solve) | Clean (1 LOW) |
| NormsOP (fProxy + iProxy) | 1 MEDIUM |
| OP.Component (fProxy + iProxy) | 1 LOW |
| OP.Dot (fProxy + iProxy) | 1 HIGH |
| OpHelpers (Shared + fProxy) | Clean |
| Optimize | Clean |
| QP (Info + solve) | Clean |
| QR (Workspace + solve) | Clean |
| QRCP (Workspace + solve) | Clean |
| Query.Shared | Clean |
| QueryCore.Metric (fProxy + iProxy) | Clean |
