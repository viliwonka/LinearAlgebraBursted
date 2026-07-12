# Release scan 2026-07-12 — area: arena

Scanned 18 template files (core). Findings: total 8 — 7 confirmed, 0 uncertain, 0 unverified, 1 refuted. Severity: 0 high, 1 medium, 6 low (non-refuted).

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/Arena.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/Arena.bool.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/Arena.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/Arena.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/ArenaConversions.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/ArenaExtensions.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/ArenaExtensions.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/ArenaExtensions.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/ArenaExtensions.FFT.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/ArenaExtensions.Generators.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/ArenaExtensions.Query.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/ArenaExtensions.Query.iProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/ChunkedRecordTable.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/Gallery.SPD.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/Gallery.Special.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/boolRecords.bool.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/fProxyRecords.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/iProxyRecords.iProxy.cs

## Findings

### 1. [medium/numerical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/ArenaExtensions.fProxy.cs:240 — fProxyHouseholderMat divides by vTv with no zero/near-zero guard, producing NaN/Inf for a zero (or tiny) input vector.

**Evidence**

```
fProxy vTv = Blas.dot(v, v);
fProxy scaleFactor = 2 / vTv;  // vTv==0 when v is the zero vector -> Inf, then every matrix entry becomes NaN. No guard, and the XML doc does not require v != 0.
```

`vTv` is zero when `v` is the zero vector, making `scaleFactor` Inf and every matrix entry NaN. There is no guard, and the XML doc does not state `v != 0` as a requirement.

**Verifier**: Line 240 computes `scaleFactor = 2 / vTv` where `vTv = Blas.dot(v, v)`, with no guard. For v==0 (or v so tiny that per-element products underflow to 0 in float32), vTv is exactly 0, scaleFactor becomes +Inf, and the rank-1 update writes NaN into every matrix entry — silently, with no thrown exception or status. Peer constructors in the same file (fProxyPermutationMat, fProxyHilbertMat) do throw on invalid preconditions, and the XML doc does not state v!=0 as a required contract, so this is a genuine validation gap consistent with the reviewer's report (severity closer to minor for a gallery helper, but the defect is real).

**Suggested fix**: Guard vTv <= epsilon (scaled to the magnitude of v) and either throw or return the identity (H = I) when v is effectively zero.

### 2. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/Arena.cs:241 — ClearCore's XML summary says it 'leaves temp pools alone', but its body calls ClearTempCore(), so it actually disposes the temp pool too (doc contradicts behavior).

**Evidence**

```
/// Disposes every PERSISTENT-pool allocation (leaves temp pools alone -- see <see cref="ClearTempCore"/>).
...
            // Calls the TEMP pool's unguarded core directly...
            ClearTempCore();
```

The XML summary claims persistent-only disposal, but the method body ends by calling `ClearTempCore()`, clearing the temp pool as well.

**Verifier**: ClearCore's XML summary at Arena.cs:240-247 states "Disposes every PERSISTENT-pool allocation (leaves temp pools alone -- see ClearTempCore)", but the method body ends at line 332 with `ClearTempCore();` (with an inline comment openly acknowledging it invokes the temp pool's core directly). This directly contradicts "leaves temp pools alone". As a side consequence, the explicit `ClearTempCore()` calls in `Clear()` (line 233) and `Dispose()` (line 433) become redundant, matching the reviewer's suggested fix direction.

**Suggested fix**: Either drop the redundant ClearTempCore() call at the end of ClearCore() (Clear()/Dispose() already call it separately) or fix the summary to state that ClearCore also clears temp.

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/ArenaExtensions.fProxy.cs:152 — Comment on fProxyRandomDiagonalMat claims 'scalar s on diagonal' but the method fills the diagonal with random values in [min,max] (copy-paste from fProxyDiagonalMat).

**Evidence**

```
// constructs diagonal matrix with scalar s on diagonal
public static fProxyMxN fProxyRandomDiagonalMat(this ref Arena arena, int N, fProxy min, fProxy max, uint seed = 65792)
```

The comment references a scalar `s` that does not exist in the signature or body; the method fills the diagonal with random entries in [min,max].

**Verifier**: At ArenaExtensions.fProxy.cs:152 the comment "constructs diagonal matrix with scalar s on diagonal" precedes fProxyRandomDiagonalMat(N, min, max, seed) whose body fills the diagonal with rand.NextFProxy(min, max) — there is no scalar s. The same stale comment is duplicated at ArenaExtensions.iProxy.cs:160 above iProxyRandomDiagonalMat. Under the project's contracts-only comment policy this is a genuine text/behavior contradiction (low-severity naming/doc defect). No codegen marker or elsewhere-guard makes it valid.

**Suggested fix**: Reword to 'diagonal matrix with random diagonal entries in [min,max]'. Same stale comment exists at ArenaExtensions.iProxy.cs:160.

### 4. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/Arena.fProxy.cs:205 — fProxyTempMat's second parameter is named M_cols instead of N_cols, contradicting the project's M_rows/N_cols convention (misleading; same in iProxy).

**Evidence**

```
internal fProxyMxN fProxyTempMat(int M_rows, int M_cols, bool uninit = false)
```

The second parameter is `M_cols` where every sibling API uses `N_cols`.

**Verifier**: Arena.fProxy.cs:205 declares `internal fProxyMxN fProxyTempMat(int M_rows, int M_cols, bool uninit = false)` and Arena.iProxy.cs:211 mirrors it with `M_cols`. Every sibling API in the project (fProxyMat/iProxyMat/boolMat/boolTempMat at Arena.*.cs:140/146/95/135, all *MxN constructors, and every Shortcuts/ArenaExtensions overload) uses `N_cols`, and MEMORY records that the M_rows/N_cols convention was explicitly kept. Behaviorally harmless (internal, forwarded positionally to the ctor on line 218) but a genuine naming inconsistency — low severity as reported.

**Suggested fix**: Rename the second parameter to N_cols for consistency. Same issue at Arena.iProxy.cs:211 (iProxyTempMat).

### 5. [low/performance/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/Gallery.SPD.fProxy.cs:129 — fProxyKMS computes rho^|i-j| with an inner multiply loop for every entry, making the Toeplitz generator O(n^3) instead of O(n^2).

**Evidence**

```
int e = math.abs(i - j);
fProxy r = (fProxy)1;
for (int k = 0; k < e; k++) r *= rho;  // per-entry power loop, sum over all i,j is O(n^3)
```

A per-entry power loop over |i-j| makes the whole fill O(n^3), even though the matrix is Toeplitz and values repeat along anti-diagonals.

**Verifier**: Lines 124-131 of fProxyKMS run an inner `for (int k = 0; k < e; k++) r *= rho;` loop for every one of the n² entries, where e = |i-j|. Summed over the grid this is n(n²-1)/3 multiplies — O(n³) — for a Toeplitz matrix whose values repeat along anti-diagonals. Sibling generators in the same file (Laplacian1D, MinIJ, Lehmer, Pei) are O(n²), so this is a real asymmetry with no guard or caching, and the suggested fix (precompute a length-n powers table) is correct and would also reduce roundoff drift under the float expansion.

**Suggested fix**: Exploit the Toeplitz structure: precompute powers rho^0..rho^(n-1) once into a length-n array, then index A[i,j] = pow[abs(i-j)].

### 6. [low/pointer/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/Arena.fProxy.cs:253 — Debug pool checks isPersistent/isTemp dereference _core with no null guard, so calling them on a default/disposed Arena null-derefs (crash), unlike every other public member here.

**Evidence**

```
public bool isPersistent(in fProxyN v) {
    for (int i = 0; i < _core->fProxyVecRecords.Count; i++)  // _core may be null (default/disposed arena)
```

`_core` is dereferenced without the null check that every other public member in this file performs, so a default or disposed Arena crashes instead of throwing cleanly.

**Verifier**: All four isPersistent/isTemp overloads at Arena.fProxy.cs:252-275 (and the parallel Arena.iProxy.cs:255-278) dereference `_core->...Records.Count` with no `_core == null` check, while every other public/internal member in the same file begins with `if (_core == null) throw new InvalidOperationException(...)`. The comment above them ("READ-ONLY: not guarded") justifies skipping only the EnterMutation/ExitMutation tripwire, not the null check — so `Arena a = default; a.isPersistent(v);` null-derefs (NullReferenceException managed, likely segfault under Burst) instead of throwing the clean InvalidOperationException that every neighboring method contracts. Severity low is fair (debug helper), but the defect is real and the suggested fix (add `if (_core == null) return false;` or throw for consistency) is appropriate.

**Suggested fix**: Add `if (_core == null) return false;` at the top of each of the four isPersistent/isTemp overloads (also in Arena.iProxy.cs:255-278).

### 7. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/Arena.fProxy.cs:133 — Code comment carries a bug postmortem ('was untracked and leaked on Dispose'), which the project's comment policy requires to live in DEVLOG, not in code.

**Evidence**

```
// forward to the (rows, cols) overload so the matrix is TRACKED in fProxyMatRecords —
// the direct `new fProxyMxN(...)` here was untracked and leaked on Dispose.
```

The past-tense "was untracked and leaked on Dispose" is development history / bug postmortem, which the strict comment policy routes to the folder's DEVLOG.md.

**Verifier**: The comment at Arena.fProxy.cs:131-136 (mirrored at Arena.iProxy.cs:137-142) contains the past-tense clause "the direct `new fProxyMxN(...)` here was untracked and leaked on Dispose" — this is a bug postmortem / development history sentence, both categories explicitly listed as forbidden in the CLAUDE.md strict comment policy ("bug postmortems and debugging narration", "development history"). A DEVLOG.md already exists in Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/, so the reviewer's fix (move the postmortem sentence there, keep the contract-only rationale about the missing guard in code, apply the same to the iProxy variant) is directly applicable. Because these files are templates, the postmortem propagates into every generated float/double/int/uint sibling in the shipped package.

**Suggested fix**: Move the historical/bug-postmortem sentence to Arena/DEVLOG.md and keep only the contract in code. Same at Arena.iProxy.cs:137-142.

## Refuted

| file:line | claim | why refuted |
|---|---|---|
| Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/Arena.cs:105 | AllocationsCount / TempAllocationsCount omit the bool families entirely, so AllAllocationsCount undercounts and a leaked/tracked bool buffer never shows in the count. | The factual observation is accurate (bool families are not summed at Arena.cs:105-123, though they are declared, tracked, and disposed), but the XML doc on AllocationsCount (lines 97-104) explicitly names this as the property's contract: "(Bool allocations were never included in this count at all -- a separate, pre-existing gap.)" Per project comment policy, code comments state contracts only, so this parenthetical IS the documented contract of the accessor. Per the review brief's rule not to report documented behavior as a bug, this is a documented limitation, not a defect. |

## Scanner notes

Verified the count-pass vs fill-pass predicates in the Query wrappers (ArenaExtensions.Query.fProxy.cs / .iProxy.cs) against QueryOP.countNonzero / countWithinRadius / RowScore / ColScore and iProxyQueryCore.IsSimilarityMetric: predicates match (math.abs > tol; iAbs with MinValue guard > tol; sim ? s>=r : s<=r), so no buffer-overflow between count and fill passes. Unity.Mathematics column-major -> library row-major Convert() mappings (2x2/3x3/4x4) are correct. Clement sub/super-diagonals are symmetric and match the documented e[i]=sqrt((i+1)(n-1-i)); Frank n=3 example, DingDong, Parter, Redheffer, Pascal, Hadamard, Magic all check out. ChunkedRecordTable slot/pointer stability, free-list recycling, generation stamping, and the Init/Clear/ClearTemp/Dispose ownership walks are sound (ClearTempCore double-walk in Dispose is idempotent, not a leak/double-free). No high-severity defects found.
