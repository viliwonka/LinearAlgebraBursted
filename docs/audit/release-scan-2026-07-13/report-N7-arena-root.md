# N7 narrow scan — Arena (18 files) + TemplateSource root (11 files)

Scanner: N7 (narrow, all dimensions + wide-pass addendum patterns). Date: 2026-07-13.
Every line of every file in the partition was read. Cross-references (UnsafeMathOP.setIndexZero/
setIndexOne, OP.Component mulInPlace, QueryOP count/score kernels, FFT.irfft, fProxyN ctors,
fProxyMxN.Shortcuts, GenUtils/TemplateConverter marker semantics, Arena & root DEVLOG.md) were
verified in their own files where a partition file's correctness depends on them.

Paths below are relative to Assets/LinearAlgebra/CodeGen/TemplateSource/.

---

## HIGH

### H1. Arena/ArenaExtensions.iProxy.cs:77-79 — iProxyLinVec interpolates in float; long/int/uint variants produce wrong interior values

    float scale = 1 / (float)(N - 1);
    for(int i = 0; i < N; i++) {
        vec[i] = (iProxy)math.lerp((iProxy)start, (iProxy)end, i * scale);
    }

math.lerp resolves to the FLOAT overload for every generated variant (long/int/uint to float is
the better conversion target than double, and scale is float). float has a 24-bit mantissa, so:
- long variant: longLinVec(5, 0, 4_611_686_018_427_387_904) (2^62) — interior values can be off
  by up to ~2^38 (~hundreds of billions). Only the two pinned endpoints are exact.
- int variant: any endpoint magnitude > 2^24 (16.7M) gives interior values off by up to ~128
  (e.g. intLinVec(3, 0, 2_000_000_001) midpoint wrong by ~tens of counts).
- uint variant (file opts into uint via alsoExpand): same beyond 2^24.
The endpoint-pinning comment shows ~1-ulp truncation was considered, but the interior-precision loss
for wide-range types was not. Fix direction: interpolate in double (double scale, double lerp
overload) — exact for all int/uint and for long up to 2^53; or a per-type choose marker.

---

## MEDIUM

### M1. Arena/Arena.cs:97-123 — AllocationsCount doc claims "every tracked family" but the property omits bool records AND Pivot/Indices buffers
AllocationsCount sums only the fProxy family (vec/mat/BSR/builders/BlockJacobi) and iProxy vec/mat.
BoolVecRecords/BoolMatRecords are not summed (the doc's own parenthetical admits it: "Bool
allocations were never included in this count at all"), and Pivots.Length / IndexBuffers.Length are
not summed AND are not mentioned as exceptions at all — yet both are "tracked" (Pivots.Add,
IndexBuffers.Add, freed by Clear/Dispose). User-visible: arena.Pivot(8); arena.boolVec(4); leaves
AllocationsCount == 0. TempAllocationsCount likewise omits TempBoolVecRecords/TempBoolMatRecords.
Fix direction: either include bool + Pivot/Indices in the counts or make the XML doc list every
excluded family truthfully (and move the "pre-existing gap" history to DEVLOG, see L2).

### M2. Arena/Arena.bool.cs:31,135,155 — bool TEMP factories are public; fProxy/iProxy temp factories are internal (public-surface accident)
public boolN boolTempVec(int N, ...), public boolMxN boolTempMat(int M_rows, int N_cols, ...),
public boolMxN boolTempMat(in boolMxN mat) — but the siblings are internal fProxyN
fProxyTempVec(...) / internal fProxyMxN fProxyTempMat(...) (Arena.fProxy.cs:87,204,224) and the same
for iProxy. The temp pool's factory surface is internal everywhere except bool, so the generated
package publicly exposes temp allocation for bool only. Fix direction: make the three bool temp
factories internal (or consciously open all of them).

### M3. Arena/ArenaExtensions.fProxy.cs:37,183,186,208,211,229,233,264 (+ ArenaExtensions.iProxy.cs:42,207,210) — exception messages name methods that do not exist, passed as ArgumentOutOfRangeException's paramName
Examples: throw new System.ArgumentOutOfRangeException("BasisVector: Index out of bounds"),
"RotationMatrix: Matrix must be at least 2x2", "PermutationMatrix: ...", "HouseholderMatrix: ...",
"HilbertMatrix: ...". The real generated methods are floatBasisVec/doubleRotationMat/etc.; the
Gallery siblings do it right ("fProxyHilbert: n must be >= 1" generates to "floatHilbert: ...").
Additionally the string is passed as ArgumentOutOfRangeException's paramName argument, so it renders
as "Parameter name: BasisVector: Index out of bounds". Fix direction: use the template method name
in the message (so substitution yields the true generated name) and the message ctor /
ArgumentException consistently.

### M4. Arena/ArenaExtensions.fProxy.cs:32-42,178-190,203-216 (+ ArenaExtensions.iProxy.cs:37-47,202-215) — validate-after-allocate: guards run only after the arena allocation
fProxyBasisVec allocates arena.fProxyVec(N) before the index bounds check; fProxyRotationMat /
fProxyPermutationMat / iProxyPermutationMat allocate arena.fProxyIdentityMat(M) before checking
M < 2 and i/j bounds. With a negative M/N the failure is a confusing UnsafeList/Malloc error (or
worse in a no-checks build) instead of the intended ArgumentException; with a merely-invalid index
the rejected buffer stays live in the arena until Clear. Siblings validate first
(fProxyHouseholderMat, fProxyHilbertMat, all of Gallery.*). Fix direction: hoist the guards above
the allocation, matching the Gallery pattern.

### M5. Arena/ArenaExtensions.Query.fProxy.cs:109-162 (+ ArenaExtensions.Query.iProxy.cs:104-149) — k-wrappers allocate scores from A's arena, not the receiver arena; doc says both come "from arena"
All eight kNearest/kFarthest wrappers do: var idx = arena.Indices(clampedK); scores =
A.fProxyVec(clampedK); — idx from the extension receiver arena, scores from A's OWNER arena
(fProxyMxN.Shortcuts routes to OwnerArena). The summary doc on each ("Allocates clamped-k Indices +
fProxyN from arena") is wrong for scores whenever the receiver differs from A's owner, and the two
outputs then have different lifetimes (one survives the other's Clear). The clampedK <= 0 early-out
has the same split (scores = A.fProxyVec(0, true)). Fix direction: allocate scores from arena too
(arena.fProxyVec(clampedK)), or document the split.

### M6. Arena/ArenaExtensions.Query.fProxy.cs:49 + ArenaExtensions.Query.iProxy.cs:47 — public parameter named tol where the canon (and the kernel it forwards to) is tolerance
fProxyNonzeroIndices(this ref Arena arena, in T x, fProxy tol) forwards to Query.countNonzero(in x,
tol) whose own parameter is named tolerance (QueryOP.fProxy.cs:953, QueryOP.iProxy.cs:1001). Rename
straggler per the settled tolerance canon (wide-pass addendum #2), visible in the generated public
API. Fix direction: rename the wrapper parameter to tolerance.

### M7. Arena/ArenaExtensions.Query.fProxy.cs:73,91 + ArenaExtensions.Query.iProxy.cs:57,74,89 — fill passes re-implement the count kernels' predicates by copy-paste; any drift overruns the exact-sized Indices buffer
The two-pass wrappers size Indices from Query.countWithinRadius/countNonzero, then re-derive the
predicate locally: bool sim = m == Metric.Cosine || m == Metric.Dot; (duplicating
fProxyQueryCore.IsSimilarityMetric), and the iProxy nonzero fill re-implements iAbs's
MinValue-to-MaxValue clamp inline (v == iProxy.MinValue ? iProxy.MaxValue : (iProxy)(-v)). Today the
copies agree with the kernels, so counts match — but the fill loop writes idx[written++] with no
bounds check, so a future one-sided edit to either copy silently under-fills (garbage trailing
indices) or over-fills (out-of-bounds write into the exact-alloc Indices). Fix direction: call
fProxyQueryCore.IsSimilarityMetric(m) / the shared iAbs in the fill passes so both passes share one
predicate.

---

## LOW

- L1. Arena/Arena.cs:35-36 — comment history: "an Arena is single-threaded by contract, but nothing
  previously enforced that". DEVLOG entry: ## Arena.cs / - 2026-07-13 | Concurrency guards were
  added later than the single-threaded contract; before them a violation corrupted record tables
  silently. (was Arena.cs:35)
- L2. Arena/Arena.cs:99-104 — XML doc dev-history: "it still uses the old value-copy tracking
  list", "(Bool allocations were never included in this count at all -- a separate, pre-existing
  gap.)". Keep the contract (what is/is not counted), move the old-model / pre-existing-gap
  narrative to DEVLOG (ties into M1).
- L3. Consts.cs:28 — dev-note in comment: "could lower this, if necessary" on doubleZeroThreshold.
  Delete or move to DEVLOG.
- L4. Arena/Arena.cs:462-467 — AtomicSafetyHandle.Release(Safety) sits after the try/finally; if
  the disposal body throws, ExitMutation runs but Release is skipped, leaking the safety handle.
  Fix direction: release in the finally (after ExitMutation) or note the throw-path contract.
- L5. Arena/Arena.fProxy.cs:172 (+ iProxy:178) — scalar-fill MATRIX factory constructs with
  uninit=false (zero-clears) then setAll overwrites everything; the vector twin (line 56) correctly
  passes true. Wasted MemClear + sibling drift.
- L6. Arena/ArenaConversions.fProxy.cs:101-111 — the fProxyN-to-fProxy2 conversion's source
  parameter is named mathVec (copy-paste from the from-math overloads; it is the library vector),
  the arena receiver is unused, and N > 2 silently truncates (only N < 2 is rejected). The to-math
  direction covers only fProxy2 (no 3/4/matrix) — presumably parked with the interop spec; open
  coverage note, not a defect.
- L7. proxyStructs.math.cs:3 — stray "using NUnit.Framework.Constraints;" in a template-only shim
  file (drags an NUnit reference into the TemplateSource compile). Delete.
- L8. Dead template-side code (all in codegen-ignored files, never ships): proxyShims.cs:15-21
  NextIProxy is unused — and if a template ever used it, caps-token substitution would produce
  Random.NextShort/NextLong, which do not exist on Unity.Mathematics.Random; proxyStructs.cs:113-156
  anyProxy struct has zero references; markers.cs Marker.CopyBgn/CopyEnd has zero references and a
  cryptic "must be same length" comment. Delete or DEVLOG the intent.
- L9. proxyStructs.cs:45-46,148-149 — commented-out Equals overrides on fProxy/anyProxy (iProxy
  keeps its live). Commented-out code, template-only; delete or align all three.
- L10. AssemblyInfo.cs:5-7 — the generated package ships
  InternalsVisibleTo("BurstLinearAlgebra.TemplateSource.Tests-firstpass") with a comment ("the
  template test assembly compiles ... against THIS assembly") that is only true for the
  TemplateSource compile, not the shipped Source assembly it also lands in. Harmless grant; comment
  reads wrong post-generation.
- L11. Arena/ArenaExtensions.fProxy.cs:58 — fProxyRandomUnitVec: 1 / math.sqrt(sum) is unguarded;
  sum == 0 (N == 0, or the astronomically-unlikely all-zero draw) yields an Inf scale / NaN vector.
  Sibling fProxyHouseholderMat guards the same pattern with Consts.fProxyZeroThreshold.
- L12. Arena/ArenaExtensions.iProxy.cs:62-64,172-174,195-196 — the max < min fallback branches
  iterate the fill loop backwards (for (int i = N - 1; i >= 0; i--)) for no functional reason (the
  forward branch fills forward). Oddity a released-source reader will stumble on.
- L13. Arena/ArenaExtensions.FFT.fProxy.cs:36-42 — fProxyIrfft computes N = (re.N - 1) << 1 and
  allocates before FFT.irfft validates; re.N == 0 produces a negative-length arena allocation
  (confusing failure) instead of irfft's clear "re.N must be >= 2" message. Hoist a re.N >= 2 check
  (same theme as M4).
- L14. Arena factories (Arena.fProxy.cs / Arena.iProxy.cs / Arena.bool.cs, all overloads) perform
  no N >= 0 / M_rows,N_cols >= 0 validation anywhere — negative sizes flow into UnsafeList/Malloc.
  The absence is CONSISTENT across all families (so not a sibling gap), but it is the root cause
  that makes M4/L13's ordering visible. Noted for the maintainer to rule on.
- L15. Arena/Arena.fProxy.cs:251-282 (+ iProxy twins) — isPersistent/isTemp match by Data.Ptr
  equality; two zero-length buffers can both carry a null/equal pointer, giving a false positive.
  Debug-only helpers; document or length-guard.
- L16. Arena/Arena.cs:549 and Arena.cs:185 — Pivot(int size) has no XML doc while its sibling
  Indices(int n) (both on Arena and ArenaCore) is documented with the arena-owns-disposal contract
  that applies to Pivot too.
- L17. Arena/Arena.fProxy.cs:17 / Arena.iProxy.cs:23 — "public unsafe partial struct Arena {"
  same-line brace vs the Allman style of every other Arena partial (Arena.cs, Arena.bool.cs).
  Formatting outlier.
- L18. Arena/ArenaExtensions.fProxy.cs:39,53 — 1f / (-1f, 1f) float-suffixed literals in an fProxy
  template generate verbatim into the double variant. Values are exact in double (1f == 1.0) so
  this is benign — recorded only because it is addendum pattern #6.

---

## Wide-pass addendum sweep results (patterns 1-7)

1. Role-swapped InPlace wrappers — checked every forwarding call in the partition
   (UnsafeMathOP.setAll/setIndexZero/setIndexOne targets, fProxyComp.mulInPlace(vec, scale),
   Generate.*(ref dest, ...), FFT.*(in re, in im, ref dest), Query.*(..., ref idx, ref scores))
   against the kernels' actual parameter semantics: all mutate the intended operand. Clean.
2. Rename stragglers — tol found (M6); no maxIter, relTol, BSM, Solvers, MatrixMetrics, StatsOP,
   Elem, Linear, _OP in the partition. (Consts.cs:62's "maxIter/maxSweeps" comment matches the
   still-current PCA/SVD parameter names — not a straggler.)
3. Missing InPlace suffix — no partition method mutates an input without the suffix; arena
   factories/generators return fresh allocations. Clean.
4. NoAlias violations — the partition passes each pointer to at most one kernel parameter per
   call. Clean.
5. Sibling-validation gaps — M4 (guard ordering), L13, L14.
6. Literal type keywords surviving substitution — H1 (float scale in the iProxy template, harmful),
   L18 (1f in fProxy templates, benign). (fProxy)(2.0 * Math.PI) and all Gallery literals
   substitute correctly for both float and double.
7. Test-template comment debt — N/A to this partition (no test templates).

## Areas confirmed clean

- Arena/Gallery.SPD.fProxy.cs + Arena/Gallery.Special.fProxy.cs: all 29 generator formulas verified
  against the literature definitions (Rosser's 8x8 entries match exactly, trace 4040; Magic's
  Siamese walk reproduces [[8,1,6],[3,5,7],[4,9,2]]; Clement sub/super-diagonal symmetry; Frank n=3
  example; Hadamard popcount rule; Circulant positive-mod; Kahan S*R; Lauchli shape (n+1)xn;
  Cauchy/GCD/Redheffer/Parter/Prolate/Grcar/Lotkin). Guards validate before allocating; exception
  messages carry the correct template method names. No findings.
- Arena/ChunkedRecordTable.cs: chunk math, unsigned-compare bounds check, long-based byte sizing,
  free-list recycling, Init/Dispose idempotence, and the Slot-layout container-of contract are all
  sound; capacity-doubling overflow needs > 2^31 slots (unreachable).
- Arena/fProxyRecords.fProxy.cs, iProxyRecords.iProxy.cs, boolRecords.bool.cs: field-for-field
  consistent; alsoExpand[uint] correctly mirrored between iProxyRecords and Arena.iProxy.
- Arena/Arena.cs codegen structure: singular-file + alsoExpand[uint] verified against
  TemplateConverter — the copyReplaceFill/copyReplace iProxy blocks do widen to uint, so uint tables
  are constructed/cleared/disposed/counted; the fProxy blocks expand float+double only, as intended.
  EnterMutation/ExitMutation pairing is finally-correct at every guarded entry point; the ClearCore/
  ClearTempCore unguarded-core split avoids self-tripping as documented.
- Arena/ArenaConversions.fProxy.cs from-math direction: Unity.Mathematics column-major to row-major
  transposition is correct in all of 2x2/3x3/4x4.
- Arena/ArenaExtensions.Generators.fProxy.cs, ArenaExtensions.FFT.fProxy.cs (except L13),
  ArenaExtensions.cs: thin wrappers match their kernels' contracts.
- Root: Assume.cs/Assume.fProxy.cs/Assume.iProxy.cs (consistent trio; iProxy alsoExpand[uint] is
  signed-clean), ChooseMarkerDemo.fProxy.cs/.iProxy.cs (choose lists line up with the type arrays),
  Consts.cs values (float eps 2^-23, double eps 2^-52, sqrt-eps values correct; per-type thresholds
  properly split; the deleteThis template-compile block is correctly stripped from output).
- Arena/DEVLOG.md and TemplateSource/DEVLOG.md exist and already hold the non-obvious decisions
  (safety-handle placement, chunk sizing, blocking gates); no code comment duplicates them beyond
  the items flagged in L1/L2.

## Summary

| Severity | Count |
|----------|-------|
| HIGH     | 1     |
| MEDIUM   | 7     |
| LOW      | 18    |

HIGH: H1 — iProxyLinVec float-precision interpolation (wrong interior values in the long/int/uint
variants).
