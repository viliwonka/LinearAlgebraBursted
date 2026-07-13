# Release scan 2026-07-13 — N9 narrow pass: small folders

Partition: `TemplateSource/Debug` (9), `Interfaces` (8), `Hash` (4), `Pivot` (2), `Indices` (1),
`Realtime` (1) — 25 template files, every line read. All dimensions applied at once, plus the
narrow-pass addendum patterns. All file:line references are TEMPLATE paths under
`Assets/LinearAlgebra/CodeGen/TemplateSource/`.

Verification work done beyond reading: xxHash32 kernel checked against the reference algorithm;
Huber/Cauchy/Tukey rho/rho'/rho'' re-derived against the scipy `optimize._lsq` s = r^2 convention;
every XML-doc cross-reference (`Kalman.numericJacobianF/H`, `ekfPredict/ekfUpdate`,
`Optimize.nlsSolve`/`curveFit` overload shapes, `Rand.randomInPlace`, `Generate.sample`,
`fProxyEasing`/`fProxyWave`/`fProxyGaussian`, `arena.Indices(n)`, `Stats.covarianceInto`,
`QR.solveInPlace`, `Eigen.decompInPlace`) resolved against the real declarations; operator
forwarding roles (`Apply`/`ApplyT`/`ApplyDot`/`ApplyBlock`) checked against `Blas.dot`,
`Blas.dotSelf`, `Blas.dotRows` parameter semantics; Pivot cycle-application algorithm traced on a
3-cycle; RollingWindow ring index math traced through wrap-around; every `//+choose`,
`//alsoExpand`, `//+skipFor`, `//+copyReplace*` block in the partition simulated against
TemplateConverter.cs for every generated variant.

## HIGH

None.

## MEDIUM

**M1 — Pivot.Operations.cs:20,40,60 — static `Apply*InPlace` overloads perform no dimension
validation; mismatch is an unchecked out-of-bounds write through a raw pointer.**
`ApplyVecInPlace(ref v, ref pivot)`, `ApplyRowInPlace(ref A, ref pivot)`,
`ApplyColumnInPlace(ref A, ref pivot)` never check `v.N`/`A.M_Rows`/`A.N_Cols` against `pivot.N`,
while every instance sibling in the same file does (`ApplyRow` line 83, `ApplyColumn` line 97,
`ApplyVec` line 110, and the three `ApplyInverse*`). Failure scenario: `pivot.N = 8`,
`A.M_Rows = 4` -> `UnsafeOP.swapRows(A.Data.Ptr, fromR, toR, A.N_Cols)` with `toR = 7` writes past
A's buffer — silent memory corruption even with collections checks enabled (Pivot.Swap only checks
pivot's own bounds; UnsafeOP kernels have none). The vector variant goes through the `v[i]`
indexer, which is only guarded under ENABLE_UNITY_COLLECTIONS_CHECKS, so player builds corrupt
memory there too. Addendum pattern 5 (sibling-validation gap).
Fix direction: add the same `pivot.N` size guards the instance wrappers use to the three statics.

**M2 — Realtime/RollingWindow.fProxy.cs:88 — `GetSample(i, ref dest)` silently returns a wrong
sample for out-of-range `i`.** It validates `dest.N` but never `i` against `_count`; `RingRow(i)`
is `(OldestRow + i) % _capacity`, so `i >= Count` wraps and reads an uninitialized or
already-overwritten ring row with no error, even with collections checks on. The sibling indexer
`this[int i, int f]` (line 75) does guard via `Assume.IndexInsideBounds(new int2(_count, ...))`.
Failure scenario: Count = 3, Capacity = 8, `GetSample(5, ...)` -> garbage row 5 returned as data.
Fix direction: add the same `i`-vs-`_count` check (unconditional, matching the throwing style of
Push/AsMatrix, or at minimum under ENABLE_UNITY_COLLECTIONS_CHECKS like the indexer).

**M3 — Interfaces/LinearOperator.fProxy.cs:23 — `ApplyT` XML doc is factually wrong and contains
an internal stage reference.** Quoted: "y = A^T x. y must be distinct from x. Needed by
CGLS/LSQR/BiCGSTAB (Phase 3)." BiCGSTAB is transpose-free — `Krylov.biCGStab` never calls
`ApplyT` (verified: no ApplyT call between its entry at Krylov.fProxy.cs:700 and its last overload
at :873; actual ApplyT consumers are cgls/lsqr/lsmr and `lstsqResidual`). And "(Phase 3)" is
internal spec/dev-speak forbidden by the comment policy.
Fix direction: correct the consumer list (CGLS/LSQR/LSMR + residual audit) and drop "(Phase 3)".

**M4 — Realtime/RollingWindow.fProxy.cs:177 — retired name `StatsOP` in a shipped comment.**
Quoted: `// Time-order into a temp matrix, then reuse the StatsOP covariance core.` The public
class is `Stats` (Statistics/StatsOP.fProxy.cs declares `public static partial class Stats`; core
is `fProxyStatsCore`). Addendum pattern 2 (rename straggler).
Fix direction: say `Stats.covarianceInto`.

**M5 — Debug/Debug.Info.cs — `Print.Log` info-struct coverage stopped growing; seven newer
diagnostics structs are missing.** The file covers SolveInfo, LstsqInfo, DirectSolveInfo,
RankInfo, EigenSolveInfo, LanczosInfo, LOBPCGInfo, Pivot, Indices. LQRInfo, KFInfo, LPInfo,
MIPInfo, MPCInfo, NLSInfo, QPInfo all expose the identical `FixedString128Bytes ToFixedString()`
shape (verified in OP/Control.Info.cs, Kalman.Info.cs, LP.Info.cs, MIP.Info.cs, MPC.Info.cs,
NLS.Info.cs, QP.Info.cs) but have no `Print.Log` overload anywhere in TemplateSource — an
accidental-looking asymmetry for users who reach for `Print.Log(info)` after using it with the
older solvers. Also the class doc (lines 6-8) lists only "OP/SolveInfo.cs, OP/Eigen.Info.cs"
though LOBPCGInfo lives in OP/LOBPCG.Info.cs.
Fix direction: add the seven one-line overloads (or record the cut-off as a deliberate decision).

**M6 — Hash/Hash.Shared.cs:10-15 (also Hash.fProxy.cs:40-43, Hash.iProxy.cs:24-30) —
template-editor hazard notes ship in the generated package; the Debug folder solved the identical
problem via DEVLOG.** Quoted: `// NOTE FOR EDITORS: this file must never contain the literal
proxy-token substrings...`. These are codegen-editing instructions, meaningless (and mildly
confusing) inside generated output whose banner already says DO NOT EDIT. Precedent:
Export.bool.cs keeps a 2-line pointer and the detail lives in Debug/DEVLOG.md. The Hash folder
has no DEVLOG.md at all.
Fix direction: create Hash/DEVLOG.md, move the hazard prose there, keep one pointer line
(matching the Export.bool.cs pattern).

**M7 — Pivot/Pivot.cs:78,88 — `Copy()`/`InverseCopy()` hardcode `Allocator.Temp` with no
parameter and no doc.** A Persistent-allocated pivot's copy is silently Temp-lifetime; holding it
across a frame/job boundary is a use-after-free. Neither method's signature nor doc mentions the
allocator (Pivot's own constructor exposes one).
Fix direction: add an allocator parameter defaulting to Temp, or document the Temp contract on
both methods.

**M8 — Pivot/Pivot.cs:117-125 — `Print()` emits indices with no separator, producing ambiguous
output, and duplicates `ToFixedString()` without its truncation guard.** Quoted:
`toPrint.Append($"{indices[i]}");` in a bare loop — pivot (10, 2, 3) prints `1023`,
indistinguishable from (1, 0, 2, 3); large N silently truncates mid-number at the FixedString4096
cap. `ToFixedString()` (line 132) already does this correctly and `Print.Log(in Pivot)` exists.
Fix direction: delete `Print()` or make it forward to `ToFixedString()`.

## LOW

**L1 — Debug/Debug.iProxy.cs:33 — sibling drift in the last-element check.** iProxy uses
`if (i == a.N - 1)` where the fProxy sibling (Debug.fProxy.cs:36) uses `i == end - 1`; with an
explicit `end < a.N` the int print gives every element a trailing ", ". Cosmetic copy-paste
divergence. Fix: use `end - 1`.

**L2 — Debug/Debug.fProxy.cs:88 — `Spy(m, 0.01f)` float-suffixed default survives into the double
variant** (becomes 0.010000000474974513). Harmless for a sparsity threshold; addendum pattern 6
straggler. Fix: `(fProxy)0.01`.

**L3 — Debug/Debug.cs:7,13,44 — `[BurstCompile]` only on the bool `Log` overloads (and the class
declaration in this one partial); the fProxy/iProxy `Log`, `Spy`, and `Histogram` siblings have
none.** Direct-call capability differs per element type for the same API surface. Fix: pick one
convention for the whole `Print` class.

**L4 — Interfaces/PredicateQuery.fProxy.cs:22,30 and PredicateQuery.iProxy.cs:5 — internal spec
taxonomy "Group-A"/"Group-D" in public XML docs.** The group letters come from the internal query
spec and are defined nowhere user-visible (the ops are at least named alongside). Fix: drop the
group labels, keep the op lists.

**L5 — Interfaces/ResidualFunction.fProxy.cs:33 — roadmap-speak in a shipped XML doc.** Quoted:
"and (shared, standalone) a future linear IRLS facade". Forward-looking dev planning, not a
contract; the same fact already lives in Interfaces/DEVLOG.md (2026-07-12 entry). Fix: delete the
"future facade" clause.

**L6 — literal proxy tokens surviving substitution inside comments.** Debug/Debug.iProxy.cs:9
("mirroring the fProxy Print.Log overloads") ships the token `fProxy` verbatim into the generated
int/short/long/uint files; Hash/Hash.fProxy.cs:42 ships `iProxy` into the float/double files;
Debug/Export.iProxy.cs:8 ships `Export.fProxy.cs`. Template jargon in package output (comment-only,
no code effect). Fix: refer to "the float/double overloads" by description, as Hash.Shared.cs:14-15
itself recommends.

**L7 — Pivot/Pivot.Operations.cs — file is classified as a multiplying int-family template and
emits the same output path three times.** It has no `//singularFile//` marker, its NAME has no
proxy token, but the body contains the float proxy token inside the `//+copyReplaceAll` block, so
TemplateConverter routes it through the int/short/long per-type loop: after CopyReplaceAll consumes
every token, three identical `Pivot/Pivot.Operations.cs` outputs are AddCode'd to one path
(currently benign — byte-identical, last write wins; verified single file in Source/Pivot/). A
stray int-proxy token added outside a marker would silently triple-diverge. Fix direction: add
`//singularFile//` to Pivot.Operations.cs (CopyReplaceAll runs in the singular path too).

**L8 — unused using directives.** Pivot/Pivot.cs:1-2 and Pivot/Pivot.Operations.cs:1-2
(`System.Collections`, `System.Collections.Generic`), Debug/Debug.cs:2 (`Unity.Mathematics`),
Debug/Debug.PCAModel.fProxy.cs:3 (`using LinearAlgebra.ML;` — the one use site is fully
qualified). Fix: trim.

**L9 — Realtime/RollingWindow.fProxy.cs — mutable-value-struct copy trap undocumented.**
`fProxyRollingWindow` carries `_head`/`_count` by value; a struct copy shares the buffer but forks
the counters, so `Push` on a copy desyncs both views. The class doc never says "store in one place
/ pass by ref" (the same trap class the demo pass hit with warm-solver structs). Fix: one doc
sentence.

**L10 — Pivot/Pivot.Operations.cs:19,39,59 — "Applies pivot" docs don't state the direction
convention.** Traced semantics are scatter: dest[pivot[i]] = src[i] (row i moves TO row pivot[i]);
`InverseCopy`/`ApplyInverse*` give the gather direction. Internally consistent, but a user
composing with external permutation data can't tell which convention this is from the docs. Fix:
one sentence stating dest[p[i]] = src[i] on the three statics.

## Areas confirmed clean

- **Hash.Shared.cs xxHash32 kernel** — accumulator init, 16-byte stripe loop bound (`p <= limit`),
  4-byte and byte tails, length fold, avalanche: all match the canonical algorithm; `combine` =
  Round with correct non-commutativity claim; zero-length contract true (data never dereferenced).
- **Hash choose/skipFor machinery** — fProxy files use 2-slot choose lists, the alsoExpand[uint]
  iProxy file uses 4-slot lists matching its int/short/long/uint rotation; the bool-sourced
  rowHashes/colHashes block is correctly confined to the uint slot via `//+skipFor[int,short,long]`
  (plain proxy tokens inside resolve to uintN/uintVec exactly as its comment claims); all
  alsoExpand marker continuation-comment lines verified stripped from output.
- **Robust losses (ResidualFunction.fProxy.cs)** — Huber/Cauchy/Tukey Rho/RhoPrime/RhoPrime2
  re-derived under the s = r^2, 0.5*sum(rho(s)) convention: all correct, continuous at the knee,
  L2Loss is the exact identity; positive-scale guards `!(scale > 0)` correctly reject NaN.
- **LinearOperator implementations** — Apply/ApplyT role mapping onto `Blas.dot(A,x)` /
  `Blas.dot(x,A)` correct; `ApplyDot` -> `dotSelf` (square-validated) consistent with the interface
  contract; `ApplyBlock` -> `dotRows(Vrows, A, AVrows, rows)` roles verified against dotRows'
  actual semantics (no role-swapped wrapper); ColScaledOperator scratch/aliasing and rectangular
  ApplyDot caveat accurate; no `[NoAlias]` double-pointer violations in the partition.
- **KalmanModel / Sampler / ScalarFunction / PredicateQuery interfaces** — every `<see cref>` and
  `<c>` reference resolves to a real declaration with the documented signature; iProxy predicate
  coverage (int/short/long, no uint) exactly matches QueryOP.Predicate.iProxy.cs's type set.
- **Indices.cs** — clean (guarded indexer both ways, truncating ToFixedString, managed-ToString
  warning present).
- **RollingWindow ring math** — OldestRow/RingRow/Push wrap-around, Mean, AsMatrix, Covariance
  (Count >= 2 guard, correct Stats.covarianceInto reuse) all verified; arena factory validates
  capacity/features >= 1.
- **Debug.Histogram** — bin clamp (including v == hi and NaN -> clamp-to-0), maxCount >= 1 div
  guard, 3500-byte flush, Temp counts disposed on every path (empty-input early-out precedes the
  allocation).
- **Export.*.cs family** — G9/G17 round-trip choose split correct per precision; int/uint 4-slot
  choose lists correct; bool exporter has no proxy tokens (hazard respected); InvariantCulture
  everywhere.
- **Comment/DEVLOG hygiene elsewhere** — Debug/DEVLOG.md and Interfaces/DEVLOG.md exist and no
  code comment in the partition duplicates their content (Export.bool.cs keeps only the sanctioned
  pointer). One DEVLOG nit, not a code finding: Interfaces/DEVLOG.md's ResidualFunction entry says
  IfProxyScalarDerivativeFunction sits "just above" in the same file — it actually lives in
  OP/Optimize.fProxy.cs:13.

## Summary

| Severity | Count |
|----------|-------|
| HIGH     | 0     |
| MEDIUM   | 8     |
| LOW      | 10    |

Partition verdict: no wrong-result or crash defects in shipping code paths under documented use.
The real risks are the two unguarded-misuse holes (M1 static pivot applies, M2 GetSample) and the
doc lie on ApplyT (M3); the rest is consistency and comment-policy debt.
