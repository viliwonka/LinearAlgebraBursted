# API / Structure Coherence Audit — TemplateSource (pre-v1.0)

Scope: `Assets/LinearAlgebra/CodeGen/TemplateSource/**` (all subdirectories, ~190 files). Read-only
audit against `docs/dev/naming-style-guide.md` and `docs/dev/codegen-refactor-lessons.md`. Settled
decisions (kept as-is, not re-litigated below): `M_Rows`/`N_Cols` field names; `fProxy`/`iProxy`
tokens as codegen machinery in template file/type names; the iterative-solver overload-ladder
count; the `decomp`/`decompInPlace`/`decompSolve`/`solveInPlace` four-token grid itself.

## Summary

The direct-solver family (LU/CHO/CHOP/QR/QRCP/LQ/LQRP/SVD/Krylov) and the `*Info`/`*Status` result-
struct family (`SolveInfo`, `LPInfo`, `MIPInfo`, `QPInfo`, `LQRInfo`, `EigenInfo`, `SVDInfo`, ...) are
excellent — genuinely one of the most internally consistent parts of the library, worth calling out
as a model for the rest. Against that high bar, the real incoherence clusters in three places: (1) a
codegen edge case where an un-substituted proxy TOKEN leaks into a shipped public class name
(`fProxyComp` in `UtilityOP.cs`'s generated output — a real bug, not a style nit); (2) tolerance/
iteration-limit parameter naming drifts four ways across the iterative solvers a user is most likely
to chain together; (3) a handful of Pascal-case predicates and a missing `LinearAlgebra.Stats`
namespace that break patterns the rest of the library follows uniformly. `ChooseMarkerDemo` is
confirmed shipping as public generated types in the actual UPM package, exactly as suspected.

---

## 1. Naming coherence

### 1.1 Tolerance / iteration-limit parameter naming drifts four ways across the iterative solvers
*(High visibility — a user chaining Krylov → Eigen → SVD → LOBPCG calls hits a different pair of
names in almost every one.)*

| File | Params used |
|---|---|
| `OP/Krylov.fProxy.cs` (cg/pcg/minres/biCGStab/cgls/lsqr/lsmr/cgne, ~40 overloads) | `int maxIterations, fProxy tolerance` — internally 100% consistent |
| `OP/Eigen.fProxy.cs` powerIteration/inversePowerIteration | `fProxy tol, int maxIter` |
| `OP/Eigen.fProxy.cs` valuesSymmetric/symmetric (:1175, :1383, :1417) | `int maxIterPerEig, fProxy eps` |
| `OP/Eigen.fProxy.cs` decompInPlace (Jacobi, :972) | `int maxSweeps, fProxy eps` |
| `OP/Eigen.fProxy.cs` valuesQR (:1679) | `int maxIterPerRoot` |
| `OP/SVD.fProxy.cs` (:27) | `int maxIter, fProxy eps` |
| `OP/SVD.LowRank.fProxy.cs` (:724) | `int maxIter` alone |
| `OP/LOBPCG.fProxy.cs` | `fProxy tol, int maxIter` |
| `OP/QRCP.fProxy.cs` (:829), `OP/LQRP.fProxy.cs`, `OP/SVD.Subspace.fProxy.cs` (:108) | `fProxy relativeTolerance` (a related but distinct rank-threshold quantity — may be legitimately different, but reads as a fifth spelling) |

Dominant convention: `Krylov.fProxy.cs`'s `maxIterations`/`tolerance`. Suggestion: standardize the
rest of the iterative-solver family on that pair; keep `relativeTolerance` only where it is provably
a different (scale-relative rank) quantity, and say so in the doc comment.

### 1.2 Query family's exception messages reference a class name that doesn't exist
`OP/QueryOP.fProxy.cs`, `QueryOP.iProxy.cs`, `QueryOP.Predicate.fProxy.cs`: the class is declared
`public static partial class Query` (QueryOP.fProxy.cs:23), but essentially every thrown exception
in the family (~30+ call sites) is string-literalled as `"QueryOP.<method>: ..."`, e.g.
`QueryOP.fProxy.cs:38` `"QueryOP.argMaxAbs: empty input"`, `:87` `"QueryOP.rowArgMin: colIndexPerRow.N
must equal A.M_Rows"`. Every other facade in the library (`Rand`, `LU`, `Eigen`, ...) uses its real
class name in exception text. High visibility: this is literally the text a v1.0 user reads the first
time they hit a validation error in this family. Fix: bulk `"QueryOP.` → `"Query.`.

### 1.3 Pascal-case predicates inside otherwise-lowercase files
`Analysis/BoolAnalysis.cs:26,36,49` — `IsAllSame<T>`, `IsAllEqualTo<T>`, `IsAnyEqualTo<T>` — sit in
the same file family as `Analysis.fProxy.cs`'s `isAnyNan`/`isZero`/`isSymmetric`/`isDiagonal`/
`isOrthogonal` and `Analysis.iProxy.cs`'s `isZero`/`isIdentity`/`isSymmetric` (all lowercase). This is
exactly the Pascal `Is...` anti-pattern the style guide explicitly rejects (`isSymmetric`, not
`IsSymmetric`, confirmed against `math.isnan`/`math.isfinite` precedent). Same file's `any`/`all` are
correctly lowercase. Also `Analysis/Analysis.fProxy.cs:70,79` — `MaxZeroError(...)` — Pascal-cased
next to lowercase siblings `trace`/`cond`/`rank`/`determinant`/`logDeterminant`. Fix: lowercase all
four (`isAllSame`, `isAllEqualTo`, `isAnyEqualTo`, `maxZeroError`) — pre-1.0, so a public rename is
cheap now.

### 1.4 `NormsOP.fProxy.cs`: same L∞ concept spelled three ways
`LInf`/`matrixLInf` (:32-34, :202) is the norm; `normalizeLMax`/`NormalizeLMax` (:73-80, :287) is
"normalize by that same norm" but swaps `Inf`→`Max`; the dispatch enum case is `Norm.Linf` (lowercase
"nf", referenced :95/:128/:168/:329). Three spellings (`LInf`, `LMax`, `Linf`) for one concept in one
file. Suggestion: rename `normalizeLMax` → `normalizeLInf`, align enum casing to `LInf`.

### 1.5 Minor / lower-signal naming items
- `OP/OP.Component.fProxy.cs` in-place ops are consistently `*InPlace` (long form) across
  `fProxyComp`/`iProxyComp`/`boolComp` — confirmed zero uses of the style guide's documented short
  `Inpl` token anywhere in `fProxy/`, `iProxy/`, `bool/`. Either the guide's `Inpl`-for-elementwise-ops
  rule is stale for this layer, or it was never actually applied — worth reconciling the doc with the
  50+ live call sites rather than the other way around.
- `Statistics/Structs.fProxy.cs:24` — `fProxyFullStats.count` is typed `fProxy` (float/double) though
  it holds an element count (`x.Data.Length`); every other count/length field in the library is `int`.
- Exception-message prefix style is inconsistent among the small grab-bag utility files:
  `OP/SwapOP.cs:20` has no method-name prefix at all; `GenOP.fProxy.cs`/`ResampleOP.fProxy.cs` use a
  bare method name (`"linspace: ..."`); `Rand`/`Query` use `Class.method:`. Minor; sweep alongside 1.2.

---

## 2. API pattern coherence

### 2.1 `Eigen.symmetric` / `Eigen.valuesSymmetric` destroy their input but don't say so in the name
`OP/Eigen.fProxy.cs:1416` — `Eigen.symmetric(ref fProxyMxN A, ref fProxyN eigenvalues, ref fProxyMxN
V, int maxIterPerEig, fProxy eps)` — its own doc comment states plainly (line 1410): **"A must be
symmetric (checked within eps) and is DESTROYED."** `valuesSymmetric` (:1175) has the identical
contract. Neither method name carries `InPlace`, and `A` isn't renamed to the documented `A_to_X`
transformation-name pattern either. Contrast the *same class's* older Jacobi path,
`Eigen.decompInPlace(ref fProxyMxN A, ...)` (:1143), which correctly signals destructiveness — and
every LU/CHO/CHOP/QR/QRCP/LQ/LQRP `decompInPlace`/`solveInPlace`. Since `symmetric`/`valuesSymmetric`
are the class doc's *recommended* replacement for the obsolete Jacobi path, this is the method a new
user reaches for first, and its name silently lies about whether `A` survives the call. Suggestion:
rename to `symmetricInPlace`/`valuesSymmetricInPlace` to match the rest of the direct-solver family.

### 2.2 `addInPlace`/`mulInPlace` scalar overloads are missing `this` — silently not usable as extension methods
`OP/OP.Component.fProxy.cs:16` `addInPlace<T>(T place, fProxy s)` and `:24` `mulInPlace<T>(T place,
fProxy s)` both lack the `this` modifier on their first parameter. Their siblings with the *identical*
`(place, scalar)` shape — `divInPlace` (:32), `modInPlace` (:85), `subInPlace` (:131) — all have `this`
and work as fluent extensions (`vec.divInPlace(5f)` compiles; `vec.addInPlace(5f)` does not — only the
static form `fProxyComp.addInPlace(vec, 5f)` compiles). The buffer-pairwise `(T,T)` overloads of the
*same* two methods (`addInPlace<T>(this T place, T from)` :50, `mulInPlace<T>(this T from, T to)`
:105) DO have `this`, so the gap is specifically the two scalar forms. Reads as a copy-paste
omission, not an intentional choice — exactly the kind of thing a v1.0 user trips over in their first
five minutes. Fix: add `this` to both.

### 2.3 `QP.solve` orders its operands differently from `LP.solve`/`MIP.solve`
`LP.fProxy.cs:51` and `MIP.fProxy.cs:122` both order `(A, b, c, senses, ...)` — constraints, then
cost. `QP.fProxy.cs:144` orders `(Q, c, A, b, senses, ...)` — cost, then constraints. Same four
semantic slots, reversed macro-order between the LP/MIP pair and QP. Not necessarily wrong (Q could
be argued QP's defining input) but currently looks unplanned rather than deliberate — worth either
aligning QP to `(A, b, Q, c, senses, ...)` or adding a one-line doc note on why QP is cost-led.

### 2.4 `KMeans.fit` doesn't carry the result-struct pattern its ML sibling `PCA` uses
`ML/KMeans.fProxy.cs:47-58` returns `void` with raw `out fProxy inertia, out int iters` — no
convergence flag at all. Its neighbor `ML/PCA.fProxy.cs:247` (`fitCov`) returns `bool` + `out
EigenInfo info`, and `ML/PCA.Model.fProxy.cs:50-56`'s `fProxyPCAModel` has `converged`/`Solved`/an
implicit-bool operator/`ToFixedString()` — the same family shape used everywhere else in the library.
A user moving from `PCA.fitCov` to `KMeans.fit` loses the `if (result)` idiom and any way to
distinguish "hit maxIter without settling" from a clean stop except by comparing `iters == maxIter`
themselves. Suggestion: give `KMeans.fit` a small result shape mirroring `fProxyPCAModel`, or
document explicitly why k-means (which always "succeeds" up to maxIter) is exempt from the family
pattern.

### 2.5 Workspace-struct family is named `*Cache`, contradicting the style guide's own documented `_WS` convention
`docs/dev/naming-style-guide.md` documents: "`_WS` suffix = a reusable zero-alloc workspace struct
(`fProxyBidiag_WS`, `fProxySVDThin_WS`)." **Neither of those two example type names — nor any `_WS`
type at all — exists anywhere in the codebase** (grepped `TemplateSource/**` for `_WS\b`: zero
matches). What actually exists, and is 100% internally consistent, is a `*Cache` family: 19 workspace
structs (`fProxyBidiagCache`, `fProxyCHOPCache`, `fProxyEigenSymCache`, `fProxyLanczosCache`,
`fProxyFFTCache`, `fProxyLOBPCGCache`, `fProxyLPCache`, `fProxyLQCache`, `fProxyLQMinNormCache`,
`fProxyLQRPCache`, `fProxyQRCache`, `fProxyQRCPCache`, `fProxySVD{,Full,Thin,Values,Truncated,
Randomized}Cache`, `fProxyKMeansCache`, ...). The library itself is coherent; the *written style
guide* is stale and would actively mislead a contributor into writing the wrong suffix. Secondary:
the files are named `*.Workspace.fProxy.cs` while the type inside is `...Cache` — a filename/type-name
mismatch a reader notices immediately jumping from the file tree to the code. Fix: update the style
guide's `_WS` section to document `Cache` (or rename the 19 types — much more expensive, not
recommended pre-ship at this scale); separately, rename the files to `*.Cache.fProxy.cs` to match.

### 2.6 `boolN` is missing two members its `fProxyN`/`iProxyN` siblings expose
- No `ToString()` override: `fProxyN.cs:202`, `fProxyMxN.cs:209`, `iProxyN.cs:199`, `iProxyMxN.cs:206`
  all override `ToString()`; `bool/boolN.cs` and `bool/boolMxN.cs` do not (confirmed — full files
  read). `Debug.Log(myBoolVec)` prints the struct type name instead of contents where the identical
  call on a float/int vector prints values.
- No public standalone-size constructor: `fProxyN.cs:71` / `iProxyN.cs:74` both expose `(int n,
  Allocator allocator = Allocator.Invalid, bool uninit = false)`; `boolN.cs` (confirmed, full file
  read) only has a copy-constructor and `internal` arena-tracked constructors — `new boolN(5)` does
  not compile where `new floatN(5)` / `new intN(5)` do. (`boolMxN` DOES have the matching
  `(M_rows, N_cols, Allocator, uninit)` constructor, so the gap is specific to the vector/N level.)

Fix: add both to `boolN` to match the other two families.

---

## 3. Structure

### 3.1 An un-substituted codegen TOKEN ships as a real public class name in the generated package
`OP/UtilityOP.cs` is marked `//singularFile//` (emit once, no per-type multiplication) but its class
declaration (`public static partial class fProxyComp`, line 8) sits OUTSIDE the file's
`//+copyReplace` block, so codegen never substitutes it. The generated output that actually ships
(`Assets/LinearAlgebra/Source/Generated/OP/UtilityOP.cs`, confirmed by direct read) is:
```csharp
namespace LinearAlgebra {
    public static partial class fProxyComp {
        public static void zeroInPlace(in floatN vec) { ... }
        public static void zeroInPlace(in doubleN vec) { ... }
    }
}
```
i.e. a public class **literally named `fProxyComp`** — the raw proxy placeholder token, not `floatComp`
or a merged partial fragment of it — ships in the v1.0 UPM package. `zeroInPlace` is a real, useful
method (matches `addInPlace`/`mulInPlace`/... in `OP.Component.fProxy.cs`, which correctly generates
`floatComp`/`doubleComp`), but it lives on a disconnected, oddly-named type a user will never find via
`floatComp.` autocomplete, and `fProxyComp` as an identifier reads as an obvious leftover/bug to
anyone browsing the shipped source. This is the single most concrete, mechanically-verifiable defect
found in this audit — not a style question. Fix: fold `zeroInPlace` into `OP.Component.fProxy.cs`
proper (a normal fProxy-multiplied file) and delete `UtilityOP.cs`; the file's name never described
its content anyway ("Utility" tells a reader nothing about "Comp").

### 3.2 `Statistics/` lives in the bare `LinearAlgebra` namespace, not the documented `LinearAlgebra.Stats`
`Statistics/StatsOP.fProxy.cs:5`, `StatsCore.fProxy.cs`, `StatsCore.iProxy.cs`, `Stats.iProxy.cs`,
`HistogramOP.fProxy.cs`, `HistogramCore.fProxy.cs`, `Structs.fProxy.cs` are all `namespace
LinearAlgebra`. The style guide names four domain sub-namespaces explicitly: `LinearAlgebra.Stats`,
`.ML`, `.Gallery`, `.Realtime`. Verified the other three ARE namespaced correctly (`ML/PCA.fProxy.cs:4`
→ `LinearAlgebra.ML`; `Sparse/SparseOP.fProxy.cs:7` → `LinearAlgebra.Sparse`; `Realtime/
RollingWindow.fProxy.cs:10` → `LinearAlgebra.Realtime`) — `Stats` is the one domain that never got
moved into its own namespace. Fix: move `Stats`/`Histogram`/their `*Core` classes into
`LinearAlgebra.Stats`.

### 3.3 `Debug/` folder's files all declare class `Print`, never `Debug`
`Debug/Debug.cs:10`, `Debug.fProxy.cs`, `Debug.Info.cs`, `Debug.PCAModel.fProxy.cs`,
`Debug.Histogram.fProxy.cs`, `Debug.iProxy.cs`, plus `Sparse/Debug.Sparse.fProxy.cs` are all `public
static partial class Print` (confirmed, Debug.cs:10) — a deliberate choice to avoid colliding with
`UnityEngine.Debug`, but the folder/file naming never signals it, so a contributor grepping for "the
Debug class" won't find it under `Debug.*.cs`. Also: `Print.Histogram(in fProxyN, ...)`
(`Debug.Histogram.fProxy.cs`, an ASCII bar-chart dumper) shares its exact name with the unrelated real
`Histogram` static class (`Statistics/HistogramOP.fProxy.cs`) — two unrelated APIs both answering to
"Histogram" in the same assembly is a mild autocomplete trap. Low severity; fix opportunistically
(rename files to `Print.*.cs`, or leave a folder-level note).

### 3.4 `boolN.Shortcuts.cs` / `boolMxN.Shortcuts.cs` drop the `[AggressiveInlining]` attribute its siblings carry
`fProxyN.Shortcuts.cs`, `fProxyMxN.Shortcuts.cs`, `iProxyN.Shortcuts.cs`, `iProxyMxN.Shortcuts.cs`
decorate every forwarding shortcut with `[MethodImpl(MethodImplOptions.AggressiveInlining)]`; the
`bool` family's Shortcuts files have identical method bodies but omit the attribute on every method.
Not user-visible via the API surface, but a real Burst-codegen performance-parity gap across the
three type families, cheap to fix.

### 3.5 Minor structural drift (low priority, batch with an unrelated pass)
- `iProxyN.Comparators.cs` orders `#region PREDICATES` before `#region COMPONENT-WISE OPERATIONS`;
  `iProxyMxN.Comparators.cs` places the same regions in the opposite order.
- `boolMxN.Operators.cs`'s `TempCopy()` names its local `vec`; `fProxyMxN.Operators.cs`/
  `iProxyMxN.Operators.cs` name the identical MxN-typed local `matrix`. Private locals only, no public
  impact.

---

## 4. Sore thumbs

### 4.1 `ChooseMarkerDemo.fProxy.cs` / `.iProxy.cs` ship as public generated types in the real UPM package
Confirmed directly: `ChooseMarkerDemo.fProxy.cs` declares `public static class fProxyChooseMarkerDemo`
in namespace `LinearAlgebra`, with its own doc comment stating "Not part of the library's public
surface" and "Covered by ChooseMarkerTests.fProxy.cs" — yet it sits at the `TemplateSource` root (not
under a Tests/Demo folder) and IS public. Its generated output —
`Assets/LinearAlgebra/Source/Generated/ChooseMarkerDemo.{float,double,int,short,long}.cs` — lives
directly inside `Assets/LinearAlgebra/Source/Generated/`, which is exactly the folder path the
README's UPM install instructions point at
(`.../LinearAlgebraBursted.git?path=Assets/LinearAlgebra/Source`). So `floatChooseMarkerDemo`,
`doubleChooseMarkerDemo`, `intChooseMarkerDemo`, `shortChooseMarkerDemo`, `longChooseMarkerDemo` — five
public classes whose only member is `DemoThreshold` — genuinely ship in the public v1.0 package and
pollute `LinearAlgebra.` autocomplete. Fix: move to `LinearAlgebra.Internal` (mirrors the
`Unsafe_OP`-style "advanced/internal" namespace signal already established) or gate it out of the
Source-facing generation path entirely; at minimum it should not resolve under the bare `LinearAlgebra`
namespace a consumer's `using LinearAlgebra;` pulls in.

### 4.2 `fProxyComp` — see §3.1. Listed here too because it is, functionally, the same class of problem as
`ChooseMarkerDemo`: internal codegen vocabulary leaking into the public, shipped API surface — just
via a different mechanism (a missed substitution instead of an un-gated demo file).

### 4.3 `LOBPCGInfo`'s workspace carries two fields the library's own comment says are dead
`OP/LOBPCG.Cache.fProxy.cs:121-127` — the struct doc states: "`AXnext`/`APnext` are allocated but
UNUSED ... dead weight — do not rely on their contents" (confirmed, lines 121, 127). They're still
allocated by `Arena.fProxyLOBPCGCache` (:193, :195), costing 2×k×n scratch per call for nothing. An
honest comment, but a public struct field the library itself says to ignore is exactly the kind of
thing a first-time user notices and wonders whether they've misunderstood the API. Fix: remove the
two fields and their allocation rather than ship known-dead public state.

### 4.4 Pascal-case predicates and namespace gaps (see §1.3, §3.2) read as accidental to a first-time user
skimming autocomplete, even though each individual instance is small.

---

## Verified coherent (checked, no finding — for awareness of what's covered)

- **Result/info-struct family** (`SolveInfo`, `LstsqInfo`, `DirectSolveInfo`, `RankInfo`, `SVDInfo`,
  `EigenInfo`, `EigenSolveInfo`, `LanczosInfo`, `LOBPCGInfo`, `LPInfo`, `QPInfo`, `MIPInfo`,
  `LQRInfo`) — every one follows the identical shape (`status` enum, `Solved` bool property, implicit
  `bool` operator, Burst-safe `ToFixedString()`, managed `ToString()`), spot-read across ~9 files.
  Genuinely excellent, model-quality consistency — worth holding up as the pattern the rest of the
  library should match.
- **4-token direct-solver grid** (`decomp`/`decompInPlace`/`decompSolve`/`solveInPlace`) — correctly
  applied across LU, CHO, CHOP, QR, QRCP, LQ, LQRP.
- **`(input matrix, rhs, output)` parameter order** — uniform across LU/CHO/CHOP/QR/QRCP/LQ/LQRP/
  SVD.Solvers/Krylov `decompSolve`/`solveInPlace`/`cg`-family calls.
- **Exception discipline** (`ArgumentException`/`ArgumentOutOfRangeException` only, static string
  literals, no custom types, no runtime concatenation) — holds everywhere checked except the `Query`
  family's `InvalidOperationException` usage (noted informally, not elevated to a top finding since
  it's confined to one file family and easy to fold into the §1.2 fix).
- **Bitwise operators** (`&`,`|`,`^`,`~`,`<<`,`>>`) correctly present only on `iProxy`/`bool`, absent
  on `fProxy` — precision-justified, not a gap.
- **`Rand.*InPlace` family** (RandomOP.fProxy/iProxy/bool + RandomMatrixOP) — zero naming drift.
- **`Arena` core struct stays lean** per its own documented rule — every domain factory
  (`fProxyPCAModel`, `fProxyKMeansCache`, `fProxyRollingWindow`, `fProxyBlockJacobi`, galleries) is
  correctly an `ArenaExtensions` method, not grafted onto `Arena` itself.
- **Indexing/comparator surfaces across fProxyN/iProxyN/boolN and their MxN counterparts** — byte-
  identical shape (linear `int`/`System.Index` access, from-end support, all four `(r,c)` int/Index
  combinations for MxN), and comparator operators (`<`,`>`,`<=`,`>=`,`==`,`!=`) named/shaped
  identically with the same commutative-overload trick across families.

---

## Parked for ruling (added 2026-07-11 after the cleanup arc)

### P.1 Warm-state structs are self-managed while all scratch caches are arena-backed
`fProxyLPCache` (LP.Cache.fProxy.cs), `LPBasis` (LP.Info.cs), and `fProxyLQRState`
(Control.fProxy.cs) use `new X(n, m, allocator)` + manual `.Dispose()` — every other workspace
struct (all 17 `*Cache` types) is created via `arena.fProxyXCache(...)` and dies with the arena.
An LP/LQR warm-solve user must remember to Dispose or leak; the arena exists to remove exactly
that footgun. Proposed resolution: add arena factories (`arena.fProxyLPCache(n, m)`,
`arena.fProxyLQRState(...)`, `arena.LPBasis(...)`) allocating the internal buffers from the arena,
keep the ctor+Dispose overloads for compatibility. Note `LPBasis` is type-agnostic like
Pivot/Indices (which are also ctor-based — precedent either way). NOT ruled on by the owner yet.

### P.2 Triangle-trust convention splits dense-vs-sparse (added 2026-07-11)
Dense SPD family (CHO, CHOP) and sparse fProxyIC0 read the LOWER triangle (upper ignored,
LAPACK 'L' convention); symmetric BSR storage (ToBSRSymmetric) canonicalizes UPPER (lower-triangle
blocks throw). One user, two opposite halves to remember. Eigen.symmetric* verifies both halves;
SSOR mirrors to full; ILU0/Blas.Triangular unaffected. Options: (a) cross-reference docs at both
sites (cheap, recommended); (b) flip symmetric BSR to lower storage (breaking, touches symmetric
spMV/builder/mirror); accepting both halves in ToBSRSymmetric was already design-rejected
("don't mask caller bugs"). RULED 2026-07-11: owner wants this FIXED (deferred to TODO, not immediate); confirm approach (doc cross-ref vs storage flip) when picked up.

RESOLVED 2026-07-12: option (b) — `ToBSRSymmetric` now canonicalizes LOWER (rejects upper-triangle
triplets), matching CHO/CHOP/fProxyIC0. spMV/spMM/MirrorToFull/transpose were already side-neutral
(no code change); `fProxyIC0` gets a real win, its symmetric-storage input is now consumed
zero-copy (no mirror-to-full pass) since the stored lower pattern IS the IC(0) pattern. See
Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/DEVLOG.md for the rationale.
