# Style / Cohesion / Usability Audit — Round 2

*Historical document — method names predate the 2026-07 solver-API rework (see
docs/spec-solver-api-rework.md for the mapping).*

Date: 2026-06-28
Scope: `Assets/LinearAlgebra/CodeGen/TemplateSource/` — complete second pass, emphasis on
Realtime, Stats, Gallery, ResampleOP, FFT, Wave, Easing, ML/KMeans, Interfaces, RandomOP.
All round-1 confirmed findings are referenced by name only; this document covers NET-NEW issues
and independent verification of round-1 claims.

---

## 1. Executive Summary — Top NET-NEW Findings

These are the highest-value issues not present in round-1.

| # | Severity | Location | One-line summary |
|---|----------|----------|-----------------|
| 1 | MED | `StatsOP.fProxy.cs:651` | `covarianceInto` is public; M==1 silently produces NaN in every output cell via `1f / 0f = Inf` then `0 * Inf = NaN`. |
| 2 | MED | `RollingWindow.fProxy.cs:81` | Indexer `this[int i, int f]` has no bounds check on `f` against `Features`. Read past ring-buffer row is silent OOB (round-1 only caught missing `i` check). |
| 3 | MED | `ArenaConversions.fProxy.cs:105-106` | `Convert(fProxyN → fProxy2)` reads `mathVec[0]` and `mathVec[1]` with no `N >= 2` guard; silent OOB for any length-1 vector. |
| 4 | MED | `FFT.fProxy.cs:178,180,186` | All three error messages inside `DftCore` are hard-coded to `"dft: ..."` even when the function is called via `idft`. Misattributed errors mislead diagnostics. |
| 5 | LOW | `FFT.fProxy.cs:189 vs :52` | `dft`/`idft` with N==0 silently no-op; `fft`/`ifft` with N==0 throw. Inconsistent empty-input contracts between the two DFT families. |
| 6 | LOW | `Gallery.Phase2.fProxy.cs:35,41` | `fProxyCauchy` allocates the arena matrix at line 35 then checks singularity inside the double loop (lines 41-42). A throw after partial fill violates the validate-before-allocate convention used everywhere else in the library. |
| 7 | LOW | `StatsOP.fProxy.cs:272-305 vs :345` | `rowSum`/`colSum` silently succeed on empty matrices; `rowMin`/`rowMax`/`colMin`/`colMax` throw `InvalidOperationException` for the same input. Inconsistent empty-input contracts within the statistics family. |
| 8 | LOW | `Arena.bool.cs:9,16,41,55` | Bool allocation methods named `BoolVector`/`TempBoolVector`/`BoolMatrix`/`TempBoolMatrix` (PascalCase). All fProxy/iProxy equivalents use camelCase (`fProxyVec`, `fProxyMat`). Naming diverges from the established convention; `IArenaShortcuts` declares `boolVec`/`boolMat` which matches neither. |
| 9 | LOW | `ArenaConversions.fProxy.cs:101-109 vs :12-97` | `CONVERSIONS_TO_MATH` has exactly one overload (fProxyN → fProxy2) while `CONVERSIONS_FROM_MATH` has seven (fProxy2/3/4 + 2x2/3x3/4x4). Round-trip is impossible for fProxy3, fProxy4, and any math matrix type. |
| 10 | LOW | `Wave.fProxy.cs` | `Cycles == 0` and `Square.Duty == 0` are silently replaced with 1 and 0.5 respectively. A zero-duty "always-low" signal is a legitimate request; silent substitution is undiscoverable without reading source. |

---

## 2. Round-1 Verification

Each claim from the round-1 audit was independently verified against the source.

### V1 — RollingWindow missing `i`-vs-Count bounds check (round-1: MEDIUM)

**CONFIRMED.**
`RollingWindow.fProxy.cs:81`: `get => _buffer[RingRow(i), f]` — no check that `0 <= i < _count`.
`RollingWindow.fProxy.cs:85-92`: `GetSample(int i, ref fProxyN dest)` — checks `dest.N != _features` but
never checks `0 <= i < Count`. `RingRow(i) = (OldestRow + i) % _capacity` wraps silently for any `i`,
returning a stale ring slot.

NET-NEW addition: neither the indexer nor `GetSample` checks `f` against `_features` (finding #2 above).

### V2 — KMeansEnums.cs O(k²·N·D) comment is stale (round-1: LOW)

**CONFIRMED.**
`KMeansEnums.cs:5`: `// KMeansPlusPlus = D²-weighted seeding (Arthur & Vassilvitskii 2007); O(k²·N·D);`
The implemented `SeedKMeansPlusPlus` computes distances incrementally (FIX 8 applied), giving O(k·N·D).
The comment predates FIX 8 and was never updated.

### V3 — HistogramOP `densityInto`/`cdfInto` docs say "[lo, hi)" but behavior is [lo, hi] (round-1: LOW)

**CONFIRMED.**
`HistogramOP.fProxy.cs:142`: doc says `"over [lo, hi)"`.
`HistogramOP.fProxy.cs:185`: doc says `"[lo, hi)"` for `cdfInto`.
Both delegate to `histogramInto`, which assigns `x == hi` to the last bin (closed upper edge), making
the actual range [lo, hi].

### V4 — `weightedPick`/`weightedPickInpl` have `rng` last while `nextUniformInpl` has it first (round-1: MEDIUM)

**CONFIRMED.**
`RandomOP.fProxy.cs`: `weightedPick(in fProxyN weights, ref Random rng)` — rng last.
`RandomOP.fProxy.cs`: `weightedPickInpl(in fProxyN weights, ref Indices dest, ref Random rng)` — rng last.
`RandomOP.fProxy.cs`: `nextUniformInpl(ref Random rng, ref fProxyN dest)` — rng first.
All other `nextXxxInpl` methods also lead with `rng`. Inconsistency is real and confirmed.

### V5 — `householderInpl` square guard makes tall-matrix guard unreachable (round-1: LOW)

**CONFIRMED.**
`OrthoOP.fProxy.cs:19-22` (exact):
```csharp
if (matrix.IsSquare == false)                             // line 19: throws for all non-square
    throw new System.Exception("...must be square");

if (matrix.M_Rows < matrix.N_Cols)                       // line 22: dead code
    throw new System.Exception("...square or tall...");
```
`IsSquare` is `M_Rows == N_Cols`. If not square → throws at line 19. If square → `M_Rows == N_Cols`,
so `M_Rows < N_Cols` is always false and line 22 never throws. The second guard, with its message
"square or tall", is unreachable dead code AND contradicts the first guard's intention.

---

## 3. NET-NEW Findings by Subsystem

### 3.1 Realtime (`Realtime\`)

**R1 — RollingWindow indexer: missing `f` bounds check (MED)**
File: `Realtime/RollingWindow.fProxy.cs:78-81`
```csharp
public fProxy this[int i, int f]
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    get => _buffer[RingRow(i), f];   // no check: 0 <= f < _features
}
```
`_buffer` is a `fProxyMxN` with `Features` columns. Nothing prevents `f >= _features`, which reads
into the memory of an adjacent row (or past the end of the allocation). `GetSample` has the same gap:
it checks `dest.N != _features` but never validates the individual column index it passes when reading.
Fix: add `if ((uint)f >= (uint)_features) throw ...` before the `_buffer` access, or rely on
`_buffer`'s own bounds check if it has one.

---

### 3.2 Statistics (`Statistics\`)

**S1 — `covarianceInto` public with no M >= 2 guard; M==1 produces NaN (MED)**
File: `Statistics/StatsOP.fProxy.cs:633-665`
```csharp
public static void covarianceInto(in fProxyMxN A, ref fProxyMxN C)
{
    int M = A.M_Rows;
    // ... compute means ...
    fProxy invDenom = 1f / (fProxy)(M - 1);   // line 651: Inf when M==1
    // ...
    fProxy cov = acc * invDenom;               // 0 * Inf = NaN when M==1
```
When M==1: `means[c] = A[0,c]`, so `A[r,i] - means[i] = 0` for every element, `acc = 0`, but
`invDenom = Inf`; IEEE 754 gives `0 * Inf = NaN`. Every cell of C is silently set to NaN.
When M==0: `invDenom = -1f` (harmless), all acc loops skip, C filled with `0 * -1 = -0`. Silent and
wrong (should throw or return zero matrix with documentation).
The function is public and the method doc has no precondition. The comment block above it says
"Public so zero-alloc callers can reuse it" with no mention of M >= 2 requirement.
Fix: add `if (M < 2) throw new ArgumentException("covarianceInto: requires at least 2 observations (M_Rows >= 2)");`.

**S2 — Empty-matrix handling inconsistency within the statistics family (LOW)**
Files: `Statistics/StatsOP.fProxy.cs:272-305` (rowSum/colSum), `:345` (rowMin)
`rowSum` (line 277: `for (int r = 0; r < A.M_Rows; r++)`) — silently no-ops for M==0; dest is
unchanged (undefined from caller's perspective).
`colSum` (line 300: dest explicitly zeroed before loop) — well-defined zero result for M==0; silently
succeeds.
`rowMin`, `rowMax`, `colMin`, `colMax` (lines 345, 369, 393) — throw `InvalidOperationException("Cannot
compute statistics of an empty matrix.")` for either dimension being zero.
A user who hits the throw from `rowMin` and adds a size-guard will also apply it to `rowSum`,
unnecessarily restricting code flow. Fix: give `rowSum` the same empty guard as `rowMin`, or document
the divergence explicitly in both sets.

---

### 3.3 Gallery (`Arena\Gallery.*.fProxy.cs`)

**G1 — `fProxyCauchy`: singularity check after arena allocation (LOW)**
File: `Arena/Gallery.Phase2.fProxy.cs:35,41-42`
```csharp
int n = x.N;
var A = arena.fProxyMat(n, true);        // line 35: arena allocation happens first

for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
    {
        fProxy denom = x[i] + y[j];
        if (denom == (fProxy)0)          // line 41-42: throw inside double loop
            throw new ArgumentException("fProxyCauchy: x[i]+y[j] must be nonzero");
```
The rest of the Gallery (and the explicit library comment in HistogramOP: "Arguments are validated
before allocating any scratch so throw paths cannot leak memory") validates inputs before the first
allocation. Here `A` is allocated at line 35; a singularity at `(i,j)` throws after partial fill,
leaving an orphaned arena slot.
Fix: pre-scan all `x[i]+y[j]` pairs before `arena.fProxyMat`:
```csharp
for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
        if (x[i] + y[j] == (fProxy)0)
            throw new ArgumentException("fProxyCauchy: x[i]+y[j] must be nonzero");
var A = arena.fProxyMat(n, true);
```

---

### 3.4 ResampleOP (`OP\ResampleOP.fProxy.cs`)

No NET-NEW issues found. Catmull-Rom coefficients verified correct. Edge modes (clamp/wrap/mirror)
handle the n==1 degenerate case. Endpoint pinning in `resampleInto` and `resample2DInto` is correct.
All allocations are preceded by full input validation.

---

### 3.5 FFT / DFT (`OP\FFT.fProxy.cs`, `Arena\ArenaExtensions.FFT.fProxy.cs`)

**F1 — DftCore error messages misattribute `idft` errors to `dft` (MED)**
File: `OP/FFT.fProxy.cs:178,180,186`
```csharp
static void DftCore(..., bool inverse)
{
    if (inIm.N != n)
        throw new ArgumentException("dft: inRe and inIm must have the same length");    // line 178
    if (outRe.N != n || outIm.N != n)
        throw new ArgumentException("dft: outRe and outIm must match the input length");// line 180
    // ...
        throw new ArgumentException("dft: output must not alias the input");            // line 186
```
When the user calls `idft(...)` and mismatches lengths, the exception says `"dft: ..."`. Both `dft`
and `idft` call `DftCore`; the message should be `inverse ? "idft: ..." : "dft: ..."`.
Fix: replace each literal with a conditional `(inverse ? "idft" : "dft") + ": ..."`.

**F2 — Inconsistent empty-input contract between FFT and DFT families (LOW)**
File: `OP/FFT.fProxy.cs:52-53 (FftCore), :189 (DftCore)`
`fft`/`ifft`/`rfft` with N==0: `IsPow2(0)` returns false (since `0 > 0` is false), so `FftCore`
throws `"fft: length must be a power of two"`.
`dft`/`idft` with N==0: `DftCore` hits `if (n == 0) return;` at line 189 — silent no-op.
The same data (empty arrays) produces a throw from the fast-path and a no-op from the slow-path.
Fix: either add `if (n == 0) return;` at the top of `FftCore` (before the pow-of-two check), or
make `DftCore` throw for N==0 to match, and document the chosen contract.

---

### 3.6 Wave / Easing (`OP\Wave.fProxy.cs`, `OP\Easing.fProxy.cs`)

**W1 — Silent substitution for zero Cycles and zero Duty (LOW)**
File: `OP/Wave.fProxy.cs`
All four wavetable types (Sine, Saw, Square, Triangle) substitute `Cycles == 0 → 1` silently.
`Square.Duty == 0` substitutes to `0.5` silently.
A user who sets `Duty = 0` to produce an "always-low" pulse train gets a 50% duty cycle instead,
with no warning. The guard should either throw `ArgumentException` for invalid values, or be
documented in the XML summary with a `<remarks>` tag so the substitution shows up in IntelliSense.

**E1 — Missing Quint easing family (LOW, incomplete API)**
File: `OP/Easing.fProxy.cs`
`EaseInQuad/Cubic/Quart` (and their Out/InOut variants) are present. The Quint (fifth-power) family
(`EaseInQuint`, `EaseOutQuint`, `EaseInOutQuint`) is absent. All standard easing libraries (easings.net,
Tween.js, GreenSock) include Quint alongside Quart. The omission creates a gap in the progression
Quad → Cubic → Quart → [missing] Expo.

---

### 3.7 ML / KMeans (`ML\`)

**M1 — Stale O(k²·N·D) complexity comment (LOW) [confirmed V2 above]**
File: `ML/KMeansEnums.cs:5`
Already confirmed as a round-1 finding. Included here for completeness.

---

### 3.8 Interfaces and Arena Naming (`Interfaces\`, `Arena\Arena.bool.cs`)

**I1 — Bool factory naming diverges from convention (LOW)**
File: `Arena/Arena.bool.cs:9,16,41,55`
```csharp
public boolN   BoolVector(int N, bool uninit = false)          // PascalCase noun
public boolN   TempBoolVector(int N, bool uninit = false)
public boolMxN BoolMatrix(int M_rows, int N_cols, bool uninit = false)
public boolMxN TempBoolMatrix(int M_rows, int N_cols, bool uninit = false)
```
The established convention for Arena factory methods is camelCase verb style: `fProxyVec`,
`tempfProxyVec`, `fProxyMat`, `tempfProxyMat`, `Indices`, `tempIndices`. The bool methods use
PascalCase noun style, which matches C# type names rather than the method naming convention.
`IArenaShortcuts` declares `boolVec`/`tempBoolVec`/`boolMat`/`tempBoolMat` — matching neither the
interface names nor the actual implementation names, and preventing the interface from being
implemented by `Arena` without explicit interface forwarding.
Fix: rename to `boolVec`/`tempBoolVec`/`boolMat`/`tempBoolMat` to match `IArenaShortcuts` and the
convention of every other allocator method.

---

### 3.9 RandomOP and Generators (`OP\RandomOP.fProxy.cs`, `OP\GenOP.fProxy.cs`)

**RO1 — `arange` does not guard against NaN/Inf step (LOW)**
File: `OP/GenOP.fProxy.cs` (exact line depends on template expansion)
If `step` is NaN or Inf, `arange` silently fills the output vector with NaN/Inf values.
IEEE 754: `NaN * i = NaN` for all i (including i==0 because `NaN * 0f = NaN`), so a NaN step
propagates to every element. An Inf step produces `start, start+Inf, start+Inf, ...` (all Inf after
the first). Analogy: `linspace` avoids this by pinning endpoints explicitly; `arange` has no such
protection and no input validation.
Fix: add `if (!math.isfinite(step)) throw new ArgumentException("arange: step must be finite");`.

---

### 3.10 Arena Conversions (`Arena\ArenaConversions.fProxy.cs`)

**AC1 — `Convert(fProxyN → fProxy2)` silent OOB for length-1 vectors (MED)**
File: `Arena/ArenaConversions.fProxy.cs:102-109`
```csharp
public static fProxy2 Convert(this ref Arena arena, in fProxyN mathVec) {
    var vec = new fProxy2();
    vec.x = mathVec[0];   // OOB if mathVec.N < 2 (e.g. a length-1 vector)
    vec.y = mathVec[1];   // OOB
    return vec;
}
```
`fProxyN` indexer does not bounds-check in release builds (Burst removes them). A caller who
passes a length-1 `fProxyN` reads one element past the end of the allocation.
Fix: add `if (mathVec.N < 2) throw new ArgumentException("Convert: fProxyN must have at least 2 elements to convert to fProxy2");`.

**AC2 — Asymmetric FROM-math / TO-math API (LOW)**
File: `Arena/ArenaConversions.fProxy.cs:12-109`
FROM-math provides: fProxy2, fProxy3, fProxy4 (3 vector overloads) + fProxy2x2, fProxy3x3, fProxy4x4
(3 matrix overloads) = 7 overloads.
TO-math provides: fProxyN → fProxy2 only (1 overload).
There is no round-trip path for fProxy3, fProxy4, or any math matrix type. A user doing
`fromMath → compute → toMath` for anything other than 2D vectors has to hand-write the conversion.
Fix: add `Convert(fProxyN → fProxy3)`, `Convert(fProxyN → fProxy4)`, and at minimum
`Convert(fProxyMxN → fProxy2x2/3x3/4x4)`.

---

## 4. Prioritized Finding Table

| Severity | File : Line | Issue | Fix |
|----------|-------------|-------|-----|
| MED | `Statistics/StatsOP.fProxy.cs:651` | `covarianceInto` public; M==1 → `invDenom=Inf` → NaN in every output cell | Add `if (M < 2) throw` before line 651 |
| MED | `Realtime/RollingWindow.fProxy.cs:81` | Indexer `this[i,f]` no bounds check on `f` vs `Features`; silent OOB | Add `(uint)f < (uint)_features` guard or let `_buffer` check propagate |
| MED | `Arena/ArenaConversions.fProxy.cs:105` | `Convert(fProxyN→fProxy2)` reads `[0]`,`[1]` without `N>=2` guard | Add `if (mathVec.N < 2) throw` |
| MED | `OP/FFT.fProxy.cs:178,180,186` | All `DftCore` error messages say `"dft:"` even when called from `idft` | `(inverse ? "idft" : "dft") + ": ..."` |
| LOW | `OP/FFT.fProxy.cs:52 vs :189` | `fft` N==0 throws; `dft` N==0 silently no-ops | Unify: either guard in `FftCore` or throw in `DftCore` |
| LOW | `Arena/Gallery.Phase2.fProxy.cs:35,41` | `fProxyCauchy` allocates matrix before completing singularity scan | Pre-scan all (i,j) pairs before `arena.fProxyMat` |
| LOW | `Statistics/StatsOP.fProxy.cs:272 vs :345` | `rowSum`/`colSum` silent on empty; `rowMin`/`rowMax`/`colMin`/`colMax` throw | Add same `InvalidOperationException` guard to rowSum/colSum, or document divergence |
| LOW | `Arena/Arena.bool.cs:9,41` | Bool factories named `BoolVector`/`BoolMatrix`; rest of API uses camelCase; `IArenaShortcuts` declares `boolVec`/`boolMat` | Rename to `boolVec`/`boolMat`/`tempBoolVec`/`tempBoolMat` |
| LOW | `Arena/ArenaConversions.fProxy.cs:101-109` | TO-math has 1 overload; FROM-math has 7 — no fProxy3/4/matrix round-trips | Add `fProxy3`/`fProxy4`/matrix overloads for TO-math |
| LOW | `OP/Wave.fProxy.cs` | `Duty=0` silently becomes 0.5; no IntelliSense-visible documentation of substitution | Throw `ArgumentException` for invalid values, or add `<remarks>` to XML doc |
| LOW | `ML/KMeansEnums.cs:5` | Comment says O(k²·N·D) — stale since FIX 8 made seeding incremental | Update to O(k·N·D) |
| LOW | `OP/GenOP.fProxy.cs` | `arange` no guard against NaN/Inf step; output silently filled with NaN/Inf | Add `if (!math.isfinite(step)) throw` |
| LOW | `OP/Easing.fProxy.cs` | Quint family (EaseInQuint/EaseOutQuint/EaseInOutQuint) absent from Quad/Cubic/Quart/Quint/Expo set | Add the three Quint variants |
