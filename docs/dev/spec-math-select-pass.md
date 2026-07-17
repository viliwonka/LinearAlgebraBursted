# Spec: branch-free `math.select` / `math.max` / `math.min` conversion pass

Status: ready for implementation. Line numbers verified 2026-07-17 against the current templates.
Anchor on the quoted expressions, not the line numbers, if the files drift again.

## Goal

Convert hot-loop ternaries and `if (x > best)` reductions to branch-free
`math.select`/`math.max`/`math.min`/`math.abs` so the loops become straight-line code Burst can
auto-vectorize (same lesson as the DetMath prototype: branch-free + straight-line = SIMD).
This is a **semantics-preserving** pass: every conversion must select the SAME value the branch
selects, bit for bit, for all inputs that can reach the site. The library's headline is
determinism (FloatMode.Strict); nothing here may change rounding, reassociate, or alter NaN
outcomes observably.

## Ground rules

- Edit ONLY templates under `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/` and
  `.../TemplateSource/OP/Blas.ColumnScaling.fProxy.cs`. Never touch `Assets/LinearAlgebra/Source/`
  (generated). Regenerate with `Tools/regen.ps1`.
- Codegen expansion: `fProxy` → {float, double}. `iProxy` → {int, short, long}, plus uint where
  the file header carries the alsoExpand marker (`UnsafeMathOP.iProxy.cs`, `UnsafeOP.iProxy.cs`,
  `SelectOP.iProxy.cs` all do). Signed-only functions are fenced by skipFor markers — leave every
  marker line exactly as-is, and never write marker tokens inside prose comments (the codegen
  parser is content-sensitive, not comment-aware; the file headers warn about this).
- `short` has NO Unity.Mathematics overloads: `math.max/min/select/abs` on short operands resolve
  via implicit widening to the `int` overload and the result MUST be cast back with `(iProxy)`.
  Precedent already in-tree: `UnsafeMathOP.iProxy.cs:68` (`clamp` writes
  `x[i] = (iProxy)math.max(min, math.min(max, x[i]));`). The `(iProxy)` cast is an identity for
  the int/long/uint expansions and the correct unchecked narrowing for short — use it on every
  iProxy-site replacement below.
- Comment policy: do not add rationale/perf comments at the sites. Keep existing contract comments
  (e.g. ColumnScaling's `// NaN-safe: !(c>0) -> 1`). Perf numbers and the pass narrative go to
  `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/DEVLOG.md` (and the Benchmarks DEVLOG for
  measurement notes).
- One commit per batch (A, B, C; D separate and optional). Full suite green between batches.

## Verified Unity.Mathematics semantics (basis for all equivalence claims)

From this project's package cache, `com.unity.mathematics@19a9377c4ffa/Unity.Mathematics/math.cs`:

```csharp
public static float  min(float x, float y)   { return float.IsNaN(y)  || x < y ? x : y; }   // :929  (double :958 same shape)
public static float  max(float x, float y)   { return float.IsNaN(y)  || x > y ? x : y; }   // :1061 (double :1090 same shape)
public static int    max(int x, int y)       { return x > y ? x : y; }                      // :987  (uint :1016, long :1045)
public static int    min(int x, int y)       { return x < y ? x : y; }                      // :855  (uint :884,  long :913)
public static int    abs(int x)              { return max(-x, x); }                         // :1813 (long :1837)
public static float  abs(float x)            { return asfloat(asuint(x) & 0x7FFFFFFF); }    // :1844 (double :1869)
public static float  select(float f, float t, bool test) { return test ? t : f; }           // :4278 (double/int/uint/long ditto)
```

Consequences used throughout:

1. **`math.select(a, b, c)` IS the ternary `c ? b : a`** — scalar select conversions are
   definitionally bit-identical, including NaN and signed-zero behavior. Zero risk.
2. **Float/double `math.max(best, x)` ≡ `if (x > best) best = x`** whenever `best` is not NaN:
   a NaN candidate `x` is ignored by both forms (`IsNaN(y)` short-circuit vs. `NaN > best` false);
   a candidate equal in value to `best` makes `max` return the *candidate* where the branch keeps
   the accumulator — a bit-pattern difference only for ±0, which cannot occur at any batch-B site
   (all candidates are `math.abs(...)` outputs or sums of them; float/double `math.abs` is a sign-bit
   mask and never yields −0). Same argument for `min` with `<`. The ONE case where they differ is a
   NaN *accumulator* (branch keeps NaN forever; `max` replaces it) — every batch-B site below either
   initializes the accumulator to a finite constant and provably never assigns NaN into it, or is
   explicitly converted with the select form instead (B8).
3. **Integer max/min are literally the ternary** — exact for every value including
   `MinValue` overflow edges (`math.abs(int.MinValue) == int.MinValue`, matching the current
   `v < 0 ? -v : v` unchecked wrap; verified equal for short/long as well).
4. FloatMode.Strict compatibility: none of these introduce reassociation, FMA, or a different
   arithmetic result — select/min/max are data selection, float abs is a bitmask. Strict forbids
   reassociation, not SIMD, so cross-arch determinism is preserved by construction.

Argument-order rule for reductions: always write the accumulator FIRST —
`best = math.max(best, x)` — so a NaN candidate lands in the `IsNaN(y)` arm and is dropped
exactly like the branch drops it.

---

## Batch A — per-element data selects (raw-pointer kernels; these actually vectorize)

Highest value, clearly safe. All sites are contiguous pointer loops already `[NoAlias]`-annotated
where applicable, so branch removal is the only thing between them and SIMD.

| # | Site (verified) | Current | Replacement |
|---|---|---|---|
| A1 | `TemplateSource/OP/SelectOP.fProxy.cs:109` (kernel `selectfProxy`) | `target[i] = c[i] ? b[i] : a[i];` | `target[i] = math.select(a[i], b[i], c[i]);` |
| A2 | `TemplateSource/OP/SelectOP.iProxy.cs:111` (kernel `selectiProxy`) | same | `target[i] = (iProxy)math.select(a[i], b[i], c[i]);` |
| A3 | `TemplateSource/OP/UnsafeMathOP.iProxy.cs:44-45` (`abs`, inside skipFor[u] fence) | `iProxy v = x[i]; x[i] = v < 0? (iProxy)(-v) : v;` | `x[i] = (iProxy)math.abs(x[i]);` (drop the local `v`) |
| A4 | `TemplateSource/OP/UnsafeMathOP.iProxy.cs:54` (`max`) | `x[i] = x[i] > y[i]? x[i]: y[i];` | `x[i] = (iProxy)math.max(x[i], y[i]);` |
| A5 | `TemplateSource/OP/UnsafeMathOP.iProxy.cs:61` (`min`) | `x[i] = x[i] < y[i] ? x[i] : y[i];` | `x[i] = (iProxy)math.min(x[i], y[i]);` |
| A6 | `TemplateSource/OP/UnsafeMathOP.iProxy.cs:83-84` (`relu`, skipFor[u] fence) | `iProxy v = x[i]; x[i] = v < 0? (iProxy)0 : v;` | `x[i] = (iProxy)math.max(x[i], (iProxy)0);` |
| A7 | `TemplateSource/OP/UnsafeMathOP.fProxy.cs:274` (`relu`) | `x[i] = x[i] < 0? 0 : x[i];` | `x[i] = math.select(x[i], (fProxy)0, x[i] < (fProxy)0);` |
| A8 | `TemplateSource/OP/UnsafeOP.iProxy.cs:34-35` (`sumAbs`, skipFor[u] fence) | `iProxy v = a[i]; sum += (iProxy)(v < 0? -v : v);` | `sum += (iProxy)math.abs(a[i]);` |
| A9 | `TemplateSource/OP/UnsafeOP.iProxy.cs:50-51` (`maxAbs`, skipFor[u] fence) | `var abs = (v < 0 ? -v : v); max = (iProxy)(max < abs? abs : max);` | `iProxy abs = (iProxy)math.abs(v); max = (iProxy)math.max(max, abs);` |
| A10 | `TemplateSource/OP/Blas.ColumnScaling.fProxy.cs:52` (`buildJacobiScale`) | `d[j] = (c > (fProxy)0) ? (fProxy)1 / math.sqrt(c) : (fProxy)1;` | `d[j] = math.select((fProxy)1, (fProxy)1 / math.sqrt(c), c > (fProxy)0);` |
| A11 (optional) | `TemplateSource/OP/SelectOP.bool.cs:65` (kernel `selectBool`) | `target[i] = c[i] ? b[i] : a[i];` | `target[i] = (c[i] & b[i]) \| (!c[i] & a[i]);` |

Per-site notes:

- **A1/A2**: `SelectOP.fProxy.cs` and `SelectOP.iProxy.cs` currently have NO
  `using Unity.Mathematics;` — add it to both. iProxy expands to int/short/long/uint; scalar
  `math.select` overloads exist for int/uint/long, short goes via int widening + the `(iProxy)`
  cast. Bit-identical by consequence 1.
- **A3/A6/A8/A9** are inside skipFor[u] fences (uint excluded) — keep the fences untouched. Integer
  equivalence is exact including `MinValue` (consequence 3). In A9, `max < abs ? abs : max` and
  `math.max(max, abs)` differ only in which operand is returned on equality — same integer value,
  no observable difference.
- **A7 (float relu) must be the select form, NOT `math.max(x, 0)`.** The current ternary maps
  NaN→NaN (`NaN < 0` is false) and −0→−0; `math.max(x[i], 0)` would map NaN→0 and −0→+0 — an
  observable, determinism-relevant change. `math.select(x[i], 0, x[i] < 0)` is bit-identical.
  (The fProxy `max`/`min` siblings at lines 147/154 already use `math.max`/`math.min` — no change.)
- **A10**: the select form computes `1/math.sqrt(c)` unconditionally; for the discarded lane
  (`c <= 0` or NaN) that speculative value is NaN or +Inf and is thrown away — Burst has no FP
  traps, and the selected value is the identical computation the ternary performs, so the function's
  documented NaN-safe contract and bit-exact output are preserved. Keep the trailing
  `// NaN-safe: !(c>0) -> 1` comment.
- **A11 (optional)**: no bool `math.select` exists; the masked form is equivalent for normalized
  bools (the only kind reachable through the public API). Take it for pattern completeness or skip
  it — it is the lowest-value site in the batch.

**Batch A benchmark obligation.** Gap: no existing benchmark exercises `Select.select`, `relu`,
`abs`, or the integer `maxAbs`/`sumAbs` kernels (checked: only `DetMathBenchmark` and
`SparseSolverBenchmark` mention these names; `KernelBenchmark.fProxy.cs` covers
dot/L1/L2/LInf/sum/GEMV only). Requirement: A/B-measure at least A1 (fProxyN select, N ≈ 10240)
and A7 (relu) before/after via `Tools/benchmark.ps1`, using a temporary section added to the
`TemplateSourceBenchmarks/KernelBenchmark.fProxy.cs` template (or a scratch benchmark file).
Record before/after numbers in the OP `DEVLOG.md`; keep the benchmark rows permanent only if they
earn their runtime (benchmark-budget concerns are on record — a reverted scratch section with
numbers in the DEVLOG is acceptable).

---

## Batch B — max/min reductions (`if (x > best) best = x` → `best = math.max(best, x)`)

Safe by consequence 2. These are index-property loops (`A[i, j]`, `fProxyN[i]`), so full SIMD may
additionally need the separate raw-pointer-hoist pass — do NOT hoist pointers here; branch removal
is still the prerequisite and stands on its own. All accumulators below start at a finite constant
and never receive NaN (a NaN candidate is dropped by both old and new forms), except B8 — see note.

| # | Site (verified) | Current | Replacement |
|---|---|---|---|
| B1 | `TemplateSource/OP/Eigen.fProxy.cs:100-101` (powerIteration residual) | `if (ri > residual) residual = ri;` | `residual = math.max(residual, ri);` |
| B2 | `TemplateSource/OP/Eigen.fProxy.cs:138-139` (powerIteration final residual) | `if (ri > finalResidual) finalResidual = ri;` | `finalResidual = math.max(finalResidual, ri);` |
| B3 | `TemplateSource/OP/Eigen.fProxy.cs:349` (inversePowerIteration vecDiff) | `if (di > vecDiff) vecDiff = di;` | `vecDiff = math.max(vecDiff, di);` |
| B4 | `TemplateSource/OP/Eigen.fProxy.cs:396` (`InversePowerResidual`) | `if (ri > residual) residual = ri;` | `residual = math.max(residual, ri);` |
| B5 | `TemplateSource/OP/NormsOP.fProxy.cs:193-194` (`matrixL1`) | `if (colSum > best) best = colSum;` | `best = math.max(best, colSum);` |
| B6 | `TemplateSource/OP/NormsOP.fProxy.cs:208-209` (`matrixLInf`) | `if (rowSum > best) best = rowSum;` | `best = math.max(best, rowSum);` |
| B7 | `TemplateSource/OP/LOBPCG.fProxy.cs:1010` (`FactorGram` ridge scale) | `{ fProxy d = math.abs(Gram[i, i]); if (d > scale) scale = d; }` | `{ fProxy d = math.abs(Gram[i, i]); scale = math.max(scale, d); }` |
| B8 | `TemplateSource/OP/LOBPCG.fProxy.cs:1027-1028` (`MinMaxDiagRatio`) | `if (d < mn) mn = d;` / `if (d > mx) mx = d;` | `mn = math.select(mn, d, d < mn);` / `mx = math.select(mx, d, d > mx);` |
| B9 | `TemplateSource/OP/LOBPCG.fProxy.cs:1071-1072` (`TryRayleighRitz` quotient envelope) | `if (q < qMin) qMin = q;` / `if (q > qMax) qMax = q;` | `qMin = math.min(qMin, q);` / `qMax = math.max(qMax, q);` |
| B10 | `TemplateSource/OP/LOBPCG.fProxy.cs:1271` (`MaxRelResidual`) | `if (rel > worst) worst = rel;` | `worst = math.max(worst, rel);` |

Per-site NaN / bit-identity notes:

- **B1-B4**: candidates are `math.abs(...)` values (≥ +0 or NaN); accumulators init `(fProxy)0`.
  NaN candidates dropped identically by both forms; no −0 possible; equal positive finite values
  are bit-unique in IEEE. Bit-identical, including the `residual` value that lands in the returned
  `EigenSolveInfo`.
- **B5/B6**: candidate is a sum of abs values — NaN possible only if the matrix contains
  NaN/±Inf, and then both forms drop the NaN candidate identically (accumulator stays non-NaN,
  consequence 2 applies). Honesty note: these are outer-loop reductions (once per row/column);
  the inner abs-sum dominates, so the win here is small — included for pattern consistency.
- **B7**: same as B1-B4 (abs candidates, zero init). Leave line 1030's
  `return mx > (fProxy)0 ? mn / mx : (fProxy)0;` alone — scalar, gates a division, no loop.
- **B8 uses the select form, not min/max, deliberately**: `mn`/`mx` initialize from DATA
  (`math.abs(L[0,0])`). If that were NaN, the branch keeps NaN forever while `math.min(mn, d)`
  would silently recover to `d` — an observable difference on a corrupted factor. The select form
  is definitionally the branch (consequence 1) — bit-identical in every case, still branch-free.
- **B9**: `qMin`/`qMax` init `fProxy.MaxValue` / `-fProxy.MaxValue` (finite); `q` NaN is dropped
  identically by both forms. Keep the `if (!(gi > (fProxy)0)) continue;` guard line as a branch —
  it is a defensive skip, not a data select. A ±0-valued `q` tie can swap the accumulator's zero
  sign, but qMin/qMax feed only comparisons (where −0 == +0) and are not returned — no observable
  difference.
- **B10**: `worst`/`rel` are double; `math.max(double)` verified NaN-aware (math.cs:1090). `rel`
  can be NaN only if `ws.lambda[i]` or `ws.residual[i]` is NaN, and then both forms drop it. Do
  NOT convert line 1269's `if (scale < (fProxy)1) scale = (fProxy)1;` — see DON'T list.

Optional same-recipe extras (take them or leave them; identical justification to B1-B7, all
one-time O(n²) setup scans with zero-init accumulators over abs values):
`Eigen.fProxy.cs:679` (Gershgorin `if (radius > bound) bound = radius;`),
`Eigen.fProxy.cs:1152` and `:1404` (`if (a > matScale) matScale = a;`),
`Eigen.fProxy.cs:1232` and `:1479` (`if (rowSum > anorm) anorm = rowSum;`).
The int counters (`if (iter > sweeps) sweeps = iter;` at 1281/1544/1724/1757) are scalar
bookkeeping, not loops over data — skip.

---

## Batch C — QR triangular R extraction (the diagonal-care site)

Two clones of the same copy loop. **Contract that must survive: the diagonal `R[r, r]` is written
earlier (unblocked: `R[d, d] = A_to_Q[d, d];` at line 138, BEFORE the reflector application
destroys it; blocked path equivalently) — by the time this loop runs, the diagonal entry in the
source matrix is STALE. The loop must not read the source at `c == r` and must not overwrite
`R[r, r]`.**

Replacement is a **loop split**, not a nested select — it is branch-free by construction, touches
exactly the same cells with exactly the same values (trivially bit-identical), and avoids the
nested-select version's pointless speculative reads:

- C1 `TemplateSource/OP/QR.fProxy.cs:147-158` (unblocked, source `A_to_Q`):

```csharp
// Copy the upper triangular part of Q into R
for (int r = 0; r < R.M_Rows; r++)
{
    int z = math.min(r, R.N_Cols);
    for (int c = 0; c < z; c++)
        R[r, c] = 0;
    for (int c = r + 1; c < R.N_Cols; c++)
        R[r, c] = A_to_Q[r, c];
}
```

- C2 `TemplateSource/OP/QR.fProxy.cs:290-297` (blocked, source `Q`): same shape with
  `R[r, c] = Q[r, c];` in the second inner loop. Keep the existing one-line comment.

The `math.min(r, R.N_Cols)` clamp preserves the current behavior for rows below the last column
(`r >= N_Cols`), where the whole row is in the `c < r` zero region. Confirm the file has
`using Unity.Mathematics;` (it uses `math.` already). No NaN concern — pure data movement.

---

## Batch D — argmax / selection-sort selections (OPTIONAL, do last or defer)

Loop-carried scalar argmax: these will NOT vectorize; the only win is removing a
hard-to-predict branch. Lower priority — implement only if batches A-C land cleanly, as its own
commit, and drop it entirely if the LU benchmark (`Tools/benchmark.ps1`, `LUBenchmark` /
`benchmark-lu.txt` baseline) shows no improvement.

Canonical pattern (definitionally the branch, consequence 1 — bit-identical selection: strict `>`
keeps the FIRST occurrence on ties, and a NaN candidate is never selected, both preserved):

```csharp
// before                                        // after
if (absValue > pivotValue) {                     bool better = absValue > pivotValue;
    pivotIndex = r;                              pivotIndex = math.select(pivotIndex, r, better);
    pivotValue = absValue;                       pivotValue = math.select(pivotValue, absValue, better);
}
```

Verified sites (all the same shape; keep every surrounding zero-pivot/singular check untouched):

- `TemplateSource/OP/LU.fProxy.cs:128-134`, `:223-229`, `:352-358`, `:441-447` (partial-pivot argmax;
  the 352/441 pair reads through the permutation `A_to_LU[P[r], k]` — same recipe).
- `TemplateSource/OP/SVD.fProxy.cs:56-59`, `:118-121`, `:238-241`, `:342-345`, `:553-556`
  (descending selection-sort argmax over `S[k]`).
- `TemplateSource/OP/Eigen.fProxy.cs:1040-1049`, `:1288-1291`, `:1565-1568` (eigenvalue
  selection-sort argmax).
- `TemplateSource/OP/LOBPCG.fProxy.cs:1244-1245` (`SortAscending` argmin):
  `bool better = ws.lambda[j] < ws.lambda[best]; best = math.select(best, j, better);`

Excluded from D: `Eigen.fProxy.cs:1874-1878` (two-key lexicographic real/imag compare — compound
condition, cold path, keep the branch for clarity) and `Eigen.fProxy.cs:1633-1637` (Hessenberg
pivot compare — cold, entangled with surrounding swaps).

Because pivot choices are bit-identical, factorizations, permutations, and (for LU inside MIP)
node counts are unchanged — any test or benchmark diff here means the conversion was done wrong.

---

## DON'T convert — leave these as real branches

Rule: convert per-element **data** selects; keep error paths, early exits, and branches that gate
an expensive alternate computation. Specifically (all verified present):

1. `UnsafeMathOP.fProxy.cs:372` `refract` — a scalar discriminant gates a whole loop's worth of
   work; select would force both sides.
2. `Eigen.fProxy.cs:982-988` Jacobi `if (absTheta > (fProxy)1)` — BOTH arms contain a `sqrt`;
   select would double the work per rotation.
3. `UnsafeOP.fProxy.cs:2262` `SiftDown` — heap bounds guard + `break`; control flow, not data.
4. All Krylov breakdown guards and LU/CHO singular/zero-pivot checks (e.g.
   `LU.fProxy.cs:137-138`) — error/early-exit paths.
5. **Every `if (scale < (fProxy)1) scale = (fProxy)1;` clamp** (`Eigen.fProxy.cs:106-107`,
   `:143-144`, `:368`; `LOBPCG.fProxy.cs:197`, `:1269`, `:1295`, `:1311`): scalar, not a loop
   reduction, zero vectorization value — and `math.max(scale, 1)` would rewrite a NaN scale to 1
   (consequence 2's accumulator caveat), changing downstream convergence comparisons on a
   corrupted lambda. Not worth a semantics risk for nothing.
6. Skip/`continue` guards (`NormsOP.fProxy.cs:175`, `LOBPCG.fProxy.cs:1069`) and the scalar
   division gate `LOBPCG.fProxy.cs:1030`.
7. `Eigen.fProxy.cs` int sweep counters (1281/1544/1724/1757) — scalar bookkeeping.

---

## Acceptance criteria

Per batch (A, B, C, and D if taken):

1. `Tools/regen.ps1` clean; all generated expansions compile (pay attention to the short/uint
   expansions of the iProxy files — the `(iProxy)` casts are what make them compile).
2. Full headless suite green via `Tools/run-tests.ps1` (baseline ~6297/6297) — this is the
   bit-identity regression gate; many tests assert exact values. Zero tolerance for a single
   numeric diff: these conversions are equivalence-preserving by the analysis above, so any
   failure means a conversion error, not an acceptable drift.
3. No new narrative comments at the sites; one DEVLOG entry per touched template folder
   summarizing the pass (dates only, per DEVLOG format).
4. Batch A additionally: the A/B benchmark obligation above, numbers recorded in
   `TemplateSource/OP/DEVLOG.md` (and Benchmarks DEVLOG if the harness was touched). No
   regression tolerated; if a site measures slower, revert that site and note it.
5. Batch D additionally: LU benchmark A/B; drop the batch on a null result.

Suggested order: A → B → C → (D). Each batch independently shippable.
