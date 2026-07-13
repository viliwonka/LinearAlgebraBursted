# Release scan 2026-07-13 — N8: core type structs (fProxy / iProxy / bool)

Partition: `Assets/LinearAlgebra/CodeGen/TemplateSource/fProxy` (10 .cs), `iProxy` (10 .cs),
`bool` (8 .cs). Every line read; siblings diffed (fProxyN vs iProxyN vs boolN, MxN twins,
fProxy vs iProxy operator files); every forwarding call verified against the actual
UnsafeOP/UnsafeBoolOP kernel parameter semantics (addendum pattern 1); addendum patterns 2-7
swept.

HIGH findings in this partition: **none**. The confirmed wide HIGH (`mulInPlace` role swap)
lives in `OP/OP.Component.{fProxy,iProxy}.cs` (N1-N5 territory); its four call sites in this
partition currently **compensate** for the swapped roles and produce correct results — see M1.

---

## MEDIUM

### M1 — `operator *` call sites are coupled to the role-swapped `mulInPlace` kernel (fix must be atomic)
- Files: `fProxy/fProxyMxN.Operators.cs:141`, `fProxy/fProxyN.Operators.cs:146`,
  `iProxy/iProxyMxN.Operators.cs:203`, `iProxy/iProxyN.Operators.cs:209`
- `fProxyComp.mulInPlace(rhs, matrix);` — every other componentwise operator passes
  `(copy, otherOperand)`; `operator *` alone passes `(otherOperand, copy)` because
  `mulInPlace<T>(this T from, T to)` (OP.Component) mutates its **argument** (`to`), not its
  receiver (`UnsafeOP.compMul(from, target, n)` writes `target[i] *= from[i]`, and the wrapper
  passes `to` as `target`). Result today is CORRECT; but when the wide-pass HIGH on
  `OP.Component`'s `mulInPlace` is fixed (receiver becomes the mutated operand), these four
  swapped call sites will silently start mutating the **user's rhs** and returning an unmodified
  copy of lhs — wrong results with no compile error.
- Fix direction: fix the kernel wrapper and these four operators in the same commit; add the
  cross-reference to whatever triage note covers the OP.Component HIGH. Note `mulInPlace`'s
  kernel also sizes the loop from `from.Data.Length` while every sibling sizes from the mutated
  operand — same commit should align that.

### M2 — Linear indexers have no bounds check while the (row, col) indexers do
- Files: `fProxy/fProxyMxN.Indexing.cs:12-22`, `fProxy/fProxyN.Indexing.cs:10-20`,
  `iProxy/iProxyMxN.Indexing.cs:14-24`, `iProxy/iProxyN.Indexing.cs:12-22`,
  `bool/boolMxN.Indexing.cs:11-21`, `bool/boolN.Indexing.cs:10-20` (12 accessors, all 6 families)
- `get => ref Data.ElementAt(index);` — `UnsafeList<T>.ElementAt` performs no bounds check in
  any build config, so `vec[n]` / `mat[bigIndex]` silently reads/writes out of bounds even in
  the editor, while the sibling `this[int r, int c]` in the very same MxN files guards via
  `Assume.IndexInsideBounds` under ENABLE_UNITY_COLLECTIONS_CHECKS. The `System.Index` overload
  additionally maps `^0` to `Data.Length` (one past the end) unguarded.
- Fix direction: add the same checks-gated bounds assert (`0 <= index < Data.Length`) to the
  linear and `System.Index` accessors in all six families.

### M3 — Comparators and cross-type Shortcuts null-deref on standalone (non-arena) instances; siblings throw an informative exception
- Files: `fProxy/fProxyN.Comparators.cs` / `fProxyMxN.Comparators.cs` /
  `iProxyN.Comparators.cs` / `iProxyMxN.Comparators.cs` (every operator body, e.g.
  `fProxyN.Comparators.cs:13` `boolN res = lhs.boolTempVec(lhs.N, true);`) and all six
  `*.Shortcuts.cs` files (every member forwards through `OwnerArena`).
- `OwnerArena => new Arena(_rec->Owner)` dereferences a null `_rec` when the instance is
  standalone (view ctor, `new fProxyN(n, allocator)`, copy ctor) — a NullReferenceException in
  managed code, undefined/crash under Burst. `Copy()`/`TempCopy()` in the same structs guard
  `_rec == null` with a clear InvalidOperationException, so `a + b` on standalone vectors throws
  a useful message while `a < b` raises a raw NRE. Failure scenario: user creates a
  `floatN` view over a NativeArray (a supported, documented path) and writes `x < 0.5f`.
- Fix direction: give the comparator/shortcut path the same `_rec == null` guard message as
  Copy()/TempCopy() (cheapest: guard inside `OwnerArena` under ENABLE_UNITY_COLLECTIONS_CHECKS).

### M4 — `operator ==`/`!=` declared without `Equals`/`GetHashCode` overrides → CS0660/CS0661 in every consumer compile
- Files: `fProxy/fProxyN.Comparators.cs:64-92`, `fProxy/fProxyMxN.Comparators.cs:68-158`,
  `iProxy/iProxyN.Comparators.cs:66-92`, `iProxy/iProxyMxN.Comparators.cs:70-96`,
  `bool/boolN.Operators.cs:34-50`, `bool/boolMxN.Operators.cs:35-51`
- All 14 generated struct types (float/double/int/short/long/uint/bool x N/MxN) declare
  `operator ==`/`!=` but no template overrides `object.Equals`/`GetHashCode` and no
  `#pragma warning disable 0660,0661` exists anywhere in generated Source — the shipped UPM
  package should emit 2 warnings per struct (28 total) in every user project compile.
- Fix direction: verify in a clean compile log; then either add trivial `Equals`/`GetHashCode`
  overrides (documenting that `==` is elementwise, NOT equality) or a scoped pragma in the
  templates.

### M5 — `boolN` has no standalone allocation constructor; every sibling does
- File: `bool/boolN.cs` (constructors at 74, 84, 105, 117)
- `fProxyN`/`iProxyN` have `(int n, Allocator allocator = Invalid, bool uninit = false)`
  (fProxyN.cs:69, iProxyN.cs:85) and even `boolMxN` has `(int M_rows, int N_cols, Allocator
  allocator, bool uninit = false)` (boolMxN.cs:90) — `boolN` offers only view/copy/internal-arena
  ctors, so a user cannot create a standalone bool vector with its own allocation.
- Fix direction: add the standard `(int n, Allocator, bool uninit)` ctor mirroring iProxyN's.

### M6 — `boolN` missing struct-to-struct `CopyTo(in boolN)`/`CopyFrom(in boolN)`
- File: `bool/boolN.cs` (only the NativeArray overloads exist, lines 146/155)
- `fProxyN.cs:163/171`, `iProxyN.cs:166/174`, `boolMxN.cs:188/197` all provide the
  dimension-checked struct-to-struct pair; boolN alone lacks it (accidental gap from the
  NativeArray-interop fix pass, which the twins received in full).
- Fix direction: add the two overloads with the same `N != vec.N` guard as iProxyN.

### M7 — `boolN.Shortcuts.cs` missing the whole int-family cross-shortcut block its MxN twin has
- File: `bool/boolN.Shortcuts.cs` (fProxy block at 9-17, then bool only) vs
  `bool/boolMxN.Shortcuts.cs:26-34` (+`//alsoExpand[uint]//`)
- Generated output confirms: `boolMxN` exposes int/short/long/uint Vec/Mat/Temp shortcuts,
  `boolN` exposes none of them — a public API asymmetry between twins with no DEVLOG
  justification (the fProxy DEVLOG documents only boolMxN's uint rationale via Hash).
- Fix direction: either add the iProxy copy-replace block (+alsoExpand if uint symmetry is
  wanted) to boolN.Shortcuts.cs, or record the deliberate omission in the DEVLOG.

---

## LOW

### L1 — `s == 0f` float literal in the integer operator templates (addendum pattern 6)
- `iProxy/iProxyN.Operators.cs:78,101`, `iProxy/iProxyMxN.Operators.cs:76,96`
- `if (s == 0f)` generates `int/short/long/uint == 0f` — compiles via implicit int-to-float
  conversion and is semantically safe (no nonzero integer converts to 0f), but it is a float
  literal surviving into all four integer variants. Fix: `s == 0`.

### L2 — Dev-history phrase in XML docs: "(matching the historical behavior)"
- `fProxy/fProxyN.cs:102`, `iProxy/iProxyN.cs:102-103` (copy-ctor `<summary>`)
- History belongs in DEVLOG. Proposed relocation:
  `## fProxyN.cs / iProxyN.cs copy constructor` /
  `- 2026-07-13 | Copy-ctor allocator fallback (arena's allocator when source is arena-backed and no allocator given) preserves the pre-arena-migration behavior. (was fProxyN.cs:102, iProxyN.cs:102)`
  and shorten the doc to the contract ("falls back to that arena's allocator").

### L3 — Stray "// Generated" first-line comment in six Shortcuts templates
- `fProxy/fProxyN.Shortcuts.cs:1`, `fProxy/fProxyMxN.Shortcuts.cs:1`,
  `iProxy/iProxyN.Shortcuts.cs:1`, `iProxy/iProxyMxN.Shortcuts.cs:1`,
  `bool/boolN.Shortcuts.cs:1`, `bool/boolMxN.Shortcuts.cs:1`
- The templates are the SOURCE, not generated; the real `<auto-generated>` banner is prepended
  by codegen, so the line also ships as a redundant second header in output. Delete it.

### L4 — `boolN`/`boolMxN` missing `ToString()` overrides
- `bool/boolN.cs`, `bool/boolMxN.cs` — all four numeric siblings override ToString()
  (fProxyN.cs:217, fProxyMxN.cs:246, iProxyN.cs:221, iProxyMxN.cs:250); the bool twins print
  the default type name. Add the same row/element formatting.

### L5 — Temp copy allocated before the divide-by-zero guard
- `fProxy/fProxyMxN.Operators.cs:63-68,83-88`, `fProxy/fProxyN.Operators.cs:65-70,88-93`,
  `iProxy/iProxyMxN.Operators.cs:72-77,92-97`, `iProxy/iProxyN.Operators.cs:74-79,97-102`
- `operator /(a, s)` and `%(a, s)` call `TempCopy()` first, then `if (s == 0f) throw` — the
  throw path leaks a temp-pool allocation until ClearTemp. Hoist the check above the copy.

### L6 — Parameter-passing drift on siblings
- `fProxy/fProxyN.Operators.cs:78,101` and `iProxy/iProxyN.Operators.cs:87,110`: scalar-first
  `/` and `%` take the vector by value while the MxN twins take `in`.
- `bool/boolN.Operators.cs:66-102` and `bool/boolMxN.Operators.cs:67-103`: componentwise rhs
  passed by value (`boolN rhs`) where every fProxy/iProxy componentwise operator uses `in`.
- Align on `in`.

### L7 — Copy-paste local names crossed between vector and matrix files
- `iProxy/iProxyN.Operators.cs:123,133,142,151,160,167`: locals named `matrix` in the vector
  struct's bitwise operators; `bool/boolMxN.Operators.cs:10-107`: locals named `vec` throughout
  the matrix struct's operators. Rename for the released source.

### L8 — bool templates lack `[MethodImpl(AggressiveInlining)]` their siblings carry
- `bool/boolN.Operators.cs`, `bool/boolMxN.Operators.cs`, `bool/boolN.Shortcuts.cs`,
  `bool/boolMxN.Shortcuts.cs` — every fProxy/iProxy operator and shortcut is attributed;
  none of the bool ones are. Accidental omission; add for consistency.

### L9 — MxN view/standalone constructors accept negative dimensions
- `fProxy/fProxyMxN.cs:77-88` (same in iProxyMxN.cs:80, boolMxN.cs:77) — the view ctor's only
  guard is `M_rows * N_cols != viewOf.Length`, so e.g. `(-2, -3)` with a length-6 array passes
  and later `r * N_Cols + c` indexes garbage; the `(int, int, Allocator)` ctors validate
  nothing. Add a `>= 0` check on both dims in all six ctor sites.

### L10 — Copy-constructor doc drift across the six families
- `bool/boolN.cs:80-83`: summary "Creates a copy of vector with new allocation" plus an EMPTY
  `<param name="orig"></param>` tag; `fProxy/fProxyMxN.cs:122-124`, `iProxy/iProxyMxN.cs:125-127`,
  `bool/boolMxN.cs` (none at all on the copy ctor): docs omit the allocator-fallback contract
  that fProxyN.cs:97-101/iProxyN.cs:100-104 document; the standalone `(int, int, Allocator)`
  MxN ctors have no doc anywhere. Unify on the N-family wording (minus the history phrase, L2).

### L11 — Modulo operator summaries copy the division parenthetical
- `fProxy/fProxyN.Operators.cs:163`, `iProxy/iProxyN.Operators.cs:226`
- `/// <summary>Component-wise modulo (dividend / divisor); ...` — should read
  `(dividend % divisor)`.

### L12 — N-family `IsCreated` doc doesn't state the recycled-slot limitation
- `fProxy/fProxyN.cs:52-53` (same iProxyN.cs:55, boolN.cs:53) — the N structs carry no
  generation stamp (DEVLOG-documented size constraint), so a stale handle whose slot was freed
  and recycled reports `IsCreated == true`; the MxN twin's generation check returns false in
  that case. One doc clause ("a recycled arena slot may still report true") would make the
  contract honest without code change.

### L13 — Minor declaration drift
- `bool/boolN.Indexing.cs:7`: partial re-declares `: IDisposable` (no other Indexing partial
  re-states interfaces).
- `fProxy/fProxyMxN.cs:90` (and iProxy/bool twins): standalone MxN ctor requires an explicit
  allocator while the N-family ctor defaults to Temp — `new floatN(5)` works, a two-arg
  `new floatMxN(2, 3)` doesn't exist. Harmless but asymmetric; consider a defaulted overload.

---

## Cross-partition note (for the OP scanners' triage, not counted here)

`OP.Component.{fProxy,iProxy}.cs` `signFlipInPlace` (`UnsafeOP.signFlip(a.Ptr, a.Ptr, n)`) and
every `BoolOP.cs` wrapper (`UnsafeBoolOP.or(a.Ptr, b.Ptr, a.Ptr, n)`, `not(a.Ptr, a.Ptr, n)`,
etc.) pass the same pointer to two `[NoAlias]` parameters — addendum pattern 4, observed while
verifying this partition's forwarding targets. Benign for same-index elementwise loops in
practice, but it is a violated compiler contract; belongs to N1-N5's files.

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH     | 0 |
| MEDIUM   | 7 (M1-M7) |
| LOW      | 13 (L1-L13) |

Areas confirmed clean:
- Row-major indexing math: every accessor computes `r * N_Cols + c`; no M/N swaps anywhere.
- Operand roles of ALL operator forwardings verified against kernel bodies (scalAdd/scalSub/
  scalDiv/scalMod/compAdd/compSub/compDiv/compMod, or/and/xor/equals/notEquals, cmpr* family,
  bitwise*Comp): every mutated pointer is the temp copy; scalar-first non-commutative forms
  (`s - v`, `s / v`, `s % v`) map to the correct `s op target[i]` kernels; comparator
  mirror identities (`s < v` == `v > s`, etc.) are all correct. Only `operator *` is coupled to
  the known role-swapped kernel (M1).
- `//+skipFor[u]` gating: unary negation is the only signed-only surface here and is gated in
  both operator files, matching the gated `signFlipInPlace` kernel; all other uint-generated
  ops (sub, div, mod, shifts, comparisons) are unsigned-clean.
- `//alsoExpand[uint]//` markers: present where needed across the 10 iProxy files and the two
  MxN shortcut providers; marker comment blocks are contiguous-`//` and strip cleanly; no
  `//+choose` blocks exist in this partition; generated output spot-checked (boolMxN uint
  shortcuts present, no leftover proxy tokens).
- Dispose ordering (cache-Data, Free, then Dispose), double-dispose safety, view Dispose no-op
  (Allocator.None), arena generation stamping in ctors: consistent across all six families.
- No rename stragglers (maxIter/tol/BSM/Solvers/MatrixMetrics/StatsOP/Elem/_OP: zero hits).
- No `[NoAlias]` violations at this partition's call sites (temp copy is always a distinct
  allocation from both operands).
- fProxy/DEVLOG.md exists, covers all six families' shared history; no code comment in the
  partition duplicates DEVLOG content.
