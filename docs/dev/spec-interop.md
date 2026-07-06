# Spec — Unity.Mathematics ↔ library interop

Status: **DRAFT (for review, no code yet)** · 2026-06-27

Goal: make it trivial to move data between native `Unity.Mathematics` value types
(`float2/3/4`, `float2x2…4x4`, and `double*` equivalents) and this library's arena-backed
`fProxyN` / `fProxyMxN`. Both directions, allocating + zero-alloc, plus per-row/column.

---

## 0. What exists today (and why it's stuck)

`Arena/ArenaConversions.fProxy.cs` (a codegen template) already has a partial layer:
- **from-math, allocating:** `arena.Convert(in fProxy2/3/4) → fProxyN`, and `arena.Convert(in fProxy2x2/3x3/4x4) → fProxyMxN`. ✓ (square matrices only)
- **to-math:** only `arena.Convert(in fProxyN) → fProxy2`. ✗ nothing else.

It stalled for a concrete reason: **C# can't overload on return type.** `Convert(in fProxyN)→fProxy2`
and a hypothetical `Convert(in fProxyN)→fProxy3` have identical signatures → won't compile. So the
to-math direction can never be completed under one `Convert` name. The fix is a naming scheme that
encodes the target arity (below). The from-math direction has no such problem (it overloads on the
parameter type), it's just incomplete (missing non-square matrices + zero-alloc + temp variants).

## 1. The codegen mechanism (already in place — reuse it)

`LinearAlgebra.mathProxies` defines `fProxy2/3/4` and all nine `fProxy{2,3,4}x{2,3,4}` structs with the
**same memory layout** as the Unity.Mathematics types (column-major: `c0,c1,…` are columns). A template
references e.g. `fProxy3`; codegen substitutes `fProxy`→`float`/`double`, yielding `float3`/`double3`
(the real Unity.Mathematics types). The template's `using LinearAlgebra.mathProxies;` is wrapped in the
`//+deleteThis … //-deleteThis` marker so it's stripped from generated output, leaving `using
Unity.Mathematics;` to bind `float3`. **All new interop reuses this exact pattern** — no new infra.

### Hard limitation: int interop is float-token-only
The `iProxy` token expands to **int / short / long**, but Unity.Mathematics has **no `short2`/`long3`/…**
types (only `int2/3/4`, `int2x2…`). So integer interop **cannot be an `iProxy` template** — it would
generate references to nonexistent `short2`/`long2`. → integer interop must be a **hand-written, int-only
singular file** (Phase 2). `bool2/3/4` and `quaternion` (float-only) are likewise out of the fProxy
template and deferred. The headline feature (the user's ask) is the **fProxy** path: `float`/`double`
2/3/4 vectors and 2x2…4x4 matrices.

## 2. Row-major ↔ column-major (correctness-critical)

Unity.Mathematics matrices are **column-major**: `float3x3.c0` is the first *column*; `floatRxC` has
`R` rows and `C` columns stored as `C` column-vectors of length `R`. The library `fProxyMxN` is
**row-major** (`this[r,c]`, `Data` row-major, `M_Rows`×`N_Cols`). Every matrix conversion is therefore a
**transpose-aware element map**, not a memcpy:

```
fProxyMxN[r, c]  ==  mathMat.c{c}[r]      // column c, component r
```

(The existing square-matrix `Convert` does this correctly; the spec keeps it for all 9 shapes and the
reverse direction.) Vectors are a straight element copy (and *could* be a memcpy since both are
contiguous `x,y,z,w` — noted as a possible micro-opt, but element copy is fine for ≤4 elements).

## 3. Naming scheme (resolves the return-type wall)

- **native → library** (overloads on the *parameter*, so one name is fine): use the **constructor
  family** for discoverability — `arena.fProxyVec(in fProxy3)`, `arena.fProxyMat(in fProxy3x3)`
  (+ `temp` variants). Drop the old `Convert` from-math overloads (or keep as thin aliases).
- **library → native** (can't overload on return type): **encode arity in the method name** —
  `v.ToFProxy3()`, `m.ToFProxy3x3()`. Codegen turns these into `ToFloat3()` / `ToDouble3()` etc., which
  are distinct methods — no collision. This is the unblock.
- **zero-alloc, both directions**: a `fProxyInteropOP` static class with `copyInto` (native→existing
  buffer) and the `ToFProxy*` readers (which allocate nothing — they return a value type).

## 4. API surface (fProxy = float + double)

### 4a. Native → library, allocating (extend `Arena/ArenaConversions.fProxy.cs`)
```csharp
fProxyN  arena.fProxyVec(in fProxy2 v);   // length 2
fProxyN  arena.fProxyVec(in fProxy3 v);   // length 3
fProxyN  arena.fProxyVec(in fProxy4 v);   // length 4
fProxyMxN arena.fProxyMat(in fProxy2x2 m); // 2x2  … all nine shapes …
fProxyMxN arena.fProxyMat(in fProxy3x4 m); // 3 rows × 4 cols
fProxyMxN arena.fProxyMat(in fProxy4x4 m); // 4x4
// + tempfProxyVec / tempfProxyMat overloads (Temp pool, reclaimed by ClearTemp)
```

### 4b. Native → existing library buffer, zero-alloc (`OP/InteropOP.fProxy.cs`)
```csharp
void fProxyInteropOP.copyInto(ref fProxyN dest, in fProxy3 v);    // dest.N == 3 (else throw)
void fProxyInteropOP.copyInto(ref fProxyMxN dest, in fProxy3x3 m); // dest is 3x3 (else throw)
// … all vec arities (2/3/4) and all nine matrix shapes …
```

### 4c. Library → native (`OP/InteropOP.fProxy.cs`, extension methods)
```csharp
fProxy2  v.ToFProxy2();   // requires v.N == 2 (else throw)
fProxy3  v.ToFProxy3();   // requires v.N == 3
fProxy4  v.ToFProxy4();   // requires v.N == 4
fProxy3x3 m.ToFProxy3x3(); // requires m is 3x3
fProxy3x4 m.ToFProxy3x4(); // requires m is 3x4
// … all nine matrix shapes …
```

### 4d. Per-row / per-column (the point-cloud / transform-stream case)
Build an `N×3` matrix from a stream of `float3`s, or read a row back as a `float3`:
```csharp
void   fProxyInteropOP.setRow(ref fProxyMxN A, int r, in fProxy3 v);  // A.N_Cols == 3
fProxy3 fProxyInteropOP.getRow3(in fProxyMxN A, int r);               // A.N_Cols == 3
void   fProxyInteropOP.setCol(ref fProxyMxN A, int c, in fProxy3 v);  // A.M_Rows == 3
fProxy3 fProxyInteropOP.getCol3(in fProxyMxN A, int c);               // A.M_Rows == 3
// 2- and 4-wide variants: setRow/getRow2/getRow4, setCol/getCol2/getCol4
```
(Row/col getters need the arity in the name for the same return-type reason — `getRow2/3/4`.)

## 5. Validation
Strict dimension match at every boundary; throw `ArgumentException("method: msg")`:
- `ToFProxy3` requires `N == 3` (not `>= 3`) — a 3-vector *is* a float3; a length-5 vector is not.
- matrix converters require exact `M_Rows×N_Cols` match to the target shape.
- `copyInto` validates the destination already has the right shape (it does not resize).
- row/col helpers validate the strided axis length and the index in range.

## 6. Placement summary (aesthetics)
- `Arena/ArenaConversions.fProxy.cs` — **all allocating native→library constructors** (`fProxyVec`/
  `fProxyMat` + temp). Reuses the existing file; removes the broken `Convert(in fProxyN)→fProxy2`.
- `OP/InteropOP.fProxy.cs` (new) — **`fProxyInteropOP`**: zero-alloc `copyInto`, all `ToFProxy*`
  library→native readers, and the row/col helpers. One cohesive "interop" surface, mirrors the
  one-concept-per-OP convention (QueryOP / NormsOP / HistogramOP / ResampleOP).
- `OP/IntInteropOP.cs` (Phase 2, hand-written, **int-only**) — `int2/3/4`, `int2x2…4x4` ↔ `intN`/`intMxN`,
  with a header note on the short/long absence. quaternion + bool deferred.

## 7. Tests (Phase after build)
- Round-trip: `float3 → fProxyVec → ToFProxy3` is identity; same for all arities and all 9 matrix shapes.
- **Transpose correctness:** a non-symmetric `float3x4` round-trips with rows/cols in the right place
  (assert `fProxyMxN[r,c] == m.c{c}[r]`), and `M_Rows==3, N_Cols==4`.
- `copyInto` writes into a preallocated buffer with no allocation; wrong-shape dest throws.
- `ToFProxy3` on a length-2/length-4 vector throws; matrix shape mismatches throw.
- Row/col set/get round-trip on an `N×3` matrix; index-out-of-range throws.
- float + double generated variants.

## 8. Build phases
1. **P1 (fProxy):** finish `ArenaConversions` (all shapes + temp), add `fProxyInteropOP`
   (copyInto / ToFProxy* / row-col), 3-agent review, tests. ← the user's ask.
2. **P2 (int):** hand-written `IntInteropOP` (int-only), tests.
3. **Deferred:** quaternion (float4-only), bool vectors, memcpy fast-path for vectors.

Open question for the user: keep the old `arena.Convert(...)` overloads as deprecated aliases for
back-compat, or replace them outright? (Nothing in the repo appears to depend on them.)
