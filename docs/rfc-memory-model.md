# RFC: Memory-Ownership Model for Arena-Backed Math Structs

Status: **DECIDED 2026-07-01 — "Core now, full sweep later"** (see §8 + decision note below)
Author: architecture sweep (Opus 4.8)
Date: 2026-07-01

> **DECISION (user):** adopt **Option A** (pointer-to-record). Do the **structural core now** —
> stable arena identity (retire the `ref Arena` footgun) + migrate **growable** types (the
> `fProxyBSMBuilder`, and the upcoming resizable/add-remove BSM) to in-arena records, making both
> failure modes structurally impossible. **Defer** the fixed-size-type template sweep and the
> **Option B** DEBUG generational overlay to a dedicated pre-release epic. Sequencing: let the
> in-flight builder point-fix land (stepping stone), run the sparse polish pass, then execute the
> "core now" epic from §6.0/§6.1/§7 steps 1–3, deliberately and reviewed.
Scope: `Arena` + `fProxyMxN` / `fProxyN` / `fProxyBSM` / `fProxyBSMBuilder` / `fProxyBlockJacobi`
and their codegen'd concrete types (`float*`, `double*`, `int*`, `long*`, `short*`, `bool*`).

---

## 0. TL;DR / Recommendation

The library's math structs are C# **value types** that store their authoritative mutable
state (`UnsafeList<T> Data`) **inline**, plus an identity-by-address `Arena* _arenaPtr`.
Copying or passing such a struct can leave copies pointing at freed/relocated memory. Two
failure modes have already bitten the project (§2).

**Recommendation (ranked):**

1. **Ship the "smallest change that kills BOTH failure modes" now (§6.0), then adopt Option A
   (pointer-to-allocation-record in a pointer-stable arena table) as the release architecture.**
   Option A resolves the `Data` accessor through a **stable in-arena record**, so a struct copy
   is just a copy of a stable pointer — it can never diverge from the one source of truth. It is
   the smallest model that makes *both* the growable-relocation bug and the identity-by-address
   bug **structurally impossible**, at ~zero hot-loop cost (the pointer is resolved once per op
   and cached, which is loop-invariant — §4.3, §5.A).

2. **Add generational validation (Option B) as a DEBUG-only overlay on top of Option A**, not as
   the release addressing mode. It buys use-after-free/double-dispose *detection* — the one thing
   Unity's safety system does **not** give you once Burst safety checks are compiled out (§3.5) —
   without paying a generation-compare on every release-build access.

3. **Do NOT convert fixed-size types that never grow to handles for safety reasons alone.**
   `fProxyMxN`/`fProxyN`/`fProxyBSM`/`Pivot`/`Indices` allocate once and never realloc; for them
   only failure mode 2 (identity-by-address) is live, and that is fixed by Option A's stable
   record. The growable types (`fProxyBSMBuilder`, and any future builder) are where relocation
   actually happens and are the real motivation.

The migration is unusually cheap because the types are **codegen'd from ~6 templates** and every
hot kernel reads through the **`Data` property**, not a field — changing the property's backing in
the template regenerates all ~847 `.Data.Ptr` sites transparently (§5, §7).

---

## 1. Current model (grounded in the code)

### 1.1 `Arena` — a struct that owns growable per-type tracking lists

`Arena` is a **struct**, not a class
(`Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/Arena.cs:10`):

```csharp
[StructLayout(LayoutKind.Sequential)]
public partial struct Arena : System.IDisposable { ... }
```

It holds one growable `UnsafeList<T>` per tracked type (declared across the partials
`Arena.fProxy.cs`, `Arena.iProxy.cs`, `Arena.bool.cs`, `Arena.Sparse.fProxy.cs`, `Arena.cs`):

- `fProxyVectors`, `fProxyMatrices`, `tempfProxyVectors`, `tempfProxyMatrices`
- `fProxyBSMs`, `fProxyBSMBuilders`, `fProxyBlockJacobis`
- `iProxyVectors`, `iProxyMatrices`, `tempiProxyVectors`, `tempiProxyMatrices`
- `BoolVectors`, `BoolMatrices`, `TempBoolVectors`, `TempBoolMatrices`
- `Pivots`, `IndexBuffers`

**Allocation + tracking** (`Arena.fProxy.cs:63`):

```csharp
public fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false) {
    var matrix = new fProxyMxN(M_rows, N_cols, in this, uninit);
    fProxyMatrices.Add(in matrix);   // <-- stores a VALUE COPY of the struct
    return matrix;                   // <-- returns ANOTHER value copy to the caller
}
```

So after a factory call there are **at least two copies** of the struct: the arena's tracking
copy and the caller's. For a fixed-size type these copies are equivalent forever (the inline
`UnsafeList` header — `{Ptr, Length, Capacity, Allocator}` — never changes after construction).
For a **growable** type they diverge the moment one copy reallocs (§2.1).

**Disposal** (`Arena.cs`, `Clear`/`ClearTemp`/`Dispose`) iterates each tracking list and calls
`[i].Dispose()` on the arena's stored copy, then `list.Clear()`. `Dispose()` then disposes the
tracking lists themselves and sets `Allocator = Allocator.Invalid`. This is a **bulk/rewind**
model — no per-object free by the caller; the arena frees everything at once (the standard arena
ownership contract — Fleury, "Untangling Lifetimes"
<https://www.dgtlgrove.com/p/untangling-lifetimes-the-arena-allocator>; Wellons
<https://nullprogram.com/blog/2023/09/27/>).

**Arena identity** is captured by child structs as a **raw address**: each math struct holds
`Arena* _arenaPtr`, set in the `in Arena` constructor via `fixed`:

```csharp
[NativeDisableUnsafePtrRestriction] private unsafe Arena* _arenaPtr;
...
fixed (Arena* arenaPtr = &arena) _arenaPtr = arenaPtr;   // captures address of `arena`
```

`_arenaPtr` is used only for the convenience methods `Copy()`/`TempCopy()` (which forward to
`_arenaPtr->fProxyVec/Mat(...)`) and for the allocator lookup in some ctors. Its correctness
depends entirely on the arena living at a **stable address** for as long as any child struct
holds the pointer.

### 1.2 The math structs — inline authoritative state + identity-by-address

`fProxyMxN` (`TemplateSource/fProxy/fProxyMxN.cs`) and `fProxyN`
(`TemplateSource/fProxy/fProxyN.cs`):

```csharp
public partial struct fProxyMxN : IDisposable, IUnsafefProxyArray, IMatrix<fProxy> {
    public int M_Rows;
    public int N_Cols;
    public UnsafeList<fProxy> Data { get; private set; }   // <-- authoritative inline state
    [NativeDisableUnsafePtrRestriction] private unsafe Arena* _arenaPtr;
    public readonly int Length;
    ...
}
```

Two constructor paths on every type:
- `(…, Allocator allocator)` — standalone, `_arenaPtr = null`, disposed by the caller.
- `(…, in Arena arena)` — arena-tracked, `_arenaPtr = &arena`.

**Hot-kernel access** is uniformly through the `Data` property, cached once at op entry. Example
shape (from `OP.Dot.fProxy.cs`, `NormsOP.fProxy.cs`, etc.):

```csharp
unsafe {
    fProxy* a = lhs.Data.Ptr;      // resolve ONCE, hoisted out of the element loop
    int n = lhs.Data.Length;
    for (int i = 0; i < n; i++) { /* touch a[i] */ }
}
```

`Data.Ptr` is the `UnsafeList<T>.Ptr` internal buffer; `Data.Length` its element count. The
per-element loop never re-reads `Data` — it works off the cached local. **This is the single most
important fact for the perf analysis (§4.3, §5): the accessor is already resolved once per
operation, not per element.**

DEBUG NaN-poison: under `LINALG_DEBUG`, `Dispose()` writes `float.NaN` across the buffer before
freeing, so a read-after-dispose surfaces as NaN instead of stale data
(`fProxyMxN.cs:88`, `fProxyN.cs:108`, `fProxyBSM.cs:108`).

### 1.3 The growable types

`fProxyBSMBuilder` (`TemplateSource/Sparse/fProxyBSMBuilder.cs`) is the one type that **grows
after construction**. It holds three growable lists:

```csharp
private UnsafeList<int>    triBlockRow;
private UnsafeList<int>    triBlockCol;
private UnsafeList<fProxy> triValues;
```

`AddBlock`/`AddValue` call `.Add(...)` on these, which reallocs when capacity is exceeded.
`fProxyBSM` (compressed) and `fProxyBlockJacobi` are **fixed-size** once built.

### 1.4 Blast radius (migration cost input)

Grep over `Assets/LinearAlgebra` (templates + generated + tests):

| Pattern | Occurrences | Files |
|---|---|---|
| `.Data.Ptr` | ~847 | 106 |
| `.Data.Length` | ~600 | 73 |
| `_arenaPtr` | ~532 | 52 |

Crucially, the bulk is **generated**: every concrete `floatMxN`/`doubleMxN`/… is a verbatim
`fProxy`→`float` expansion of its template (compare `TemplateSource/fProxy/fProxyMxN.cs` with
`Source/Generated/float/floatMxN.cs` — identical field/property/ctor shape). The template set that
actually needs hand-editing is small: the `fProxy*`/`iProxy*`/`bool*` struct templates (~6 files)
plus `Arena*` partials. `fProxy` fans out to `{float, double}` (2×), `iProxy` to
`{int, long, short}` (3×). **All ~847 access sites read through the `Data` property**, which is
already a `{ get; private set; }` auto-property — so redefining the getter in the template
regenerates every site with no call-site churn (§5).

---

## 2. The two confirmed failure modes (with current code)

### 2.1 Failure mode 1 — growable state relocates; the arena's value-copy dangles

`fProxyBSMBuilder` is created and tracked by value:

```csharp
// Arena.Sparse.fProxy.cs:33
public fProxyBSMBuilder fProxyBSMBuilder(int blockRows, int blockCols, int BR, int BC, int capacityHint = 8) {
    var builder = new fProxyBSMBuilder(blockRows, blockCols, BR, BC, in this, capacityHint);
    fProxyBSMBuilders.Add(in builder);   // arena stores a VALUE COPY (snapshot of the list headers)
    return builder;                      // caller gets a SEPARATE copy
}
```

The caller then grows *their* copy:

```csharp
builder.AddBlock(br, bc, block);   // triValues.Add(...) reallocs -> caller's triValues.Ptr moves
```

`UnsafeList<T>.Add` "increases the capacity if necessary" by **allocating a new buffer, copying,
and freeing the old one** (NativeList Capacity remarks, Collections 0.7:
"Changing Capacity creates a new array … copies … then deallocates the original array memory";
this remark was scrubbed from newer doc pages but the mechanism is unchanged). After a growth:

- the **caller's** copy has the new `Ptr`,
- the **arena's** tracking copy still has the **old, freed** `Ptr`.

On `arena.Dispose()`, the arena disposes its stale copy → **double-free / use-after-free** on the
already-freed old buffer, and the caller's live buffer is **leaked**. Confirmed class of bug; it
is exactly why `slab`-style "store a snapshot" is unsafe for growable state.

### 2.2 Failure mode 2 — identity-by-address captures a stack temporary

Every `in Arena` constructor does `fixed (Arena* p = &arena) _arenaPtr = p;`. Because `Arena` is a
**struct** and its allocator methods are **not `readonly`**, passing it through an `in` parameter
to a factory that then *mutates* it forces the C# compiler to make a **defensive copy** — and the
struct captures the address of that dead temporary. This is documented in the code itself
(`fProxyBSM.cs:117-129`):

```csharp
/// Takes the arena by `ref`, NOT `in`: it calls the mutating arena.fProxyMat(...) allocator
/// internally, and `Arena`'s allocator methods are not `readonly` -- an `in` parameter here
/// would force the compiler to defensively copy the arena before that call, so the returned
/// matrix's internal arena pointer would capture the address of a dead temporary instead of the
/// caller's real arena (a dangling-pointer bug caught by the test suite).
public fProxyMxN ToDense(ref Arena arena) { ... }
```

The same hazard exists for `&localArena` (a stack arena that dies at end of frame) and for storing
a math struct beyond the arena's stack lifetime. The current mitigation — "remember to take
`ref Arena`, not `in Arena`, on any factory-shaped method" — is a **latent footgun encoded as a
convention**, not a structural guarantee. It has already been tripped once and caught only by the
test suite.

**Root cause common to both:** the struct is treated as the source of truth (it stores state
inline and identity by address), but a value type is *copied freely* by the language. Any model
where the truth lives in the copy is fragile. The fix is to make the struct a **stable handle to a
truth that lives elsewhere** — inside the arena.

---

## 3. External patterns (evidence base)

All confirmed items were fetched during research; see the source list at the end. `[C]` =
confirmed against a fetched page, `[I]` = reasoned inference.

### 3.1 Generational-index handles / slotmaps

- **floooh, "Handles are the better pointers"** <https://floooh.github.io/2018/06/17/handles-vs-pointers.html>
  `[C]`: replace pointers with `{index, generation}` handles owned by a central system; items
  packed tightly; handle→pointer does range-check + occupied-check + **generation compare**; a
  stale handle into a reused slot fails the compare. Discipline: **pointers obtained from handles
  "should never be stored"** — use them only in a local block. This is precisely the
  "resolve once, use locally" pattern the library already follows for `Data.Ptr`.
- **Rust `slotmap`** <https://docs.rs/slotmap/> `[C]`: key = `{ idx: u32, version: NonZeroU32 }`;
  a key is valid only when its version matches the slot's stored version; removed keys are invalid
  forever even if the slot is physically reused (ABA-safe). `DenseSlotMap` gives contiguous values
  (fast iteration) at the cost of **two indirections** on random access.
- **Rust `slab`** <https://docs.rs/slab/> `[C]`: the **counter-example** — Vec-backed, key is a
  bare `usize` with **no version**; a stale key silently accesses the new occupant. This is exactly
  the failure-mode-1 shape (store a snapshot, no validation). Cheaper, unsafe.
- **Sebastian Aaltonen** (DoD threads) `[C, via threadreader]`
  <https://threadreaderapp.com/thread/1080069784644059139.html>: pack data in arrays for cache-
  friendly linear access; "index them with a handle. Handle can include generation bits for
  lifetime check"; "a generational arena … is a nice data container [without] a massive ECS
  framework." (X/Twitter originals return HTTP 402; wording is from search snippets.)
- **fitzgen/generational-arena** <https://github.com/fitzgen/generational-arena> `[C]`: "a safe
  arena allocator that allows deletion without suffering from the ABA problem by using generational
  indices"; lookup requires matching index **and** generation.

Tradeoff summary:

| | raw pointer | `{index, gen}` handle | `slab` (index only) |
|---|---|---|---|
| UAF/double-free detection | none (UB) | **yes** (gen compare) `[C]` | **none** (silent) `[C]` |
| Deref cost | 1 load | index + gen compare `[I]` | 1 index load `[C]` |
| Extra memory | 0 | generation counter/slot | 0 `[C]` |
| Survives store relocation | no | **yes** (index is position-independent) `[C]` | yes for index, but unsafe |

### 3.2 Unity precedent — ECS `Entity` is a generational handle

`Entity` <https://docs.unity3d.com/Packages/com.unity.entities@1.0/api/Unity.Entities.Entity.html>
`[C]`: two public fields, `Index` and `Version` ("the generational version of the entity"). Indexes
are recycled on destroy and Version is incremented; a stale reference has the same Index but a
different Version. Doc explicitly warns Version can **wrap**, so "newer ≠ larger." Bevy's `Entity`
is the same idea packed into a `u64` (index low 32, generation high 32; generation wraps)
<https://github.com/bevyengine/bevy/blob/main/crates/bevy_ecs/src/entity/mod.rs> `[C]`. **The
`{index, generation}` handle is a first-class, battle-tested Unity-ecosystem pattern**, not exotic.

### 3.3 Unity precedent — AllocatorManager / RewindableAllocator

- `AllocatorManager.AllocatorHandle`
  <https://docs.unity3d.com/Packages/com.unity.collections@1.2/api/Unity.Collections.AllocatorManager.AllocatorHandle.html>
  `[C]`: itself an `{ Index: ushort, Version: ushort (low 15 bits) }` generational handle for
  allocators. Registration uses managed delegates and is **`[ExcludeFromBurstCompatTesting]`** —
  not Burst-compatible — but a `UnmanagedUnregister` exists `[C]`.
- `RewindableAllocator`
  <https://docs.unity3d.com/Packages/com.unity.collections@2.2/manual/allocator-rewindable.html>
  `[C]`: "works … like a linear allocator … fast and thread safe"; `Rewind()` frees everything at
  once, keeps blocks for reuse, and **"invalidates all its child safety handles"** so post-rewind
  access throws under safety checks. This is Unity's own **rewindable-arena-with-invalidation** —
  the closest in-engine analogue to what this library's `Arena` is.

### 3.4 `NativeList`/`UnsafeList` relocation-on-grow

`[C]` (NativeList 0.7 Capacity remarks): growing capacity allocates a new array, copies, and frees
the old. `UnsafeList<T>.Ptr` is the internal buffer; `Add` "increases the capacity if necessary."
`[I, well-founded]` a cached `.Ptr` is **dangling after any growth** — Unity does not state this
invalidation explicitly, but it follows directly from the realloc mechanism. This is the mechanical
root of failure mode 1.

### 3.5 Unity's safety net is compiled out in shipped Burst code

- `DisposeSentinel` <https://github.com/Unity-Technologies/UnityCsReference/blob/master/Runtime/Export/NativeArray/DisposeSentinel.cs>
  `[C]`: managed, finalizer-based leak detector, usable **only when `ENABLE_UNITY_COLLECTIONS_CHECKS`
  is defined**, and its methods are `[BurstDiscard]` → **Burst strips the calls entirely**.
- `AtomicSafetyHandle` `[C]`: coordinates safe container access + race detection, active only when
  safety checks are enabled.
- Burst `[BurstCompile(DisableSafetyChecks = true)]` `[C]`: "Burst will remove all safety check
  code, resulting in code-generation that is generally faster."

**Conclusion for this RFC** `[I from the above]`: in a shipped Burst build with safety checks off,
Unity gives you **no** UAF/dangling/double-free detection. If the library wants any release-time
detection of these bugs, it must provide it itself — which is the argument for a generational
overlay (Option B), but only where the cost is acceptable (§6.2).

### 3.6 Cost of indirection — resolve-once is effectively free

- Drepper, "What Every Programmer Should Know About Memory" <https://lwn.net/Articles/252125/>
  `[C]`: register ≤1 cyc, L1d ~3, L2 ~14, **main memory ~240 cyc**; **dependent** pointer chasing
  that defeats the prefetcher rises to "450 cycles and more." curiouscoding
  <https://curiouscoding.nl/posts/cpu-benchmarks/> `[C]`: pointer-chase RAM latency ~204 cyc,
  but **independent** accesses pipeline under out-of-order execution.
- Loop-invariant code motion <https://en.wikipedia.org/wiki/Loop-invariant_code_motion> `[C]`:
  computations whose result doesn't change are hoisted; "constants … reside in registers rather
  than requiring repeated memory access."
- `[I, well-supported]` Resolving a handle→pointer **once** before a tight loop is loop-invariant:
  hoisted into a register, the per-iteration cost is ~a register-relative load, i.e. amortized to
  ~0. This is categorically different from **per-element** chasing (dependent ~200-cyc misses).
  **The library already resolves `Data.Ptr` once per op (§1.2), so an extra indirection at op entry
  is in the noise.**

### 3.7 Arena/region allocators & composition with handles

`[C]` Fleury/Wellons/bumpalo: arena = buffer + bump offset; free everything by resetting the
offset; caller picks the arena (lifetime = arena's lifetime); real arenas chain chunks; scratch
sub-arenas via begin/end markers (cf. `RewindableAllocator.Rewind`, and this library's
`Clear`/`ClearTemp`). `[C]` fitzgen/Sardois/floooh: a "generational arena" fuses arena allocation
with `{index, generation}` handles; remove bumps the generation; slots recycled via a free-list.
`[I, honest limit]` The *specific* fusion "a **relocating bump arena** whose live objects are
addressed **exclusively** by generational handles" has **no single canonical source** — the
bump-arena sources hand back raw pointers and don't relocate live objects; the generational-handle
sources are slot/pool-backed. They compose (an integer index is position-independent and
self-validating), but treat the exact fusion as a supported design inference, not a quoted
precedent.

---

## 4. Design space

The invariant we want: **a copy of a math struct must never be able to diverge from the one source
of truth, and identity must not depend on a stack address.** All options below move the
authoritative state (`Data`, and for growable types the growable lists) **into the arena**, and
leave the struct holding only a **stable reference** to an in-arena **record**.

The record (per allocation) holds what is today inline in the struct:

```csharp
// lives INSIDE the arena, at a pointer-stable location; never copied by user code
internal struct AllocRecord {          // one per allocation
    public UnsafeList<fProxy> Data;    // (or the 3 growable lists for a builder record)
    public int Generation;             // Option B / DEBUG only
    public bool Alive;                 // disposed?  (DEBUG / detection)
}
```

The struct becomes:

```csharp
public partial struct fProxyMxN {
    public int M_Rows, N_Cols;
    public readonly int Length;
    // EITHER a stable pointer (Option A) OR a handle (Option B):
    private unsafe AllocRecord* _rec;          // Option A
    // private RecordHandle _rec;              // Option B  ({int index, int generation})

    public UnsafeList<fProxy> Data {           // <-- the ONE line that changes for all 847 sites
        get { unsafe { return _rec->Data; } }  // Option A
        // get { return _arena->Resolve(_rec).Data; }   // Option B
        private set { unsafe { _rec->Data = value; } }
    }
}
```

Because every kernel reads `x.Data.Ptr`/`x.Data.Length` through this property, **the accessor
change is the migration** (§5, §7).

### Option A — pointer-to-allocation-record (stable arena table)

The struct holds `AllocRecord* _rec` pointing into a **pointer-stable** arena-owned table.
"Pointer-stable" is the whole trick: the current tracking lists are `UnsafeList`, which **relocate
on grow**, so a naive `&list[i]` would itself dangle (recreating failure mode 1 one level up).
Two ways to get stability:

- **A1 — chunked/block storage**: store records in fixed-size chunks (e.g. `UnsafeList<Chunk>`
  where each `Chunk` is a `256×AllocRecord` block allocated once and never moved). Records get a
  stable address for life; only the chunk *directory* grows (and the directory holds pointers to
  chunks, so growing it doesn't move the chunks). This is the bumpalo "chain of chunks" pattern
  (<https://docs.rs/bumpalo/>).
- **A2 — over-reserve + assert**: `SetCapacity` the record table to a max at arena creation and
  hard-error on overflow. Simpler, but caps allocation count. Reasonable for a first cut given
  arenas are typically sized per algorithm.

Properties:
- **Kills failure mode 1**: growable lists live *in the record*; the struct copy holds the same
  `_rec`, so both "copies" mutate the same list. There is no second copy to go stale.
- **Kills failure mode 2**: the struct no longer captures `&arena`. Copy/Temp helpers use the
  record's arena back-reference (a stable pointer/handle stored once in the record, or the arena
  passed explicitly). `in Arena` vs `ref Arena` stops mattering for correctness.
- **Burst**: a raw pointer with `[NativeDisableUnsafePtrRestriction]` is exactly what the code
  already uses for `_arenaPtr`; fully Burst-compatible.
- **Per-access cost**: one extra load (`_rec->Data`) at op entry, hoisted; ~0 in the loop (§3.6).
- **Detection**: none by itself (raw pointer). Add `Alive`/`Generation` in the record for DEBUG
  checks (this is where Option B bolts on).

### Option B — generational handle + arena slotmap

The struct holds `RecordHandle { int Index; int Generation }` instead of a pointer; the arena is a
**slotmap** (`records[Index]` + per-slot `generation`, free-list of dead slots). `Data` resolves
via `arena.Resolve(handle)` which checks `records[Index].generation == handle.Generation`.

Properties:
- Everything Option A gives, **plus** use-after-free / double-dispose / stale-handle **detection**:
  disposing bumps the slot generation, so any later `Resolve` of an old handle fails the compare.
  This is the `slotmap`/`Entity` model (§3.1, §3.2).
- **Cost**: resolve needs the arena base (the handle carries no pointer), so either the struct also
  stores an arena pointer/handle, or ops receive the arena. Plus a branch (generation compare) at
  op entry. Both are hoistable → ~0 in the loop, but the compare is real work on every op boundary.
- **Burst**: fully compatible (integers + a bounds-checked array index; the compare is trivial).
  Note Unity's own `Entity` and `AllocatorHandle` are exactly this shape.
- **Downside**: the handle can't be dereferenced without the arena, so `Data` becomes
  arena-relative; every place that currently does `x.Data` "for free" needs the arena in scope, OR
  the struct caches an arena pointer too (giving back some of Option A's directness). Generation
  **wrap** is a real (if remote) correctness caveat Unity documents for `Entity` (§3.2).

### Hybrid (recommended long-term)

**Option A addressing in release; Option B validation in DEBUG.** Struct holds `AllocRecord* _rec`
(Option A). The record carries `Generation` + `Alive`. Under `LINALG_DEBUG` the `Data` getter
asserts `_rec->Alive` (and, if the record can be recycled, checks a generation stamp the struct
also stores). In release the assert compiles out and you have raw Option A speed. This mirrors
Unity's own split: raw-pointer speed in shipped Burst, full detection only when checks are on
(§3.5). It also composes with the existing NaN-poison DEBUG facility (§1.2).

---

## 5. Why the migration is cheap (codegen leverage)

The decisive fact: **all state is read through the `Data` property, and the types are generated
from templates.** Concretely:

1. The property `public UnsafeList<T> Data { get; private set; }` appears **once per template**
   (`fProxyMxN.cs`, `fProxyN.cs`, `fProxyBSM.cs`, `iProxy*`, `bool*` — ~6 files). Redefine its
   getter/setter to resolve through `_rec` there.
2. Regenerate. All ~847 `.Data.Ptr` and ~600 `.Data.Length` sites in `Source/Generated` (and the
   OP templates that consume them) are **untouched** — they still say `x.Data.Ptr`, which now
   resolves through the record. **Zero call-site churn in kernels.**
3. Replace the `Arena* _arenaPtr` field with `AllocRecord* _rec` (Option A) in the same ~6
   templates; update the ~2 constructor bodies per template and the `Copy()`/`TempCopy()` helpers.
4. Change the arena factories (`fProxyMat`, `fProxyVec`, `fProxyBSMBuilder`, …) to **allocate a
   record, store the state in it, and return a struct that points at the record** instead of
   `list.Add(in value)`. This is the ~18 factory methods across the `Arena.*` partials.
5. Change `Clear`/`ClearTemp`/`Dispose` to walk the record table instead of the value-copy lists.

The mechanical part (steps 1-2) covers the overwhelming majority of the code. The hand-touch is
steps 3-5: ~6 struct templates + the `Arena` partials — on the order of a few hundred lines of
template, all in one subsystem. Everything downstream is regeneration.

**Risk note:** `Data`'s `set` is `private` and only called in constructors and the few in-place
buffer swaps — grep `Data =`/`Data.Resize` to confirm no external writer relies on value
semantics before flipping the backing. (Initial grep shows writes are confined to ctors.)

---

## 6. Concrete proposals

### 6.0 Smallest change that kills BOTH failure modes (baseline for comparison)

If we wanted the *minimum* diff, independent of the full record model:

1. **Failure mode 2** (identity-by-address): make `Arena` live at a **stable heap address** and
   have structs capture *that*, not a stack `&arena`. Cheapest concrete form: give `Arena` a single
   heap-pinned "self" record (an `UnsafeList<ArenaCore>` of length 1, or a `GCHandle`-pinned box)
   and store `ArenaCore* _arena` = that stable address. `in Arena` vs `ref Arena` then no longer
   affects the captured pointer, because the pointer targets the pinned core, not the parameter.
   This alone removes the `ref Arena` footgun everywhere.
2. **Failure mode 1** (growable relocation): make the arena track growable builders **by reference,
   not by value** — store the builder's growable lists in an arena-owned, pointer-stable record and
   have `fProxyBSMBuilder` hold `Record*`. (For the fixed-size types, tracking-by-value is already
   safe — see §6.3 — so they need no change for FM1.)

Observe that (1)+(2) **are** Option A, applied minimally. There is no cheaper structural fix that
kills both: FM2 fundamentally needs a stable arena identity, and FM1 fundamentally needs the
growable truth to have one owner. Anything less (e.g. "just always take `ref Arena`" or "reserve
huge builder capacity so it never grows") is a **convention or a gamble**, not a guarantee — and
conventions are what failed here already. **So the smallest robust change and the recommended
architecture are the same thing; that is the argument for doing Option A now.**

### 6.1 Release architecture: Option A (pointer-to-record, chunked table)

- Arena owns a **chunked record table** (A1) per category (matrices/vectors/temp/BSM/builder/…),
  or one heterogeneous table if records are tagged. Records never move.
- Structs hold `AllocRecord* _rec`; `Data` resolves `_rec->Data`.
- Arena also lives at a stable address (its core in a length-1 pinned allocation), so the record
  can hold a valid arena back-reference for `Copy()`/`TempCopy()` and `_arenaPtr` is retired.
- Growable builders store their lists in the record → FM1 gone. Fixed-size types store their single
  `Data` in the record → FM2 gone.

### 6.2 DEBUG overlay: Option B validation (hybrid)

- Record carries `Generation` + `Alive`; struct also carries the generation stamp it was created
  with.
- Under `LINALG_DEBUG`: `Data` getter asserts `Alive && _rec->Generation == _stamp` → **detects
  double-dispose and use-after-free**, complementing the existing NaN-poison.
- In release: compiled out; raw Option A speed.
- This gives the library its *own* detection in a regime where Unity's `DisposeSentinel`/
  `AtomicSafetyHandle` are stripped (§3.5), without a release-time generation compare on every op.

### 6.3 Explicitly leave fixed-size types alone (where the current design is fine)

`fProxyMxN`, `fProxyN`, `fProxyBSM`, `fProxyBlockJacobi`, `Pivot`, `Indices` allocate **once** and
never realloc. For these, failure mode 1 **cannot occur** — the inline `UnsafeList` header is
immutable after construction, so the arena's value-copy and the caller's copy stay identical
forever. Only failure mode 2 (identity-by-address for `Copy()`/`ToDense(ref Arena)`) is live, and
that is fixed purely by the stable-arena part of §6.0/§6.1. **Converting these to generational
handles purely for safety would add indirection and complexity for a bug that can't happen** — so
under Option A they still benefit (state moves to the record, `_arenaPtr` retired) but they do
**not** need Option B's generation machinery on the hot path. This is the honest "the current
design is actually fine here" carve-out.

---

## 7. Migration plan

**Incremental, one type-family at a time, behind the shared `Data` accessor** — not big-bang:

1. **Stable arena core first** (§6.0.1). Land it, run the suite. This alone kills FM2 and de-risks
   the `ref Arena` convention. Low blast radius (Arena partials + the `_arenaPtr` initializers).
2. **Introduce `AllocRecord` + chunked table** in the arena, unused at first.
3. **Migrate `fProxyBSMBuilder` (the growable type) to a record.** This is the FM1 fix and the
   highest-value change. Gate with the existing `SparseBSMTests` + a **new regression test that
   reproduces FM1** (build a builder, force several grows via `AddBlock` past `capacityHint`, then
   `arena.Dispose()` and assert no double-free/leak — under `LINALG_DEBUG` the NaN-poison + `Alive`
   assert should fire on a stale access).
4. **Migrate the fixed-size struct templates** (`fProxyMxN`/`fProxyN`, then `iProxy*`, then
   `bool*`, then `fProxyBSM`/`fProxyBlockJacobi`) by changing the `Data` getter + field + ctors in
   each template and regenerating. Run the full **3209-test suite** after each family — because the
   accessor name is unchanged, a green suite is strong evidence the resolution is transparent.
5. **Flip disposal** to walk the record table; delete the per-type value-copy tracking lists.
6. **Add the FM2 regression test**: a factory returning a struct whose arena was a stack temporary,
   asserting the struct still resolves correctly (record-based identity) where the old code would
   dangle.

**Test strategy / regression gates:**
- The **two known repro bugs become permanent regression tests** (steps 3 & 6).
- The **3209-test suite** is the transparency gate for the mechanical accessor swap (step 4).
- Keep **NaN-poison** and add the **`Alive`/generation asserts** (§6.2) so DEBUG runs actively hunt
  UAF during migration.
- Bit-identical numerics: since `Data.Ptr` semantics are unchanged, existing bit-preservation
  tests (matMatDot etc.) should stay green — a good early smoke signal.

**Where migration is genuinely risky (call it out):**
- **`fixed`-pointer aliasing / struct size**: `[StructLayout(Sequential)]` on the structs is
  relied on somewhere (e.g. interop transpose boundary — see the row-major convention note).
  Swapping fields changes struct size/layout; audit any code that assumes a specific layout or
  reinterprets these structs.
- **Standalone (`Allocator`, non-arena) construction path** must still work — those structs have
  no record/arena. Either keep an inline fallback (`_rec == null` → a private inline `Data`) or
  require all buffers to be arena-owned. The former preserves the current dual-path API at the cost
  of a branch in the getter; the latter is cleaner but a breaking change (acceptable pre-release).
- **Chunked table addressing** must be genuinely non-relocating; a subtle bug here re-creates FM1
  one level up. Unit-test the table in isolation (allocate past a chunk boundary, assert earlier
  record addresses are unchanged).

---

## 8. Recommendation, ranked

1. **Do Option A now** (stable arena core + chunked record table + `Data` resolves through the
   record), because it is simultaneously the *smallest robust change* (§6.0) and the *right release
   architecture* — it makes both failure modes structurally impossible at ~0 hot-loop cost, and the
   codegen structure makes it cheap (§5, §7).
2. **Layer Option B as a DEBUG-only overlay** (record `Generation`/`Alive` + asserts) for
   use-after-free/double-dispose detection Unity won't give you in shipped Burst (§3.5). Do **not**
   pay generation-compare on the release hot path.
3. **Leave fixed-size types on value-copy tracking if you must ship sooner** — they are not where
   the bugs live (§6.3). But since Option A migrates them almost for free (accessor swap), prefer
   migrating them too for uniformity and to retire `_arenaPtr` everywhere.
4. **Reject** pure-handle-everywhere-in-release (Option B as the primary addressing mode): it adds
   arena-relative resolution and a per-op compare with no release benefit over Option A, given the
   library already resolves the pointer once per op. Reserve generations for DEBUG.

Honest bottom line: the current design is a value-type-copies-the-truth model, which is inherently
fragile; the two bugs are symptoms, not one-offs. Moving the truth into a pointer-stable in-arena
record (Option A) is the minimal principled fix, it is Burst-friendly, its hot-loop cost is in the
noise because the accessor is already hoisted once per op, and the codegen makes the diff small and
mostly mechanical. Generational handles are worth adopting for **detection**, but scoped to DEBUG.

---

## Sources (all fetched during research)

- floooh, "Handles are the better pointers" — https://floooh.github.io/2018/06/17/handles-vs-pointers.html
- Rust `slotmap` — https://docs.rs/slotmap/ ; source https://docs.rs/slotmap/latest/src/slotmap/lib.rs.html ; https://github.com/orlp/slotmap
- Rust `slab` — https://docs.rs/slab/
- Sebastian Aaltonen DoD thread — https://threadreaderapp.com/thread/1080069784644059139.html
- fitzgen/generational-arena — https://github.com/fitzgen/generational-arena
- Sardois, generational-indices guide — https://lucassardois.medium.com/generational-indices-guide-8e3c5f7fd594
- Unity ECS `Entity` — https://docs.unity3d.com/Packages/com.unity.entities@1.0/api/Unity.Entities.Entity.html
- Bevy `Entity` source — https://github.com/bevyengine/bevy/blob/main/crates/bevy_ecs/src/entity/mod.rs
- Unity `AllocatorManager` — https://docs.unity3d.com/Packages/com.unity.collections@2.1/api/Unity.Collections.AllocatorManager.html
- Unity `AllocatorManager.AllocatorHandle` — https://docs.unity3d.com/Packages/com.unity.collections@1.2/api/Unity.Collections.AllocatorManager.AllocatorHandle.html
- Unity `RewindableAllocator` — https://docs.unity3d.com/Packages/com.unity.collections@2.2/manual/allocator-rewindable.html
- Unity `NativeList`/`UnsafeList` — Collections 2.1/1.4/0.7 API pages
- Unity `AtomicSafetyHandle` / `DisposeSentinel` — ScriptReference + https://github.com/Unity-Technologies/UnityCsReference/blob/master/Runtime/Export/NativeArray/DisposeSentinel.cs
- Unity Burst AdvancedUsages (DisableSafetyChecks) — Burst 1.7 manual
- Drepper, "What Every Programmer Should Know About Memory" — https://lwn.net/Articles/252125/
- curiouscoding CPU benchmarks — https://curiouscoding.nl/posts/cpu-benchmarks/
- Loop-invariant code motion — https://en.wikipedia.org/wiki/Loop-invariant_code_motion
- Ryan Fleury, "Untangling Lifetimes: The Arena Allocator" — https://www.dgtlgrove.com/p/untangling-lifetimes-the-arena-allocator
- Chris Wellons, arena allocator — https://nullprogram.com/blog/2023/09/27/
- Rust `bumpalo` — https://docs.rs/bumpalo/
