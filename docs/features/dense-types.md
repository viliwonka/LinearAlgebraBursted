# Dense types & the Arena allocator

`floatN`/`floatMxN` and their `double`/`int`/`short`/`long`/`uint`/`bool` counterparts are the
library's vector and matrix types. Matrices are row-major (`Data[r*N_Cols+c]`) — the opposite of
`Unity.Mathematics`' column-major layout, so any conversion between the two is a transpose, not a
reinterpret-cast; see [spec-interop.md](../spec-interop.md)'s "Row-major ↔ column-major" section for
the full correctness argument.

## Arena

`Arena` is a cheap-to-copy handle (a pointer to a heap-allocated core) to a bump/tracking allocator.
All copies of an `Arena` share one core, so passing it by value is safe — but **disposing two copies
of the same handle double-frees**; treat an `Arena` like a single owned resource.

- `new Arena(Allocator allocator)` — `Allocator.Persistent` for long-lived state, `Allocator.Temp`
  for a single frame/job.
- `arena.floatVec(int N, bool uninit = false)` / `arena.floatVec(int N, float s)` — vector factories
  (zeroed, or filled with `s`). `arena.floatMat(int rows, int cols, ...)` — matrix factories, plus a
  square `floatMat(int dim)` overload.
- `ArenaExtensions` adds everything else as `this ref Arena` extensions: `floatIdentityMat`,
  `floatRandomMat`, `floatRandomDiagonalMat`, `floatBasisVec`, `floatLinVec`, `floatHouseholderMat`,
  `floatHilbertMat`, and per-feature workspace factories (`floatSVDThinCache`, `floatFFTCache`, …).
- `arena.ClearTemp()` — disposes only the **temp** pool (the scratch every allocating op/operator
  produces); call it once per frame/loop iteration to keep temp allocations from accumulating.
- `arena.Dispose()` — disposes everything (persistent + temp) and frees the core itself.

**Threading contract (from the type's own doc comment):** an `Arena` is single-threaded by contract
— like Unity's native containers, a single instance must not be mutated (allocated from, cleared, or
disposed) from more than one thread/job concurrently. Concretely:

- **Allocate before you schedule.** Do an arena's persistent allocations (and any `Pivot`/`Indices`
  buffers a factorization needs) on the scheduling thread before handing data derived from it to a
  job. A job's `Execute()` can itself call arena factories/`Clear`/`ClearTemp` (they're
  Burst-compilable), but only if no other thread is touching the same arena at the same time.
- **One arena per concurrently-running job/thread.** If two jobs may run at the same time (no
  dependency between them), give each its own `Arena`. Sharing one arena across concurrent jobs
  races the record tables' chunk-directory/free-list bookkeeping — silently, with no exception, just
  wrong answers or a crash later.
- **Complete jobs before `Clear()`/`ClearTemp()`/`Dispose()`.** Wait on a job's `JobHandle` before
  tearing down (or clearing) an arena it might still touch.

### Concurrency guards (detection, not prevention)

Nothing about `Arena`/`ArenaCore` makes concurrent misuse safe — the contract above still has to be
followed. What exists is *detection*: under `ENABLE_UNITY_COLLECTIONS_CHECKS`, every mutating entry
point (the `arena.floatVec`/`floatMat`/... factories, `fProxyBSR`/`fProxyBSRBuilder`/
`fProxyBlockJacobi`, `Clear`, `ClearTemp`, `Dispose`, `Pivot`, `Indices`) is wrapped by two
independent mechanisms so a contract violation throws instead of corrupting memory silently:

1. **Race tripwire.** An `int` flag inside `ArenaCore` (heap-resident, so it doesn't affect
   `sizeof(Arena)`) is armed via `Interlocked.CompareExchange` on entry to a guarded body and
   released via `Interlocked.Exchange` on exit. If a second thread/job enters a guarded body while
   the flag is already armed, it throws `InvalidOperationException` immediately — this is the
   mechanism that actually catches two jobs racing the same arena. It is not a mutex: it never
   blocks, it only detects overlap. Reentrancy (one guarded method legitimately calling another,
   e.g. `Clear()` internally clearing the temp pool too) is handled structurally, by routing
   internal calls through unguarded "core" helpers instead of nesting the guard — not by counting,
   since Burst has no `Thread.CurrentThread` to distinguish legitimate same-thread nesting from a
   genuine second thread.
2. **`AtomicSafetyHandle`.** `ArenaCore` also owns an `AtomicSafetyHandle`, created in `Init` and
   released in `Dispose`, checked at the top of every guarded entry point. This catches a live
   handle used (from any thread) after `Dispose()` already released it.

**Why not Unity's `[NativeContainer]` job-scheduling protocol** (the mechanism that makes the job
debugger reject two `IJob`s capturing the same container without a dependency, *at Schedule time*)?
That protocol requires the `AtomicSafetyHandle` to be a field directly on the struct a job captures
by value — Unity's own `NativeList<T>`/`NativeReference<T>` carry `m_Safety` inline for exactly this
reason. The struct a job captures here is `Arena`, which is deliberately pinned to a single
pointer's width (`ArenaLayoutTests.Arena_IsPointerSized`) so it stays cheap to copy and pass around;
adding an `AtomicSafetyHandle` field to it would grow `sizeof(Arena)` under
`ENABLE_UNITY_COLLECTIONS_CHECKS` and break that pin. The handle instead lives inside the
heap-allocated `ArenaCore` that `Arena` points to, which keeps `Arena` pointer-sized but means there
is no automatic Schedule-time rejection — only the manual checks above, which fire once the
conflicting code actually runs, not the moment it's scheduled.

**Known gap:** an individual buffer's own `Dispose()` (e.g. `floatN.Dispose()`) also mutates a
record table (frees its slot) but is *not* guarded by the tripwire — it sits at a different altitude
than the Arena/ArenaCore entry points above (every numeric/bool/sparse type's own per-instance
lifecycle method, closer in spirit to "element access"). Two threads disposing different allocations
from the same arena concurrently is therefore still a real, undetected race.

## Vectors & matrices

Both types carry either an arena-tracked record or a standalone `UnsafeList` (for `new floatN(n,
Allocator.Temp)`-style construction outside an arena) behind one indexer surface:

- Indexers: linear `this[int]` / `this[System.Index]` (from-end supported), and for matrices
  `this[int r, int c]` (bounds-checked only under `ENABLE_UNITY_COLLECTIONS_CHECKS`).
- Fields: `floatMxN.M_Rows`, `.N_Cols`, `.Length`, `.IsSquare`.
- Operators: `+ - * / %` (unary, scalar, and component-wise) and comparators (`< > <= >= == !=`,
  returning `boolN`/`boolMxN`) are all **allocating** — each is sugar over a `TempCopy()` plus the
  matching `Comp`/`UnsafeBoolOP` kernel. For a hot loop, call the `*InPlace` methods directly on a
  buffer you own instead (see [comp-elementwise](comp-elementwise.md)).
- `Copy()`/`TempCopy()`/`Dispose()` — only valid on arena-tracked instances (standalone instances
  dispose their own `UnsafeList` directly).

## Temp pool & threading in practice

The temp pool is a convenience, not a hard requirement — every allocating op has a `ref`-destination
primitive that writes into a buffer you already own and allocates nothing (see
[zero-alloc-ops](../zero-alloc-ops.md)). Reach for the temp pool in one-shot/setup code and the
zero-alloc primitives inside per-frame loops.
