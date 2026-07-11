# Dense types & the Arena allocator

`floatN`/`floatMxN` and their `double`/`int`/`short`/`long`/`uint`/`bool` counterparts are the
library's vector and matrix types. Matrices are row-major (`Data[r*N_Cols+c]`) — the opposite of
`Unity.Mathematics`' column-major layout, so any conversion between the two is a transpose, not a
reinterpret-cast; see [spec-interop.md](../dev/spec-interop.md)'s "Row-major ↔ column-major" section for
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

### Concurrency guards

Don't share one `Arena` across concurrently-running jobs/threads — under
`ENABLE_UNITY_COLLECTIONS_CHECKS`, a violation throws instead of corrupting memory; in a player
build without checks it's silent corruption, so the contract above still needs to be followed by
construction (one arena per concurrent job). See [rfc-memory-model.md](../dev/rfc-memory-model.md)
for the detection mechanism.

## Vectors & matrices

Both types carry either an arena-tracked record or a standalone `UnsafeList` (for `new floatN(n,
Allocator.Temp)`-style construction outside an arena) behind one indexer surface:

- Indexers: linear `this[int]` / `this[System.Index]` (from-end supported), and for matrices
  `this[int r, int c]` (bounds-checked only under `ENABLE_UNITY_COLLECTIONS_CHECKS`).
- Fields: `floatMxN.M_Rows`, `.N_Cols`, `.Length`, `.IsSquare`. Both vectors and matrices expose
  `.IsCreated` (false for `default` and after `Dispose()`, like any native container).
- Operators: `+ - * / %` (unary, scalar, and component-wise) and comparators (`< > <= >= == !=`,
  returning `boolN`/`boolMxN`) are all **allocating** — each is sugar over a `TempCopy()` plus the
  matching `Comp`/`UnsafeBoolOP` kernel. For a hot loop, call the `*InPlace` methods directly on a
  buffer you own instead (see [comp-elementwise](comp-elementwise.md)).
- `Copy()`/`TempCopy()`/`Dispose()` — only valid on arena-tracked instances (standalone instances
  dispose their own `UnsafeList` directly). For a job-safe standalone copy use the copy constructor:
  `new floatMxN(in orig, Allocator.Temp)` / `new floatN(in orig, Allocator.Temp)`.
- `CopyTo`/`CopyFrom` — into/from a same-shape vector or matrix, or a `NativeArray<float>`
  (row-major for matrices, lengths must match).
- **NativeArray views** — `new floatN(array)` and `new floatMxN(rows, cols, array)` wrap an existing
  `NativeArray<float>`'s memory with no copy and no ownership: reads/writes go straight to the
  array, `Dispose()` releases nothing, and the view is only valid while the array is alive. The
  view does not carry the array's job-safety handle — the caller owns the aliasing discipline.
  This is the zero-copy bridge for keeping game state in `NativeArray`s while solving in place
  through library types.

## The two-tier model: authoring vs compute

The API is really two tiers, and the threading contract falls out of which tier you're in:

- **Authoring tier (main thread): the arena.** Operators (`a + b`), `Copy()`/`TempCopy()`, the
  cross-type shortcuts, and the temp pool. Easy arithmetic structurally *requires* the arena: a C#
  operator receives only its operands, so the result's allocator must ride inside them — that is
  what the struct's internal owner reference is for. This tier is for setup, orchestration, and
  gameplay-level math on the scheduling thread.
- **Compute tier (jobs, any thread): pre-allocated buffers + in-place APIs.** `Comp.xxxInPlace`,
  `Blas.dot(..., ref dest)`, solver workspace forms — plus standalone `new floatN(n, Allocator.Temp)`
  scratch, which is thread-local and job-legal. Every kernel in the library itself lives in this
  tier.

The trap to know: **the arena rides invisibly inside every arena-tracked struct**, so an
innocent-looking `var c = a + b;` inside a job is an arena mutation from a worker thread — exactly
the race the contract forbids, reached without ever "passing the arena" anywhere. Inside jobs, use
the in-place APIs on buffers allocated before scheduling; leave the operators to the authoring tier.
(Under `ENABLE_UNITY_COLLECTIONS_CHECKS` the tripwire above turns an actual collision into a thrown
exception; in player builds it is silent corruption.)

## Temp pool & threading in practice

The temp pool is a convenience, not a hard requirement — every allocating op has a `ref`-destination
primitive that writes into a buffer you already own and allocates nothing. Reach for the temp pool in
one-shot/setup code and the zero-alloc primitives inside per-frame loops.

## Performance

The record indirection (`Data.Ptr` resolved once per op, not per element) is free in hot loops —
GEMM, factorizations, and sparse spMV/CG all run at the same throughput whether allocated standalone
or arena-tracked.
