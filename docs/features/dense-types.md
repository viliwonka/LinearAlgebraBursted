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

**Threading contract (from the type's own doc comment):** an `Arena` is not thread-safe — like
Unity's native containers, a single instance must not be allocated from or disposed from more than
one thread concurrently. Use one arena per job/thread rather than sharing one across threads.

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
