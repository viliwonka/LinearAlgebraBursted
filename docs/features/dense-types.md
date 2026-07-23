# Dense types, allocation & lifetime

`floatN`/`floatMxN` and their `double`/`int`/`short`/`long`/`uint`/`bool` counterparts are the
library's vector and matrix types. Matrices are row-major (`Data[r*N_Cols+c]`) — the opposite of
`Unity.Mathematics`' column-major layout, so any conversion between the two is a transpose, not a
reinterpret-cast.

## Allocation

Every type is a plain unmanaged struct over an `UnsafeList`. Allocation is explicit — you pick the
`Allocator`, you own the lifetime:

- `new floatN(int n, Allocator allocator = Allocator.Temp, bool uninit = false)` — zeroed (or
  uninitialized) vector. `new floatMxN(int rows, int cols, Allocator allocator, bool uninit = false)`
  — matrix.
- `Allocator.Temp` is freed automatically at end of frame (main thread) or end of job — no
  `Dispose()` needed. Use it for scratch and one-shot work.
- `Allocator.Persistent` (and `TempJob`) must be `Dispose()`d. Use it for state that lives across
  frames.
- `GenerateOP` holds the filled/structured factories, same names and arguments library-wide:
  `GenerateOP.floatVec(n, s)` (fill with `s`), `floatIdentityMat`, `floatRandomMat`,
  `floatRandomDiagonalMat`, `floatBasisVec`, `floatLinVec`, `floatHouseholderMat`, `floatLinspace`,
  kernels/windows, and integer/bool counterparts. Each takes a trailing
  `Allocator allocator = Allocator.Temp`.
- Test matrices live on `floatGallery` (`floatGallery.floatHilbert(n)`, …); conversions to/from
  `Unity.Mathematics` fixed-size types on `ConvertOP`.
- Workspaces/caches construct directly: `new floatQRCache(m, n, allocator)`,
  `new floatFFTCache(n, allocator)`, `new floatLOBPCGCache(n, k, allocator)`, … each has a matching
  `Dispose()`.

## Vectors & matrices

- Indexers: linear `this[int]` / `this[System.Index]` (from-end supported), and for matrices
  `this[int r, int c]` (bounds-checked only under `ENABLE_UNITY_COLLECTIONS_CHECKS`).
- Fields: `floatMxN.M_Rows`, `.N_Cols`, `.Length`, `.IsSquare`. Both vectors and matrices expose
  `.IsCreated` (false for `default` and after `Dispose()`, like any native container).
- Comparators: `< > <= >= == !=` return a freshly allocated `boolN`/`boolMxN`. All arithmetic is
  explicit — the `Comp` in-place kernels (`floatComp.addInPlace(dst, src)`, …) mutate a buffer you
  own and allocate nothing (see [comp-elementwise](comp-elementwise.md)).
- `Copy()`/`TempCopy()` return an independent `Allocator.Temp` copy. For a copy on a specific
  allocator use the copy constructor: `new floatMxN(in orig, allocator)`.
- `CopyTo`/`CopyFrom` — into/from a same-shape vector or matrix, or a `NativeArray<float>`
  (row-major for matrices, lengths must match).
- **NativeArray views** — `new floatN(array)` and `new floatMxN(rows, cols, array)` wrap an existing
  `NativeArray<float>`'s memory with no copy and no ownership: reads/writes go straight to the
  array, `Dispose()` releases nothing, and the view is only valid while the array is alive. The
  view does not carry the array's job-safety handle — the caller owns the aliasing discipline.
  This is the zero-copy bridge for keeping game state in `NativeArray`s while solving in place
  through library types.

## Jobs

The structs copy by value into a job, but the copy shares the same native memory — element writes
inside `Execute()` persist and are visible to the caller after the job completes. Two rules:

- **Write through a matrix, never reassign it.** Reassigning the struct field itself
  (`A = other;`) or calling `Dispose()` inside a job only affects the job's private copy of the
  handle — the caller's copy still points at the old (possibly freed) memory.
- **`Allocator.Temp` inside a job is thread-local and job-legal** — allocate scratch freely; it is
  freed when the job ends. For buffers a job returns to the caller, allocate them before
  scheduling (or use `Persistent` and dispose on the main thread after the job completes).

## Performance

Allocation-free hot loops: every allocating op has a `ref`-destination primitive
(`Blas.dot(..., ref dest)`, `Comp.xxxInPlace`, solver workspace forms) that writes into a buffer you
already own. Reach for the allocating conveniences in one-shot/setup code and the zero-alloc
primitives inside per-frame loops.
