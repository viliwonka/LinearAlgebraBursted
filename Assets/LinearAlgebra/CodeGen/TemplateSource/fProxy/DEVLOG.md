# DEVLOG — fProxy
Code comments state contracts only; history lives here (see CLAUDE.md).

Note: the arena-tracking doc comments covered below (`_rec`, `AssertRecordAlive`/
`AssertRecordValid`, `_gen`, `Dispose()`, `OwnerArena`, the MxN `StructLayout` rationale, and the
standalone-source allocator guard) were near-identical clones across all six parallel type-family
files: `fProxyN.cs`, `fProxyMxN.cs`, `iProxy\iProxyN.cs`, `iProxy\iProxyMxN.cs`, `bool\boolN.cs`,
`bool\boolMxN.cs`. Each was rewritten to one contract-only version and applied consistently to all
six; this single entry captures the shared history/rationale that used to be inline in all of them.

## CopyFrom(in Self) / CopyTo(in Self) -- silent-resize footgun (N and MxN family)
- 2026-07-19 | Root cause: `Data` returns the backing `UnsafeList<T>` BY VALUE (plain fields, not
  indirected through a pointer). `dst.Data.CopyFrom(src.Data)` resizes `dst` off `src.Data.Length` --
  for a narrowed same-buffer view (the `View`/`RowsView`/`RectView` pattern used by
  `Krylov.Block.fProxy.cs`/`LOBPCG.fProxy.cs`: copy the struct, overwrite `M_Rows`/`N_Cols`, leave the
  readonly `Length` field pointing at the ORIGINAL backing buffer's size), `Data.Length` reports the
  FULL backing size, not the narrowed logical `M_Rows*N_Cols`. When `dst` must grow to match that
  stale size, the reallocation lands on the DISCARDED temporary `UnsafeList` the `Data` getter
  returned -- `dst`'s real storage is left at its original (often zeroed) content. Found via
  `LQRP.decomp`'s `W.Data.CopyFrom(A.Data)`; see OP/DEVLOG.md's bcgrq section for the concrete repro
  (`L[0,0]` came back exactly 0 despite a real residual).
  FIX: `CopyFrom(in Self)`/`CopyTo(in Self)` on `fProxyN`/`fProxyMxN`/`iProxyN`/`iProxyMxN`/`boolMxN`
  (mirrors the already-safe `CopyFrom(NativeArray<T>)` overload just below each) now validate the
  LOGICAL size -- `N` for the vector family (always live: `N => Data.Length`), or `M_Rows`/`N_Cols`
  for the matrix family (the pre-existing dimension check already used those, not the possibly-stale
  `Length` field) -- and `UnsafeUtility.MemCpy` exactly that many elements: fixed-size, never resizes
  either side, throws `ArgumentException` on a genuine mismatch. `boolN` has no struct-to-struct
  `CopyFrom`/`CopyTo` overload (only the `NativeArray<bool>` ones), so nothing changed there.

## `_rec` (arena-tracked record pointer), N and MxN family
- 2026-07-11 | Design cites docs/dev/rfc-memory-model.md §4 Option A (the record-pointer design).
  Replaces an older `Arena _arena` handle field: retiring that field kept the struct's size
  unchanged (both are a single pointer-width field), and the record's own `Owner` back-pointer is
  what `Copy()`/`TempCopy()`/the cross-type shortcuts resolve through instead of a stored `_arena`.
  (was fProxyN.cs:11-16, fProxyMxN.cs:24-26, iProxyN.cs:14-19, iProxyMxN.cs:27-29, boolN.cs:12-17,
  boolMxN.cs:24-26)

## `AssertRecordAlive` (N family) / `AssertRecordValid` (MxN family)
- 2026-07-11 | N-family structs (fProxyN/iProxyN/boolN) have no spare bits to pack a generation
  stamp into: 32B = 8 (`_rec`) + 24 (`UnsafeList<T>`), exactly -- confirmed by
  `ArenaLayoutTests.VectorStructsAreExpectedSize` (docs/dev/rfc-memory-model.md §6.2) -- so
  `AssertRecordAlive` only checks Alive: it catches a read after Dispose()/Clear()/ClearTemp() on
  the record, but not a stale handle into a slot recycled by a fresh Allocate() (that needs the
  generation stamp the MxN family carries in its padding hole). Both guards are
  ENABLE_UNITY_COLLECTIONS_CHECKS-only (auto-defined by the Unity Editor/every test run, compiled
  out of player builds; struct size is identical either way since neither adds a field). Both use
  `ChunkedRecordTable`'s `IsAliveFast`/`GenerationFast(TRecord*)` -- direct pointer casts, no
  index, no chunk-scan -- rather than the index-based `IsAlive(int)`/`GetGeneration(int)`, because
  the getter runs on every read (per element, via the indexer), so the index-based chunk scan
  would be a real per-element cost; see `IsAliveFast`'s own doc comment (ChunkedRecordTable.cs)
  for the container-of rationale. (was fProxyN.cs:37-53, fProxyMxN.cs:136-152, iProxyN.cs:40-56,
  iProxyMxN.cs:139-155, boolN.cs:39-54, boolMxN.cs:134-149; test-name/spec citations also at
  iProxyN.cs:45-46, boolN.cs:43, iProxyMxN.cs:54-65, boolMxN.cs:56)

## `_gen` (MxN family generation stamp)
- 2026-07-11 | Byte-size derivation for why `_gen` is free: with `[StructLayout(Sequential)]` and
  natural alignment, `M_Rows(4)+N_Cols(4)+_rec(8)+_inlineData(24)+Length(4) = 44` bytes, and the
  struct's 8-byte alignment (forced by the pointer/UnsafeList fields) rounds that up to 48
  regardless -- there were already 4 unused trailing bytes (docs/dev/rfc-memory-model.md §6.2's
  padding analysis, confirmed by `ArenaLayoutTests.MatrixStructsAreExpectedSize` staying at 48
  with the field present). (was fProxyMxN.cs:51-62, iProxyMxN.cs:54-65, boolMxN.cs:51-62)

## `Dispose()` (N and MxN family)
- 2026-07-11 | LINALG_DEBUG NaN-poison-on-dispose removed 2026-07-05: the symbol was defined
  nowhere in the project, so that block was dead code that had never executed. Superseded by the
  record table's own unconditional guards -- a double-dispose (aliased or not) throws
  deterministically via `Free()`'s double-Free check in every build config, not just a debug one --
  plus the ENABLE_UNITY_COLLECTIONS_CHECKS generational overlay on the `Data` getter, which catches
  a stale read instead of returning garbage. Before that removal, disposing had a silent
  double-free through a stale value-copy in the arena's tracking list. `Arena.cs`'s
  `Clear()`/`ClearTemp()` use the opposite order (dispose-then-Free) safely, for a different
  reason -- see the comment there. (was fProxyN.cs:168-190, fProxyMxN.cs:176-200,
  iProxyN.cs:171-197, iProxyMxN.cs:179-204, boolN.cs:134-156 "same ordering rationale as every
  other migrated family's N.Dispose()", boolMxN.cs:173-191 "same ordering rationale as
  boolN.Dispose()...")

## `TempCopy()` (N family)
- 2026-07-11 | Trailing comment "(was wrongly the persistent Copy path)" on fProxyN.cs:149 /
  iProxyN.cs:152 referred to a fixed bug where `TempCopy()` incorrectly routed through the
  persistent-allocation `Copy()` path instead of the arena's temp pool.

## Standalone-source allocator guard (N and MxN family, all six files)
- 2026-07-11 | The "guard a standalone (null-record) source" comment on the standalone copy
  constructors used to read "— was dereferencing null for the default allocator", describing a
  fixed bug (the old code dereferenced a null `_rec`/`_arena` when no allocator was given for a
  standalone source) rather than the current guard's behavior. (was fProxyN.cs:96,
  fProxyMxN.cs:107, iProxyN.cs:99, iProxyMxN.cs:110, boolN.cs:78, boolMxN.cs:104 -- only the boolN/
  boolMxN occurrences were in the original audit list; the identical text in the other four files
  was condensed for consistency)

## `StructLayout` rationale (MxN family, all three files)
- 2026-07-11 | `[StructLayout(Sequential)]` pins field order/packing explicitly instead of leaving
  it to the compiler's default Auto layout; this is what makes the trailing padding hole -- and
  therefore `_gen`'s placement in it -- a guarantee rather than an implementation detail (Auto
  layout is free to reorder/repack fields, so relying on "the compiler currently happens to leave
  4 bytes at the end" without Sequential would be fragile). (was fProxyMxN.cs:12-17,
  iProxyMxN.cs:15-20, boolMxN.cs:12-17 -- boolMxN's copy wasn't in the original audit list but is
  the same clone, condensed for consistency)

## `fProxyN.Shortcuts.cs` / `fProxyMxN.Shortcuts.cs`
- 2026-07-11 | `fProxyBSRTranspose` (fProxyN.Shortcuts.cs:23-26 originally): named the specific
  downstream consumer (`Krylov.fProxy.cs`, `b.fProxyBSRTranspose(in A)`) and the private `_rec`
  field it lets callers avoid touching directly.
- 2026-07-11 | `//alsoExpand[uint]//` block (fProxyMxN.Shortcuts.cs:7-11 originally): explained
  that it widens the iProxy-family copy-replace block to a 4th (uint) copy, giving fProxyMxN
  uintVec/uintTempVec/uintMat/uintTempMat shortcuts alongside int/short/long, mirroring
  iProxyMxN.Shortcuts.cs/iProxyN.Shortcuts.cs, because Hash.fProxy.cs's rowHashes/colHashes
  wrappers need it to allocate their uintN result from A's own arena without touching fProxyMxN's
  private `_rec` field directly.
