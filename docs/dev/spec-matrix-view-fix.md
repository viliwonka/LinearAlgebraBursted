# Mini-spec: fix `fProxyMxN` narrowed-view `Data.Length` footgun at the root

## 0. Status of the bug (verified by reading the code, not assumed)

Two point-patches already shipped for symptoms of this bug:

- `Assets/LinearAlgebra/CodeGen/TemplateSource/fProxy/fProxyMxN.cs:191-224` -- `CopyFrom(in Self)`/
  `CopyTo(in Self)` now validate against `M_Rows`/`N_Cols` (not the stale `Length` field). Root-caused
  in `fProxy/DEVLOG.md` ("CopyFrom(in Self) / CopyTo(in Self) -- silent-resize footgun").
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/NormsOP.fProxy.cs:18,26,32` -- `L2`/`L1`/`LInf(in
  fProxyMxN)` now scan `a.M_Rows * a.N_Cols` instead of `a.Data.Length`. Root-caused in `OP/DEVLOG.md`
  ("NormsOP" / "Krylov.bgmres" sections -- a real wrong-answer bug in `BlockGmresTests`, traced to
  `Norms.LInf` over-reading a `RectView`/`RowsView`'s stale full-buffer `Data.Length`).

Both fixes are correct but local. The root cause is still live: `View`/`RowsView`/`RectView`
(`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.Common.fProxy.cs:121-147`, duplicated in
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LOBPCG.fProxy.cs:931-949`) still produce a **struct copy
with only `M_Rows`/`N_Cols` overwritten** -- `Data` (hence `Data.Length`) and the readonly `Length` field
stay at the full backing buffer's size. Any current or future code that reads `.Data.Length` or `.Length`
on one of these views (rather than recomputing `M_Rows*N_Cols`) silently operates over the full backing
buffer, not the logical view -- over-reading (uninitialized/stale tail data corrupts a computed result,
as `Norms.LInf` did) or over-writing (a generic elementwise op clobbers rows/cols the view's caller
believes are untouched). This is unbounded whack-a-mole: every kernel that is ever handed one of these
views needs individual auditing forever.

**Confirmed live, not yet patched, exposure** (same root cause, not yet hit by a failing test): the
entire generic elementwise surface in
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/OP.Component.fProxy.cs` --
`zeroInPlace`/`fillInPlace`/`addInPlace`/`mulInPlace`/`divInPlace`/`subInPlace`/`addScaledInPlace`/
`scaleAddInPlace`/`clamp`/`abs`/`sign`/`sqrt`/every trig+exp helper, ~30 methods -- is generic over
`IUnsafefProxyArray` (shared by both `fProxyN` and `fProxyMxN`) and reads `place.Data.Length`. That's
safe for `fProxyN` (`N => Data.Length` always live, no narrowing concept exists for vectors) but is the
exact same footgun as `Norms` for `fProxyMxN` the moment any of these is called on a narrowed view. A
grep-based audit (Section 6) found no current call site inside `Krylov.Block.*`/`LOBPCG.fProxy.cs` doing
this today (those files use hand-rolled loops keyed off `M_Rows`/`N_Cols`, e.g. `BlockAdd`), so it is not
an active wrong-answer bug today -- but it is exactly the kind of landmine a future kernel or caller will
step on next, and this whole file's contract cannot be trusted until the root cause is fixed.

## 1. Root-cause finding that changes the fix's shape

The task brief assumes `View(buf, sa)` (`sa < s`) and any `RectView` with `cols < buf.N_Cols` are
**strided** (row pitch != logical width) and therefore fundamentally incompatible with the library's
pointer-based/SIMD kernels, requiring a copy-compaction fallback for that case. Reading the indexer
disproves this:

```
// fProxy/fProxyMxN.Indexing.cs:37-48
public ref fProxy this[int r, int c] {
    get { ... return ref Data.ElementAt(r * N_Cols + c); }
}
```

`View`/`RowsView`/`RectView` overwrite `N_Cols` itself (not just `M_Rows`), so every subsequent `[r,c]`
read/write already uses the **new, narrower** `N_Cols` as the row stride -- it addresses
`Data[r*newN_Cols + c]`, i.e. a **tightly-packed reinterpretation of the buffer's own leading
`rows*cols` flat elements**, not a strided leading corner of the original wider matrix that would need
`Data[r*buf.N_Cols + c]`. This is exactly what `View`'s own doc comment already (correctly) says:
"the leading m*m elements of buf's storage... not a new allocation, not a strided sub-block of a
larger stride". A concrete illustration: for a `4x4` `buf` (flat indices `0..15`) and `View(buf, 2)`,
`v[1,0]` reads `Data[2]` -- the buffer's 3rd flat element -- not `Data[4]` (which is what a
stride-preserving "top-left 2x2 corner of the 4x4" would read).

Consequence: **every one of `View`/`RowsView`/`RectView`'s existing call sites is already a contiguous
row-major prefix of the flat backing array**, whatever the (rows, cols) split. That means all three can
be turned into genuinely correctly-sized views via a zero-copy pointer+length reslice -- **no strided
view type, no per-call compaction, needed anywhere**, including the small `s x s` coefficient solves the
brief anticipated would need it (`BlockSolveSPD`'s `View(workBuf, saSearch)` fed to `CHO.decompInPlace`,
etc.). This supersedes the brief's Option 1/4 framing (which is safe but does unnecessary compaction
work) with a strictly better option that has the same blast radius and zero added cost. Section 6's
implementation checklist includes a one-time audit step to confirm no call site actually relies on the
(unused, and current-code-doesn't-provide-it-anyway) "leading corner preserving the wide buffer's own
pitch" interpretation before relying on this finding.

## 2. Option comparison

| # | Approach | Blast radius | Perf | Correctness | Ends whack-a-mole? |
|---|---|---|---|---|---|
| A | Universal real reslice via a new `fProxyMxN` constructor; `View`/`RowsView`/`RectView` become 1-line wrappers over it. No compaction anywhere. | 1 new ctor + 2 files' worth of 3-line helper bodies (`Krylov.Block.Common`, `LOBPCG`). Zero solver-body call-site changes (same `View(...)`/`RowsView(...)`/`RectView(...)` call shape). | Zero-copy, same as today -- no new allocation, no per-iteration copy, not even for the `s x s` case. | `Data.Length`/`Length` exactly right for every view, always; fixes Norms/elementwise/CopyFrom/linear-indexer-bounds-check/any future reader uniformly. | Yes -- closes the whole class by construction (Section 4). RECOMMENDED. |
| B | Brief's original Option 1/4: real reslice for `RowsView` (contiguous n-dim case) + compact-copy into an exact scratch buffer for the `s x s` `View`/narrow-`RectView` case | Same files as A, plus per-call-site compaction logic and scratch-buffer lifetime bookkeeping at every `View`/`RectView` call (~15 sites across `Krylov.Block.BCGrQ`/`BFBCG`/`GMRES`) | Reintroduces the exact per-iteration `O(s*s)` copy Phase-1 "Fix 5" deliberately removed (see `OP/DEVLOG.md`'s bcgrq/bfbcg sections) -- small in absolute terms (s is a handful to a few dozen) but pure waste since A shows it is unnecessary | Correct, but only because it happens to not need the compaction it pays for | Yes, but doing strictly more work than needed |
| C | Copy-into-exact-buffers everywhere (revert Fix 5): delete `View`/`RowsView`/`RectView`, allocate/copy a fresh exact buffer at every use | No new construct, but touches every call site (~50) and reintroduces the `O(sLive*n)` per-iteration copy on the hot `s x n` path that Fix 5 specifically removed (`bcgrq`/`bfbcg` DEVLOG entries: "one fewer `Allocator.Temp` allocation and one fewer O(sLive*n) copy per iteration") | Real, measured regression on the hot path, for both the cheap and expensive cases | Correct | Yes, but pure regression vs. the status quo's already-shipped optimization |
| D | First-class row-stride field on `fProxyMxN`, kernels honor it | Every pointer/SIMD kernel that assumes `stride == N_Cols` (`Blas.dot`, `CHO`, `QR`, `LQRP`, `UnsafeOP.*`, ...) would need a stride parameter threaded through -- dozens of files | Unknown/negative -- stride-aware inner loops are typically slower or need a slow-path fallback | Correct, most general | Yes, but wildly disproportionate to the actual need (Section 1 shows no current view is genuinely strided) |
| E | Status quo: keep patching individual `Data.Length` readers as they're found (Norms, CopyFrom already done) | Unbounded, per the user's own framing | No cost, no benefit | Correct only where already patched | No -- this is the problem, not a fix |

Recommendation: Option A. It has the same footprint as the brief's own Option 1/4 (a new `fProxyMxN`
constructor + rewriting the three helper functions) but, per Section 1's finding, needs no compaction
anywhere -- strictly cheaper and simpler than B/C, far less invasive than D, and (unlike E) actually ends
the whack-a-mole by making the invariant hold for every `fProxyMxN` in existence (Section 4).

## 3. Design: the new `fProxyMxN` constructor

`Length` is a `readonly int` field, settable only from a constructor (not a method) -- this is why a
proper constructor overload is required rather than an instance method (`buf.RowsView(rows)` cannot
assign `Length`). Add to `Assets/LinearAlgebra/CodeGen/TemplateSource/fProxy/fProxyMxN.cs`, next to the
existing `(int, int, NativeArray<fProxy> viewOf)` view constructor (~line 77):

```csharp
/// <summary>
/// Creates a standalone VIEW over the leading rows*cols elements of <paramref name="source"/>'s own
/// backing storage, repacked row-major as a rows x cols matrix -- a tightly-packed reinterpretation of
/// source's flat storage prefix, not a strided sub-block that preserves source's own N_Cols pitch. No
/// copy, no ownership; Dispose() on the view releases nothing. Aliases source's memory: reads/writes
/// through the view touch source's own storage in place, visible through any other handle to the same
/// storage. Valid only while source's backing memory is alive; outside the job-safety/generation-stamp
/// system, same contract as the NativeArray view constructor above -- caller owns the aliasing/lifetime
/// discipline. Throws if rows*cols exceeds source's own logical length.
/// </summary>
public unsafe fProxyMxN(in fProxyMxN source, int rows, int cols)
{
#if ENABLE_UNITY_COLLECTIONS_CHECKS
    if (!source.IsCreated)
        throw new InvalidOperationException("fProxyMxN view: source is not created / has been disposed");
#endif
    if (rows < 0 || cols < 0 || rows * cols > source.Data.Length)
        throw new ArgumentException("fProxyMxN view: rows*cols must not exceed source's own logical length");

    _rec = null;
    _gen = 0; // standalone (non-arena): never read (AssertRecordValid short-circuits on _rec == null)
    M_Rows = rows;
    N_Cols = cols;
    Length = rows * cols;
    _inlineData = new UnsafeList<fProxy>(source.Data.Ptr, Length);
}
```

Notes tied to the brief's specific questions:

- Arena (`_rec != null`) vs. standalone (`_inlineData`) source. The view is always constructed as
  standalone (`_rec = null`), regardless of whether `source` is arena-tracked. This is required, not just
  convenient: the `Data` property setter for an arena-backed instance writes through `_rec->Data = value`
  -- a shared pointer with every other handle to that record. If a narrowed view kept `_rec` and ever
  set `Data`, it would silently truncate the original (non-view) matrix's `Data` for every other holder
  of that same record. Going standalone with a raw `Ptr,length` `UnsafeList` (the exact mechanism the
  existing `NativeArray` view constructor already uses) sidesteps this entirely: the view never touches
  `_rec`, so it cannot corrupt the source's record no matter what.
- IsCreated/job-safety. `IsCreated` on the view resolves via the standalone branch
  (`_inlineData.IsCreated`) -- correct, no arena generation check involved. The view forfeits the
  generation-stamp staleness check (`AssertRecordValid`) that a live `_rec` would otherwise provide on
  every read -- this is not a new regression: it is the exact same, already-accepted tradeoff the
  shipped `NativeArray` view constructor documents ("outside the job-safety system... caller owns the
  aliasing/race discipline"). Every current `View`/`RowsView`/`RectView` call site is short-lived
  `Allocator.Temp`/arena scratch whose lifetime strictly encloses the view's use within a single solver
  call -- never spanning a `Clear()`/`Dispose()` of the backing buffer -- so this is safe in practice, not
  merely in theory.
- Dispose(). Because `_inlineData` is built from the `UnsafeList<T>(T* ptr, int length)` constructor
  (not an allocating one), disposing the view is a documented no-op (mirrors the `NativeArray` view
  constructor's "Dispose() releases nothing" contract) -- calling `.Dispose()` on a view by mistake is
  harmless. (Audited: no current `View`/`RowsView`/`RectView` result is ever explicitly disposed --
  `.Dispose()` calls in `Krylov.Block.*`/`LOBPCG.fProxy.cs` are all on the original caller-owned buffers.)
- ENABLE_UNITY_COLLECTIONS_CHECKS. Two checks, matching the existing split in this file between
  always-on argument validation (the exact-length check on the `NativeArray` ctor) and debug-only
  liveness checks (`AssertRecordValid`): the `rows*cols` bound is a cheap `ArgumentException`, thrown
  unconditionally (construction-time only, not per-element, so no perf concern in player builds); the
  `source.IsCreated` liveness check is gated behind `ENABLE_UNITY_COLLECTIONS_CHECKS` like
  `AssertRecordValid` itself.
- Ambiguity check. `(in fProxyMxN, int, int)` does not collide with any existing overload
  (`(int,int,NativeArray<fProxy>)`, `(int,int,Allocator,bool)`, the internal record ctor, or the
  `(in fProxyMxN, Allocator)` copy ctor) -- distinct arities/types.

## 4. The universal invariant this establishes

`instance.Data.Length == instance.M_Rows * instance.N_Cols` holds for every live `fProxyMxN`, always --
this is the acceptance property. Verified by construction for every path that can produce an instance:

1. `(int,int,NativeArray<fProxy>)` -- `Length = viewOf.Length`, checked equal to `M_rows*N_cols` at entry.
2. `(int,int,Allocator,bool)` -- `Length = M_Rows*N_Cols` directly.
3. `(int,int,fProxyMatRecord*,Allocator,bool)` (internal, arena) -- same.
4. `(in fProxyMxN orig, Allocator)` / `(in fProxyMxN orig, fProxyMatRecord*, Allocator)` (copy ctors) --
   `Length = orig.Length`, `M_Rows`/`N_Cols` copied verbatim from `orig` -- holds by induction on `orig`.
5. New `(in fProxyMxN source, int rows, int cols)` (Section 3) -- `Length = rows*cols` directly.
6. `default(fProxyMxN)` -- `0 == 0*0`, trivially.

No other way to construct an `fProxyMxN` exists. Any future kernel that reads `.Data.Length` or
`.Length` on any `fProxyMxN` -- raw allocation or view, arena or standalone -- gets the right answer
with no per-kernel audit required. This is what ends the whack-a-mole.

Side benefit: the linear indexer's debug-only bounds check
(`Assume.IndexInsideBounds(Data.Length, index)` in `fProxyMxN.Indexing.cs:18`) was previously checking
against the stale full-buffer length for a view, so an out-of-bounds linear read past a narrowed
view's logical region silently succeeded in `ENABLE_UNITY_COLLECTIONS_CHECKS` builds. After this fix it
correctly throws.

## 5. Rewriting `View`/`RowsView`/`RectView`

`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.Common.fProxy.cs:121-147` (shared by `bcg`,
`bcgrq`, `bfbcg`, and `bgmres` -- `Krylov.Block.GMRES.fProxy.cs` calls these same three functions):

```csharp
static fProxyMxN View(in fProxyMxN buf, int m) => new fProxyMxN(in buf, m, m);

static fProxyMxN RowsView(in fProxyMxN buf, int rows) => new fProxyMxN(in buf, rows, buf.N_Cols);

static fProxyMxN RectView(in fProxyMxN buf, int rows, int cols) => new fProxyMxN(in buf, rows, cols);
```

`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LOBPCG.fProxy.cs:931-949` -- identical duplicate
implementation (no `RectView` there), same rewrite:

```csharp
[MethodImpl(MethodImplOptions.AggressiveInlining)]
static fProxyMxN View(in fProxyMxN buf, int m) => new fProxyMxN(in buf, m, m);

[MethodImpl(MethodImplOptions.AggressiveInlining)]
static fProxyMxN RowsView(in fProxyMxN buf, int rows) => new fProxyMxN(in buf, rows, buf.N_Cols);
```

Update each function's doc comment to state the corrected contract (real view, `Data.Length`/`Length`
now exactly `rows*cols`) instead of the old "not a new allocation" framing that said nothing about
`Data.Length`. Do not rename or move these functions, and do not touch any call site that uses
`View`/`RowsView`/`RectView` (`FactorLiveResidual`/`FactorLiveSearch`/`BlockSolveSPD`/
`FactorGramOnce` in `Krylov.Block.Common.fProxy.cs`; every call in `Krylov.Block.BCGrQ.fProxy.cs`,
`Krylov.Block.BFBCG.fProxy.cs`, `Krylov.Block.GMRES.fProxy.cs`, `LOBPCG.fProxy.cs`) -- their call shape
(`View(buf, m)`, `RowsView(buf, rows)`, `RectView(buf, rows, cols)`) is unchanged; only what happens
inside the helper changes. This is what keeps the blast radius to two files' worth of 3-line function
bodies while retroactively fixing every block solver that calls them.

## 6. Implementation checklist (ordered)

1. Add the new constructor to `fProxy/fProxyMxN.cs` (Section 3).
2. Audit (grep, read, do not skip): confirm no call site reads any of `Lbuf`/`Gbuf`/`workBuf`/`PQbuf`/
   `alphaBuf`/`betaBuf`/`HijBuf`/`YiBuf`/`Rscratch`/`HQscratch`/`QtGscratch`/`Yscratch` (the buffers fed
   through `View`/`RectView` with a narrower `N_Cols` than their own allocation) directly at their
   native shape anywhere -- i.e. confirm every access to these specific buffers goes through
   `View`/`RowsView`/`RectView` first, never `buf[i,j]` on the raw un-viewed struct. This confirms
   Section 1's finding holds for every current call site before relying on it. (Buffers only ever passed
   through `RowsView`, e.g. `Gbuf`/`Hbuf` in `bgmres`, need no such check -- `RowsView` never changes
   `N_Cols`, so it is always safe regardless.)
3. Rewrite `View`/`RowsView`/`RectView` in `Krylov.Block.Common.fProxy.cs` (Section 5).
4. Rewrite `View`/`RowsView` in `LOBPCG.fProxy.cs` (Section 5).
5. Do not modify `NormsOP.fProxy.cs`'s or `fProxyMxN.cs`'s already-shipped `CopyFrom`/`Norms` fixes
   (Section 7) -- leave them exactly as they are.
6. Add the regression tests (Section 8).
7. Add `DEVLOG.md` entries (`fProxy/DEVLOG.md` for the new constructor -- new top entry above the
   existing `CopyFrom` entry; `OP/DEVLOG.md` for the `View`/`RowsView`/`RectView` rewrite -- new top entry
   that cross-references and supersedes-in-spirit the `NormsOP`/`Krylov.bgmres` entries, without deleting
   them). Per `CLAUDE.md`: rationale/history goes in `DEVLOG.md` only; code comments on the new
   constructor and the rewritten helpers state contracts only (Section 3/5's comment text above is
   contract-only and can be used close to verbatim).
8. Run `Tools/regen.ps1` to regenerate `Source/Generated` (float/double) from the templates. Do not
   hand-edit generated output.
9. Run the full headless test suite (`Tools/*.ps1`). Green gate = exact `Result="Passed"`, no silent Mono
   fallback (see the project's own Burst-test-compile-gotchas lesson -- a real Burst compile error here
   would show up as a suite-duration spike and scattered 1-ULP fails in untouched tests, not a clean red;
   if that pattern appears, it is a compile problem, not a semantic regression -- fix the compile error,
   don't chase ULPs).

## 7. Disposition of the already-shipped patches

- `fProxyMxN.CopyFrom(in Self)`/`CopyTo(in Self)` (Section 0): keep as-is, no change. These validate
  against `M_Rows`/`N_Cols` directly, which is correct both before and after this fix, and is the right
  place for a same-type dimension-mismatch check regardless of the `Data.Length` invariant. Not
  redundant -- it is a distinct, still-necessary shape check (not a `Data.Length` trust question).
- `NormsOP.fProxy.cs`'s `L2`/`L1`/`LInf(in fProxyMxN)` (Section 0): keep as-is, no revert. After this
  fix, `a.Data.Length == a.M_Rows*a.N_Cols` always, so `a.M_Rows*a.N_Cols` and `a.Data.Length` are now
  interchangeable here -- reverting to `a.Data.Length` would be pure churn with no behavior change and no
  benefit. Leave the code and its `OP/DEVLOG.md` entries untouched.
- The `uninit: true` -> `uninit: false` workarounds in `Krylov.Block.GMRES.fProxy.cs` (`Wbuf`/`HQscratch`,
  per `OP/DEVLOG.md`'s "Krylov.bgmres" entry) were already noted there as "now unnecessary but harmless" --
  still true after this fix; no action needed.

## 8. Regression tests to add

Test 1 -- constructor correctness and bounds, in
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/BridgeFillTests.fProxy.cs`, alongside the
existing `NativeArray`-view-constructor tests (~line 99):
- Build a `4x4` `fProxyMxN` (`Allocator.Temp`), fill flat elements `0..15` with `0,1,2,...,15`.
- `view = new fProxyMxN(in buf, 2, 2)`. Assert `view.Length == 4`, `view.Data.Length == 4`,
  `view.M_Rows == 2`, `view.N_Cols == 2`.
- Assert `view[0,0]==0`, `view[0,1]==1`, `view[1,0]==2`, `view[1,1]==3` (the tightly-packed prefix
  reading, per Section 1 -- explicitly not `view[1,0]==4`, which would be the stride-preserving
  "top-left corner of the 4x4" reading).
- Assert `new fProxyMxN(in buf, 5, 4)` (20 > 16) throws `ArgumentException`.
- Assert `new fProxyMxN(in buf, 4, 4)` (exactly 16) succeeds and is elementwise equal to `buf`.

Test 2 -- poisoned-tail invariant through `Norms` and the generic elementwise surface, new file
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/MatrixViewInvariantTests.fProxy.cs` (Burst
`[BurstCompile] IJob`, per the project's existing test-file convention):
- Allocate `buf` (`s x s`, e.g. `s=6`, `Allocator.Temp`, `uninit: true`), fill the entire buffer with a
  distinctive sentinel (e.g. `(fProxy)12345`) via `buf.fillInPlace(...)`, then overwrite only the leading
  `m*m` flat elements (`m=3`) with a known test pattern (any deterministic nonzero values).
- `view = new fProxyMxN(in buf, m, m)`.
- Assert `Norms.L1(in view)`/`Norms.L2(in view)`/`Norms.LInf(in view)` equal the values independently
  computed from the known `m*m` pattern alone (i.e. the sentinel-filled tail must not participate).
- Call `view.zeroInPlace()`. Assert every element of `view` (via `view[i,j]`) is now `0`. Assert flat
  buffer elements at index `>= m*m` (read via `buf`, not `view`) still equal the sentinel `12345`,
  unchanged -- proving `zeroInPlace` did not over-write past the view's logical region.

Test 3 -- retroactive fix through `LQRP`/`QR`/`CHO`, same new file:
- Build `wide` (`s x n` e.g. `s=8, n=5`, `Allocator.Temp`, `uninit: true`), fill entirely with a sentinel,
  then overwrite rows `[0, sLive)` (`sLive=3`) with a fixed, deterministic test matrix.
- Build an independent `exact` (`sLive x n`, no sentinel anywhere) with the same values as `wide`'s
  first `sLive` rows.
- `view = new fProxyMxN(in wide, sLive, n)` (mirrors `RowsView`). Run `LQRP.decomp` on `view` and on
  `exact` (separate `L`/`Q`/`Pivot` outputs each) and assert the two `L`/`Q` results match to tight
  (`1e-5f`/`1e-10` per type) tolerance.
- Repeat the same pattern (poisoned-tail wide buffer vs. clean exact buffer, same-shape narrowed view)
  for `QR.decompInPlace` and for `CHO.decompInPlace`+`CHO.decompSolve` using a `View(buf, m)`-style
  square view (`m < s`, buffer allocated `s x s`) instead of a `RowsView`, to also cover the `N_Cols`-
  narrowing case.

Existing coverage that must stay green (no new test needed, just confirm in the suite run):
`BlockCGrQTests.fProxy.cs`, `BlockBiCGStabTests.fProxy.cs`, `BlockGmresTests.fProxy.cs`,
`LOBPCGRobustnessTests.fProxy.cs`/`LOBPCGSmokeTests.fProxy.cs` -- these already exercise
`View`/`RowsView`/`RectView` under Burst through many iterations of shrinking/deflating active widths
(exactly the pattern this fix targets); they are the "retroactively fixes the shipped block solvers"
proof, and the full-suite green gate (Section 6, step 9) is what certifies it.

## 9. Out of scope

- Do not touch `iProxyMxN`/`boolMxN` -- they have the identical `_rec`/`_inlineData`/readonly-`Length`
  structure (per `fProxy/DEVLOG.md`'s note that the six type-family files are near-identical clones), but
  grep confirms no `View`/`RowsView`/`RectView`-style helper exists for them anywhere in the codebase
  today (the pattern is `fProxy`-only, used solely by the block-Krylov/LOBPCG solvers). Add the same
  reslice constructor to those types only if/when a real caller needs it.
- Do not add a first-class stride field or touch any pointer/SIMD kernel's addressing logic (Option D,
  rejected -- Section 1/2).
- Do not rename, relocate, or change the call signature of `View`/`RowsView`/`RectView`, and do not
  touch any of their call sites in `Krylov.Block.BCGrQ.fProxy.cs`/`Krylov.Block.BFBCG.fProxy.cs`/
  `Krylov.Block.GMRES.fProxy.cs`/`LOBPCG.fProxy.cs` -- only the three (two in LOBPCG) helper function
  bodies change.
- Do not revert or modify the already-shipped `CopyFrom`/`Norms` patches (Section 7).
- Do not deduplicate the two separate `View`/`RowsView` implementations (`Krylov.Block.Common` vs.
  `LOBPCG.fProxy.cs`) into one shared location -- both call the same new constructor after this fix, so
  they are already just two thin wrappers; merging their homes is a separate, purely cosmetic follow-up
  with no correctness payoff, left for later if ever wanted.
- Do not flip any README checkbox or write user-facing docs -- this is an internal correctness fix with
  no API-surface change (the new constructor is additive; nothing existing is removed or renamed).

## 10. Acceptance criteria

1. `fProxyMxN` (template) has the new `(in fProxyMxN source, int rows, int cols)` constructor exactly as
   specified in Section 3, including both guard checks.
2. `View`/`RowsView`/`RectView` in `Krylov.Block.Common.fProxy.cs` and `View`/`RowsView` in
   `LOBPCG.fProxy.cs` are rewritten as one-line wrappers over the new constructor (Section 5); no other
   line in either file changes except their doc comments.
3. `Data.Length == M_Rows * N_Cols` and `Length == M_Rows * N_Cols` hold for every `fProxyMxN` produced
   by any constructor, including a `View`/`RowsView`/`RectView` result -- Test 1 (Section 8) asserts this
   directly for a `View`-style narrowing.
4. Test 1, Test 2, and Test 3 (Section 8) exist and pass.
5. The full existing test suite passes with `Result="Passed"` (no new failures, no Mono-fallback compile
   errors) after `Tools/regen.ps1` regeneration -- in particular `BlockCGrQTests`, `BlockBiCGStabTests`,
   `BlockGmresTests`, `LOBPCGRobustnessTests`, `LOBPCGSmokeTests` all still pass unmodified.
6. No `.Data.CopyFrom`/allocation/per-iteration-copy is added anywhere in `Krylov.Block.*.fProxy.cs` or
   `LOBPCG.fProxy.cs` as part of this fix (grep diff review: the only new allocation-shaped code is the
   `UnsafeList<fProxy>(ptr, length)` construction inside the new `fProxyMxN` constructor itself, which
   performs no allocation).
7. `fProxy/DEVLOG.md` and `OP/DEVLOG.md` each have a new top entry describing this fix, per `CLAUDE.md`'s
   comment policy; no history/rationale prose is left in code comments on the new constructor or the
   rewritten helpers.

## Executive summary (for the orchestrator)

Chosen approach: give `fProxyMxN` one new zero-copy reslice constructor
(`fProxyMxN(in fProxyMxN source, int rows, int cols)`) and rewrite the existing `View`/`RowsView`/
`RectView` helpers (in `Krylov.Block.Common.fProxy.cs` and duplicated in `LOBPCG.fProxy.cs`) to be
one-line wrappers over it, instead of the current bare struct-copy that leaves `Data`/`Length` stale.
Reading the `(r,c)` indexer proved the brief's assumed "strided `s x s` view" case doesn't actually
exist in this codebase: `View`/`RectView` already overwrite `N_Cols` itself, so every current use is
already a tightly-packed reinterpretation of the buffer's own leading flat prefix, not a stride-
preserving sub-block -- so no compaction/copy is needed anywhere, not even for the small
coefficient-matrix solves, contra the brief's Option 1/4 framing. This establishes a universal, provable-
by-induction invariant: `Data.Length == M_Rows*N_Cols` for every `fProxyMxN` that can ever exist (raw
allocation, `NativeArray` view, or reslice view) -- which is exactly what ends the whack-a-mole, since
every current and future kernel that reads `Data.Length`/`.Length` (the already-patched `Norms`/
`CopyFrom`, and the ~30-method generic elementwise surface in `OP.Component.fProxy.cs` that isn't hit by
a failing test yet but shares the identical exposure) becomes correct automatically, with zero code
changes to those files. Blast radius: one new constructor + two files' worth of 3-line helper-function
bodies; zero changes to any block-solver call site, so this retroactively fixes `bcg`/`bcgrq`/`bfbcg`/
`bbiCGStab`/`bgmres` and LOBPCG's own views without touching their algorithms. Perf impact: none --
still zero-copy/zero-allocation, identical to today's cost, with no reintroduced per-iteration copy on
any path (unlike the compaction fallback the brief's Option 1/4 anticipated needing). The already-shipped
`Norms`/`CopyFrom` patches are not reverted -- they become redundant-but-harmless under the new
invariant and are left exactly as-is to avoid pure-churn diffs.
