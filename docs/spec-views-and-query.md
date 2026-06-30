# Spec (research): Views/Slicing and Search/Query

> **SUPERSEDED 2026-06-25.** Decision: **Views = DROPPED** (strided views can't feed contiguous kernels →
> materialise anyway; more machinery than games need). **Query = KEPT** — converged, coder-ready spec moved
> to **`docs/spec-query.md`**. This file is retained only for the research grounding (reference-library
> survey, Burst notes, sources) and the now-abandoned views design.

Status: RESEARCH DRAFT (iteration 1). No code yet. Games-oriented Burst library — favour simple,
zero-alloc, Burst-compatible designs over completeness. The two open README items:
`🔳 Vector/matrix views (slicing)` and `🔳 Find/query (e.g. the row with the largest L2 norm)`.

## Burst ground rules (constrain every option below)
- No managed types in jobs: **no C# lambdas/delegates/closures**, no LINQ, no `List<T>`, no exceptions
  as control flow. Everything is unmanaged structs + pointers.
- The library's existing data types (`fProxyN`/`fProxyMxN`) are **contiguous, row-major** over an
  `UnsafeList` (`A[i,j] = Data[i*N_Cols + j]`). Every existing kernel (`Unsafe_OP.*`) assumes this
  contiguity. A strided/non-contiguous view therefore CANNOT be fed to those kernels as-is.
- The library already has the Burst-native "lambda": the **struct-functor pattern** used by the
  optimizers — `gradientDescent<F>(... ref F f) where F : struct, IfProxyGradientFunction`. The caller
  defines a small struct implementing an interface; the generic is monomorphised → zero overhead, no
  allocation, fully inlinable. This is the key enabler for query predicates.

---

## Part A — Views / Slicing

### How the reference libraries do it
- **NumPy**: *basic slicing* (`a[2:5]`, `a[:,1]`, `a[1:3,2:4]`) returns a **view** — same buffer,
  described by `offset + strides + shape`. *Fancy/advanced indexing* (integer arrays, boolean masks
  `a[a>0]`) returns a **copy**, because the result can't be expressed as offset+stride+count.
- **Eigen**: lightweight expression objects, no copy, usable as lvalues (assignable): `.block(i,j,p,q)`,
  `.row(i)`, `.col(j)`, `.segment(i,n)`, `.head(n)`, `.tail(n)`, corners (`topLeftCorner`, `topRows`…),
  and (3.4+) a general `A(rowIndices, colIndices)` strided/indexed slicing. Fixed- or dynamic-size.
- **MathNet.Numerics**: `SubMatrix(...)`, `Row(i)`, `Column(j)` mostly **copy** (return new Vector/Matrix).
- **Julia**: `A[1:3,:]` copies; `view(A, 1:3, :)` / `@view` returns a `SubArray` (view).

### The core decision: strided view (no copy) vs copy
- **Option A — strided views** (NumPy/Eigen): a `VectorView`/`MatrixView` struct holding
  `(basePtr, offset, len/rows/cols, strides)`. Indexing `view[i,j] = base[offset + i*rowStride + j*colStride]`.
  - Pros: zero-alloc; write-through to parent (NumPy semantics); natural row/col/block/diagonal slices;
    assignable.
  - Cons: a view is **non-contiguous**, so it can't be passed to any existing `dot`/`matmul`/decomp
    kernel without either (a) a parallel set of stride-aware kernels (huge surface) or (b) materialising
    (copying) it first. It's also a *separate type* from `fProxyMxN`, so it won't satisfy the existing
    generic `IUnsafefProxyArray` ops unless that interface is generalised. No `Dispose` (doesn't own
    memory); dangles if the parent is freed/reallocated (UB, not an exception).
- **Option B — copy extract/write-back** (MathNet): `getRow(i)`, `getCol(j)`, `getBlock(r0,c0,rows,cols)`
  return a fresh contiguous arena vec/mat; `setRow/setCol/setBlock/setDiagonal` write back.
  - Pros: result is a normal contiguous `fProxyN/MxN` → works with ALL existing ops immediately; dead
    simple; Burst-safe.
  - Cons: allocates + copies (not zero-alloc); no write-through (must explicitly write back).
- **Option C — HYBRID (recommended)**: lightweight VIEW structs for the cheap read/write cases that do
  NOT need to flow into heavy LA ops, PLUS copy-extract/write-back for the cases that do.
  - Views: `M.row(i)`, `M.col(j)`, `M.diagonal()`, `v.segment(start,len)` / `v.head(n)` / `v.tail(n)`,
    `M.block(r0,c0,rows,cols)`. Support get/set indexing (write-through) and a `.CopyTo(dest)` /
    `.Copy()` to materialise a contiguous matrix when an op needs one.
  - Copy path: `getBlock`/`setBlock` etc. for "pull a sub-matrix out, run a solve, write it back."
  - Rationale: in a game, slicing is overwhelmingly *read/write a row/column/block/diagonal*, not
    *feed a slice into matmul*. The hybrid covers the 90% case with zero alloc and keeps every kernel
    untouched. Heavy ops on a slice cost one explicit copy — acceptable and rare.

### Scope guard (don't overcomplicate)
- v1: positive indices only, half-open `[start, start+len)`, NO negative indices, NO step/stride-skip.
  (Add `step` later if a real need appears — games rarely slice every k-th element.)
- View lifetime: valid only while the parent lives and is not reallocated. Document it; don't try to
  track it (no GC here). Arena vecs/mats are allocated once and never resized, so this is safe in
  practice.

---

## Part B — Search / Query

### How the reference libraries do it
- **NumPy**: `where(cond)`, `nonzero`, `argwhere`, `argmax`/`argmin`, `count_nonzero`, `searchsorted`
  (binary search on sorted), boolean masking `a[a>cond]`, `any`/`all`.
- **Eigen**: `minCoeff(&i,&j)`/`maxCoeff(&i,&j)` (value + index), visitors, `(A>0).count()`, functors.
- **MathNet**: `.Find(predicate)`, `.MaximumIndex()`, `.MinimumIndex()`, LINQ-ish.

### "Can we do lambdas?" — YES, as struct-functors (not C# lambdas)
Managed lambdas are out under Burst. But the **struct-functor predicate** is the exact same pattern the
optimizers already use and IS Burst-compatible:
```
public interface IfProxyPredicate { bool Invoke(fProxy x); }
// caller:
struct GreaterThan : IfProxyPredicate { public fProxy t; public bool Invoke(fProxy x) => x > t; }
int idx = QueryOP.findFirst(in v, new GreaterThan { t = 5 });
```
More verbose than `v.findFirst(x => x > 5)` but zero-alloc, inlinable, and consistent with the rest of
the library. This is the recommended flexible path.

### Three complementary mechanisms (recommend all, minimally)
1. **Struct-functor predicate API** (flexible): `findFirst<P>`, `count<P>`, `any<P>`, `all<P>`,
   `findAll<P>(in v, ref intN outIndices, out int n)` where `P : struct, IfProxyPredicate`.
   - Pros: NumPy-`where`-like flexibility, zero-alloc, no managed lambda. Cons: caller writes a struct.
2. **Concrete built-ins** (no functor for the common cases): `argMin`/`argMax` (already in StatsOP),
   `minCoeff`/`maxCoeff` returning value+index, `findValue(v, x[, tol])` → first matching index,
   `nonzero(v) → indices`, and the README example `argMaxRowNorm(M)` / `argMaxColNorm` (row/col index
   with the largest L2 norm). Covers the frequent game queries without a functor.
3. **Boolean-mask path (already half-built)**: the library already produces `boolN`/`boolMxN` from
   comparisons (`v > 5`) and has `select` + bool reductions. Add `count(boolN)`, `any(boolN)`,
   `all(boolN)`, `nonzero(boolN) → indices`. Reuses existing machinery; composes with `select`.
   - Pro: composable, mostly already there. Con: the mask itself allocates a `boolN` (the functor path
     avoids that).

### Variable-length results under Burst (the `findAll`/`nonzero` problem)
Can't return a managed list. Use the library's existing ref-dest idiom: **caller provides an `intN`
index buffer (length ≥ N worst case) + `out int count`**, the op fills `[0,count)`. Zero-alloc, matches
the rest of the API. (Alternative: a two-pass count-then-arena-allocate-exact — more convenient, one
alloc; offer later if wanted.)

### Scope guard
- Implement the above small set; do NOT build full NumPy `where`(broadcast-select), `argwhere`
  multi-dim, or `searchsorted` unless a concrete need appears (sorted binary search is cheap to add if
  someone needs it for spatial/lookup work).

---

## Leaning recommendation (to be confirmed over next iterations)
- **Views**: Option C (hybrid). Lightweight view structs (`row`/`col`/`diagonal`/`segment`/`block`,
  write-through, `.Copy()` to materialise) + copy `getBlock`/`setBlock`. No step/negative in v1.
- **Query**: struct-functor predicates (`findFirst`/`count`/`any`/`all`/`findAll`) + a handful of
  concrete helpers (`min/maxCoeff`+index, `findValue`, `nonzero`, `argMaxRowNorm`) + bool-mask
  reductions. Variable-length out via caller buffer + `out count`.

## Open questions for later iterations
- Should views satisfy a shared read/write interface so a few key ops (e.g. `dot`) can accept them
  directly via stride-aware overloads, or always materialise? (Lean: always materialise in v1.)
- Index type for results: `intN` (existing) vs a raw `int*`/`NativeArray<int>`. (Lean: `intN`.)
- Naming: Eigen (`block`/`segment`/`head`/`tail`) vs NumPy (`slice`) — pick one consistent set.
- Do we want a `mask`-free fused `countWhere<P>`/`firstWhere<P>` only, dropping the boolN path to keep
  the surface small? (Revisit after sketching call sites.)

---

## Iteration 2 — grounded in the real types; open questions resolved

### NativeSlice precedent + what backs a view here
Unity's `NativeSlice<T>` is the proven model: a strided window (byte-stride = multiple of element size),
**owns no memory, can't be Disposed, Burst-safe, write-through**. But `NativeSlice` wraps a `NativeArray`,
whereas this library's `Data` is an **`UnsafeList<fProxy>`** (confirmed: `IUnsafefProxyArray.Data` is
`UnsafeList<fProxy>`), and `NativeSlice` is 1-D only. So:
- **1-D views** (row, column, diagonal, vector segment) map cleanly onto NativeSlice *semantics* — a small
  custom struct `(fProxy* base, int offset, int length, int stride)` over `Data.Ptr`. (Could literally be a
  `NativeSlice<fProxy>` via `NativeSliceUnsafeUtility.ConvertExistingDataToNativeSlice`, but a custom struct
  keeps the public API in the library's own types and avoids leaking Unity's slice type.)
- **2-D block views** need a custom struct `(fProxy* base, int offset, int rows, int cols, int rowStride,
  int colStride)` — NativeSlice can't express 2-D.

### RESOLVED open questions
- **Views can't feed existing ops → materialise in v1.** `IUnsafefProxyArray` requires a contiguous
  `UnsafeList<fProxy> Data`; a strided view has no such contiguous buffer, so it CANNOT implement that
  interface and CANNOT be passed to `dot`/`matmul`/decompositions. v1 = the view exposes get/set indexing +
  `Copy()`/`CopyTo`/`CopyFrom`; heavy ops take an explicit materialised copy. (No stride-aware kernel fork.)
- **Naming = Eigen.** `row`/`col`/`diagonal`/`block`/`segment`/`head`/`tail` — clear, standard, and C#
  can't do NumPy's `[a:b, c:d]` ergonomically for 2-D anyway.
- **Keep the boolN path AND the functor path.** Comparisons (`v > 5` → `boolN`) already exist and compose
  with `select`; the functor path is the zero-alloc primary. Not dropping either.
- **value+index returns use `out` params**, not ValueTuple (safest under Burst).

### Concrete API sketch (illustrative, not final)
Views (custom structs; factory methods are extensions on `fProxyN`/`fProxyMxN`):
```
struct fProxyVecView { fProxy* b; int off, len, stride;  fProxy this[int i]{get;set;}  int N;
                       void CopyTo(ref fProxyN dst); void CopyFrom(in fProxyN src); fProxyN Copy(); }
struct fProxyMatView { fProxy* b; int off, rows, cols, rowStride, colStride;  fProxy this[int r,int c]{...}
                       fProxyMxN Copy(); ... }

v.segment(start,len) / v.head(n) / v.tail(n)          -> fProxyVecView      (stride 1)
M.row(i)      -> fProxyVecView   (off i*N_Cols, stride 1,        len N_Cols)
M.col(j)      -> fProxyVecView   (off j,        stride N_Cols,   len M_Rows)
M.diagonal()  -> fProxyVecView   (off 0,        stride N_Cols+1, len min(M,N))
M.block(r0,c0,rows,cols) -> fProxyMatView
```
Query (`QueryOP`, currently an empty stub — good, nothing to break):
```
interface IfProxyPredicate { bool Test(fProxy x); }     // the Burst "lambda" (matches optimizer functors)

int  findFirst<P>(in fProxyN v, ref P p) where P:struct,IfProxyPredicate    // -1 if none
int  count<P>(in fProxyN v, ref P p);  bool any<P>(...);  bool all<P>(...)
int  findAll<P>(in fProxyN v, ref intN outIdx, ref P p)  // returns count; outIdx.N >= v.N (worst case)

void minCoeff(in fProxyN v, out fProxy val, out int i)   // + maxCoeff; + (in fProxyMxN, out val, out r, out c)
int  findValue(in fProxyN v, fProxy x, fProxy tol)       // first |v[i]-x|<=tol, else -1
int  nonzero(in fProxyN v, ref intN outIdx, fProxy tol)  // indices where |v[i]|>tol; returns count
int  argMaxRowNorm(in fProxyMxN M)  // README example; + argMaxColNorm

int  count(in boolN mask);  bool any(in boolN mask);  bool all(in boolN mask)   // reuse v>5 -> boolN
int  nonzero(in boolN mask, ref intN outIdx)
```
Example caller predicate (zero-alloc, no managed lambda):
```
struct GreaterThan : IfProxyPredicate { public fProxy t; public bool Test(fProxy x) => x > t; }
var gt = new GreaterThan { t = 5 };
int n = QueryOP.findAll(in v, ref idxBuf, ref gt);
```

### Still open (next iteration)
- `intN` result buffer vs an exact-size arena alloc (two-pass) — pick one and note the trade-off.
- Whether `head`/`tail`/`segment` are worth it given `block`/`row`/`col` cover most game needs (trim surface?).
- A single `viewOf`/indexer sugar vs the explicit method names.

---

## Iteration 3 — final resolutions, phasing, and recommendation

### pandas — deliberately not a model here
pandas is **label-indexed dataframes** (heterogeneous columns, `.loc`/`.iloc`, index alignment). Its
*positional* `.iloc` slicing is just NumPy's; the label/alignment machinery is a different domain
(tabular data, not homogeneous numeric arrays) and contributes nothing for a Burst numeric library.
NumPy + Eigen are the right references; pandas is noted and set aside.

### Last open questions — RESOLVED
- **Variable-length results = primitive + convenience** (same pattern as CG / SVD workspaces):
  - primitive (zero-alloc): `findAll(in v, ref intN outIdx, …) -> int count`; caller sizes `outIdx.N >= v.N`.
  - convenience: `findAll(in v, …) -> intN` that does a count pass, arena-allocates the exact size, fills.
  Ship the primitive first; add the convenience if callers want it.
- **Keep `segment` as the 1-D primitive; `head`/`tail` are 2-line sugar over it** (Eigen-familiar, not bloat).
  Matrix slicing primitives are `block`/`row`/`col`/`diagonal`.
- **No `Range`/`a[1..3,2..4]` indexer sugar in v1** — C# can't do 2-D range indexing cleanly; explicit
  Eigen-style method names are clearer. Revisit if desired.
- **Views are one-level and read-write (write-through)** in v1. View-of-a-view composition and a read-only
  marker are future niceties, not needed for games.

### Suggested phasing (query first — lower risk, higher immediate value)
- **Phase 1 — Query** (no new types, no kernel changes, reuses existing patterns):
  concrete built-ins (`min/maxCoeff`+index, `findValue`, `nonzero`, `argMaxRowNorm/argMaxColNorm`),
  functor predicates (`findFirst`/`count`/`any`/`all`/`findAll<P>`), bool-mask reductions
  (`count`/`any`/`all`/`nonzero` over `boolN`). Fills the empty `QueryOP` stub. Cheap, immediately useful.
- **Phase 2 — Views** (new `fProxyVecView`/`fProxyMatView` structs + factory methods + `Copy`/`CopyFrom`;
  needs codegen for the new structs). Materialise-only, so contained — no kernel surgery.

### FINAL RECOMMENDATION (one paragraph)
For a games-oriented Burst library: implement **query first** as a small `QueryOP` surface — struct-functor
predicates (the Burst-native "lambda", identical in shape to the optimizer functors, so no managed lambdas)
plus a handful of concrete helpers, with variable-length results returned via a caller `intN` buffer + count
(zero-alloc) and an optional arena convenience. Then add **hybrid views** — lightweight strided
`VecView`/`MatView` structs (`row`/`col`/`diagonal`/`segment`/`block`, write-through, no Dispose, mirroring
`NativeSlice` semantics over `Data.Ptr`) for cheap read/write, with `.Copy()` to materialise a contiguous
matrix whenever a heavy op needs one. Do NOT fork the kernels for strides, do NOT build NumPy
`where`/`argwhere`/broadcast, and skip negative indices / steps in v1. This stays fully Burst-compatible and
small while covering the real game use-cases (slice a row/column/block, find/count/argmax with a custom
predicate).

## Sources
- Unity NativeSlice (stride, no-dispose, Burst): https://docs.unity3d.com/ScriptReference/Unity.Collections.NativeSlice_1.html
- NumPy copies/views: https://numpy.org/doc/stable/user/basics.copies.html
- NumPy sorting/searching/counting: https://numpy.org/devdocs/reference/routines.sort.html
- Eigen block ops: https://libeigen.gitlab.io/eigen/docs-nightly/group__TutorialBlockOperations.html ; slicing (3.4): https://eigen.tuxfamily.org/dox/group__TutorialSlicingIndexing.html
- NumPy sorting/searching/counting: https://numpy.org/devdocs/reference/routines.sort.html ;
  argwhere: https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
- Eigen block operations: https://libeigen.gitlab.io/eigen/docs-nightly/group__TutorialBlockOperations.html ;
  slicing/indexing (3.4): https://eigen.tuxfamily.org/dox/group__TutorialSlicingIndexing.html
