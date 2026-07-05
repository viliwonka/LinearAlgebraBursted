# Spec: complete the Debug / Print / Export surface

*Historical document — method names predate the 2026-07 solver-API rework (see
docs/spec-solver-api-rework.md for the mapping).*

Goal: fill the gaps in the library's inspection surface so every result type and matrix type
can be printed (Burst) and exported (managed). Everything lives in `public static partial class
Print` (namespace `LinearAlgebra`; sparse helpers may use `LinearAlgebra.Sparse` overloads but keep
the `Print` class name).

## Architecture (the two surfaces + how they share)

Two surfaces, by hard constraint:
- **Burst surface** — `Print.Log(...)`, `Print.Spy(...)`, and per-type `ToFixedString()`. Uses
  `Unity.Collections.FixedStringNBytes`. Bounded (≤ ~4KB). `$"{x:G3}"` interpolation INTO a
  FixedString is Burst-legal (see existing `Debug.fProxy.cs`). Logs via
  `UnityEngine.Debug.Log(fixedString)`. **enum.ToString() is NOT Burst-legal** — map status enums
  with a manual `switch` returning a `FixedString32Bytes` literal per case.
- **Managed surface** — `Print.ToText/ToCsv/SaveCsv`. `System.String` + `System.IO.File`. Unbounded,
  main-thread ONLY, never Burst. This is correct and intended: export is a post-job operation.

**Shared core rule:** for a SMALL object, write formatting once as a Burst-safe `ToFixedString()`
(returns `FixedString128Bytes` for scalars/info, larger for grids). Managed `public override string
ToString() => ToFixedString().ToString();` wraps it — one implementation serves both. Large/unbounded
output stays managed-only.

## Item 1 — ToString on the info/result structs (Burst-friendly)

Structs: `SolveInfo`, `LstsqInfo`, `DirectSolveInfo`, `RankRevealingInfo` (OP/Solvers.Info.cs, singular),
`EigenSolveInfo`, `LanczosInfo` (OP/Eigen.Info.cs, singular), `fProxyPCAModel` (ML/PCA.Model.fProxy.cs,
templated).

For each, add:
- `public FixedString128Bytes ToFixedString()` — Burst-safe. Format compactly, e.g.:
  - SolveInfo: `SolveInfo(Converged, iters=42, rnorm=1.23e-08)`
  - LstsqInfo: `LstsqInfo(Converged, iters=17, rnorm=..., Arnorm=..., xnorm=...)`
  - DirectSolveInfo: `DirectSolveInfo(Success)`
  - RankRevealingInfo: `RankRevealingInfo(RankDeficient, rank=3)`
  - EigenSolveInfo: `EigenSolveInfo(Converged, iters=..., residual=...)`
  - LanczosInfo: `LanczosInfo(Converged, produced=20)`
  - fProxyPCAModel: `fProxyPCAModel(k=3, p=8, converged=true)` (buffer struct — summarize dims/k/converged,
    do NOT dump the component matrix)
  Use `:G3`/`:G6` for the doubles. Guard NaN gracefully (interpolation handles it).
- `public override string ToString() => ToFixedString().ToString();`
- A `Print.Log(in XInfo info)` overload (Burst) => `UnityEngine.Debug.Log(info.ToFixedString());`

Status-name helpers (Burst-safe), put next to the enums in OP/Solvers.SolveStatus.cs (singular) and
OP/Eigen.Info.cs: a static method `FixedString32Bytes Name(this IterativeSolveStatus s)` /
`Name(this DirectSolveStatus s)` with a `switch` returning `"Converged"`/`"MaxIterations"`/`"Breakdown"`
etc. Reuse in every ToFixedString. (If an extension method on an enum is awkward in a singular file,
a plain static `StatusName(IterativeSolveStatus)` is fine.)

Note the singular-file structs are NOT templated — put their helpers in the same singular files. The
templated fProxyPCAModel goes in its .fProxy template.

## Item 2 — Sparse block-structure printing (MATLAB spy)

New template file `Sparse/Debug.Sparse.fProxy.cs` (namespace LinearAlgebra.Sparse; class `Print` —
verify partial-class visibility across namespaces; if it must be `LinearAlgebra.Print`, add
`using`/qualify). fProxyBSM structure: BlockRows×BlockCols grid, BR×BC blocks, RowPtr(len BlockRows+1),
ColInd(block col per stored block), Values(nnzb*BR*BC row-major). Symmetric => only upper block-triangle
stored (mirror when displaying).

Add (all Burst-safe, FixedString4096Bytes buffer, cap gracefully if grid too big):
- `Print.Spy(in fProxyBSM m)` — BLOCK sparsity grid: one char per block, `X` if the block is stored
  (nonzero), `.` if absent. Header line: dims (M_Rows×N_Cols), block size BR×BC, block grid
  BlockRows×BlockCols, Nnzb, Symmetric flag, density = Nnzb/(BlockRows*BlockCols). For Symmetric,
  reflect stored upper blocks into the lower triangle in the display.
- `Print.Log(in fProxyBSM m)` — the Spy grid PLUS, if it fits the FixedString budget, the actual
  block values (iterate stored blocks, print `block(br,bc): [values...]`). Cap when the buffer nears
  full and append `...` so it never overflows.

Implement the block-present lookup per block-row via RowPtr[br]..RowPtr[br+1) scanning ColInd.

## Item 3 — Print + export for ALL matrix/vector types (managed)

Today `PrintExport.cs` (Source/Debug, hand-written) covers only float/double MxN+N. Extend coverage to
`iProxyMxN`/`iProxyN` (int/short/long) and `boolMxN`/`boolN`, plus a sparse export. Managed only (answer
to "does it work in Burst": NO — and it must not; File.IO + unbounded string are main-thread).

Preferred approach (matches codegen philosophy): MOVE the export into templates so type coverage is
automatic —
- `Debug/Export.fProxy.cs` => float/double (port existing ToText/ToCsv/SaveCsv for MxN+N).
- `Debug/Export.iProxy.cs` => int/short/long (numeric formatting; integers need no `:G` rounding).
- `Debug/Export.bool.cs` (singular or bool-typed) => bool matrices/vectors (print `1`/`0` or `T`/`F`
  for CSV; `True`/`False` for ToText — pick one and be consistent).
- Sparse export: `Print.ToText(in fProxyBSM)` / `SaveCsv(in fProxyBSM, path)` — densify via ToDense
  (needs an arena) OR emit a coordinate/triplet list (block br,bc + values). Coordinate list avoids the
  arena dependency; prefer that for the sparse CSV. Provide at least ToText (dense-ish preview) and a
  triplet CSV.
Then DELETE the now-superseded hand-written float/double bodies in PrintExport.cs (keep the file only if
something non-templatable remains; otherwise remove it and its .meta). Keep all method names/signatures
API-compatible (`Print.ToText`, `Print.ToCsv`, `Print.SaveCsv`).

If templatizing PrintExport proves too invasive, fallback: keep PrintExport.cs hand-written and ADD
int/short/long/bool/sparse overloads by hand. Coverage is the requirement; the mechanism is your call.

## Item 4 — Pivot + Indices debug (ignore RollingWindow)

- `Pivot` (Pivot/Pivot.cs, singular): add `ToFixedString()` (e.g. `Pivot[N=5, sign=+1]: (2 0 1 4 3)`),
  `ToString()` wrapper, `Print.Log(in Pivot)`. Access the permutation via its public indexer/N (the
  backing `indices` is private — use the public surface; add a read accessor only if none exists).
- `Indices` (Indices/Indices.cs, singular): same — `Indices[N=4]: (7 2 9 0)`, ToString, Print.Log.

## Verification

Regen (`Tools/regen-and-test.ps1`) must stay green (3673/3673+). Add a small test file per surface where
practical (info-struct ToString content sanity is testable on the managed side outside Burst; sparse Spy
can assert the grid string for a known small BSM). Keep codegen directives untouched. Commit on main.

## Burst gotchas (do not trip)
- No `System.String`, `StringBuilder`, `string.Format`, or `enum.ToString()` inside anything reachable
  from a [BurstCompile] job. FixedString + manual enum switch only.
- `ToString()` managed overrides are fine as long as they are never CALLED from inside a Burst job.
- FixedString has a byte cap — always guard appends near the limit and truncate with `...`.
- `//singularFile//` must remain line 1 of the singular files; never touch codegen directives.
