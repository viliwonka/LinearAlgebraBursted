# Print & export

## Burst `Log`/`Spy` (bounded, callable from inside a job)

- `Print.Log(in floatN a, int start = 0, int end = -1)` / `Log(in floatMxN m)` — uses
  `FixedString4096Bytes` internally, so output **silently truncates** past 4KB; fine for
  spot-checking a value mid-algorithm, not for dumping large matrices.
- `Print.Spy(in floatMxN m[, float absThreshold = 0.01f])` — an ASCII sparsity grid (`X`/space).
- Every solver/eigensolver diagnostics struct — `DirectSolveInfo`, `RankInfo`, `SolveInfo`,
  `LstsqInfo` (see [solvers.md](solvers.md)), `EigenSolveInfo`, `LanczosInfo`, `LOBPCGInfo` (see
  [eigen.md](eigen.md#diagnostics-structs)) — has a matching `Print.Log(in <Struct>)`: a Burst-safe
  compact summary, e.g. `DirectSolveInfo(Success)`, never allocates.
- `boolN`/`boolMxN` overloads, and [sparse](sparse-bsr.md) equivalents: `Print.Spy(in floatBSR m)`
  (one character per block, mirrors symmetric storage for display) and `Print.Log(in floatBSR m)`.

## Managed export (unbounded, NOT Burst-callable)

- `Print.ToText(in floatMxN|floatN) : string` — `G7`, human-readable preview.
- `Print.ToCsv(in floatMxN|floatN) : string` — `G9`, round-trip-exact.
- `Print.SaveCsv(in floatMxN|floatN, string path)` — writes via `File.WriteAllText`.
- Sparse: `Print.ToText(in floatBSR m)` (densifies via `ToDense(Allocator.Temp)`, then reuses the
  dense path), `Print.ToCsv(in floatBSR m)` (block-triplet CSV — `blockRow,blockCol,v0..v(BR·BC-1)`,
  no densification needed), `Print.SaveCsv(in floatBSR m, path)`.

## Histogram quick-look

`Print.Histogram(in floatN data, int bins = 16, int width = 40)` — auto min/max range, a horizontal
ASCII bar chart via `UnityEngine.Debug.Log`. Distinct from the [`Histogram`](stats.md) class's
binning API, which is for computing counts/density/CDF.
