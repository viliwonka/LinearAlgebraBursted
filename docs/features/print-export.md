# Print & export

`Print` — split into a Burst-callable inspection surface and a managed unbounded export surface.
Design doc: [spec-debug-print.md](../spec-debug-print.md).

## Burst `Log`/`Spy` (bounded, callable from inside a job)

- `Print.Log(in floatN a, int start = 0, int end = -1)` / `Log(in floatMxN m)` — uses
  `FixedString4096Bytes` internally, so output **silently truncates** past 4KB; fine for
  spot-checking a value mid-algorithm, not for dumping large matrices.
- `Print.Spy(in floatMxN m[, float absThreshold = 0.01f])` — an ASCII sparsity grid (`X`/space).
- Every solver/eigensolver diagnostics struct — `DirectSolveInfo`, `RankInfo`, `SolveInfo`,
  `LstsqInfo` (see [solvers.md](solvers.md)), `EigenSolveInfo`, `LanczosInfo` (see
  [eigen.md](eigen.md#diagnostics-structs)) — has a matching `Print.Log(in <Struct>)`: a Burst-safe
  compact summary, e.g. `DirectSolveInfo(Success)`, never allocates. `LOBPCGInfo` doesn't have one
  yet as of this writing but is expected to follow the same convention.
- `boolN`/`boolMxN` overloads, and [sparse](sparse-bsr.md) equivalents: `Print.Spy(in floatBSR m)`
  (one character per block, mirrors symmetric storage for display) and `Print.Log(in floatBSR m)`.

## Managed export (unbounded, NOT Burst-callable)

- `Print.ToText(in floatMxN|floatN) : string` — `G7`, human-readable preview.
- `Print.ToCsv(in floatMxN|floatN) : string` — `G9`, round-trip-exact.
- `Print.SaveCsv(in floatMxN|floatN, string path)` — writes via `File.WriteAllText`.
- Sparse: `Print.ToText(in floatBSR m)` (densifies via a throwaway internal `Arena`, then reuses the
  dense path), `Print.ToCsv(in floatBSR m)` (block-triplet CSV — `blockRow,blockCol,v0..v(BR·BC-1)`,
  no densification needed), `Print.SaveCsv(in floatBSR m, path)`.

## Histogram quick-look

`Print.Histogram(in floatN data, int bins = 16, int width = 40)` — auto min/max range, a horizontal
ASCII bar chart via `UnityEngine.Debug.Log`. Distinct from (and simpler than) the
[`Histogram`](stats.md) class's binning API, which is meant for computing counts/density/CDF, not
just eyeballing a distribution.

## Benchmarks

Not applicable — this is inspection/export tooling, not a hot-path feature.
