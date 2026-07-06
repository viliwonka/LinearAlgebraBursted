# Query

`Query` (bare, de-genericized class). Search and selection over the rows/columns of a matrix, or
flat over a vector. Full design rationale: [spec-query.md](../dev/spec-query.md) and
[spec-predicate-queries.md](../dev/spec-predicate-queries.md).

- **Enums** — `Metric{Manhattan, Euclidean, SqEuclidean, Chebyshev, Cosine, Dot}` (Cosine/Dot are
  similarities — higher is nearer; the rest are distances — lower is nearer); `Norm{L1, L2, Linf}`.
- **Extremes** — `argMaxAbs`/`argMinAbs`, `rowArgMin`/`rowArgMax`/`colArgMin`/`colArgMax(in A, ref
  Indices idx[, ref floatN val])`, `argMaxRowNorm`/`argMaxColNorm(in A, Norm n)`.
- **Search over rows/columns** — `nearestRow`/`nearestColumn(in A, in q, Metric m, out index, out
  score)`, `farthestRow`/`farthestColumn`, `kNearestRows`/`kNearestColumns`/`kFarthestRows`/
  `kFarthestColumns(in A, in q, int k, Metric m, ref Indices idx, ref floatN scores)` (bounded
  insertion sort, O(M·k)), `countWithinRadius`/`rowsWithinRadius(in A, in q, float r, Metric m, ...)`.
- **Value/mask** — `nonzero`, `findValue`, `countNonzero` (flat, vector or matrix).
- **Predicate-filtered variants** — every search above has a masked twin taking a struct-functor
  predicate (`IfloatPredicate`/`IfloatRowPredicate`/`IfloatColPredicate`): `whichRows`/`whichColumns`,
  `findAll`, `nearestRowWhere`/`nearestColumnWhere`/`kNearestRowsWhere` (empty-result contract:
  index = -1), and score-based `argMaxRowBy`/`argMinRowBy`/`topKRowsBy` (`IfloatRowScore` functor) +
  column twins.

## Performance

Not benchmarked as a standalone feature. `nearestRow`/`kNearestRows` are linear scans (O(M) per
query, O(M·k) for k-nearest via bounded insertion) — no spatial index; that tradeoff is deliberate
for a games-oriented library operating on modest-sized in-memory matrices, not a gap to be filled.
