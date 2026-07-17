# DEVLOG — ML
Code comments state contracts only; history lives here (see CLAUDE.md).

## KMeans.fProxy.cs — raw-pointer hoist (spec-raw-pointer-hoist-pass batch 4)
- 2026-07-17 | The O(N·D) / O(k·D) per-iteration loops (PointNormSq, CentNormSq, Gram-patch,
  zero-accumulators, accumulate-points, divide-to-centroids, and the two seeding distance loops) were on
  the fProxyMxN struct indexer. Hoisted X/centroids/ws.* row pointers before the loops; bodies verbatim
  (pure hoist, bit-identical; suite 6317/6317). Bigger win than the "GEMM-dominated" framing suggested —
  at D=64/k=16 the per-point O(N·D) work (norms + accumulate) is a real fraction of the iteration.
  Measured (9950X3D, Uniform init): N=1024 float 1.89→0.49 ms (3.9×), N=512 1.94→0.26 ms (7.4×), double
  N=1024 2.01→0.61 ms (3.3×). Farthest-point scan + Gram[n,assignment[n]] gather left scalar (index
  capture / gather, won't SIMD).

## KMeans.fProxy.cs
- 2026-07-12 | k-means++ seeding's incremental D2Weights update (O(k·N·D)) replaced an earlier
  from-scratch recompute per new centroid, which was O(k²·N·D). (was KMeans.fProxy.cs:348-350)
