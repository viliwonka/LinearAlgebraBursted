# DEVLOG — ML
Code comments state contracts only; history lives here (see CLAUDE.md).

## KMeans.fProxy.cs
- 2026-07-12 | k-means++ seeding's incremental D2Weights update (O(k·N·D)) replaced an earlier
  from-scratch recompute per new centroid, which was O(k²·N·D). (was KMeans.fProxy.cs:348-350)
