# Sparse — Block Sparse Matrix (BSR)

`LinearAlgebra.Sparse`. A uniform grid of `BlockRows × BlockCols` dense blocks (each `BR × BC`),
stored block-CSR. Design rationale and full API: [spec-sparse-bsm.md](../spec-sparse-bsm.md).
Vectors stay dense — no sparse vector type.

## Storage & assembly

- **`floatBSR`** — `RowPtr`/`ColInd`/`Values` (flat, row-major per block). `Symmetric = true` stores
  only the upper block-triangle (requires square blocks and grid) — halves memory. `ToDense(ref
  arena)` materializes it (mirroring implicit lower blocks when symmetric).
- **`floatBSRBuilder`** — COO-of-blocks assembly: `AddBlock(br, bc, block)` / `AddValue(row, col, v)`,
  then `ToBSR(ref arena)` / `ToBSRSymmetric(ref arena)`. **Duplicate triplets at the same
  (blockRow, blockCol) are summed on compression** — the standard sparse-assembly contract, so you
  can add contributions from multiple sources without pre-merging them yourself.
- **`floatBSROperator`** — implements `IfloatLinearOperator` (`Apply`/`ApplyT`), so it's a drop-in
  `TOp` for every solver/eigensolver in the library. The two-argument constructor
  `floatBSROperator(in A, in AT)` takes a precomputed transpose so `ApplyT` runs as a cache-friendly
  forward `spMV` over `AT` instead of an on-the-fly scatter; `arena.floatBSRTranspose(in A)` builds it
  (a no-op returning `A` itself when `A.Symmetric`).
- **`floatBlockJacobi`** — block-diagonal preconditioner (`BR == 1` degenerates to point-Jacobi),
  built via a per-block LU factorization.

## Matvec kernels

`BSR.spMV`/`spMVT(in A, in x, ref y)` dispatch to compile-time-unrolled kernels for square block
sizes `BR ∈ {1,2,3,4,6}` (the block interior is fully unrolled, no runtime trip count), falling back
to a general kernel for other sizes or rectangular blocks.

## Solvers & eigensolvers

Every solver in [solvers.md](solvers.md)/[least-squares.md](least-squares.md)/[eigen.md](eigen.md) is
generic over `IfloatLinearOperator` — the dense and BSR overloads share one body (`floatDenseOperator`
and `floatBSROperator` are both just thin `TOp` wrappers), not a forked sparse implementation.

## Benchmarks

All measured on a 9950X3D, single CCD pinned for repeatability (commit `c3df68b`).

**Block-size-unrolled spMV vs. the general (runtime-block-size) kernel** — same result, commit `6481455`:

| Case | General kernel | Unrolled (b=3) | Speedup | vs. dense GEMV |
|---|---|---|---|---|
| float, 768², 7% fill | 0.96ms | 0.34ms | 2.8× | 38× (was 14× pre-unroll) |
| float, 768², 33% fill | 4.46ms | 1.53ms | 2.9× | 8.5× (was 3×) |
| double, 768², 7% fill | 1.08ms | 0.34ms | 3.2× | — |

**Iterative solve, sparse (BSR) vs. dense, same system** — commit `754ff4a`, `d3f65c1`:

| Solver family | Fill | Speedup vs. dense |
|---|---|---|
| Square (CG/MINRES) | 7% | ~11–13× (e.g. double CG N=768: 8.59ms → 0.66ms) |
| Square (CG/MINRES) | 33% | ~2.7–3× |
| Rectangular (CGLS/LSQR) | 7% | ~7–8× (undershoots the ideal ~14×; `ApplyT`'s transpose scatter is the gap — materializing `AT` per above measured perf-neutral on this benchmark, commit `724ceb0`) |

**CG at a genuine N=1024, block size b=4 (7% fill)** — the b=3 sweep above tops out at N=768 since
1024 isn't divisible by 3; b=4 is another compile-time-unrolled kernel size (`bsrMatVecB4`), giving a
real 1024×1024 dense-vs-sparse CG case (`Benchmarks/SparseSolverBenchmark.cs`, Section 1x). AMD Ryzen
9 9950X3D, single CCD pinned, 2026-07-06, commit `f938c66`, Unity Editor batchmode (checks likely on):

| dtype | CG-dense med(ms) | CG-sparse med(ms) | speedup |
|---|---|---|---|
| float | 3.68 | 0.09 | ~40× |
| double | 15.05 | 0.37 | ~40× |

**Symmetric vs. full storage spMV** — modest, ~1.05–1.22× before the block-unroll work (commit
`9c0ae85`, general kernels only); after both the symmetric and full kernels were unrolled equally,
the gap closes further to ~break-even (~1.02×, commit `6481455`). Symmetric storage is a **memory**
win (½ footprint), not a compute win — each stored upper block still does two block-multiplies.

**Block-Jacobi PCG vs. plain CG, same BSR system** (`Benchmarks/PCGBenchmark.cs` — `Solvers.pcg`
wasn't covered by any benchmark before this; a block-tridiagonal SPD system, b=3, nb=256, N=768,
K=40 fixed iterations, tol=0). AMD Ryzen 9 9950X3D, single CCD pinned, 2026-07-05, commit `0714c97`,
Unity Editor batchmode (checks likely on):

| dtype | solver | med(ms) | residual |
|---|---|---|---|
| float | CG | 0.030 | 9.25×10⁻⁸ |
| float | PCG (block-Jacobi) | 0.045 | 8.64×10⁻⁸ |
| double | CG | 0.079 | 2.15×10⁻¹⁶ |
| double | PCG (block-Jacobi) | 0.122 | 2.07×10⁻¹⁶ |

On this well-conditioned, already-diagonally-strong test system the block-Jacobi preconditioner adds
~50% per-iteration overhead (one extra `Apply` for `M`) without buying back enough iterations to pay
for itself — expected, since Jacobi preconditioning's real win shows up on ill-conditioned systems
(see `least-squares.md`'s AᵀA-Jacobi preconditioner, measured on a purpose-built ill-conditioned case)
not on a benchmark's synthetic well-conditioned one.
