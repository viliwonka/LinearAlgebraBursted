# Sparse — Block Sparse Matrix (BSR)

`BULA.Sparse`. A uniform grid of `BlockRows × BlockCols` dense blocks (each `BR × BC`),
stored block-CSR. Vectors stay dense — no sparse vector type.

## Storage & assembly

- **`floatBSR`** — `RowPtr`/`ColInd`/`Values` (flat, row-major per block). `Symmetric = true` stores
  only the lower block-triangle (requires square blocks and grid) — halves memory. `ToDense(Allocator
  allocator = Allocator.Temp)` materializes it (mirroring implicit upper blocks when symmetric).
- **`floatBSRBuilder`** — COO-of-blocks assembly: `AddBlock(br, bc, block)` / `AddValue(row, col, v)`,
  then `ToBSR(allocator)` / `ToBSRSymmetric(allocator)`. **Duplicate triplets at the same
  (blockRow, blockCol) are summed on compression** — the standard sparse-assembly contract, so you
  can add contributions from multiple sources without pre-merging them yourself.
- **`floatBSROperator`** — implements `IfloatLinearOperator` (`Apply`/`ApplyT`), so it's a drop-in
  `TOp` for every solver/eigensolver in the library. The two-argument constructor
  `floatBSROperator(in A, in AT)` takes a precomputed transpose so `ApplyT` runs as a cache-friendly
  forward `spMV` over `AT` instead of an on-the-fly scatter; `A.Transpose(Allocator.Temp)` builds it
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

## Performance

Ryzen 9 9950X3D, single-thread Burst, median of 9.

**Block-size-unrolled spMV** (compile-time block sizes b∈{1,2,3,4,6}), 768², b=3:

| Case | spMV (ms) | vs. dense GEMV |
|---|---|---|
| float, 7% fill | 0.34 | 38× |
| float, 33% fill | 1.53 | 8.5× |
| double, 7% fill | 0.34 | — |

**Iterative solve, sparse (BSR) vs. dense, same system:**

| Solver family | Fill | Speedup vs. dense |
|---|---|---|
| Square (CG/MINRES) | 7% | ~11–13× |
| Square (CG/MINRES) | 33% | ~2.7–3× |
| Rectangular (CGLS/LSQR) | 7% | ~7–8× (below the ideal ~14× — `ApplyT`'s transpose scatter is the gap) |

**CG at N=1024, block size b=4 (7% fill)**, dense vs. sparse (`Benchmarks/SparseSolverBenchmark.cs`):

| dtype | CG-dense med(ms) | CG-sparse med(ms) | speedup |
|---|---|---|---|
| float | 3.66 | 0.09 | ~40× |
| double | 15.02 | 0.37 | ~40× |

Symmetric storage (lower blocks only) is a **memory** win — ½ the footprint at ~break-even spMV
throughput, since each stored lower block still does two block-multiplies.

**Block-Jacobi PCG vs. plain CG, same BSR system** (`Benchmarks/PCGBenchmark.cs`; block-tridiagonal
SPD system, b=3, nb=256, N=768, K=40 fixed iterations, tol=0):

| dtype | solver | med(ms) | residual |
|---|---|---|---|
| float | CG | 0.031 | 9.25×10⁻⁸ |
| float | PCG (block-Jacobi) | 0.045 | 8.64×10⁻⁸ |
| double | CG | 0.080 | 2.15×10⁻¹⁶ |
| double | PCG (block-Jacobi) | 0.122 | 2.07×10⁻¹⁶ |

On this well-conditioned system, block-Jacobi adds ~50% per-iteration overhead (one extra `Apply` for
`M`) without buying back enough iterations to justify the cost — preconditioning's real payoff shows
up on ill-conditioned systems (see [least-squares.md](least-squares.md)'s AᵀA-Jacobi example), not here.
