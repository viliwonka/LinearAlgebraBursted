# Sparse - Block Sparse Matrix (BSR)

`BULA.Sparse`. A uniform grid of `BlockRows × BlockCols` dense blocks (each `BR × BC`),
stored block-CSR. Vectors stay dense - no sparse vector type.

## Storage & assembly

- **`floatBSR`** - `RowPtr`/`ColInd`/`Values` (flat, row-major per block). `Symmetric = true` stores
  only the lower block-triangle (requires square blocks and grid) - halves memory. `ToDense(Allocator
  allocator = Allocator.Temp)` materializes it (mirroring implicit upper blocks when symmetric).
- **`floatBSRBuilder`** - COO-of-blocks assembly: `AddBlock(br, bc, block)` / `AddValue(row, col, v)`,
  then `ToBSR(allocator)` / `ToBSRSymmetric(allocator)`. **Duplicate triplets at the same
  (blockRow, blockCol) are summed on compression** - the standard sparse-assembly contract, so you
  can add contributions from multiple sources without pre-merging them yourself.
- **`floatBSROperator`** - implements `IfloatLinearOperator` (`Apply`/`ApplyT`), so it's a drop-in
  `TOp` for every solver/eigensolver in the library. The two-argument constructor
  `floatBSROperator(in A, in AT)` takes a precomputed transpose so `ApplyT` runs as a cache-friendly
  forward `spMV` over `AT` instead of an on-the-fly scatter; `A.Transpose(Allocator.Temp)` builds it
  (a no-op returning `A` itself when `A.Symmetric`).
- **`floatBlockJacobi`** - block-diagonal preconditioner (`BR == 1` degenerates to point-Jacobi),
  built via a per-block LU factorization.

## Matvec kernels

`BSR.spMV`/`spMVT(in A, in x, ref y)` dispatch to compile-time-unrolled kernels for square block
sizes `BR ∈ {1,2,3,4,6}` (the block interior is fully unrolled, no runtime trip count), falling back
to a general kernel for other sizes or rectangular blocks.

## Solvers & eigensolvers

Every solver in [solvers.md](solvers.md)/[least-squares.md](least-squares.md)/[eigen.md](eigen.md) is
generic over `IfloatLinearOperator` - the dense and BSR overloads share one body (`floatDenseOperator`
and `floatBSROperator` are both just thin `TOp` wrappers), not a forked sparse implementation.
