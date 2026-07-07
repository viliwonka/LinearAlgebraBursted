# Blas, Norms & Analysis — core LA primitives

Three merged (bare-name) classes covering the primitives everything else is built from.

## Blas — dot, transpose, outer product, GEMM

Every op has a zero-alloc `ref`-destination primitive (the `ref` result must be distinct from the
inputs) and an allocating wrapper of the same name:

- `dot(floatN a, floatN b)` / `dot(a, b, start, end)` - vector dot,
- `dot(in floatMxN A, in floatN x, ref floatN result)` mat vec dot,
- `dot(in floatN y, in floatMxN A, ref floatN result)` - vec mat dot,
- `dot(in floatMxN a, in floatMxN b, ref floatMxN c, bool transposeA = false)` mat mat dot (GEMM), dispatching to `matMatDot` or `matMatDotTransA`,
- `outerDot(in floatN a, in floatN b, ref floatMxN result)` - outer product,
- `trans(in floatMxN A, ref floatMxN T)` - transpose,
- `householderInPlace(ref floatMxN matrix, in floatN u)` - apply a Householder reflection directly,


**GEMM tiling:** for performance, `matMatDot` & `matMatDotTransA` hold an 8×16 block of the output in named scalar
accumulators across the whole k-reduction (each `A` value reused 16×, each `B` value 8×).

## Norms

Basic norms `L1`, `L2`, `LInf` are supported as calculation or inplace normalization.
Helpers such as per-axis (rows or columns) normalization.

Special matrix norms exist too.

## Analysis

Mostly matrix but also vector functions: 
- `isAnyNan(x)`/`isAnyInf(x)`, 
- `isZero(x, ε)`, 
- `isIdentity`/`isSymmetric`/`isDiagonal`/`isUpperTriangular`/`isLowerTriangular`/`isOrthogonal(A, ε)`, 
- `any(x)`/`all(x)`,
- `trace(A)`,
- `cond(A)`,
- `rank(A, ε)`, 
- `determinant(A)`,
- `logDeterminant(A, out sign)` — `log|det|` + sign (slogdet); robust where `determinant` over/underflows,

## Performance

`matMatDot`, basic matrix matrix operation, benched on Ryzen 9 9950X3D.

| Size | float (GFLOP/s) | double (GFLOP/s) |
|---|---|---|
| 512² | 93 | 50 |
| 1024² | 86 | 50 |

