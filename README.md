# LinearAlgebraBursted

A linear algebra library for Unity, compiled entirely through [Burst](https://docs.unity3d.com/Packages/com.unity.burst@latest).
It extends `Unity.Mathematics` past its fixed 4×4 ceiling: arbitrary-size vectors and matrices for
`float`/`double`/`int`/`short`/`long`/`uint`/`bool`, an arena allocator, dense and block-sparse
factorizations/solvers, eigensolvers, FFT, statistics, and a small ML layer.

The public API is still being reviewed and may change before `1.0` (current version `0.1.0`).

## Installation

Install via UPM — only the generated runtime source ships, no templates or codegen tooling. In
*Window → Package Manager → + → Add package from git URL*:

```
https://github.com/viliwonka/LinearAlgebraBursted.git?path=Assets/LinearAlgebra/Source
```

or add it to `Packages/manifest.json`:

```json
"com.viliwonka.burst-linear-algebra": "https://github.com/viliwonka/LinearAlgebraBursted.git?path=Assets/LinearAlgebra/Source"
```

Or clone the repo and copy `Assets/LinearAlgebra/Source` into your project. Either way the only
dependency is `com.unity.collections` (pulls in Burst and Mathematics). Requires Unity 6000.3+. To
work on the library itself (templates + codegen), open the repo directly in Unity.

## Quick start

```csharp
// Arena owns every allocation below.
var arena = new Arena(Allocator.Persistent);

int dim = 128;
floatN vecA = arena.floatVec(dim);          // zero vector
floatN vecB = arena.floatVec(dim, 1f);      // filled with 1
floatN vecAdd = vecA + vecB;                // per-component, allocates a temp

floatMxN matI = arena.floatIdentityMat(16);
floatMxN matRand = arena.floatRandomMat(16, 16);
floatMxN sum = matI + matRand;              // allocates
floatComp.addInPlace(sum, 1f);              // in place, allocates nothing
floatComp.mulInPlace(sum, matRand);         // in place, allocates nothing

floatMxN A = arena.floatRandomDiagonalMat(dim, -3f, 3f);
floatMxN B = arena.floatRandomDiagonalMat(dim, -3f, 3f);
floatMxN C = Blas.dot(A, B);                // matrix multiply, allocates
C[0, 0] += 5f;

floatN b = arena.floatVec(dim, 1f);
floatN x = arena.floatVec(dim);
// Solve Ax = b via QR; returns a diagnostics struct.
DirectSolveInfo info = QR.qrDirectSolve(ref A, ref b, ref x);
Print.Log(info);                            // "DirectSolveInfo(Success)"
float norm = Norms.L1(x);

boolMxN cmp = C > A;                        // element-wise compare, allocates
cmp = !cmp;                                 // negate, allocates

arena.ClearTemp();                          // free the temporaries above
arena.Dispose();                            // free everything, dispose the arena
```

## Benchmarks at a glance

Measured, single-machine numbers for the solvers most people call directly. Machine and commit are
noted per row; full tables (more sizes, both precisions) live in each feature's mini-doc.

| Solver | Representative case | Measured result | Source |
|---|---|---|---|
| Direct solve, dense QR (`QR.qrDirectSolve`) | 1024×1024, float | 74.7× faster after a cache-locality fix (2684ms→35.9ms; 0.53→39.9 GFLOP/s) | commit `eadf6a8` |
| Least-squares, QRCP (`QR.qrcpDirectSolve`) | 1024×1024, float | 61.6× faster after the same fix (5889ms→95.5ms; 0.24→15.0 GFLOP/s) | commit `eadf6a8` |
| Iterative solve, CG, dense vs. sparse (`Solvers.cg`) | 768×768 SPD, 7% block fill, double | 13× faster on the sparse (BSR) operand (8.59ms → 0.66ms) | commit `754ff4a` |
| SVD, truncated GKL (`SVD.svdTruncated`) | 2048×256, k≈21%, float | 6.2–6.9× faster after vectorizing the Lanczos basis | commit `4203c72` |
| Eigensolve, symmetric (`Eigen.eigenSymmetric`, values only) | 256×256, float | ~75× faster than cyclic Jacobi | commit `4902032` |
| FFT, real input (`FFT.rfft` vs. full `fft`) | N = 1,048,576, float | 1.5× faster (24.0ms → 15.9ms) | commit `dc3bd3f` |
| Eigensolve, k smallest (`LOBPCG.lobpcg`) | dense SPD 512×512, k=4, float | 84.9ms median for a fixed 50-iteration budget | 2026-07-05, commit `0714c97` |
| Iterative solve, block-Jacobi PCG vs. plain CG (`Solvers.pcg`) | 768×768 SPD BSR, b=3, 7% fill, float | 0.030ms (CG) vs. 0.045ms (PCG) — preconditioning overhead exceeds its payoff on this well-conditioned case | 2026-07-05, commit `0714c97` |

See [docs/features](docs/features) — each linked doc below carries the deeper benchmark table.

## Features

- **Dense types** — [vectors, matrices, the `Arena` allocator](docs/features/dense-types.md)
- **Element-wise ops** — [`Comp`: arithmetic, math functions, clamp, in-place](docs/features/comp-elementwise.md)
- **Core linear algebra** — [`Blas`/`Norms`/`Analysis`: dot, GEMM, transpose, outer product, norms, matrix metrics](docs/features/blas.md)
- **Decompositions** — [LU, Cholesky (+ pivoted), QR/LQ (+ QRCP), Bidiag](docs/features/decompositions.md)
- **Direct solvers** — [triangular/LU/Cholesky/QR solve, the diagnostics-struct convention](docs/features/solvers.md)
- **Least squares** — [QR/QRCP/SVD routes, CGLS/LSQR/LSMR, Tikhonov damping, Jacobi preconditioning](docs/features/least-squares.md)
- **SVD** — [thin/values/truncated-GKL/randomized, pseudo-inverse, low-rank approximation](docs/features/svd.md)
- **Eigensolvers** — [symmetric Jacobi & Householder, non-symmetric QR, matrix-free power/inverse/Lanczos/LOBPCG](docs/features/eigen.md)
- **Sparse (BSR)** — [block-CSR storage, builder assembly, unrolled spMV, sparse solvers/eigensolvers](docs/features/sparse-bsr.md)
- **FFT** — [power-of-two FFT/IFFT, radix-4, real-input rfft/irfft, arbitrary-N DFT](docs/features/fft.md)
- **Statistics** — [whole-array & row/col reductions, covariance/correlation, transforms](docs/features/stats.md)
- **Random** — [distribution samplers, weighted pick/shuffle, multivariate normal, structured matrices](docs/features/random.md)
- **Query** — [nearest/k-nearest/radius search, argmax/argmin, predicate-filtered variants](docs/features/query.md)
- **Select & bit ops** — [element-wise select, integer bit intrinsics, bool logic](docs/features/select-bits.md)
- **Hash** — [xxHash32 over vectors/matrices, lockstep desync-checksum use case](docs/features/hash.md)
- **Realtime** — [`RollingWindow`: ring-buffer moving average/covariance](docs/features/realtime.md)
- **ML** — [k-means, PCA (4 fit routes)](docs/features/ml.md)
- **Generators** — [linspace, easing curves, LFO/wave, DSP windows, kernels](docs/features/generators.md)
- **Print & export** — [Burst `Print.Log`/`Print.Spy`, managed CSV/text export](docs/features/print-export.md)

## Determinism

The core algorithms are single-threaded with a fixed reduction order (the only reassociation used is
a documented, rounding-only multi-accumulator dot). Compiled under Burst's `FloatMode.Strict` (which
disables FP reassociation), results are reproducible run to run and across CPU architectures for a
fixed Burst version — what a deterministic lockstep multiplayer sim needs — **provided the path uses
only correctly-rounded operations**: the core factorizations and solvers do, but FFT and random
sampling both use transcendental functions (`sin`/`cos`/`exp`/...) and do not carry this guarantee.
This isn't a project-wide default either way: the library's own benchmarks compile under
`FloatMode.Default`, so a caller who needs determinism must compile their own jobs under
`FloatMode.Strict`.

## License

[MIT](LICENSE).
