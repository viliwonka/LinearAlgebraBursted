# LinearAlgebraBursted

A linear algebra library for Unity, supported entirely by [Burst](https://docs.unity3d.com/Packages/com.unity.burst@latest).
It extends `Unity.Mathematics` past its fixed 4×4 ceiling: arbitrary-size vectors and matrices for
`float`/`double`/`int`/`bool` and others. 

It's meant to be deterministic, simple, tested, optimized.

It has things you would expect from linear algebra / math library:
- Dense / Sparse matrices,
- Solvers / eigen,
- Decompositions,
- Statistics,
- FFT

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
// Solve Ax = b via QR; fastest path, but DESTROYS A and b (both become scratch).
DirectSolveInfo info = QR.solveInPlace(ref A, ref b, ref x);
// To keep A and b intact instead: QR.decomp(in A, ref Q, ref R) then QR.decompSolve(ref Q, ref R, ref b, ref x).
Print.Log(info);                            // "DirectSolveInfo(Success)"
float norm = Norms.L1(x);

boolMxN cmp = C > A;                        // element-wise compare, allocates
cmp = !cmp;                                 // negate, allocates

arena.ClearTemp();                          // free the temporaries above
arena.Dispose();                            // free everything, dispose the arena
```

## Benchmarks

Benchmarked on a Ryzen 9 9950X3D (pinned to a non-V-Cache core), single-threaded Burst, median of
9 runs. Full tables — more sizes, double precision — live in each feature's doc under
[docs/features](docs/features).

| Algorithm | Case | Results |
|---|---|---|
| `LU.decomp` + `LU.decompSolve` LU | 1024×1024, float | 12.2 ms |
| `CHO.decomp` + `CHO.decompSolve` Cholesky | 1024×1024, float | 7.5 ms |
| `QR.solveInPlace` QR | 1024×1024, float | 34.9 ms |
| `QR.solveInPlace` QR least squares | 2048×512, float | 29.8 ms |
| `QRCP.solveInPlace` pivoted QR, least squares | 2048×512, float | 32.5 ms |
| `LQ.minNormSolve` LQ, underdetermined system | 512×1024, float | 8.9 ms |
| `Krylov.cg` CG, iterative SPD solver, dense | 1024×1024, double; dense storage of the same 7%-fill SPD matrix, 40 iterations | 2.4 ms |
| `Krylov.cg` CG, iterative SPD solver, sparse | 1024×1024, double; sparse BSR (4×4 blocks, 7% fill), 40 iterations | 0.36 ms |
| `Eigen.symmetricInPlace` Symmetric eigen decomp  | 1024×1024, float, values + vectors | 168.8 ms |
| `Eigen.valuesSymmetricInPlace` Symmetric eigen, values only | 1024×1024, float | 63.3 ms |
| `Eigen.lobpcg` Smallest eigenpairs, SPD solver | 512×512, k=4, float, 50 iterations | 33.6 ms (0.67 ms/iter) |
| `SVD.thin` full SVD | 2048×512, float | 186.8 ms |
| `SVD.truncated` truncated SVD w/ top-k only | 2048×512, k=21, float | 17.6 ms |
| `SVD.randomized` randomized SVD w/ top-k only  | 2048×512, k=21, float | 29.5 ms |
| `FFT.fft` FFT  | N = 1,048,576, float | 24.4 ms |
|`FFT.rfft` Real FFT | N = 1,048,576, float | 17.9 ms |

## Features

- [**Types**](docs/features/dense-types.md): vectors, matrices, the `Arena` allocator
- [**Element-wise ops**](docs/features/comp-elementwise.md): Per component arithmetic, math functions, clamp
- [**LA primitives**](docs/features/la-primitives.md): `Blas`/`Norms`/`Analysis`: dot, GEMM, transpose, outer product, norms, matrix metrics
- [**Decompositions**](docs/features/decompositions.md): LU, CHO/CHOP, QR/QRCP,LQ
- [**Direct solvers**](docs/features/solvers.md): tri/LU/CHO/QR solve
- [**Least squares**](docs/features/least-squares.md): QR/QRCP/SVD routes, CGLS/LSQR/LSMR, Tikhonov damping, Jacobi preconditioning
- [**SVD**](docs/features/svd.md): thin/values/truncated-GKL/randomized, pseudo-inverse, low-rank approximation
- [**Eigensolvers**](docs/features/eigen.md): symmetric Jacobi & Householder, non-symmetric QR, matrix-free power/inverse/Lanczos/LOBPCG
- [**Sparse (BSR)**](docs/features/sparse-bsr.md): block-CSR storage, builder assembly, sparse solvers/eigensolvers
- [**LP / LAD**](docs/features/lp-lad.md): linear programming (simplex/revised/dual/interior-point, sparse), exact L1/quantile regression, warm-started re-solve
- [**QP / MIP**](docs/features/qp-mip.md): convex quadratic programs (active-set), mixed-integer programs (branch & bound)
- [**Control**](docs/features/control.md): discrete-time LQR (cold/warm/finite-horizon gain schedule)
- [**FFT**](docs/features/fft.md): real valued rfft/irfft, complex fft/ifft, dft
- [**Statistics**](docs/features/stats.md): vector/row/col reductions, covariance/correlation, transforms
- [**Random**](docs/features/random.md): distribution samplers, weighted pick/shuffle, multivariate normal, structured matrices
- [**Realtime**](docs/features/realtime.md): fixed-capacity rolling window, moving mean/covariance
- [**Query**](docs/features/query.md): nearest/k-nearest/radius search, argmax/argmin, predicate-filtered variants
- [**Select & bit ops**](docs/features/select-bits.md): element-wise select, integer bit intrinsics, bool logic
- [**Hash**](docs/features/hash.md): vector/matrix, col/row reduction
- [**ML**](docs/features/ml.md): k-means, PCA
- [**Generators**](docs/features/generators.md): linspace, easing curves, LFO/wave, DSP windows, kernels
- [**Print & export**](docs/features/print-export.md): `Print.Log`/`Print.Spy`, managed CSV/text export

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
