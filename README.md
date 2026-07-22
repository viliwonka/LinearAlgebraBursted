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

## Install

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

## Features

- [**Types**](docs/features/dense-types.md): vectors, matrices, the `Arena` allocator
- [**Element-wise ops**](docs/features/comp-elementwise.md): Per component arithmetic, math functions, clamp, integer bit ops, bool logic
- [**LA primitives**](docs/features/la-primitives.md): `Blas`/`Norms`/`Analysis`: dot, GEMM, transpose, outer product, norms, matrix metrics
- [**Decompositions & direct solvers**](docs/features/decompositions.md): LU, CHO/CHOP, QR/QRCP, LQ/LQRP, Bidiag; direct solve (tri/LU/CHO/QR)
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
- [**Select**](docs/features/select-bits.md): element-wise select
- [**Hash**](docs/features/hash.md): vector/matrix, col/row reduction
- [**ML**](docs/features/ml.md): k-means, PCA
- [**Generators**](docs/features/generators.md): linspace, easing curves, LFO/wave, DSP windows, kernels
- [**Print & export**](docs/features/print-export.md): `Print.Log`/`Print.Spy`, managed CSV/text export


## Example code

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
// Solve Ax = b via QR; fastest path, but modifies A and b (both become scratch).
DirectSolveInfo info = QR.solveInPlace(ref A, ref b, ref x);
Print.Log(info);                            // "DirectSolveInfo(Success)"
float norm = Norms.L1(x);

boolMxN cmp = C > A;                        // element-wise compare, allocates
cmp = !cmp;                                 // negate, allocates

arena.ClearTemp();                          // free the temporaries above
arena.Dispose();                            // free everything, dispose the arena
```


## Determinism

The idea of this library is that besides being fast and accurate it's also deterministic run to run; and across different CPU architectures. (To be confirmed).

All code in project is intended to be compiled under Burst's `FloatMode.Strict`, (which disables FP reassociation), results should be reproducible run to run and across different CPU architectures, **provided the code uses only basic operations** (`+ - * /` and `sqrt`). The core factorizations and solvers qualify, and so does the FFT and others.

Transcendental functions (`log`/`exp`/`sin`/...) are reimplemented to cover the determinism gap. They are comparable in accuracy and speed to `Unity.Mathematics` functions.

These deterministic transcendentals are used by default. To route them back to `Unity.Mathematics`' native `math.*` instead, faster to Burst-compile, but no longer cross-architecture deterministic - add `LINALG_NATIVE_MATH` to **Project Settings → Player → Scripting Define Symbols**.

If you do not care about determinism, you still can just compile your job with `FloatMode.Fast`. 


## Benchmarks

Benchmarked on a Ryzen 9 9950X3D (pinned to a non-V-Cache core), single-threaded, median scores.

| Dense solvers & decomposition | Case | Results ||
|---|---|---|---|
| `LU.solveInPlace` LU | 1024×1024, float | 12.0 ms |
| `CHO.solveInPlace` Cholesky | 1024×1024, float | 7.7 ms |
| `CHOP.solveInPlace` pivoted Cholesky | 1024×1024, float | 14.5 ms |
| `QR.solveInPlace` QR, square | 1024×1024, float | 34.6 ms |
| `QR.solveInPlace` QR, overdetermined | 2048×512, float | 29.9 ms |
| `QRCP.solveInPlace` pivoted QR, overdetermined | 2048×512, float | 32.3 ms |
| `LQ.minNormSolveInPlace` underdetermined min-norm, full row rank | 512×2048, float | 20.8 ms |
| `LQRP.minNormSolveInPlace` underdetermined min-norm, rank-revealing (COD) | 512×2048, float | 29.4 ms |
|||||
| **SVD** | **Case** | **Result** ||
| `SVD.thin` full SVD | 2048×512, float | 100.9 ms |
| `SVD.truncated` truncated SVD w/ top-k only | 2048×512, k=21, float | 2.8 ms |
| `SVD.randomized` randomized SVD w/ top-k only  | 2048×512, k=21, float | 29.5 ms |
|||||
| **Fourier Transform** | **Case** | **Result** ||
| `arena.floatFFTCache(n)` one-time twiddle-workspace build | N = 2^20, float | 1.0 ms |
| `FFT.fft` complex forward,  | N = 2^20, float | 6.3 ms |
| `FFT.ifft` complex inverse | N = 2^20, float | 6.4 ms |
| `FFT.rfft` real forward | N = 2^20, float | 3.4 ms |
| `FFT.irfft` real inverse | N = 2^20, float | 3.4 ms |
| `FFT.rfft` real forward | N = 2^14, float | 0.039 ms |
|||||
| **Eigen solvers** | **Case** | **Result** ||
| `Eigen.symmetricInPlace` Symmetric eigen decomp  | 1024×1024, float, values + vectors | 163.7 ms |
| `Eigen.valuesSymmetricInPlace` Symmetric eigen, values only | 1024×1024, float | 60.3 ms |
| `Eigen.lobpcg` smallest-k eigenpairs, dense SPD | 512×512, k=4, float, 50 iterations | 33.1 ms (0.66 ms/iter) |
| `Eigen.lobpcg` smallest-k eigenpairs, dense SPD | 1024×1024, k=4, float, 50 iterations | 74.7 ms (1.49 ms/iter) |
| `Eigen.lobpcg` smallest-k, sparse BSR, IC(0)-preconditioned, converged | 1024×1024, k=4, float | 44.8 ms (38 iters) |
| `Eigen.lobpcg` smallest-k, sparse 2D-grid Laplacian (96×96 grid), SSOR, converged | 9216×9216, k=8, float | 2.51 s (38 iters) |
|||||
| **Krylov solver** | **Iterations** | **Result** ||
| Sparse iterative solvers, N = 10240 BSR, 1.5% fill, float | | | 
| `Krylov.cg` SPD | 40 (fixed budget) | 14.1 ms |
| `Krylov.minres` symmetric-indefinite | 30 (converged) | 10.7 ms |
| `Krylov.biCGStab` nonsymmetric | 11 (converged) | 4.0 ms |
| Sparse least squares, D = 20480×10240. 1.5% fill, float |  |
| `Krylov.lsqr` / `Krylov.lsmr`  | 25 (converged) | 12.4 / 12.5 ms |
| `Krylov.cgls` | 40 (fixed budget) | 19.5 ms |
|||||
| **Preconditioner** | **N = 1024** | **N = 10201** ||
| Example: square 2D Laplacian, solve to tolerance = √eps, double.
| `Krylov.cg` no preconditioner | 101 iters, 2.8 ms | 305 iters, 310 ms |
| `Krylov.pcg` SSOR | 30 iters, 2.1 ms | 83 iters, 227 ms |
| `Krylov.pcg` IC(0) | 1 iter, 0.13 ms | 1 iter, 5.9 ms |
|||||
| **Programming** | **Case** | **Result** ||
| `LP.solve` revised / dual simplex, cold solve | 192×96 dense, float | 0.62 ms |
| `LP.solve` warm re-solve (LPBasis reuse), 16 RHS-perturbed re-solves | 192×96 dense, float | cold 22.8 → warm 2.2 ms |
| `QP.solve` active-set, cold (facade: phase-1 start + solve) | n = 192, m = 96, float | 72.0 ms |
| `QP.solve` active-set, warm (feasible start, incremental reduced space) | n = 192, m = 96, float | 34.9 ms |
| `MIP.solve` branch & bound over warm-started dual simplex | p0033 (MIPLIB), double | 74.8 ms, 404 nodes |
| `MIP.solve` branch & bound over warm-started dual simplex | stein15 (MIPLIB), double | 55.4 ms, 263 nodes |
|||||
| **Control** | **Case** | **Per step** | **120-steps sum** |
| `LQR.lqr` Riccati gain solve — once (LTI) or re-solved per frame (adaptive/time-varying) | n = 12, m = 4, float | cold 26 µs → warm 6 µs | ≈ 0.03 ms (gain solved once) |
| `MPC.solve` receding-horizon step, box bounds | N = 40, n = 12, m = 4, float | cold 0.39 → warm 0.23 ms | ≈ 28 ms |
| `MPC.solve` receding-horizon step, box + soft wall | N = 40, n = 12, m = 4, float | cold 1.14 → warm 0.27 ms | ≈ 32 ms |
| `Kalman.ekfPredict` + `ekfUpdate` per step | n = 12, m = 6, float | 4.5 µs | ≈ 0.54 ms |
| `Kalman.ukfPredict` + `ukfUpdate` per step | n = 12, m = 6, float | 14 µs | ≈ 1.7 ms |
|||||
| Regression fitting — L2 (least squares) vs exact L1 (LAD) vs approximate L1 (IRLS), 2048 observations.
| **Fitting** | **Case** | **Result** ||
| `QR.solveInPlace` — L2 least squares | 2048×4, float | 0.12 ms |
| `LP.lad` — exact L1 (LAD) | 2048×4, float | 0.97 ms |
| `Optimize.ladIRLS` — approximate L1 | 2048×4, float | 0.063 ms |
| `QR.solveInPlace` — L2 least squares | 2048×64, float | 1.72 ms |
| `LP.lad` — exact L1 (LAD) | 2048×64, float | 6.78 ms |
| `Optimize.ladIRLS` — approximate L1 | 2048×64, float | 7.09 ms |

## License

[MIT](LICENSE). (except LAD code, pending permission)
