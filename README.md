# Linear Algebra Library for Unity

A linear algebra library for Unity, fully written in [Burst](https://docs.unity3d.com/Packages/com.unity.burst@latest).
It's designed as a natural extension of `Unity.Mathematics` — where that stops at 4×4, this
takes over: arbitrary-size dense and block-sparse vectors and matrices, factorizations, solvers,
eigensolvers, FFT, statistics and a small ML layer, all Burst-compiled and allocation-conscious.

Getting ready for production — the feature set is complete and heavily tested, but the public API
is still being reviewed and may change before `1.0`.

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

Alternatively, clone the repo and copy `Assets/LinearAlgebra/Source` into your project. Either way the
only dependency is `com.unity.collections` (which pulls in Burst and Mathematics). Requires Unity 6000.3+.

To work on the library itself (templates + codegen), open the repo directly in Unity.

## Core types

- **`Arena`** — a struct for managing memory: allocating vectors and matrices, and disposing them.
- **Vectors & matrices** — `floatN` / `floatMxN` and their `double` / `int` / `short` / `long` /
  `bool` counterparts. Matrices are row-major.
- **Workspaces** — reusable scratch buffers reserved from the arena so hot loops allocate nothing.
- **Info / diagnostic structs** — small result structs returned by solvers and decompositions
  (status, iterations, residual norms, rank, …), Burst-printable.

## Usage

```csharp
    // memory management struct — owns all allocations below
    var arena = new Arena(Allocator.Persistent);

    int dim = 128;
    floatN vecA = arena.floatVec(dim);        // zero vector
    floatN vecB = arena.floatVec(dim, 1f);    // filled with 1

    floatN vecAdd = vecA + vecB;              // per-component (allocates a temp)
    floatN vecMul = vecA * vecB;              // per-component (allocates a temp)

    floatMxN matI   = arena.floatIdentityMatrix(16);
    floatMxN matRand = arena.floatRandomMatrix(16, 16);

    floatMxN compSumMat = matI + matRand;     // allocates
    floatElem_OP.addInpl(compSumMat, 1f);     // in place, allocates nothing
    floatElem_OP.mulInpl(compSumMat, matI);   // in place, allocates nothing

    floatMxN A = arena.floatRandomDiagonalMatrix(dim, -3f, 3f);
    floatMxN B = arena.floatRandomDiagonalMatrix(dim, -3f, 3f);
    floatMxN C = Linear_OP.dot(A, B);         // matrix multiply (allocates)
    C[0, 0] += 5f;

    floatN b = arena.floatVec(dim, 1f);
    floatN x = arena.floatVec(dim);

    // solve Ax = b in place via QR; returns a diagnostics struct
    DirectSolveInfo info = QR.qrDirectSolve(ref A, ref b, ref x);
    Print.Log(info);                          // "DirectSolveInfo(Success, ...)"

    float norm = Norms_OP.L1(x);

    boolMxN cmp = C > A;                       // element-wise compare (allocates)
    cmp = !cmp;                                // negate (allocates)

    arena.ClearTemp();                         // free the temporaries above
    arena.Dispose();                           // free everything, dispose arena
```

## Features

- **Types** — float, double, int, short, long, bool vectors & matrices
- **Core ops** — dot, matrix multiply (register-tiled GEMM), transpose, outer product, element-wise, select, comparisons
- **Decompositions** — LU, Cholesky, QR / LQ, QRCP (pivoted), SVD (thin / truncated / randomized)
- **Solvers** — direct, least-squares, min-norm, iterative (CG/PCG, MINRES, BiCGSTAB, CGLS/LSQR/LSMR); every solver returns a diagnostics struct
- **Eigen** — power iteration, symmetric (`eigenSymmetric`), non-symmetric (Francis QR), and matrix-free sparse eigensolvers
- **Sparse** — block-sparse (BSR) matrices with rectangular blocks; solvers and eigensolvers run matrix-free over dense *or* sparse operands through a shared linear-operator interface
- **ML** — PCA + k-means
- **Statistics** — mean, var/std, median, min/max, argmin/max, covariance, correlation, row/col reductions
- **[FFT](docs/fft.md)** — power-of-two FFT/IFFT, real-input rfft/irfft, arbitrary-N DFT
- **Random** — distribution samplers + structured / multivariate matrix generators (Gaussian, orthogonal, SPD, …)
- **Signal & data** — histograms, resampling (nearest / linear / Catmull-Rom), transforms (normalize / standardize / softmax / …), generators (linspace, easing/LFO, DSP windows), find/query, 1D optimizers
- **Debug** — Burst `Print.Log` / `Print.Spy` (incl. block-sparse spy) + managed CSV/text export for every type
- **Zero-allocation** — preallocated-output / reusable-workspace variants of ops & solvers for hot loops

## Status & roadmap

Version `0.1` — everything above is built and tested. The API is still being read through and
tightened ahead of `1.0`, so names and signatures may still change.

**Under design**

- **Realtime** — a rolling window (ring buffer + zero-alloc moving average/covariance) exists, but
  the broader design is unsettled: frame-amortized solvers, resumable iterative state (CG/PCG
  stepping), online covariance / PCA, Kalman.
- **Sparse smallest-eigenpair (LOBPCG)** — for structural-stability use cases (buckling, modal
  analysis, the Fiedler vector). Dense small-scale versions are already covered by `eigenSymmetric`.

Vector/matrix views (slicing) were evaluated and intentionally dropped: a non-owning view can't
feed the contiguous Burst kernels directly, so callers materialize anyway — the query ops cover the
real need.
