# Linear Algebra Library for Unity

## Description
A linear algebra library for Unity, fully written in Burst. It's designed to be a natural extension of Unity.Mathematics, offering a bit more of functionalities. Currently in an experimental stage and not yet ready for production use.

## Installation

To open the repo in Unity, follow these steps:

1. Clone the repo
2. Open the project in Unity

To use library in your own project:

1. Clone the repo to separate project
2. Copy Assets/LinearAlgebra/Source into your own project

## Usage

Here's a simple example:

```csharp
    // memory management struct
    var arena = new Arena(Allocator.Persistent);

    int dim = 128;
    // creates a zero vector of 128 dimensions 
    floatN vecA = arena.floatVec(dim);
    // creates a vector of 128 dimensions with all elements set to 1
    floatN vecB = arena.floatVec(dim, 1f);

    // add per component (will allocate a new temporary vec)
    floatN vecAdd = vecA + vecB;
    // mul per component (will allocate a new temporary vec)
    floatN vecMul = vecA * vecB;

    // create identity matrix
    floatMxN matI = arena.floatIdentityMatrix(16);
    floatMxN matRand = arena.floatRandomMatrix(16, 16);

    // per component sum, allocates new matrix
    floatMxN compSumMat = matI + matRand;

    // adds 1f to compSumMat inplace, allocating nothing
    floatElem_OP.addInpl(compSumMat, 1f);

    // mulls matI into compSumMat inplace, allocating nothing 
    floatElem_OP.mulInpl(compSumMat, matI);

    // creates random matrix with range from -3f to 3f
    floatMxN A = arena.floatRandomDiagonalMatrix(dim, -3f, 3f);
    floatMxN B = arena.floatRandomDiagonalMatrix(dim, -3f, 3f);

    // dot multiply A and B, will allocate new matrix
    floatMxN C = Linear_OP.dot(A, B);

    // adds 5f to element on [0, 0] coords
    C[0, 0] += 5f;

    floatN b = arena.floatVec(dim, 1f);
    floatN x_result = arena.floatVec(dim, 1f);


    // solves linear system Ax = b inplace using QR, will allocate nothing permament
    // but will modify A and b
    QR.qrDirectSolve(ref A, ref b, ref x_result);

    // calculate L1 norm
    float norm = Norms_OP.L1(x_result);

    // prints C matrix, although it will be cutoff because of big dimensions
    Print.Log(C);

    // returns true for all elements c_ij > a_ij, else false
    // will allocate
    boolMxN matCompare = C > A;

    // flips booleans, will allocate
    matCompare = !matCompare;

    // creates 3 new allocations
    boolMxN matCompare2 = C > A | C < B;

    // clears all temporary allocations
    arena.ClearTemp();

    // creates new int vector with dimensions of 10 and valued at 32
    intN intVec = arena.intVec(10, 32);

    // applies bitwise OR to elements, allocates new vector
    intVec |= 64;

    // also allocates, inplace methods do exist though
    intVec = 2 + (intVec << 2) + intVec;

    // creates new integer matrix
    intMxN intMat = arena.intRandomMatrix(10, 10, 0, 10);

    // creates new double matrix
    doubleMxN doubleMat = arena.doubleRandomMatrix(10, 10, 0, 10);

    // creates new short matrix
    shortMxN shortMat = arena.shortRandomMatrix(10, 10, 0, 10);

    // creates new long matrix
    longMxN longMat = arena.longRandomMatrix(10, 10, 0, 10);

    // mean of a vec
    double mean = doubleStats_OP.mean(in doubleMat);

    // mean of a vec
    double max = doubleStats_OP.max(in doubleMat);

    // vector of means of each row
    doubleN rowMean = doubleStats_OP.rowMean(in doubleMat);

    // clears and dispose all allocated vectors/matrices, disposes also arena
    arena.Dispose();
```

## Features

- **Types** — float, double, int, short, long, bool vectors & matrices
- **Core ops** — dot, matrix multiply, transpose, outer product, element-wise, select, comparisons
- **Decompositions** — LU, Cholesky, QR (all with pivoted variants), SVD
- **Solvers** — direct, least-squares (over-determined), min-norm (under-determined), iterative (CG/PCG, MINRES, BiCGSTAB, CGLS/LSQR)
- **Eigen** — dominant eigenpair (power iteration), full symmetric (Jacobi), non-symmetric eigenvalues (Francis QR)
- **Sparse** — block-sparse (BSR) matrices with rectangular blocks + a COO builder; block-Jacobi preconditioner; the iterative solvers run matrix-free on dense *or* sparse operands through a shared linear-operator interface
- **Numerical LA** — norms, condition number, determinant, trace, rank
- **Statistics** — mean, var/std, median, min/max, argmin/max, row/col reductions, covariance, correlation
- **Random** — distribution samplers + structured/multivariate matrix generators (Gaussian, orthogonal, SPD, …)
- **Histogram** — binning, density, CDF, 2D heatmaps — feeds the weighted sampler
- **Transforms** — normalize (L1/L2/L∞), standardize, rescale, center, softmax, clamp
- **Generators** — linspace/arange, curve / easing / LFO functors, convolution kernels, DSP windows
- **FFT** — power-of-two FFT/IFFT & real-input rfft/irfft (radix-4 workspace) + direct DFT for any N · [docs](docs/fft.md)
- **Find / query** — arg-min/max, nearest / k-nearest by metric, within-radius, find-value
- **Resampling** — sample vectors/matrices as continuous functions, resize 1D/2D (nearest / linear / Catmull-Rom)
- **Optimizers** — 1D root-finding, 1D minimization, gradient descent
- **Zero-allocation** — preallocated-output / reusable-workspace variants of ops & solvers for hot loops

## WIP / TODO

**Polish & cleanup**
- Better arena management + standalone (non-arena) vec/mat lifetime
- Name unification & simplification
- More safety checks, fuller docs

**Not yet built / under design**
- **Realtime** — a rolling window (ring buffer + zero-alloc moving average/covariance) exists, but the broader design is unsettled: frame-amortized solvers, resumable iterative state (CG/PCG stepping), online covariance / PCA, Kalman
- **PCA convenience** — covariance → symmetric eigen → sorted components + explained variance
- **Sparse eigensolvers** — sparse power iteration / LOBPCG (Fiedler vector, low vibration modes, λ_min); symmetric upper-block BSR storage

Vector/matrix views (slicing) were evaluated and intentionally dropped: a non-owning
view can't feed the contiguous Burst kernels directly, so callers materialize anyway —
the query ops cover the real need.

