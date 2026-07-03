using System;
using Unity.Mathematics;
using LinearAlgebra;

namespace LinearAlgebra.Gallery
{
    /// <summary>
    /// Batch A — SPD / symmetric family. Arena extension methods that construct famous test matrices
    /// with known closed-form properties (eigenvalues, determinant, condition number, inverse,
    /// definiteness). Opt in with <c>using LinearAlgebra.Gallery;</c>.
    /// </summary>
    public static partial class fProxyGallery
    {
        // ────────────────────────────────────────────────────────────────────────────────
        // 1. Hilbert
        // ────────────────────────────────────────────────────────────────────────────────

        /// <summary>
        /// n×n Hilbert matrix: H[i,j] = 1/(i+j+1) (0-based indices).
        /// SPD, totally positive, severely ill-conditioned (cond(H₃) ≈ 524.06).
        /// Classic stress-test for Cholesky and linear solvers.
        /// </summary>
        public static fProxyMxN fProxyHilbert(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("fProxyHilbert: n must be >= 1");

            var A = arena.fProxyMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (fProxy)1 / (fProxy)(i + j + 1);
            return A;
        }

        // ────────────────────────────────────────────────────────────────────────────────
        // 2. Pascal
        // ────────────────────────────────────────────────────────────────────────────────

        /// <summary>
        /// n×n symmetric Pascal matrix: P[i,j] = C(i+j, i) (0-based indices).
        /// Built via the recurrence P[i,0]=P[0,j]=1, P[i,j]=P[i-1,j]+P[i,j-1] to stay
        /// integer-exact and overflow-safe for moderate n.
        /// Symmetric, det=1, SPD, integer-valued; eigenvalues come in reciprocal pairs.
        /// </summary>
        public static fProxyMxN fProxyPascal(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("fProxyPascal: n must be >= 1");

            var A = arena.fProxyMat(n, true);

            // Seed the first row and column to 1.
            for (int k = 0; k < n; k++)
            {
                A[k, 0] = (fProxy)1;
                A[0, k] = (fProxy)1;
            }

            // Fill the interior by the Pascal recurrence (row-major order is safe: both
            // A[i-1,j] and A[i,j-1] are already written before A[i,j] is needed).
            for (int i = 1; i < n; i++)
                for (int j = 1; j < n; j++)
                    A[i, j] = A[i - 1, j] + A[i, j - 1];

            return A;
        }

        // ────────────────────────────────────────────────────────────────────────────────
        // 3. Lehmer
        // ────────────────────────────────────────────────────────────────────────────────

        /// <summary>
        /// n×n Lehmer matrix: L[i,j] = (min(i,j)+1) / (max(i,j)+1) (0-based indices).
        /// SPD, totally nonnegative, cond &lt; 4n²; inverse is tridiagonal.
        /// </summary>
        public static fProxyMxN fProxyLehmer(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("fProxyLehmer: n must be >= 1");

            var A = arena.fProxyMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (fProxy)(math.min(i, j) + 1) / (fProxy)(math.max(i, j) + 1);
            return A;
        }

        // ────────────────────────────────────────────────────────────────────────────────
        // 4. MinIJ
        // ────────────────────────────────────────────────────────────────────────────────

        /// <summary>
        /// n×n min-index matrix: A[i,j] = min(i,j)+1 (0-based indices).
        /// SPD, det=1; inverse is the tridiagonal (−1, 2, −1) with last diagonal entry 1.
        /// </summary>
        public static fProxyMxN fProxyMinIJ(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("fProxyMinIJ: n must be >= 1");

            var A = arena.fProxyMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (fProxy)(math.min(i, j) + 1);
            return A;
        }

        // ────────────────────────────────────────────────────────────────────────────────
        // 5. KMS (Kac–Murdock–Szegö)
        // ────────────────────────────────────────────────────────────────────────────────

        /// <summary>
        /// n×n Kac–Murdock–Szegö (KMS) Toeplitz matrix: A[i,j] = ρ^|i−j| (0-based indices).
        /// SPD for |ρ| &lt; 1; det = (1−ρ²)^(n−1); inverse is tridiagonal.
        /// SPD is the caller's responsibility — the generator is intentionally permissive, so
        /// |ρ| ≥ 1 or negative ρ can be used as test inputs (same convention as <see cref="fProxyHilbert"/>).
        /// </summary>
        public static fProxyMxN fProxyKMS(this ref Arena arena, int n, fProxy rho)
        {
            if (n < 1)
                throw new ArgumentException("fProxyKMS: n must be >= 1");

            var A = arena.fProxyMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    int e = math.abs(i - j);
                    fProxy r = (fProxy)1;
                    for (int k = 0; k < e; k++) r *= rho;
                    A[i, j] = r;
                }
            return A;
        }

        // ────────────────────────────────────────────────────────────────────────────────
        // 6. Pei
        // ────────────────────────────────────────────────────────────────────────────────

        /// <summary>
        /// n×n Pei matrix: αI + J (all-ones). Diagonal entry α+1, off-diagonal entry 1.
        /// Eigenvalues: α+n (multiplicity 1) and α (multiplicity n−1).
        /// det = αⁿ⁻¹(α+n); SPD if α &gt; 0.
        /// SPD is the caller's responsibility — the generator is intentionally permissive, so
        /// degenerate/indefinite α can be used as test inputs (same convention as <see cref="fProxyHilbert"/>).
        /// </summary>
        public static fProxyMxN fProxyPei(this ref Arena arena, int n, fProxy alpha)
        {
            if (n < 1)
                throw new ArgumentException("fProxyPei: n must be >= 1");

            var A = arena.fProxyMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (i == j) ? (alpha + (fProxy)1) : (fProxy)1;
            return A;
        }

        // ────────────────────────────────────────────────────────────────────────────────
        // 7. Moler
        // ────────────────────────────────────────────────────────────────────────────────

        /// <summary>
        /// n×n Moler matrix A = UᵀU where U is upper-triangular with 1 on the diagonal and
        /// α on every strictly-upper entry (0-based). Entry formula (derived without
        /// materializing U): A[i,j] = min(i,j)·α² + (i==j ? 1 : α).
        /// SPD and det=1 hold for ALL α (not only α = −1); one eigenvalue is tiny for large |α|,
        /// making this the classic "Triw-based" ill-conditioning example.
        /// The generator is intentionally permissive, following the same convention as
        /// <see cref="fProxyHilbert"/>. Default α = −1.
        /// </summary>
        public static fProxyMxN fProxyMoler(this ref Arena arena, int n, fProxy alpha)
        {
            if (n < 1)
                throw new ArgumentException("fProxyMoler: n must be >= 1");

            var A = arena.fProxyMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (fProxy)math.min(i, j) * alpha * alpha
                             + (i == j ? (fProxy)1 : alpha);
            return A;
        }

        /// <summary>
        /// n×n Moler matrix with default α = −1.
        /// SPD, det=1, one tiny eigenvalue.
        /// </summary>
        public static fProxyMxN fProxyMoler(this ref Arena arena, int n)
            => arena.fProxyMoler(n, (fProxy)(-1));

        // ────────────────────────────────────────────────────────────────────────────────
        // 8. Laplacian1D
        // ────────────────────────────────────────────────────────────────────────────────

        /// <summary>
        /// n×n Strang 2nd-difference tridiagonal (1D discrete Laplacian):
        /// diagonal 2, sub/super-diagonal −1, all other entries 0.
        /// SPD; eigenvalues λ_k = 2 − 2cos(kπ/(n+1)) for k=1…n; det = n+1.
        /// Standard benchmark for CG solvers and eigenvalue algorithms.
        /// </summary>
        public static fProxyMxN fProxyLaplacian1D(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("fProxyLaplacian1D: n must be >= 1");

            var A = arena.fProxyMat(n, true);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    int diff = i - j;
                    if (diff == 0)
                        A[i, j] = (fProxy)2;
                    else if (diff == 1 || diff == -1)
                        A[i, j] = (fProxy)(-1);
                    else
                        A[i, j] = (fProxy)0;
                }
            }
            return A;
        }
    }
}
