using System;
using Unity.Mathematics;
using LinearAlgebra;

namespace LinearAlgebra.Gallery
{
    /// <summary>
    /// Gallery of famous test matrices — eigenvalue / nonsymmetric / structured / rank family.
    /// Batch B of the Literature Gallery (Batch A: Gallery.SPD.double.cs).
    /// Targets: eigenDecomposition, eigenvaluesQR, SVD, QR/QRCP, least-squares, FFT cross-check, det.
    /// Opt in via <c>using LinearAlgebra.Gallery;</c> then call e.g. <c>arena.doubleFrank(n)</c>.
    /// </summary>
    public static partial class doubleGallery
    {
        // =========================================================================
        // 1. Clement — symmetric tridiagonal, zero diagonal
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Clement matrix: symmetric tridiagonal with zero diagonal and
        /// off-diagonal entries e[i] = √((i+1)(n−1−i)) for i = 0…n−2.
        /// Known property: eigenvalues are exactly {n−1, n−3, …, −(n−1)} (integer-spaced symmetric
        /// about 0); trace = 0.
        /// </summary>
        public static doubleMxN doubleClement(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("doubleClement: n must be >= 1");

            var mat = arena.doubleMat(n, true);

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (j == i + 1)
                        mat[i, j] = math.sqrt((double)(i + 1) * (double)(n - 1 - i));
                    else if (j == i - 1)
                        mat[i, j] = math.sqrt((double)i * (double)(n - i));
                    else
                        mat[i, j] = (double)0;
                }
            }

            return mat;
        }

        // =========================================================================
        // 2. WilkinsonPlus — symmetric tridiagonal with near-equal top eigenvalues
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Wilkinson W+ matrix (n must be odd): symmetric tridiagonal with
        /// diagonal entries |m − i| where m = (n−1)/2, and all sub/super-diagonal entries = 1.
        /// Known property: the two largest eigenvalues are nearly equal (a classic near-pair),
        /// stressing eigenvalue separation in QR iteration.
        /// </summary>
        public static doubleMxN doubleWilkinsonPlus(this ref Arena arena, int n)
        {
            if (n < 3 || (n & 1) == 0)
                throw new ArgumentException("doubleWilkinsonPlus: n must be an odd integer >= 3");

            int m = (n - 1) / 2;
            var mat = arena.doubleMat(n, true);

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (j == i)
                        mat[i, j] = (double)math.abs(m - i);
                    else if (j == i + 1 || j == i - 1)
                        mat[i, j] = (double)1;
                    else
                        mat[i, j] = (double)0;
                }
            }

            return mat;
        }

        // =========================================================================
        // 3. Fiedler — |i − j| distance matrix
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Fiedler matrix: F[i,j] = |i − j| (0-based).
        /// Known property: symmetric; exactly one positive eigenvalue and n−1 negative eigenvalues;
        /// det = (−1)^(n−1) · (n−1) · 2^(n−2).
        /// </summary>
        public static doubleMxN doubleFiedler(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("doubleFiedler: n must be >= 1");

            var mat = arena.doubleMat(n, true);

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    mat[i, j] = (double)math.abs(i - j);

            return mat;
        }

        // =========================================================================
        // 4. DingDong — symmetric Hankel with eigenvalues near ±π/2
        // =========================================================================

        /// <summary>
        /// Allocates the n×n DingDong matrix: symmetric Hankel A[i,j] = 0.5 / (n − i − j − 0.5).
        /// Known property: all eigenvalues lie in (−π/2, π/2) and cluster near ±π/2.
        /// The denominator is always a non-zero half-integer for integer n, i, j so no division by
        /// zero can occur.
        /// </summary>
        public static doubleMxN doubleDingDong(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("doubleDingDong: n must be >= 1");

            var mat = arena.doubleMat(n, true);

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    mat[i, j] = (double)0.5 / ((double)(n - i - j) - (double)0.5);

            return mat;
        }

        // =========================================================================
        // 5. Frank — upper Hessenberg with det = 1
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Frank matrix: upper Hessenberg with F[i,j] = n − max(i,j) for
        /// i ≤ j+1, and 0 elsewhere (0-based indices).
        /// For n=3: [[3,2,1],[2,2,1],[0,1,1]].
        /// Known property: det = 1; all eigenvalues are real, positive, and come in reciprocal
        /// pairs; the matrix is ill-conditioned.
        /// </summary>
        public static doubleMxN doubleFrank(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("doubleFrank: n must be >= 1");

            var mat = arena.doubleMat(n, true);

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    mat[i, j] = (i <= j + 1) ? (double)(n - math.max(i, j)) : (double)0;

            return mat;
        }

        // =========================================================================
        // 6. Vandermonde — V[i,j] = nodes[i]^j
        // =========================================================================

        /// <summary>
        /// Allocates an n×n Vandermonde matrix from the given node vector (n = nodes.N):
        /// V[i,j] = nodes[i]^j (so column 0 is all-ones, column 1 is nodes itself).
        /// Known property: det = ∏_{i &lt; j} (nodes[j] − nodes[i]); useful for polynomial
        /// interpolation and condition-number studies.
        /// </summary>
        public static doubleMxN doubleVandermonde(this ref Arena arena, in doubleN nodes)
        {
            if (nodes.N < 1)
                throw new ArgumentException("doubleVandermonde: nodes.N must be >= 1");

            int n = nodes.N;
            var mat = arena.doubleMat(n, true);

            for (int i = 0; i < n; i++)
            {
                double v = (double)1;
                mat[i, 0] = v;
                for (int j = 1; j < n; j++) { v *= nodes[i]; mat[i, j] = v; }
            }

            return mat;
        }

        // =========================================================================
        // 7. Companion — companion matrix of a monic polynomial
        // =========================================================================

        /// <summary>
        /// Allocates the n×n companion matrix of the monic polynomial
        /// x^n + coeffs[n−1]·x^(n−1) + … + coeffs[0] (n = coeffs.N).
        /// Layout: last column C[i,n−1] = −coeffs[i]; sub-diagonal C[i,i−1] = 1 for i = 1…n−1;
        /// all other entries 0.
        /// Known property: eigenvalues of C equal the roots of the polynomial.
        /// </summary>
        public static doubleMxN doubleCompanion(this ref Arena arena, in doubleN coeffs)
        {
            if (coeffs.N < 1)
                throw new ArgumentException("doubleCompanion: coeffs.N must be >= 1");

            int n = coeffs.N;
            var mat = arena.doubleMat(n, true);

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (j == n - 1)
                        mat[i, j] = -coeffs[i];
                    else if (i > 0 && j == i - 1)
                        mat[i, j] = (double)1;
                    else
                        mat[i, j] = (double)0;
                }
            }

            return mat;
        }

        // =========================================================================
        // 8. Hadamard — Sylvester–Walsh ±1 matrix
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Sylvester–Walsh Hadamard matrix (n must be a power of two):
        /// H[i,j] = +1 if popcount(i &amp; j) is even, −1 if odd (0-based).
        /// Known property: H^T H = n·I (orthogonal up to scale √n); cond = 1;
        /// |det| = n^(n/2).
        /// Throws <see cref="ArgumentException"/> if n is not a power of two (including n &lt; 1).
        /// </summary>
        public static doubleMxN doubleHadamard(this ref Arena arena, int n)
        {
            if (n < 1 || (n & (n - 1)) != 0)
                throw new ArgumentException("doubleHadamard: n must be a power of two");

            var mat = arena.doubleMat(n, true);

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    mat[i, j] = ((math.countbits((uint)(i & j)) & 1) == 0) ? (double)1 : (double)(-1);

            return mat;
        }

        // =========================================================================
        // 9. Circulant — C[i,j] = c[(j−i) mod n]
        // =========================================================================

        /// <summary>
        /// Allocates the n×n circulant matrix defined by the first-row vector c (n = c.N):
        /// C[i,j] = c[(j − i) mod n].
        /// Known property: eigenvalues equal the DFT of c, making this matrix ideal for
        /// cross-checking the library FFT.
        /// </summary>
        public static doubleMxN doubleCirculant(this ref Arena arena, in doubleN c)
        {
            if (c.N < 1)
                throw new ArgumentException("doubleCirculant: c.N must be >= 1");

            int n = c.N;
            var mat = arena.doubleMat(n, true);

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    mat[i, j] = c[((j - i) % n + n) % n];

            return mat;
        }

        // =========================================================================
        // 10. Kahan — ill-conditioned upper triangular (QRCP stress)
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Kahan matrix K = S·R where S = diag(s^0,…,s^(n−1)),
        /// s = sin(theta), c = cos(theta):
        /// K[i,i] = s^i; K[i,j] = −c·s^i for j &gt; i; K[i,j] = 0 for j &lt; i.
        /// Known property: ill-conditioned; classic counterexample for column-pivoted QR —
        /// unpivoted Householder QR produces a poor rank-revealing factorisation.
        /// </summary>
        public static doubleMxN doubleKahan(this ref Arena arena, int n, double theta)
        {
            if (n < 1)
                throw new ArgumentException("doubleKahan: n must be >= 1");

            double s  = math.sin(theta);
            double cc = math.cos(theta);
            var mat = arena.doubleMat(n, true);

            double si = (double)1;               // s^0
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (j < i)       mat[i, j] = (double)0;
                    else if (j == i) mat[i, j] = si;
                    else             mat[i, j] = -cc * si;
                }
                si *= s;                          // advance to s^(i+1)
            }

            return mat;
        }

        // =========================================================================
        // 11. Triw — upper triangular, constant off-diagonal alpha
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Triw matrix: upper triangular with 1 on the diagonal and
        /// alpha on every strictly upper-triangular entry.
        /// Known property: det = 1; all eigenvalues = 1; severely ill-conditioned for large
        /// |alpha| or alpha ≪ 0.
        /// </summary>
        public static doubleMxN doubleTriw(this ref Arena arena, int n, double alpha)
        {
            if (n < 1)
                throw new ArgumentException("doubleTriw: n must be >= 1");

            var mat = arena.doubleMat(n, true);

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (j == i)
                        mat[i, j] = (double)1;
                    else if (j > i)
                        mat[i, j] = alpha;
                    else
                        mat[i, j] = (double)0;
                }
            }

            return mat;
        }

        // =========================================================================
        // 12. Lauchli — rectangular rank-stress matrix
        // =========================================================================

        /// <summary>
        /// Allocates the (n+1)×n Läuchli matrix: row 0 is all ones; rows 1…n are eps·I
        /// (i.e. A[i+1,j] = eps if i == j, else 0).
        /// Known property: full column rank but near rank-deficient for small eps — a standard
        /// stress test comparing QR vs SVD for least-squares solve quality.
        /// </summary>
        public static doubleMxN doubleLauchli(this ref Arena arena, int n, double eps)
        {
            if (n < 1)
                throw new ArgumentException("doubleLauchli: n must be >= 1");

            var mat = arena.doubleMat(n + 1, n, true);

            for (int r = 0; r < n + 1; r++)
            {
                for (int c = 0; c < n; c++)
                {
                    if (r == 0)
                        mat[r, c] = (double)1;
                    else
                        mat[r, c] = (r - 1 == c) ? eps : (double)0;
                }
            }

            return mat;
        }
    }
}
