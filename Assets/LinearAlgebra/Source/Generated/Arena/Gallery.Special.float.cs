using System;
using Unity.Mathematics;
using LinearAlgebra;

namespace LinearAlgebra.Gallery
{
    /// <summary>
    /// Gallery of famous test matrices — eigenvalue / nonsymmetric / structured / rank family,
    /// plus the number-theoretic / combinatorial / additional-structured set (Cauchy, GCD, Redheffer,
    /// Magic, Rosser, Parter, Prolate, Grcar, Lotkin — sections 13-21 below).
    /// Batch B of the Literature Gallery (Batch A: Gallery.SPD.float.cs).
    /// Targets: Eigen.decompInPlace, Eigen.valuesQR, SVD, QR/QRCP, least-squares, FFT cross-check, det.
    /// Opt in via <c>using LinearAlgebra.Gallery;</c> then call e.g. <c>arena.floatFrank(n)</c>.
    /// </summary>
    public static partial class floatGallery
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
        public static floatMxN floatClement(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("floatClement: n must be >= 1");

            var mat = arena.floatMat(n, true);

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (j == i + 1)
                        mat[i, j] = math.sqrt((float)(i + 1) * (float)(n - 1 - i));
                    else if (j == i - 1)
                        mat[i, j] = math.sqrt((float)i * (float)(n - i));
                    else
                        mat[i, j] = (float)0;
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
        public static floatMxN floatWilkinsonPlus(this ref Arena arena, int n)
        {
            if (n < 3 || (n & 1) == 0)
                throw new ArgumentException("floatWilkinsonPlus: n must be an odd integer >= 3");

            int m = (n - 1) / 2;
            var mat = arena.floatMat(n, true);

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (j == i)
                        mat[i, j] = (float)math.abs(m - i);
                    else if (j == i + 1 || j == i - 1)
                        mat[i, j] = (float)1;
                    else
                        mat[i, j] = (float)0;
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
        public static floatMxN floatFiedler(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("floatFiedler: n must be >= 1");

            var mat = arena.floatMat(n, true);

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    mat[i, j] = (float)math.abs(i - j);

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
        public static floatMxN floatDingDong(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("floatDingDong: n must be >= 1");

            var mat = arena.floatMat(n, true);

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    mat[i, j] = (float)0.5 / ((float)(n - i - j) - (float)0.5);

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
        public static floatMxN floatFrank(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("floatFrank: n must be >= 1");

            var mat = arena.floatMat(n, true);

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    mat[i, j] = (i <= j + 1) ? (float)(n - math.max(i, j)) : (float)0;

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
        public static floatMxN floatVandermonde(this ref Arena arena, in floatN nodes)
        {
            if (nodes.N < 1)
                throw new ArgumentException("floatVandermonde: nodes.N must be >= 1");

            int n = nodes.N;
            var mat = arena.floatMat(n, true);

            for (int i = 0; i < n; i++)
            {
                float v = (float)1;
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
        public static floatMxN floatCompanion(this ref Arena arena, in floatN coeffs)
        {
            if (coeffs.N < 1)
                throw new ArgumentException("floatCompanion: coeffs.N must be >= 1");

            int n = coeffs.N;
            var mat = arena.floatMat(n, true);

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (j == n - 1)
                        mat[i, j] = -coeffs[i];
                    else if (i > 0 && j == i - 1)
                        mat[i, j] = (float)1;
                    else
                        mat[i, j] = (float)0;
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
        public static floatMxN floatHadamard(this ref Arena arena, int n)
        {
            if (n < 1 || (n & (n - 1)) != 0)
                throw new ArgumentException("floatHadamard: n must be a power of two");

            var mat = arena.floatMat(n, true);

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    mat[i, j] = ((math.countbits((uint)(i & j)) & 1) == 0) ? (float)1 : (float)(-1);

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
        public static floatMxN floatCirculant(this ref Arena arena, in floatN c)
        {
            if (c.N < 1)
                throw new ArgumentException("floatCirculant: c.N must be >= 1");

            int n = c.N;
            var mat = arena.floatMat(n, true);

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
        public static floatMxN floatKahan(this ref Arena arena, int n, float theta)
        {
            if (n < 1)
                throw new ArgumentException("floatKahan: n must be >= 1");

            float s  = math.sin(theta);
            float cc = math.cos(theta);
            var mat = arena.floatMat(n, true);

            float si = (float)1;               // s^0
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (j < i)       mat[i, j] = (float)0;
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
        public static floatMxN floatTriw(this ref Arena arena, int n, float alpha)
        {
            if (n < 1)
                throw new ArgumentException("floatTriw: n must be >= 1");

            var mat = arena.floatMat(n, true);

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (j == i)
                        mat[i, j] = (float)1;
                    else if (j > i)
                        mat[i, j] = alpha;
                    else
                        mat[i, j] = (float)0;
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
        public static floatMxN floatLauchli(this ref Arena arena, int n, float eps)
        {
            if (n < 1)
                throw new ArgumentException("floatLauchli: n must be >= 1");

            var mat = arena.floatMat(n + 1, n, true);

            for (int r = 0; r < n + 1; r++)
            {
                for (int c = 0; c < n; c++)
                {
                    if (r == 0)
                        mat[r, c] = (float)1;
                    else
                        mat[r, c] = (r - 1 == c) ? eps : (float)0;
                }
            }

            return mat;
        }

        // =========================================================================
        // 13. Cauchy — C[i,j] = 1 / (x[i] + y[j])
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Cauchy matrix from node vectors x and y (n = x.N = y.N):
        /// C[i,j] = 1 / (x[i] + y[j]) (0-based).
        /// Known property: det = ∏_{i&lt;j}(x[j]−x[i])(y[j]−y[i]) / ∏_{i,j}(x[i]+y[j])
        /// (Cauchy determinant formula). The floatHilbert matrix is a special case
        /// with x[i] = y[i] = i + 0.5 (giving 1/(i+j+1)).
        /// Throws if x.N != y.N, x.N &lt; 1, or any x[i]+y[j] == 0.
        /// </summary>
        public static floatMxN floatCauchy(this ref Arena arena, in floatN x, in floatN y)
        {
            if (x.N != y.N)
                throw new ArgumentException("floatCauchy: x.N and y.N must be equal");
            if (x.N < 1)
                throw new ArgumentException("floatCauchy: n must be >= 1");

            int n = x.N;
            var A = arena.floatMat(n, true);

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    float denom = x[i] + y[j];
                    if (denom == (float)0)
                        throw new ArgumentException("floatCauchy: x[i]+y[j] must be nonzero");
                    A[i, j] = (float)1 / denom;
                }

            return A;
        }

        // =========================================================================
        // 14. GCD — A[i,j] = gcd(i+1, j+1)
        // =========================================================================

        /// <summary>
        /// Allocates the n×n GCD matrix: A[i,j] = gcd(i+1, j+1) (0-based indices).
        /// Known property: SPD; det = ∏_{k=1}^n φ(k) where φ is Euler's totient
        /// function (Smith's theorem).
        /// </summary>
        public static floatMxN floatGCD(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("floatGCD: n must be >= 1");

            var A = arena.floatMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (float)gcd(i + 1, j + 1);
            return A;
        }

        /// <summary>Euclidean GCD helper used by <see cref="floatGCD"/>.</summary>
        private static int gcd(int a, int b)
        {
            while (b != 0) { int t = b; b = a % b; a = t; }
            return a;
        }

        // =========================================================================
        // 15. Redheffer — R[i,j] = 1 if j==0 or (i+1)|(j+1), else 0
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Redheffer matrix: R[i,j] = 1 if j == 0 or (i+1) divides
        /// (j+1), else 0 (0-based). Equivalently: 1 if j == 0 or (j+1) % (i+1) == 0.
        /// Known property: det = Mertens M(n) = Σ_{k=1}^n μ(k) where μ is the Möbius
        /// function. M(1..8) = 1, 0, −1, −1, −2, −1, −2, −2.
        /// </summary>
        public static floatMxN floatRedheffer(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("floatRedheffer: n must be >= 1");

            var A = arena.floatMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (j == 0 || (j + 1) % (i + 1) == 0) ? (float)1 : (float)0;
            return A;
        }

        // =========================================================================
        // 16. Magic — odd-order Siamese magic square
        // =========================================================================

        /// <summary>
        /// Allocates the n×n magic square (n must be a positive odd integer) using the
        /// Siamese (de la Loubère) method: place 1…n² starting at row 0, middle column;
        /// after each placement move up-and-right (r=(r−1+n)%n, c=(c+1)%n); if the
        /// target cell is already occupied, drop down one row instead (r=(r+1)%n, c
        /// unchanged).
        /// Known property: every row sum, column sum, and both main-diagonal sums equal
        /// n(n²+1)/2 (the magic constant). For n=3: [[8,1,6],[3,5,7],[4,9,2]].
        /// </summary>
        public static floatMxN floatMagic(this ref Arena arena, int n)
        {
            if (n < 1 || (n & 1) == 0)
                throw new ArgumentException("floatMagic: n must be a positive odd integer");

            // Zero-cleared alloc (default uninit=false): the Siamese occupancy check below relies
            // on zero == empty, and the values placed are 1…n² which are always nonzero.
            var A = arena.floatMat(n);

            // Siamese (de la Loubère) walk
            int r = 0, c = n / 2;
            for (int val = 1; val <= n * n; val++)
            {
                A[r, c] = (float)val;
                int nr = (r - 1 + n) % n;
                int nc = (c + 1) % n;
                if (A[nr, nc] != (float)0)
                {
                    // Target cell occupied: drop down one row, keep column.
                    r = (r + 1) % n;
                }
                else
                {
                    r = nr;
                    c = nc;
                }
            }

            return A;
        }

        // =========================================================================
        // 17. Rosser — fixed 8×8 symmetric eigensolver stress test
        // =========================================================================

        /// <summary>
        /// Allocates the fixed 8×8 Rosser matrix (hardcoded integer entries).
        /// Known property: symmetric; trace = 4040; eigenvalues are approximately
        /// {−1020.0532, −0.1705, 0.2180, 999.9469, 1000.1207, 1019.5244, 1019.9936,
        /// 1020.4202}, with near-equal pairs near 0, 1000, and 1020 — a classic
        /// eigensolver stress test for near-degenerate eigenvalue separation.
        /// </summary>
        public static floatMxN floatRosser(this ref Arena arena)
        {
            var A = arena.floatMat(8, true);

            // Row 0
            A[0, 0] = (float)611;   A[0, 1] = (float)196;   A[0, 2] = (float)(-192); A[0, 3] = (float)407;
            A[0, 4] = (float)(-8);  A[0, 5] = (float)(-52); A[0, 6] = (float)(-49);  A[0, 7] = (float)29;
            // Row 1
            A[1, 0] = (float)196;   A[1, 1] = (float)899;   A[1, 2] = (float)113;    A[1, 3] = (float)(-192);
            A[1, 4] = (float)(-71); A[1, 5] = (float)(-43); A[1, 6] = (float)(-8);   A[1, 7] = (float)(-44);
            // Row 2
            A[2, 0] = (float)(-192); A[2, 1] = (float)113;  A[2, 2] = (float)899;    A[2, 3] = (float)196;
            A[2, 4] = (float)61;     A[2, 5] = (float)49;   A[2, 6] = (float)8;      A[2, 7] = (float)52;
            // Row 3
            A[3, 0] = (float)407;   A[3, 1] = (float)(-192); A[3, 2] = (float)196;   A[3, 3] = (float)611;
            A[3, 4] = (float)8;     A[3, 5] = (float)44;    A[3, 6] = (float)59;     A[3, 7] = (float)(-23);
            // Row 4
            A[4, 0] = (float)(-8);  A[4, 1] = (float)(-71); A[4, 2] = (float)61;     A[4, 3] = (float)8;
            A[4, 4] = (float)411;   A[4, 5] = (float)(-599); A[4, 6] = (float)208;   A[4, 7] = (float)208;
            // Row 5
            A[5, 0] = (float)(-52); A[5, 1] = (float)(-43); A[5, 2] = (float)49;     A[5, 3] = (float)44;
            A[5, 4] = (float)(-599); A[5, 5] = (float)411;  A[5, 6] = (float)208;    A[5, 7] = (float)208;
            // Row 6
            A[6, 0] = (float)(-49); A[6, 1] = (float)(-8);  A[6, 2] = (float)8;      A[6, 3] = (float)59;
            A[6, 4] = (float)208;   A[6, 5] = (float)208;   A[6, 6] = (float)99;     A[6, 7] = (float)(-911);
            // Row 7
            A[7, 0] = (float)29;    A[7, 1] = (float)(-44); A[7, 2] = (float)52;     A[7, 3] = (float)(-23);
            A[7, 4] = (float)208;   A[7, 5] = (float)208;   A[7, 6] = (float)(-911); A[7, 7] = (float)99;

            return A;
        }

        // =========================================================================
        // 18. Parter — Toeplitz C[i,j] = 1 / (i − j + 0.5)
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Parter matrix: Toeplitz C[i,j] = 1 / (i − j + 0.5)
        /// (0-based). The denominator is always a nonzero half-integer, so no division
        /// by zero can occur.
        /// Known property: nonsymmetric; singular values cluster near π (all less than π).
        /// </summary>
        public static floatMxN floatParter(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("floatParter: n must be >= 1");

            var A = arena.floatMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (float)1 / ((float)(i - j) + (float)0.5);
            return A;
        }

        // =========================================================================
        // 19. Prolate — symmetric Toeplitz with eigenvalues in (0, 1)
        // =========================================================================

        /// <summary>
        /// Allocates the n×n prolate matrix: symmetric Toeplitz A[i,j] = a_{|i−j|}
        /// where a_0 = 2w and a_k = sin(2π w k) / (π k) for k ≥ 1. Requires 0 &lt; w &lt; 0.5.
        /// Known property: symmetric; all eigenvalues lie in (0, 1) and cluster near 0
        /// and 1; ill-conditioned for w near 0 or 0.5.
        /// </summary>
        public static floatMxN floatProlate(this ref Arena arena, int n, float w)
        {
            if (n < 1)
                throw new ArgumentException("floatProlate: n must be >= 1");
            if (w <= (float)0 || w >= (float)0.5)
                throw new ArgumentException("floatProlate: w must satisfy 0 < w < 0.5");

            var A = arena.floatMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    int k = math.abs(i - j);
                    if (k == 0)
                        A[i, j] = (float)2 * w;
                    else
                        A[i, j] = math.sin((float)(2.0 * Math.PI) * w * (float)k)
                                 / ((float)Math.PI * (float)k);
                }
            return A;
        }

        // =========================================================================
        // 20. Grcar — nonsymmetric banded Toeplitz
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Grcar matrix: Toeplitz with 1 on the diagonal and k
        /// superdiagonals, −1 on the first subdiagonal, and 0 elsewhere.
        /// For d = j − i: G[i,j] = 1 if d == 0 or 1 ≤ d ≤ k; −1 if d == −1; 0 otherwise.
        /// Known property: nonsymmetric banded; highly sensitive pseudospectra — a standard
        /// structural test for non-normal matrix behaviour.
        /// </summary>
        public static floatMxN floatGrcar(this ref Arena arena, int n, int k = 3)
        {
            if (n < 1)
                throw new ArgumentException("floatGrcar: n must be >= 1");
            if (k < 1)
                throw new ArgumentException("floatGrcar: k must be >= 1");

            var A = arena.floatMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    int d = j - i;
                    if (d == 0 || (d >= 1 && d <= k))
                        A[i, j] = (float)1;
                    else if (d == -1)
                        A[i, j] = (float)(-1);
                    else
                        A[i, j] = (float)0;
                }
            return A;
        }

        // =========================================================================
        // 21. Lotkin — Hilbert with first row replaced by all ones
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Lotkin matrix: identical to the floatHilbert matrix except
        /// that the first row (i = 0) is replaced by all ones — A[0,j] = 1 for all j;
        /// A[i,j] = 1/(i+j+1) for i ≥ 1 (0-based).
        /// Known property: nonsymmetric; severely ill-conditioned (large condition number).
        /// </summary>
        public static floatMxN floatLotkin(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("floatLotkin: n must be >= 1");

            var A = arena.floatMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    if (i == 0)
                        A[i, j] = (float)1;
                    else
                        A[i, j] = (float)1 / (float)(i + j + 1);
                }
            return A;
        }
    }
}
