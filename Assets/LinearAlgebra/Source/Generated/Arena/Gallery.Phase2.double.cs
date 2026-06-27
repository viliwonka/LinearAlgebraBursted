using System;
using Unity.Mathematics;
using LinearAlgebra;

namespace LinearAlgebra.Gallery
{
    /// <summary>
    /// Gallery of famous test matrices — Phase 2: Cauchy, GCD, Redheffer, Magic, Rosser,
    /// Parter, Prolate, Grcar, Lotkin.
    /// Same partial class as Batch A (Gallery.SPD.double.cs) and Batch B (Gallery.Special.double.cs).
    /// Opt in via <c>using LinearAlgebra.Gallery;</c> then call e.g. <c>arena.doubleCauchy(x, y)</c>.
    /// </summary>
    public static partial class doubleGallery
    {
        // =========================================================================
        // 1. Cauchy — C[i,j] = 1 / (x[i] + y[j])
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Cauchy matrix from node vectors x and y (n = x.N = y.N):
        /// C[i,j] = 1 / (x[i] + y[j]) (0-based).
        /// Known property: det = ∏_{i&lt;j}(x[j]−x[i])(y[j]−y[i]) / ∏_{i,j}(x[i]+y[j])
        /// (Cauchy determinant formula). The doubleHilbert matrix is a special case
        /// with x[i] = y[i] = i + 0.5 (giving 1/(i+j+1)).
        /// Throws if x.N != y.N, x.N &lt; 1, or any x[i]+y[j] == 0.
        /// </summary>
        public static doubleMxN doubleCauchy(this ref Arena arena, in doubleN x, in doubleN y)
        {
            if (x.N != y.N)
                throw new ArgumentException("doubleCauchy: x.N and y.N must be equal");
            if (x.N < 1)
                throw new ArgumentException("doubleCauchy: n must be >= 1");

            int n = x.N;
            var A = arena.doubleMat(n, true);

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double denom = x[i] + y[j];
                    if (denom == (double)0)
                        throw new ArgumentException("doubleCauchy: x[i]+y[j] must be nonzero");
                    A[i, j] = (double)1 / denom;
                }

            return A;
        }

        // =========================================================================
        // 2. GCD — A[i,j] = gcd(i+1, j+1)
        // =========================================================================

        /// <summary>
        /// Allocates the n×n GCD matrix: A[i,j] = gcd(i+1, j+1) (0-based indices).
        /// Known property: SPD; det = ∏_{k=1}^n φ(k) where φ is Euler's totient
        /// function (Smith's theorem).
        /// </summary>
        public static doubleMxN doubleGCD(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("doubleGCD: n must be >= 1");

            var A = arena.doubleMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (double)gcd(i + 1, j + 1);
            return A;
        }

        /// <summary>Euclidean GCD helper used by <see cref="doubleGCD"/>.</summary>
        private static int gcd(int a, int b)
        {
            while (b != 0) { int t = b; b = a % b; a = t; }
            return a;
        }

        // =========================================================================
        // 3. Redheffer — R[i,j] = 1 if j==0 or (i+1)|(j+1), else 0
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Redheffer matrix: R[i,j] = 1 if j == 0 or (i+1) divides
        /// (j+1), else 0 (0-based). Equivalently: 1 if j == 0 or (j+1) % (i+1) == 0.
        /// Known property: det = Mertens M(n) = Σ_{k=1}^n μ(k) where μ is the Möbius
        /// function. M(1..8) = 1, 0, −1, −1, −2, −1, −2, −2.
        /// </summary>
        public static doubleMxN doubleRedheffer(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("doubleRedheffer: n must be >= 1");

            var A = arena.doubleMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (j == 0 || (j + 1) % (i + 1) == 0) ? (double)1 : (double)0;
            return A;
        }

        // =========================================================================
        // 4. Magic — odd-order Siamese magic square
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
        public static doubleMxN doubleMagic(this ref Arena arena, int n)
        {
            if (n < 1 || (n & 1) == 0)
                throw new ArgumentException("doubleMagic: n must be a positive odd integer");

            // Zero-cleared alloc (default uninit=false): the Siamese occupancy check below relies
            // on zero == empty, and the values placed are 1…n² which are always nonzero.
            var A = arena.doubleMat(n);

            // Siamese (de la Loubère) walk
            int r = 0, c = n / 2;
            for (int val = 1; val <= n * n; val++)
            {
                A[r, c] = (double)val;
                int nr = (r - 1 + n) % n;
                int nc = (c + 1) % n;
                if (A[nr, nc] != (double)0)
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
        // 5. Rosser — fixed 8×8 symmetric eigensolver stress test
        // =========================================================================

        /// <summary>
        /// Allocates the fixed 8×8 Rosser matrix (hardcoded integer entries).
        /// Known property: symmetric; trace = 4040; eigenvalues are approximately
        /// {−1020.0532, −0.1705, 0.2180, 999.9469, 1000.1207, 1019.5244, 1019.9936,
        /// 1020.4202}, with near-equal pairs near 0, 1000, and 1020 — a classic
        /// eigensolver stress test for near-degenerate eigenvalue separation.
        /// </summary>
        public static doubleMxN doubleRosser(this ref Arena arena)
        {
            var A = arena.doubleMat(8, true);

            // Row 0
            A[0, 0] = (double)611;   A[0, 1] = (double)196;   A[0, 2] = (double)(-192); A[0, 3] = (double)407;
            A[0, 4] = (double)(-8);  A[0, 5] = (double)(-52); A[0, 6] = (double)(-49);  A[0, 7] = (double)29;
            // Row 1
            A[1, 0] = (double)196;   A[1, 1] = (double)899;   A[1, 2] = (double)113;    A[1, 3] = (double)(-192);
            A[1, 4] = (double)(-71); A[1, 5] = (double)(-43); A[1, 6] = (double)(-8);   A[1, 7] = (double)(-44);
            // Row 2
            A[2, 0] = (double)(-192); A[2, 1] = (double)113;  A[2, 2] = (double)899;    A[2, 3] = (double)196;
            A[2, 4] = (double)61;     A[2, 5] = (double)49;   A[2, 6] = (double)8;      A[2, 7] = (double)52;
            // Row 3
            A[3, 0] = (double)407;   A[3, 1] = (double)(-192); A[3, 2] = (double)196;   A[3, 3] = (double)611;
            A[3, 4] = (double)8;     A[3, 5] = (double)44;    A[3, 6] = (double)59;     A[3, 7] = (double)(-23);
            // Row 4
            A[4, 0] = (double)(-8);  A[4, 1] = (double)(-71); A[4, 2] = (double)61;     A[4, 3] = (double)8;
            A[4, 4] = (double)411;   A[4, 5] = (double)(-599); A[4, 6] = (double)208;   A[4, 7] = (double)208;
            // Row 5
            A[5, 0] = (double)(-52); A[5, 1] = (double)(-43); A[5, 2] = (double)49;     A[5, 3] = (double)44;
            A[5, 4] = (double)(-599); A[5, 5] = (double)411;  A[5, 6] = (double)208;    A[5, 7] = (double)208;
            // Row 6
            A[6, 0] = (double)(-49); A[6, 1] = (double)(-8);  A[6, 2] = (double)8;      A[6, 3] = (double)59;
            A[6, 4] = (double)208;   A[6, 5] = (double)208;   A[6, 6] = (double)99;     A[6, 7] = (double)(-911);
            // Row 7
            A[7, 0] = (double)29;    A[7, 1] = (double)(-44); A[7, 2] = (double)52;     A[7, 3] = (double)(-23);
            A[7, 4] = (double)208;   A[7, 5] = (double)208;   A[7, 6] = (double)(-911); A[7, 7] = (double)99;

            return A;
        }

        // =========================================================================
        // 6. Parter — Toeplitz C[i,j] = 1 / (i − j + 0.5)
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Parter matrix: Toeplitz C[i,j] = 1 / (i − j + 0.5)
        /// (0-based). The denominator is always a nonzero half-integer, so no division
        /// by zero can occur.
        /// Known property: nonsymmetric; singular values cluster near π (all less than π).
        /// </summary>
        public static doubleMxN doubleParter(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("doubleParter: n must be >= 1");

            var A = arena.doubleMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (double)1 / ((double)(i - j) + (double)0.5);
            return A;
        }

        // =========================================================================
        // 7. Prolate — symmetric Toeplitz with eigenvalues in (0, 1)
        // =========================================================================

        /// <summary>
        /// Allocates the n×n prolate matrix: symmetric Toeplitz A[i,j] = a_{|i−j|}
        /// where a_0 = 2w and a_k = sin(2π w k) / (π k) for k ≥ 1. Requires 0 &lt; w &lt; 0.5.
        /// Known property: symmetric; all eigenvalues lie in (0, 1) and cluster near 0
        /// and 1; ill-conditioned for w near 0 or 0.5.
        /// </summary>
        public static doubleMxN doubleProlate(this ref Arena arena, int n, double w)
        {
            if (n < 1)
                throw new ArgumentException("doubleProlate: n must be >= 1");
            if (w <= (double)0 || w >= (double)0.5)
                throw new ArgumentException("doubleProlate: w must satisfy 0 < w < 0.5");

            var A = arena.doubleMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    int k = math.abs(i - j);
                    if (k == 0)
                        A[i, j] = (double)2 * w;
                    else
                        A[i, j] = math.sin((double)(2.0 * Math.PI) * w * (double)k)
                                 / ((double)Math.PI * (double)k);
                }
            return A;
        }

        // =========================================================================
        // 8. Grcar — nonsymmetric banded Toeplitz
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Grcar matrix: Toeplitz with 1 on the diagonal and k
        /// superdiagonals, −1 on the first subdiagonal, and 0 elsewhere.
        /// For d = j − i: G[i,j] = 1 if d == 0 or 1 ≤ d ≤ k; −1 if d == −1; 0 otherwise.
        /// Known property: nonsymmetric banded; highly sensitive pseudospectra — a standard
        /// structural test for non-normal matrix behaviour.
        /// </summary>
        public static doubleMxN doubleGrcar(this ref Arena arena, int n, int k = 3)
        {
            if (n < 1)
                throw new ArgumentException("doubleGrcar: n must be >= 1");
            if (k < 1)
                throw new ArgumentException("doubleGrcar: k must be >= 1");

            var A = arena.doubleMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    int d = j - i;
                    if (d == 0 || (d >= 1 && d <= k))
                        A[i, j] = (double)1;
                    else if (d == -1)
                        A[i, j] = (double)(-1);
                    else
                        A[i, j] = (double)0;
                }
            return A;
        }

        // =========================================================================
        // 9. Lotkin — Hilbert with first row replaced by all ones
        // =========================================================================

        /// <summary>
        /// Allocates the n×n Lotkin matrix: identical to the doubleHilbert matrix except
        /// that the first row (i = 0) is replaced by all ones — A[0,j] = 1 for all j;
        /// A[i,j] = 1/(i+j+1) for i ≥ 1 (0-based).
        /// Known property: nonsymmetric; severely ill-conditioned (large condition number).
        /// </summary>
        public static doubleMxN doubleLotkin(this ref Arena arena, int n)
        {
            if (n < 1)
                throw new ArgumentException("doubleLotkin: n must be >= 1");

            var A = arena.doubleMat(n, true);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    if (i == 0)
                        A[i, j] = (double)1;
                    else
                        A[i, j] = (double)1 / (double)(i + j + 1);
                }
            return A;
        }
    }
}
