#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra
{
    /// <summary>
    /// Inpl = inplace
    /// </summary>
    public static partial class Eigen {

        /// <summary>
        /// Power iteration with Rayleigh-quotient eigenvalue estimate.
        /// Finds the dominant eigenpair (lambda, v) of a square matrix A.
        ///
        /// On input: v (length n) is the initial guess for the eigenvector; w (length n)
        /// is caller-provided scratch storage — it is overwritten and must NOT be the same
        /// array as v. On output: v is the unit eigenvector estimate; lambda is the
        /// Rayleigh quotient estimate (v^T A v).
        ///
        /// If the supplied v has zero 2-norm it is seeded deterministically as
        /// v[i] = 1 + (i &amp; 3), then normalized before iterating.
        ///
        /// Convergence criterion: the infinity norm of the residual r = A*v - lambda*v
        /// satisfies r &lt;= tol * max(1, |lambda|). Returns true on convergence.
        ///
        /// Notes:
        ///   - Converges to the dominant eigenpair when |lambda_1| &gt; |lambda_2|;
        ///     the rate is |lambda_2 / lambda_1| per iteration.
        ///   - For a negative dominant eigenvalue the eigenvector sign may alternate
        ///     between iterations, but the residual still converges.
        ///   - When the dominant eigenvalue is a complex conjugate pair (e.g. rotation
        ///     matrices) the iteration cannot converge and the method returns false after
        ///     maxIter iterations.
        ///   - Inputs of extreme magnitude (entries whose squares overflow the type) are
        ///     not rescaled in this version; keep element magnitudes moderate.
        ///   - Does not allocate.
        /// </summary>
        public static bool powerIteration(in floatMxN A, ref floatN v, ref floatN w,
                                          out float lambda, float tol, int maxIter)
        {
            if (!A.IsSquare)
                throw new ArgumentException("Eigen.powerIteration: A must be square");

            if (v.N != A.N_Cols)
                throw new ArgumentException("Eigen.powerIteration: v.N must equal A.N_Cols");

            if (w.N != A.N_Cols)
                throw new ArgumentException("Eigen.powerIteration: w.N must equal A.N_Cols");

            unsafe {
                if (v.Data.Ptr == w.Data.Ptr)
                    throw new ArgumentException("Eigen.powerIteration: w must not alias v");
            }

            if (maxIter < 1)
                throw new ArgumentException("Eigen.powerIteration: maxIter must be >= 1");

            if (tol <= (float)0)
                throw new ArgumentException("Eigen.powerIteration: tol must be > 0");

            int n = A.N_Cols;

            // Seed v deterministically if the caller supplied the zero vector
            float vNormSq = (float)0;
            for (int i = 0; i < n; i++)
                vNormSq += v[i] * v[i];

            if (vNormSq == (float)0) {
                for (int i = 0; i < n; i++)
                    v[i] = (float)(1 + (i & 3));
                vNormSq = (float)0;
                for (int i = 0; i < n; i++)
                    vNormSq += v[i] * v[i];
            }

            // Normalize v to unit length
            float vNorm = math.sqrt(vNormSq);
            float invVNorm = (float)1 / vNorm;
            for (int i = 0; i < n; i++)
                v[i] = v[i] * invVNorm;

            lambda = (float)0;

            for (int iter = 0; iter < maxIter; iter++) {

                // Step 1: w = A * v (manual matvec — no allocation)
                for (int i = 0; i < n; i++) {
                    float sum = (float)0;
                    for (int j = 0; j < n; j++)
                        sum += A[i, j] * v[j];
                    w[i] = sum;
                }

                // Step 2: lambda = v . w (Rayleigh quotient; ||v||_2 = 1)
                lambda = (float)0;
                for (int i = 0; i < n; i++)
                    lambda += v[i] * w[i];

                // Step 3: residual r = max_i |w[i] - lambda * v[i]|  (infinity norm)
                float residual = (float)0;
                for (int i = 0; i < n; i++) {
                    float ri = math.abs(w[i] - lambda * v[i]);
                    if (ri > residual)
                        residual = ri;
                }

                // Step 4: convergence check
                float scale = math.abs(lambda);
                if (scale < (float)1)
                    scale = (float)1;
                if (residual <= tol * scale)
                    return true;

                // Step 5: compute ||w||_2; handle exact null-space case
                float nw = (float)0;
                for (int i = 0; i < n; i++)
                    nw += w[i] * w[i];
                nw = math.sqrt(nw);

                if (nw == (float)0) {
                    lambda = (float)0;
                    return true;
                }

                // Step 6: v = w / ||w||
                float invNw = (float)1 / nw;
                for (int i = 0; i < n; i++)
                    v[i] = w[i] * invNw;
            }

            // Post-loop: recompute w = A*v, lambda, residual with final v
            for (int i = 0; i < n; i++) {
                float sum = (float)0;
                for (int j = 0; j < n; j++)
                    sum += A[i, j] * v[j];
                w[i] = sum;
            }

            lambda = (float)0;
            for (int i = 0; i < n; i++)
                lambda += v[i] * w[i];

            float finalResidual = (float)0;
            for (int i = 0; i < n; i++) {
                float ri = math.abs(w[i] - lambda * v[i]);
                if (ri > finalResidual)
                    finalResidual = ri;
            }

            float finalScale = math.abs(lambda);
            if (finalScale < (float)1)
                finalScale = (float)1;
            return finalResidual <= tol * finalScale;
        }

        /// <summary>powerIteration with default maxIter (1000).</summary>
        public static bool powerIteration(in floatMxN A, ref floatN v, ref floatN w,
                                          out float lambda, float tol)
            => powerIteration(in A, ref v, ref w, out lambda, tol, 1000);

        /// <summary>powerIteration with default tol (Consts.floatZeroTreshold) and maxIter (1000).</summary>
        public static bool powerIteration(in floatMxN A, ref floatN v, ref floatN w,
                                          out float lambda)
            => powerIteration(in A, ref v, ref w, out lambda, Consts.floatZeroTreshold, 1000);

        /// <summary>
        /// Full symmetric eigendecomposition via classical two-sided (cyclic) Jacobi iteration.
        /// Computes A = V * diag(eigenvalues) * V^T where V is orthonormal.
        ///
        /// On input: A must be square and symmetric. On output: A is DESTROYED (driven to
        /// approximately diagonal); eigenvalues (length n) holds the eigenvalues;
        /// V (n x n) holds the eigenvectors as columns (V is overwritten and initialized
        /// to the identity internally).
        ///
        /// Eigenvalues are sorted in DESCENDING ORDER BY VALUE (not magnitude), so
        /// lambda[0] &gt;= lambda[1] &gt;= ... &gt;= lambda[n-1]. This means negative eigenvalues
        /// appear last. The corresponding eigenvector columns of V are reordered to match.
        ///
        /// Returns true if convergence was reached within maxSweeps (a sweep with zero
        /// Jacobi rotations), false if the sweep limit was exhausted.
        ///
        /// Notes:
        ///   - Works for any real symmetric matrix including indefinite ones; eigenvalues
        ///     are always real.
        ///   - For positive semi-definite matrices the result matches SVD up to column
        ///     sign differences.
        ///   - Does not allocate.
        /// </summary>
        public static bool eigenDecomposition(ref floatMxN A, ref floatN eigenvalues,
                                              ref floatMxN V, int maxSweeps, float eps)
        {
            if (!A.IsSquare)
                throw new ArgumentException("Eigen.eigenDecomposition: A must be square");

            int n = A.N_Cols;

            if (eigenvalues.N != n)
                throw new ArgumentException("Eigen.eigenDecomposition: eigenvalues.N must equal A dimension");

            if (!V.IsSquare || V.M_Rows != n)
                throw new ArgumentException("Eigen.eigenDecomposition: V must be square with side equal to A dimension");

            if (maxSweeps < 1)
                throw new ArgumentException("Eigen.eigenDecomposition: maxSweeps must be >= 1");

            if (eps <= (float)0)
                throw new ArgumentException("Eigen.eigenDecomposition: eps must be > 0");

            // Symmetry guard: check that A is symmetric within eps-relative tolerance
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    float aij = A[i, j];
                    float aji = A[j, i];
                    float diff = math.abs(aij - aji);
                    float relScale = (float)1 + math.abs(aij) + math.abs(aji);
                    if (diff > eps * relScale)
                        throw new ArgumentException("Eigen.eigenDecomposition: Matrix must be symmetric");
                }
            }

            if (n == 0)
                return true;

            // Initialize V to identity
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    V[i, j] = (i == j) ? (float)1 : (float)0;

            bool converged = false;

            for (int sweep = 0; sweep < maxSweeps; sweep++) {

                int rotations = 0;

                for (int p = 0; p < n - 1; p++) {
                    for (int q = p + 1; q < n; q++) {

                        float apq = A[p, q];

                        // Skip exact zeros
                        if (apq == (float)0)
                            continue;

                        // Skip when off-diagonal is negligible relative to the diagonal
                        if (math.abs(apq) <= eps * (float)0.5 * (math.abs(A[p, p]) + math.abs(A[q, q])))
                            continue;

                        // Compute rotation angle: theta = (A[q,q] - A[p,p]) / (2 * A[p,q])
                        float theta = (A[q, q] - A[p, p]) / ((float)2 * apq);

                        // sign(theta) with 0 -> +1
                        float signTheta = theta >= (float)0 ? (float)1 : (float)(-1);
                        float absTheta = math.abs(theta);

                        float t;
                        if (absTheta > (float)1) {
                            // Factor out |theta| to avoid theta*theta overflow
                            float inv = (float)1 / theta;
                            t = signTheta / (absTheta * ((float)1 + math.sqrt((float)1 + inv * inv)));
                        } else {
                            // |theta| <= 1 -> theta*theta <= 1, safe
                            t = signTheta / (absTheta + math.sqrt((float)1 + theta * theta));
                        }

                        float c = (float)1 / math.sqrt((float)1 + t * t);
                        float s = t * c;

                        // Apply symmetric rotation to A
                        float app = A[p, p];
                        float aqq = A[q, q];
                        A[p, p] = app - t * apq;
                        A[q, q] = aqq + t * apq;
                        A[p, q] = (float)0;
                        A[q, p] = (float)0;

                        for (int i = 0; i < n; i++) {
                            if (i == p || i == q)
                                continue;
                            float aip = A[i, p];
                            float aiq = A[i, q];
                            float newAip = c * aip - s * aiq;
                            float newAiq = s * aip + c * aiq;
                            A[i, p] = newAip;
                            A[p, i] = newAip;
                            A[i, q] = newAiq;
                            A[q, i] = newAiq;
                        }

                        // Rotate columns p and q of V
                        for (int i = 0; i < n; i++) {
                            float vip = V[i, p];
                            float viq = V[i, q];
                            V[i, p] = c * vip - s * viq;
                            V[i, q] = s * vip + c * viq;
                        }

                        rotations++;
                    }
                }

                if (rotations == 0) {
                    converged = true;
                    break;
                }
            }

            // Extract diagonal of (now approximately diagonal) A into eigenvalues
            for (int i = 0; i < n; i++)
                eigenvalues[i] = A[i, i];

            // Selection sort: descending by value (not magnitude)
            for (int j = 0; j < n; j++) {
                int maxIdx = j;
                float maxVal = eigenvalues[j];

                for (int k = j + 1; k < n; k++) {
                    if (eigenvalues[k] > maxVal) {
                        maxIdx = k;
                        maxVal = eigenvalues[k];
                    }
                }

                if (maxIdx != j) {
                    // Swap eigenvalues
                    float tmp = eigenvalues[j];
                    eigenvalues[j] = eigenvalues[maxIdx];
                    eigenvalues[maxIdx] = tmp;

                    // Swap corresponding columns of V only (A's diagonal traveled into eigenvalues)
                    SwapOP.Columns(ref V, j, maxIdx);
                }
            }

            return converged;
        }

        /// <summary>eigenDecomposition with default eps (Consts.floatZeroTreshold).</summary>
        public static bool eigenDecomposition(ref floatMxN A, ref floatN eigenvalues,
                                              ref floatMxN V, int maxSweeps)
            => eigenDecomposition(ref A, ref eigenvalues, ref V, maxSweeps, Consts.floatZeroTreshold);

        /// <summary>eigenDecomposition with default maxSweeps (30) and eps (Consts.floatZeroTreshold).</summary>
        public static bool eigenDecomposition(ref floatMxN A, ref floatN eigenvalues,
                                              ref floatMxN V)
            => eigenDecomposition(ref A, ref eigenvalues, ref V, 30, Consts.floatZeroTreshold);

        // copysign: magnitude of a with the sign of b (b >= 0 -> +|a|). EISPACK SIGN(a,b).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float copysign(float a, float b) => b >= (float)0 ? math.abs(a) : -math.abs(a);

        /// <summary>
        /// All eigenvalues of a GENERAL (non-symmetric) real square matrix, via the QR algorithm:
        /// reduction to upper Hessenberg form (elimination with partial pivoting) followed by the
        /// Francis double-shift QR iteration to the real Schur form (EISPACK elmhes + hqr). Real
        /// arithmetic only — complex-conjugate eigenvalue pairs are produced from the 2x2 Schur
        /// blocks, so NO complex number type is needed.
        ///
        /// Unlike eigenDecomposition (symmetric-only Jacobi) and powerIteration (dominant pair only),
        /// this handles arbitrary real matrices including those with complex eigenvalues (e.g.
        /// rotations). It returns eigenVALUES only (no eigenvectors).
        ///
        /// On input A must be square; A is DESTROYED (overwritten during reduction/iteration).
        /// On output eigenvaluesReal[i] / eigenvaluesImag[i] are the real and imaginary parts of the
        /// i-th eigenvalue. Results are sorted by (real, then imaginary) DESCENDING, so a conjugate
        /// pair a±bi appears as (a,+b) immediately before (a,-b). Read the outputs only when the
        /// method returns true.
        ///
        /// Returns true if every eigenvalue converged within maxIterPerRoot iterations; false if the
        /// iteration limit was hit (outputs then undefined). Does not allocate.
        /// </summary>
        public static bool eigenvaluesQR(ref floatMxN A, ref floatN eigenvaluesReal,
                                         ref floatN eigenvaluesImag, int maxIterPerRoot)
        {
            if (!A.IsSquare)
                throw new ArgumentException("Eigen.eigenvaluesQR: A must be square");

            int n = A.N_Cols;

            if (eigenvaluesReal.N != n)
                throw new ArgumentException("Eigen.eigenvaluesQR: eigenvaluesReal.N must equal A dimension");

            if (eigenvaluesImag.N != n)
                throw new ArgumentException("Eigen.eigenvaluesQR: eigenvaluesImag.N must equal A dimension");

            if (maxIterPerRoot < 1)
                throw new ArgumentException("Eigen.eigenvaluesQR: maxIterPerRoot must be >= 1");

            if (n == 0)
                return true;

            // ---- Step 1: reduce A to upper Hessenberg form (elmhes: Gaussian elimination with
            //      partial pivoting via similarity transforms; preserves eigenvalues). ----
            for (int m = 1; m < n - 1; m++)
            {
                // pivot: largest |A[j, m-1]| over rows j >= m.
                float x = (float)0;
                int piv = m;
                for (int j = m; j < n; j++)
                {
                    if (math.abs(A[j, m - 1]) > math.abs(x))
                    {
                        x = A[j, m - 1];
                        piv = j;
                    }
                }

                // interchange rows and columns piv <-> m (a similarity transform).
                if (piv != m)
                {
                    for (int j = m - 1; j < n; j++)
                    {
                        float tmp = A[piv, j]; A[piv, j] = A[m, j]; A[m, j] = tmp;
                    }
                    for (int j = 0; j < n; j++)
                    {
                        float tmp = A[j, piv]; A[j, piv] = A[j, m]; A[j, m] = tmp;
                    }
                }

                // eliminate below the subdiagonal in column m-1.
                if (x != (float)0)
                {
                    for (int i = m + 1; i < n; i++)
                    {
                        float y = A[i, m - 1];
                        if (y != (float)0)
                        {
                            y /= x;
                            A[i, m - 1] = y;                          // store multiplier (cleared below)
                            for (int j = m; j < n; j++)
                                A[i, j] -= y * A[m, j];
                            for (int j = 0; j < n; j++)
                                A[j, m] += y * A[j, i];
                        }
                    }
                }
            }

            // clear the stored multipliers below the subdiagonal -> clean upper Hessenberg H in A.
            for (int i = 2; i < n; i++)
                for (int j = 0; j < i - 1; j++)
                    A[i, j] = (float)0;

            // ---- Step 2: Francis double-shift QR on the Hessenberg matrix (hqr). ----
            float anorm = (float)0;
            for (int i = 0; i < n; i++)
                for (int j = math.max(i - 1, 0); j < n; j++)
                    anorm += math.abs(A[i, j]);

            int nn = n - 1;     // index of the current bottom-right active row/col
            float t = (float)0;

            while (nn >= 0)
            {
                int its = 0;
                int l;
                do
                {
                    // look for a single negligible subdiagonal element to split off.
                    for (l = nn; l >= 1; l--)
                    {
                        float s0 = math.abs(A[l - 1, l - 1]) + math.abs(A[l, l]);
                        if (s0 == (float)0) s0 = anorm;
                        if (math.abs(A[l, l - 1]) + s0 == s0)
                        {
                            A[l, l - 1] = (float)0;
                            break;
                        }
                    }
                    if (l < 0) l = 0;

                    float x = A[nn, nn];

                    if (l == nn)
                    {
                        // one real root.
                        eigenvaluesReal[nn] = x + t;
                        eigenvaluesImag[nn] = (float)0;
                        nn--;
                    }
                    else
                    {
                        float y = A[nn - 1, nn - 1];
                        float w = A[nn, nn - 1] * A[nn - 1, nn];

                        if (l == nn - 1)
                        {
                            // two roots from the trailing 2x2 block.
                            float p = (float)0.5 * (y - x);
                            float q = p * p + w;
                            float z = math.sqrt(math.abs(q));
                            x += t;
                            if (q >= (float)0)
                            {
                                // real pair.
                                z = p + copysign(z, p);
                                eigenvaluesReal[nn - 1] = x + z;
                                eigenvaluesReal[nn] = (z != (float)0) ? (x - w / z) : (x + z);
                                eigenvaluesImag[nn - 1] = (float)0;
                                eigenvaluesImag[nn] = (float)0;
                            }
                            else
                            {
                                // complex-conjugate pair a +/- bi.
                                eigenvaluesReal[nn - 1] = x + p;
                                eigenvaluesReal[nn] = x + p;
                                eigenvaluesImag[nn - 1] = z;
                                eigenvaluesImag[nn] = -z;
                            }
                            nn -= 2;
                        }
                        else
                        {
                            // no root yet: perform a double-shift QR sweep.
                            if (its >= maxIterPerRoot)
                                return false;   // not converged

                            if (its == 10 || its == 20)
                            {
                                // exceptional shift to break a cycle.
                                t += x;
                                for (int i = 0; i <= nn; i++)
                                    A[i, i] -= x;
                                float s1 = math.abs(A[nn, nn - 1]) + math.abs(A[nn - 1, nn - 2]);
                                y = x = (float)0.75 * s1;
                                w = (float)(-0.4375) * s1 * s1;
                            }
                            its++;

                            // find two consecutive negligible subdiagonals to start the sweep.
                            float p = (float)0, q = (float)0, r = (float)0;
                            int m;
                            for (m = nn - 2; m >= l; m--)
                            {
                                float z = A[m, m];
                                float rr = x - z;
                                float ss = y - z;
                                p = (rr * ss - w) / A[m + 1, m] + A[m, m + 1];
                                q = A[m + 1, m + 1] - z - rr - ss;
                                r = A[m + 2, m + 1];
                                float s2 = math.abs(p) + math.abs(q) + math.abs(r);
                                // guard the normalization (matches the guarded analog in the QR sweep
                                // below): if p,q,r are all exactly zero, leave them zero rather than
                                // dividing 0/0 -> NaN, which would poison the convergence test.
                                if (s2 != (float)0) { p /= s2; q /= s2; r /= s2; }
                                if (m == l) break;
                                float u = math.abs(A[m, m - 1]) * (math.abs(q) + math.abs(r));
                                float v = math.abs(p) * (math.abs(A[m - 1, m - 1]) + math.abs(z) + math.abs(A[m + 1, m + 1]));
                                if (u + v == v) break;
                            }

                            for (int i = m + 2; i <= nn; i++)
                            {
                                A[i, i - 2] = (float)0;
                                if (i != m + 2) A[i, i - 3] = (float)0;
                            }

                            // the double QR step over rows/cols m..nn.
                            for (int k = m; k <= nn - 1; k++)
                            {
                                if (k != m)
                                {
                                    p = A[k, k - 1];
                                    q = A[k + 1, k - 1];
                                    r = (float)0;
                                    if (k != nn - 1) r = A[k + 2, k - 1];
                                    x = math.abs(p) + math.abs(q) + math.abs(r);
                                    if (x != (float)0)
                                    {
                                        p /= x; q /= x; r /= x;
                                    }
                                }

                                float s = copysign(math.sqrt(p * p + q * q + r * r), p);
                                if (s != (float)0)
                                {
                                    if (k == m)
                                    {
                                        if (l != m)
                                            A[k, k - 1] = -A[k, k - 1];
                                    }
                                    else
                                    {
                                        A[k, k - 1] = -s * x;
                                    }
                                    p += s;
                                    float xx = p / s;
                                    float yy = q / s;
                                    float zz = r / s;
                                    q /= p;
                                    r /= p;

                                    // row modification.
                                    for (int j = k; j <= nn; j++)
                                    {
                                        p = A[k, j] + q * A[k + 1, j];
                                        if (k != nn - 1)
                                        {
                                            p += r * A[k + 2, j];
                                            A[k + 2, j] -= p * zz;
                                        }
                                        A[k + 1, j] -= p * yy;
                                        A[k, j] -= p * xx;
                                    }

                                    int mmin = nn < k + 3 ? nn : k + 3;
                                    // column modification.
                                    for (int i = l; i <= mmin; i++)
                                    {
                                        p = xx * A[i, k] + yy * A[i, k + 1];
                                        if (k != nn - 1)
                                        {
                                            p += zz * A[i, k + 2];
                                            A[i, k + 2] -= p * r;
                                        }
                                        A[i, k + 1] -= p * q;
                                        A[i, k] -= p;
                                    }
                                }
                            }
                        }
                    }
                } while (l < nn - 1);
            }

            // ---- sort by (real, then imaginary) descending; keep re/im paired. ----
            for (int a = 0; a < n - 1; a++)
            {
                int best = a;
                for (int b = a + 1; b < n; b++)
                {
                    if (eigenvaluesReal[b] > eigenvaluesReal[best] ||
                        (eigenvaluesReal[b] == eigenvaluesReal[best] && eigenvaluesImag[b] > eigenvaluesImag[best]))
                        best = b;
                }
                if (best != a)
                {
                    float tr = eigenvaluesReal[a]; eigenvaluesReal[a] = eigenvaluesReal[best]; eigenvaluesReal[best] = tr;
                    float ti = eigenvaluesImag[a]; eigenvaluesImag[a] = eigenvaluesImag[best]; eigenvaluesImag[best] = ti;
                }
            }

            return true;
        }

        /// <summary>eigenvaluesQR with default maxIterPerRoot (30, the EISPACK hqr limit).</summary>
        public static bool eigenvaluesQR(ref floatMxN A, ref floatN eigenvaluesReal,
                                         ref floatN eigenvaluesImag)
            => eigenvaluesQR(ref A, ref eigenvaluesReal, ref eigenvaluesImag, 30);
    }
}
