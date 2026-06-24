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
        public static bool powerIteration(in fProxyMxN A, ref fProxyN v, ref fProxyN w,
                                          out fProxy lambda, fProxy tol, int maxIter)
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

            if (tol <= (fProxy)0)
                throw new ArgumentException("Eigen.powerIteration: tol must be > 0");

            int n = A.N_Cols;

            // Seed v deterministically if the caller supplied the zero vector
            fProxy vNormSq = (fProxy)0;
            for (int i = 0; i < n; i++)
                vNormSq += v[i] * v[i];

            if (vNormSq == (fProxy)0) {
                for (int i = 0; i < n; i++)
                    v[i] = (fProxy)(1 + (i & 3));
                vNormSq = (fProxy)0;
                for (int i = 0; i < n; i++)
                    vNormSq += v[i] * v[i];
            }

            // Normalize v to unit length
            fProxy vNorm = math.sqrt(vNormSq);
            fProxy invVNorm = (fProxy)1 / vNorm;
            for (int i = 0; i < n; i++)
                v[i] = v[i] * invVNorm;

            lambda = (fProxy)0;

            for (int iter = 0; iter < maxIter; iter++) {

                // Step 1: w = A * v (manual matvec — no allocation)
                for (int i = 0; i < n; i++) {
                    fProxy sum = (fProxy)0;
                    for (int j = 0; j < n; j++)
                        sum += A[i, j] * v[j];
                    w[i] = sum;
                }

                // Step 2: lambda = v . w (Rayleigh quotient; ||v||_2 = 1)
                lambda = (fProxy)0;
                for (int i = 0; i < n; i++)
                    lambda += v[i] * w[i];

                // Step 3: residual r = max_i |w[i] - lambda * v[i]|  (infinity norm)
                fProxy residual = (fProxy)0;
                for (int i = 0; i < n; i++) {
                    fProxy ri = math.abs(w[i] - lambda * v[i]);
                    if (ri > residual)
                        residual = ri;
                }

                // Step 4: convergence check
                fProxy scale = math.abs(lambda);
                if (scale < (fProxy)1)
                    scale = (fProxy)1;
                if (residual <= tol * scale)
                    return true;

                // Step 5: compute ||w||_2; handle exact null-space case
                fProxy nw = (fProxy)0;
                for (int i = 0; i < n; i++)
                    nw += w[i] * w[i];
                nw = math.sqrt(nw);

                if (nw == (fProxy)0) {
                    lambda = (fProxy)0;
                    return true;
                }

                // Step 6: v = w / ||w||
                fProxy invNw = (fProxy)1 / nw;
                for (int i = 0; i < n; i++)
                    v[i] = w[i] * invNw;
            }

            // Post-loop: recompute w = A*v, lambda, residual with final v
            for (int i = 0; i < n; i++) {
                fProxy sum = (fProxy)0;
                for (int j = 0; j < n; j++)
                    sum += A[i, j] * v[j];
                w[i] = sum;
            }

            lambda = (fProxy)0;
            for (int i = 0; i < n; i++)
                lambda += v[i] * w[i];

            fProxy finalResidual = (fProxy)0;
            for (int i = 0; i < n; i++) {
                fProxy ri = math.abs(w[i] - lambda * v[i]);
                if (ri > finalResidual)
                    finalResidual = ri;
            }

            fProxy finalScale = math.abs(lambda);
            if (finalScale < (fProxy)1)
                finalScale = (fProxy)1;
            return finalResidual <= tol * finalScale;
        }

        /// <summary>powerIteration with default maxIter (1000).</summary>
        public static bool powerIteration(in fProxyMxN A, ref fProxyN v, ref fProxyN w,
                                          out fProxy lambda, fProxy tol)
            => powerIteration(in A, ref v, ref w, out lambda, tol, 1000);

        /// <summary>powerIteration with default tol (Consts.fProxyZeroTreshold) and maxIter (1000).</summary>
        public static bool powerIteration(in fProxyMxN A, ref fProxyN v, ref fProxyN w,
                                          out fProxy lambda)
            => powerIteration(in A, ref v, ref w, out lambda, Consts.fProxyZeroTreshold, 1000);

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
        public static bool eigenDecomposition(ref fProxyMxN A, ref fProxyN eigenvalues,
                                              ref fProxyMxN V, int maxSweeps, fProxy eps)
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

            if (eps <= (fProxy)0)
                throw new ArgumentException("Eigen.eigenDecomposition: eps must be > 0");

            // Symmetry guard: check that A is symmetric within eps-relative tolerance
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    fProxy aij = A[i, j];
                    fProxy aji = A[j, i];
                    fProxy diff = math.abs(aij - aji);
                    fProxy relScale = (fProxy)1 + math.abs(aij) + math.abs(aji);
                    if (diff > eps * relScale)
                        throw new ArgumentException("Eigen.eigenDecomposition: Matrix must be symmetric");
                }
            }

            if (n == 0)
                return true;

            // Initialize V to identity
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    V[i, j] = (i == j) ? (fProxy)1 : (fProxy)0;

            bool converged = false;

            for (int sweep = 0; sweep < maxSweeps; sweep++) {

                int rotations = 0;

                for (int p = 0; p < n - 1; p++) {
                    for (int q = p + 1; q < n; q++) {

                        fProxy apq = A[p, q];

                        // Skip exact zeros
                        if (apq == (fProxy)0)
                            continue;

                        // Skip when off-diagonal is negligible relative to the diagonal
                        if (math.abs(apq) <= eps * (fProxy)0.5 * (math.abs(A[p, p]) + math.abs(A[q, q])))
                            continue;

                        // Compute rotation angle: theta = (A[q,q] - A[p,p]) / (2 * A[p,q])
                        fProxy theta = (A[q, q] - A[p, p]) / ((fProxy)2 * apq);

                        // sign(theta) with 0 -> +1
                        fProxy signTheta = theta >= (fProxy)0 ? (fProxy)1 : (fProxy)(-1);
                        fProxy absTheta = math.abs(theta);

                        fProxy t;
                        if (absTheta > (fProxy)1) {
                            // Factor out |theta| to avoid theta*theta overflow
                            fProxy inv = (fProxy)1 / theta;
                            t = signTheta / (absTheta * ((fProxy)1 + math.sqrt((fProxy)1 + inv * inv)));
                        } else {
                            // |theta| <= 1 -> theta*theta <= 1, safe
                            t = signTheta / (absTheta + math.sqrt((fProxy)1 + theta * theta));
                        }

                        fProxy c = (fProxy)1 / math.sqrt((fProxy)1 + t * t);
                        fProxy s = t * c;

                        // Apply symmetric rotation to A
                        fProxy app = A[p, p];
                        fProxy aqq = A[q, q];
                        A[p, p] = app - t * apq;
                        A[q, q] = aqq + t * apq;
                        A[p, q] = (fProxy)0;
                        A[q, p] = (fProxy)0;

                        for (int i = 0; i < n; i++) {
                            if (i == p || i == q)
                                continue;
                            fProxy aip = A[i, p];
                            fProxy aiq = A[i, q];
                            fProxy newAip = c * aip - s * aiq;
                            fProxy newAiq = s * aip + c * aiq;
                            A[i, p] = newAip;
                            A[p, i] = newAip;
                            A[i, q] = newAiq;
                            A[q, i] = newAiq;
                        }

                        // Rotate columns p and q of V
                        for (int i = 0; i < n; i++) {
                            fProxy vip = V[i, p];
                            fProxy viq = V[i, q];
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
                fProxy maxVal = eigenvalues[j];

                for (int k = j + 1; k < n; k++) {
                    if (eigenvalues[k] > maxVal) {
                        maxIdx = k;
                        maxVal = eigenvalues[k];
                    }
                }

                if (maxIdx != j) {
                    // Swap eigenvalues
                    fProxy tmp = eigenvalues[j];
                    eigenvalues[j] = eigenvalues[maxIdx];
                    eigenvalues[maxIdx] = tmp;

                    // Swap corresponding columns of V only (A's diagonal traveled into eigenvalues)
                    SwapOP.Columns(ref V, j, maxIdx);
                }
            }

            return converged;
        }

        /// <summary>eigenDecomposition with default eps (Consts.fProxyZeroTreshold).</summary>
        public static bool eigenDecomposition(ref fProxyMxN A, ref fProxyN eigenvalues,
                                              ref fProxyMxN V, int maxSweeps)
            => eigenDecomposition(ref A, ref eigenvalues, ref V, maxSweeps, Consts.fProxyZeroTreshold);

        /// <summary>eigenDecomposition with default maxSweeps (30) and eps (Consts.fProxyZeroTreshold).</summary>
        public static bool eigenDecomposition(ref fProxyMxN A, ref fProxyN eigenvalues,
                                              ref fProxyMxN V)
            => eigenDecomposition(ref A, ref eigenvalues, ref V, 30, Consts.fProxyZeroTreshold);

        // copysign: magnitude of a with the sign of b (b >= 0 -> +|a|). EISPACK SIGN(a,b).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static fProxy copysign(fProxy a, fProxy b) => b >= (fProxy)0 ? math.abs(a) : -math.abs(a);

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
        public static bool eigenvaluesQR(ref fProxyMxN A, ref fProxyN eigenvaluesReal,
                                         ref fProxyN eigenvaluesImag, int maxIterPerRoot)
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
                fProxy x = (fProxy)0;
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
                        fProxy tmp = A[piv, j]; A[piv, j] = A[m, j]; A[m, j] = tmp;
                    }
                    for (int j = 0; j < n; j++)
                    {
                        fProxy tmp = A[j, piv]; A[j, piv] = A[j, m]; A[j, m] = tmp;
                    }
                }

                // eliminate below the subdiagonal in column m-1.
                if (x != (fProxy)0)
                {
                    for (int i = m + 1; i < n; i++)
                    {
                        fProxy y = A[i, m - 1];
                        if (y != (fProxy)0)
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
                    A[i, j] = (fProxy)0;

            // ---- Step 2: Francis double-shift QR on the Hessenberg matrix (hqr). ----
            fProxy anorm = (fProxy)0;
            for (int i = 0; i < n; i++)
                for (int j = math.max(i - 1, 0); j < n; j++)
                    anorm += math.abs(A[i, j]);

            int nn = n - 1;     // index of the current bottom-right active row/col
            fProxy t = (fProxy)0;

            while (nn >= 0)
            {
                int its = 0;
                int l;
                do
                {
                    // look for a single negligible subdiagonal element to split off.
                    for (l = nn; l >= 1; l--)
                    {
                        fProxy s0 = math.abs(A[l - 1, l - 1]) + math.abs(A[l, l]);
                        if (s0 == (fProxy)0) s0 = anorm;
                        if (math.abs(A[l, l - 1]) + s0 == s0)
                        {
                            A[l, l - 1] = (fProxy)0;
                            break;
                        }
                    }
                    if (l < 0) l = 0;

                    fProxy x = A[nn, nn];

                    if (l == nn)
                    {
                        // one real root.
                        eigenvaluesReal[nn] = x + t;
                        eigenvaluesImag[nn] = (fProxy)0;
                        nn--;
                    }
                    else
                    {
                        fProxy y = A[nn - 1, nn - 1];
                        fProxy w = A[nn, nn - 1] * A[nn - 1, nn];

                        if (l == nn - 1)
                        {
                            // two roots from the trailing 2x2 block.
                            fProxy p = (fProxy)0.5 * (y - x);
                            fProxy q = p * p + w;
                            fProxy z = math.sqrt(math.abs(q));
                            x += t;
                            if (q >= (fProxy)0)
                            {
                                // real pair.
                                z = p + copysign(z, p);
                                eigenvaluesReal[nn - 1] = x + z;
                                eigenvaluesReal[nn] = (z != (fProxy)0) ? (x - w / z) : (x + z);
                                eigenvaluesImag[nn - 1] = (fProxy)0;
                                eigenvaluesImag[nn] = (fProxy)0;
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
                                fProxy s1 = math.abs(A[nn, nn - 1]) + math.abs(A[nn - 1, nn - 2]);
                                y = x = (fProxy)0.75 * s1;
                                w = (fProxy)(-0.4375) * s1 * s1;
                            }
                            its++;

                            // find two consecutive negligible subdiagonals to start the sweep.
                            fProxy p = (fProxy)0, q = (fProxy)0, r = (fProxy)0;
                            int m;
                            for (m = nn - 2; m >= l; m--)
                            {
                                fProxy z = A[m, m];
                                fProxy rr = x - z;
                                fProxy ss = y - z;
                                p = (rr * ss - w) / A[m + 1, m] + A[m, m + 1];
                                q = A[m + 1, m + 1] - z - rr - ss;
                                r = A[m + 2, m + 1];
                                fProxy s2 = math.abs(p) + math.abs(q) + math.abs(r);
                                // guard the normalization (matches the guarded analog in the QR sweep
                                // below): if p,q,r are all exactly zero, leave them zero rather than
                                // dividing 0/0 -> NaN, which would poison the convergence test.
                                if (s2 != (fProxy)0) { p /= s2; q /= s2; r /= s2; }
                                if (m == l) break;
                                fProxy u = math.abs(A[m, m - 1]) * (math.abs(q) + math.abs(r));
                                fProxy v = math.abs(p) * (math.abs(A[m - 1, m - 1]) + math.abs(z) + math.abs(A[m + 1, m + 1]));
                                if (u + v == v) break;
                            }

                            for (int i = m + 2; i <= nn; i++)
                            {
                                A[i, i - 2] = (fProxy)0;
                                if (i != m + 2) A[i, i - 3] = (fProxy)0;
                            }

                            // the double QR step over rows/cols m..nn.
                            for (int k = m; k <= nn - 1; k++)
                            {
                                if (k != m)
                                {
                                    p = A[k, k - 1];
                                    q = A[k + 1, k - 1];
                                    r = (fProxy)0;
                                    if (k != nn - 1) r = A[k + 2, k - 1];
                                    x = math.abs(p) + math.abs(q) + math.abs(r);
                                    if (x != (fProxy)0)
                                    {
                                        p /= x; q /= x; r /= x;
                                    }
                                }

                                fProxy s = copysign(math.sqrt(p * p + q * q + r * r), p);
                                if (s != (fProxy)0)
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
                                    fProxy xx = p / s;
                                    fProxy yy = q / s;
                                    fProxy zz = r / s;
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
                    fProxy tr = eigenvaluesReal[a]; eigenvaluesReal[a] = eigenvaluesReal[best]; eigenvaluesReal[best] = tr;
                    fProxy ti = eigenvaluesImag[a]; eigenvaluesImag[a] = eigenvaluesImag[best]; eigenvaluesImag[best] = ti;
                }
            }

            return true;
        }

        /// <summary>eigenvaluesQR with default maxIterPerRoot (30, the EISPACK hqr limit).</summary>
        public static bool eigenvaluesQR(ref fProxyMxN A, ref fProxyN eigenvaluesReal,
                                         ref fProxyN eigenvaluesImag)
            => eigenvaluesQR(ref A, ref eigenvaluesReal, ref eigenvaluesImag, 30);
    }
}
