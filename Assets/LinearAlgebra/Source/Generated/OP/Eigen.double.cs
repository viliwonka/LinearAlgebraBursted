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
        public static bool powerIteration(in doubleMxN A, ref doubleN v, ref doubleN w,
                                          out double lambda, double tol, int maxIter)
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

            if (tol <= (double)0)
                throw new ArgumentException("Eigen.powerIteration: tol must be > 0");

            int n = A.N_Cols;

            // Seed v deterministically if the caller supplied the zero vector
            double vNormSq = (double)0;
            for (int i = 0; i < n; i++)
                vNormSq += v[i] * v[i];

            if (vNormSq == (double)0) {
                for (int i = 0; i < n; i++)
                    v[i] = (double)(1 + (i & 3));
                vNormSq = (double)0;
                for (int i = 0; i < n; i++)
                    vNormSq += v[i] * v[i];
            }

            // Normalize v to unit length
            double vNorm = math.sqrt(vNormSq);
            double invVNorm = (double)1 / vNorm;
            for (int i = 0; i < n; i++)
                v[i] = v[i] * invVNorm;

            lambda = (double)0;

            for (int iter = 0; iter < maxIter; iter++) {

                // Step 1: w = A * v (manual matvec — no allocation)
                for (int i = 0; i < n; i++) {
                    double sum = (double)0;
                    for (int j = 0; j < n; j++)
                        sum += A[i, j] * v[j];
                    w[i] = sum;
                }

                // Step 2: lambda = v . w (Rayleigh quotient; ||v||_2 = 1)
                lambda = (double)0;
                for (int i = 0; i < n; i++)
                    lambda += v[i] * w[i];

                // Step 3: residual r = max_i |w[i] - lambda * v[i]|  (infinity norm)
                double residual = (double)0;
                for (int i = 0; i < n; i++) {
                    double ri = math.abs(w[i] - lambda * v[i]);
                    if (ri > residual)
                        residual = ri;
                }

                // Step 4: convergence check
                double scale = math.abs(lambda);
                if (scale < (double)1)
                    scale = (double)1;
                if (residual <= tol * scale)
                    return true;

                // Step 5: compute ||w||_2; handle exact null-space case
                double nw = (double)0;
                for (int i = 0; i < n; i++)
                    nw += w[i] * w[i];
                nw = math.sqrt(nw);

                if (nw == (double)0) {
                    lambda = (double)0;
                    return true;
                }

                // Step 6: v = w / ||w||
                double invNw = (double)1 / nw;
                for (int i = 0; i < n; i++)
                    v[i] = w[i] * invNw;
            }

            // Post-loop: recompute w = A*v, lambda, residual with final v
            for (int i = 0; i < n; i++) {
                double sum = (double)0;
                for (int j = 0; j < n; j++)
                    sum += A[i, j] * v[j];
                w[i] = sum;
            }

            lambda = (double)0;
            for (int i = 0; i < n; i++)
                lambda += v[i] * w[i];

            double finalResidual = (double)0;
            for (int i = 0; i < n; i++) {
                double ri = math.abs(w[i] - lambda * v[i]);
                if (ri > finalResidual)
                    finalResidual = ri;
            }

            double finalScale = math.abs(lambda);
            if (finalScale < (double)1)
                finalScale = (double)1;
            return finalResidual <= tol * finalScale;
        }

        /// <summary>powerIteration with default maxIter (1000).</summary>
        public static bool powerIteration(in doubleMxN A, ref doubleN v, ref doubleN w,
                                          out double lambda, double tol)
            => powerIteration(in A, ref v, ref w, out lambda, tol, 1000);

        /// <summary>powerIteration with default tol (Consts.doubleZeroTreshold) and maxIter (1000).</summary>
        public static bool powerIteration(in doubleMxN A, ref doubleN v, ref doubleN w,
                                          out double lambda)
            => powerIteration(in A, ref v, ref w, out lambda, Consts.doubleZeroTreshold, 1000);

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
        public static bool eigenDecomposition(ref doubleMxN A, ref doubleN eigenvalues,
                                              ref doubleMxN V, int maxSweeps, double eps)
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

            if (eps <= (double)0)
                throw new ArgumentException("Eigen.eigenDecomposition: eps must be > 0");

            // Symmetry guard: check that A is symmetric within eps-relative tolerance
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    double aij = A[i, j];
                    double aji = A[j, i];
                    double diff = math.abs(aij - aji);
                    double relScale = (double)1 + math.abs(aij) + math.abs(aji);
                    if (diff > eps * relScale)
                        throw new ArgumentException("Eigen.eigenDecomposition: Matrix must be symmetric");
                }
            }

            if (n == 0)
                return true;

            // Initialize V to identity
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    V[i, j] = (i == j) ? (double)1 : (double)0;

            bool converged = false;

            for (int sweep = 0; sweep < maxSweeps; sweep++) {

                int rotations = 0;

                for (int p = 0; p < n - 1; p++) {
                    for (int q = p + 1; q < n; q++) {

                        double apq = A[p, q];

                        // Skip exact zeros
                        if (apq == (double)0)
                            continue;

                        // Skip when off-diagonal is negligible relative to the diagonal
                        if (math.abs(apq) <= eps * (double)0.5 * (math.abs(A[p, p]) + math.abs(A[q, q])))
                            continue;

                        // Compute rotation angle: theta = (A[q,q] - A[p,p]) / (2 * A[p,q])
                        double theta = (A[q, q] - A[p, p]) / ((double)2 * apq);

                        // sign(theta) with 0 -> +1
                        double signTheta = theta >= (double)0 ? (double)1 : (double)(-1);
                        double absTheta = math.abs(theta);

                        double t;
                        if (absTheta > (double)1) {
                            // Factor out |theta| to avoid theta*theta overflow
                            double inv = (double)1 / theta;
                            t = signTheta / (absTheta * ((double)1 + math.sqrt((double)1 + inv * inv)));
                        } else {
                            // |theta| <= 1 -> theta*theta <= 1, safe
                            t = signTheta / (absTheta + math.sqrt((double)1 + theta * theta));
                        }

                        double c = (double)1 / math.sqrt((double)1 + t * t);
                        double s = t * c;

                        // Apply symmetric rotation to A
                        double app = A[p, p];
                        double aqq = A[q, q];
                        A[p, p] = app - t * apq;
                        A[q, q] = aqq + t * apq;
                        A[p, q] = (double)0;
                        A[q, p] = (double)0;

                        for (int i = 0; i < n; i++) {
                            if (i == p || i == q)
                                continue;
                            double aip = A[i, p];
                            double aiq = A[i, q];
                            double newAip = c * aip - s * aiq;
                            double newAiq = s * aip + c * aiq;
                            A[i, p] = newAip;
                            A[p, i] = newAip;
                            A[i, q] = newAiq;
                            A[q, i] = newAiq;
                        }

                        // Rotate columns p and q of V
                        for (int i = 0; i < n; i++) {
                            double vip = V[i, p];
                            double viq = V[i, q];
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
                double maxVal = eigenvalues[j];

                for (int k = j + 1; k < n; k++) {
                    if (eigenvalues[k] > maxVal) {
                        maxIdx = k;
                        maxVal = eigenvalues[k];
                    }
                }

                if (maxIdx != j) {
                    // Swap eigenvalues
                    double tmp = eigenvalues[j];
                    eigenvalues[j] = eigenvalues[maxIdx];
                    eigenvalues[maxIdx] = tmp;

                    // Swap corresponding columns of V only (A's diagonal traveled into eigenvalues)
                    SwapOP.Columns(ref V, j, maxIdx);
                }
            }

            return converged;
        }

        /// <summary>eigenDecomposition with default eps (Consts.doubleZeroTreshold).</summary>
        public static bool eigenDecomposition(ref doubleMxN A, ref doubleN eigenvalues,
                                              ref doubleMxN V, int maxSweeps)
            => eigenDecomposition(ref A, ref eigenvalues, ref V, maxSweeps, Consts.doubleZeroTreshold);

        /// <summary>eigenDecomposition with default maxSweeps (30) and eps (Consts.doubleZeroTreshold).</summary>
        public static bool eigenDecomposition(ref doubleMxN A, ref doubleN eigenvalues,
                                              ref doubleMxN V)
            => eigenDecomposition(ref A, ref eigenvalues, ref V, 30, Consts.doubleZeroTreshold);

        // copysign: magnitude of a with the sign of b (b >= 0 -> +|a|). EISPACK SIGN(a,b).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double copysign(double a, double b) => b >= (double)0 ? math.abs(a) : -math.abs(a);

        // sqrt(a^2 + b^2) computed so neither square overflows/underflows prematurely.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double pythag(double a, double b)
        {
            double aa = math.abs(a), ab = math.abs(b);
            if (aa > ab) { double r = ab / aa; return aa * math.sqrt((double)1 + r * r); }
            if (ab == (double)0) return (double)0;
            { double r = aa / ab; return ab * math.sqrt((double)1 + r * r); }
        }

        /// <summary>
        /// All eigenVALUES of a SYMMETRIC real matrix, via Householder tridiagonalization followed by
        /// the implicit-shift QL iteration (EISPACK tred1 + tql1, GVL Alg. 8.3.1). Much faster than the
        /// cyclic-Jacobi eigenDecomposition: the O(n^3) reduction is a sequence of gemv + symmetric
        /// rank-2 updates (the rank-2 update is axpy → vectorises), and the QL sweep that follows is
        /// only O(n^2). No eigenvectors (use eigenDecomposition if you need them).
        ///
        /// A must be symmetric (checked within eps-relative tolerance) and is DESTROYED. On output
        /// eigenvalues[i] holds the i-th eigenvalue, sorted DESCENDING. Returns true on convergence;
        /// false if QL hit maxIterPerEig for some eigenvalue (outputs then undefined). Does not allocate
        /// beyond three length-n Temp scratch vectors.
        /// </summary>
        public static bool eigenvaluesSymmetric(ref doubleMxN A, ref doubleN eigenvalues, int maxIterPerEig, double eps)
        {
            if (!A.IsSquare)
                throw new ArgumentException("Eigen.eigenvaluesSymmetric: A must be square");

            int n = A.M_Rows;

            if (eigenvalues.N != n)
                throw new ArgumentException("Eigen.eigenvaluesSymmetric: eigenvalues.N must equal A dimension");

            if (maxIterPerEig < 1)
                throw new ArgumentException("Eigen.eigenvaluesSymmetric: maxIterPerEig must be >= 1");

            if (eps <= (double)0)
                throw new ArgumentException("Eigen.eigenvaluesSymmetric: eps must be > 0");

            // Symmetry guard (same as eigenDecomposition). The reduction reads the full symmetric
            // matrix (the gemv uses whole rows), so both triangles must agree.
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    double aij = A[i, j], aji = A[j, i];
                    double diff = math.abs(aij - aji);
                    double relScale = (double)1 + math.abs(aij) + math.abs(aji);
                    if (diff > eps * relScale)
                        throw new ArgumentException("Eigen.eigenvaluesSymmetric: Matrix must be symmetric");
                }

            if (n == 0) return true;
            if (n == 1) { eigenvalues[0] = A[0, 0]; return true; }

            var eVec = new doubleN(n, Allocator.Temp, false);   // off-diagonal e[i] couples d[i], d[i+1]
            var vVec = new doubleN(n, Allocator.Temp, false);   // Householder vector (entries m0..n-1)
            var pVec = new doubleN(n, Allocator.Temp, false);   // p = beta*A*v, then q = p - K v

            unsafe
            {
                double* ap = A.Data.Ptr;
                double* v  = vVec.Data.Ptr;
                double* p  = pVec.Data.Ptr;

                // ---- Householder tridiagonalization (full symmetric storage, values only) ----
                // The trailing submatrix stays symmetric; column k below the subdiagonal is never read
                // again, so (values-only) we record the subdiagonal in e[k] and skip zeroing it.
                for (int k = 0; k < n - 2; k++)
                {
                    int m0 = k + 1;

                    // x = A[m0.., k]; sigma = ||x[1..]||^2 (entries strictly below the leading one).
                    double sigma = 0;
                    for (int i = m0 + 1; i < n; i++)
                    {
                        double aik = ap[(long)i * n + k];
                        sigma += aik * aik;
                    }
                    double x0 = ap[(long)m0 * n + k];

                    if (sigma == (double)0)
                    {
                        // column already in tridiagonal form
                        eVec[k] = x0;
                        continue;
                    }

                    double xnorm = math.sqrt(x0 * x0 + sigma);
                    double alpha = (x0 >= (double)0) ? -xnorm : xnorm;   // -sign(x0)*||x||

                    // Householder vector v (entries m0..n-1): v[m0] = x0 - alpha, v[i>m0] = x[i].
                    v[m0] = x0 - alpha;
                    for (int i = m0 + 1; i < n; i++) v[i] = ap[(long)i * n + k];

                    double vtv  = v[m0] * v[m0] + sigma;
                    double beta = (double)2 / vtv;

                    // p = beta * A_sub * v   (A_sub = A[m0:n, m0:n], symmetric). Row dots (contiguous).
                    for (int r = m0; r < n; r++)
                    {
                        double* arow = ap + (long)r * n;
                        double s = 0;
                        for (int c = m0; c < n; c++) s += arow[c] * v[c];
                        p[r] = beta * s;
                    }

                    // K = beta * (vᵀp) / 2;  q = p - K v   (overwrite p with q)
                    double vp = 0;
                    for (int i = m0; i < n; i++) vp += v[i] * p[i];
                    double K = beta * vp / (double)2;
                    for (int i = m0; i < n; i++) p[i] -= K * v[i];

                    // Symmetric rank-2 update: A_sub -= v qᵀ + q vᵀ  (two contiguous axpys per row).
                    int len = n - m0;
                    for (int r = m0; r < n; r++)
                    {
                        double* arow = ap + (long)r * n;
                        UnsafeOP.axpy(arow + m0, p + m0, -v[r], len);   // -= v[r] * q
                        UnsafeOP.axpy(arow + m0, v + m0, -p[r], len);   // -= q[r] * v
                    }

                    eVec[k] = alpha;
                }

                // trailing subdiagonal + diagonal
                eVec[n - 2] = ap[(long)(n - 1) * n + (n - 2)];
                eVec[n - 1] = (double)0;
                for (int i = 0; i < n; i++) eigenvalues[i] = ap[(long)i * n + i];
            }

            // ---- implicit-shift QL on the tridiagonal (d = eigenvalues, e), values only ----
            // e[i] couples d[i] and d[i+1]; e[n-1] = 0.
            for (int l = 0; l < n; l++)
            {
                int iter = 0;
                int m;
                do
                {
                    for (m = l; m < n - 1; m++)
                    {
                        double dd = math.abs(eigenvalues[m]) + math.abs(eigenvalues[m + 1]);
                        if (math.abs(eVec[m]) <= eps * dd) break;
                    }
                    if (m != l)
                    {
                        if (iter++ >= maxIterPerEig) { eVec.Dispose(); vVec.Dispose(); pVec.Dispose(); return false; }

                        double g = (eigenvalues[l + 1] - eigenvalues[l]) / ((double)2 * eVec[l]);
                        double r = pythag(g, (double)1);
                        g = eigenvalues[m] - eigenvalues[l] + eVec[l] / (g + copysign(r, g));
                        double s = 1, c = 1, pp = 0;
                        int i;
                        for (i = m - 1; i >= l; i--)
                        {
                            double f = s * eVec[i];
                            double b = c * eVec[i];
                            r = pythag(f, g);
                            eVec[i + 1] = r;
                            if (r == (double)0) { eigenvalues[i + 1] -= pp; eVec[m] = 0; break; }
                            s = f / r; c = g / r;
                            g = eigenvalues[i + 1] - pp;
                            r = (eigenvalues[i] - g) * s + (double)2 * c * b;
                            pp = s * r;
                            eigenvalues[i + 1] = g + pp;
                            g = c * r - b;
                        }
                        if (r == (double)0 && i >= l) continue;
                        eigenvalues[l] -= pp; eVec[l] = g; eVec[m] = 0;
                    }
                } while (m != l);
            }

            eVec.Dispose();
            vVec.Dispose();
            pVec.Dispose();

            // sort descending (selection sort, matching eigenDecomposition)
            for (int j = 0; j < n; j++)
            {
                int maxIdx = j;
                double maxVal = eigenvalues[j];
                for (int k = j + 1; k < n; k++)
                    if (eigenvalues[k] > maxVal) { maxIdx = k; maxVal = eigenvalues[k]; }
                if (maxIdx != j)
                {
                    double tmp = eigenvalues[j];
                    eigenvalues[j] = eigenvalues[maxIdx];
                    eigenvalues[maxIdx] = tmp;
                }
            }

            return true;
        }

        /// <summary>eigenvaluesSymmetric with default maxIterPerEig (30) and eps (Consts.doubleZeroTreshold).</summary>
        public static bool eigenvaluesSymmetric(ref doubleMxN A, ref doubleN eigenvalues)
            => eigenvaluesSymmetric(ref A, ref eigenvalues, 30, Consts.doubleZeroTreshold);

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
        public static bool eigenvaluesQR(ref doubleMxN A, ref doubleN eigenvaluesReal,
                                         ref doubleN eigenvaluesImag, int maxIterPerRoot)
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
                double x = (double)0;
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
                        double tmp = A[piv, j]; A[piv, j] = A[m, j]; A[m, j] = tmp;
                    }
                    for (int j = 0; j < n; j++)
                    {
                        double tmp = A[j, piv]; A[j, piv] = A[j, m]; A[j, m] = tmp;
                    }
                }

                // eliminate below the subdiagonal in column m-1.
                if (x != (double)0)
                {
                    for (int i = m + 1; i < n; i++)
                    {
                        double y = A[i, m - 1];
                        if (y != (double)0)
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
                    A[i, j] = (double)0;

            // ---- Step 2: Francis double-shift QR on the Hessenberg matrix (hqr). ----
            double anorm = (double)0;
            for (int i = 0; i < n; i++)
                for (int j = math.max(i - 1, 0); j < n; j++)
                    anorm += math.abs(A[i, j]);

            int nn = n - 1;     // index of the current bottom-right active row/col
            double t = (double)0;

            while (nn >= 0)
            {
                int its = 0;
                int l;
                do
                {
                    // look for a single negligible subdiagonal element to split off.
                    for (l = nn; l >= 1; l--)
                    {
                        double s0 = math.abs(A[l - 1, l - 1]) + math.abs(A[l, l]);
                        if (s0 == (double)0) s0 = anorm;
                        if (math.abs(A[l, l - 1]) + s0 == s0)
                        {
                            A[l, l - 1] = (double)0;
                            break;
                        }
                    }
                    if (l < 0) l = 0;

                    double x = A[nn, nn];

                    if (l == nn)
                    {
                        // one real root.
                        eigenvaluesReal[nn] = x + t;
                        eigenvaluesImag[nn] = (double)0;
                        nn--;
                    }
                    else
                    {
                        double y = A[nn - 1, nn - 1];
                        double w = A[nn, nn - 1] * A[nn - 1, nn];

                        if (l == nn - 1)
                        {
                            // two roots from the trailing 2x2 block.
                            double p = (double)0.5 * (y - x);
                            double q = p * p + w;
                            double z = math.sqrt(math.abs(q));
                            x += t;
                            if (q >= (double)0)
                            {
                                // real pair.
                                z = p + copysign(z, p);
                                eigenvaluesReal[nn - 1] = x + z;
                                eigenvaluesReal[nn] = (z != (double)0) ? (x - w / z) : (x + z);
                                eigenvaluesImag[nn - 1] = (double)0;
                                eigenvaluesImag[nn] = (double)0;
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
                                double s1 = math.abs(A[nn, nn - 1]) + math.abs(A[nn - 1, nn - 2]);
                                y = x = (double)0.75 * s1;
                                w = (double)(-0.4375) * s1 * s1;
                            }
                            its++;

                            // find two consecutive negligible subdiagonals to start the sweep.
                            double p = (double)0, q = (double)0, r = (double)0;
                            int m;
                            for (m = nn - 2; m >= l; m--)
                            {
                                double z = A[m, m];
                                double rr = x - z;
                                double ss = y - z;
                                p = (rr * ss - w) / A[m + 1, m] + A[m, m + 1];
                                q = A[m + 1, m + 1] - z - rr - ss;
                                r = A[m + 2, m + 1];
                                double s2 = math.abs(p) + math.abs(q) + math.abs(r);
                                // guard the normalization (matches the guarded analog in the QR sweep
                                // below): if p,q,r are all exactly zero, leave them zero rather than
                                // dividing 0/0 -> NaN, which would poison the convergence test.
                                if (s2 != (double)0) { p /= s2; q /= s2; r /= s2; }
                                if (m == l) break;
                                double u = math.abs(A[m, m - 1]) * (math.abs(q) + math.abs(r));
                                double v = math.abs(p) * (math.abs(A[m - 1, m - 1]) + math.abs(z) + math.abs(A[m + 1, m + 1]));
                                if (u + v == v) break;
                            }

                            for (int i = m + 2; i <= nn; i++)
                            {
                                A[i, i - 2] = (double)0;
                                if (i != m + 2) A[i, i - 3] = (double)0;
                            }

                            // the double QR step over rows/cols m..nn.
                            for (int k = m; k <= nn - 1; k++)
                            {
                                if (k != m)
                                {
                                    p = A[k, k - 1];
                                    q = A[k + 1, k - 1];
                                    r = (double)0;
                                    if (k != nn - 1) r = A[k + 2, k - 1];
                                    x = math.abs(p) + math.abs(q) + math.abs(r);
                                    if (x != (double)0)
                                    {
                                        p /= x; q /= x; r /= x;
                                    }
                                }

                                double s = copysign(math.sqrt(p * p + q * q + r * r), p);
                                if (s != (double)0)
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
                                    double xx = p / s;
                                    double yy = q / s;
                                    double zz = r / s;
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
                    double tr = eigenvaluesReal[a]; eigenvaluesReal[a] = eigenvaluesReal[best]; eigenvaluesReal[best] = tr;
                    double ti = eigenvaluesImag[a]; eigenvaluesImag[a] = eigenvaluesImag[best]; eigenvaluesImag[best] = ti;
                }
            }

            return true;
        }

        /// <summary>eigenvaluesQR with default maxIterPerRoot (30, the EISPACK hqr limit).</summary>
        public static bool eigenvaluesQR(ref doubleMxN A, ref doubleN eigenvaluesReal,
                                         ref doubleN eigenvaluesImag)
            => eigenvaluesQR(ref A, ref eigenvaluesReal, ref eigenvaluesImag, 30);
    }
}
