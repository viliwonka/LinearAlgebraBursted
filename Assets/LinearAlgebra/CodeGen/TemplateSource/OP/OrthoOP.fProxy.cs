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
    public static partial class OrthoOP {

        public static void householderInpl(ref fProxyMxN matrix, in fProxyN u)
        {
            if(matrix.IsSquare == false)
                throw new System.Exception("OrthoOP.householder: Matrix must be square");

            if(matrix.M_Rows < matrix.N_Cols)
                throw new System.Exception("OrthoOP.householder: Matrix must be square or tall (more or equal rows than cols)");

            var maxDim = math.max(matrix.M_Rows, matrix.N_Cols);

            if(u.N < maxDim)
                throw new System.Exception("OrthoOP.householder: Vector must be at least as long as the largest dimension of the matrix");

            fProxy vTv = fProxyOP.dot(u, u); // Inline dot product calculation

            // Degenerate (zero / near-zero) reflector -> identity transform; leave matrix unchanged.
            // NaN-safe (!(vTv > t) is true for NaN); avoids 2/0 = Inf poisoning the matrix.
            if (!(vTv > Consts.fProxyZeroTreshold))
                return;

            fProxy scaleFactor = 2 / vTv;

            for (int i = 0; i < matrix.M_Rows; i++)
            {
                for (int j = 0; j < matrix.N_Cols; j++)
                {
                    fProxy vvT_element = scaleFactor * u[i] * u[j];
                    matrix[i, j] -= vvT_element; // Apply directly to matrix
                }
            }
        }

        static fProxy sign(fProxy x) {
            return x < 0 ? -1 : 1;
        }

        // zeroThreshold is the ABSOLUTE column-norm below which a column is treated as zero. Callers
        // pass a SCALE-RELATIVE value (Consts.fProxyZeroTreshold * matrix magnitude) so QR is
        // scale-invariant — a fixed absolute constant mis-classifies every column of a uniformly
        // tiny-magnitude matrix as a zero column and silently produces a garbage decomposition.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void genHouseholderPete(ref fProxyMxN Q, ref fProxyN u, int k, fProxy zeroThreshold) {

            // copy column d of A into u
            // here we are forming x vector
            for (int r = k; r < u.N; r++)
                u[r] = Q[r, k];

            fProxy xNorm = fProxyNormsOP.L2Range(u, k, u.N);

            if (math.abs(xNorm) > zeroThreshold) {

                for (int r = k; r < u.N; r++)
                    u[r] = u[r] / xNorm;

                u[k] = u[k] + sign(u[k]);

                var div = math.sqrt(math.abs(u[k]));
                for (int r = k; r < u.N; r++) {
                    u[r] = u[r] / div;
                }
            }
            else {

                u[k] = math.SQRT2;
                //for (int r = k; r < v.N; r++)
                //    v[k] = (r == k) ? math.SQRT2 : 0;
            }
        }

        // Q is original matrix A, R is identity matrix
        // Q becomes orthogonal matrix, R becomes upper triangular matrix
        // Caller-provided scratch overload (zero-alloc): u is a workspace vector of length
        // EXACTLY Q.M_Rows. Hoist u out of a hot loop to skip the per-call Allocator.Temp alloc.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void qrDecomposition(ref fProxyMxN Q, ref fProxyMxN R, ref fProxyN u)
        {
            if (Q.M_Rows < Q.N_Cols)
                throw new System.Exception("OrthoOP.qrDecomposition: Matrix R must be square or tall (more or equal rows than cols)");

            if (u.N != Q.M_Rows)
                throw new System.Exception("OrthoOP.qrDecomposition: scratch vector u.N must equal Q.M_Rows");

            int qrSteps = Q.N_Cols;

            // scale-relative zero-column threshold (see genHouseholderPete): keyed off the original
            // matrix magnitude so QR is scale-invariant. LInf(Q) == max |entry|.
            fProxy zeroThreshold = Consts.fProxyZeroTreshold * fProxyNormsOP.LInf(in Q);

            // forming R inside Q (will be copied into R later)
            // d = step and diagonal index
            for (int d = 0; d < qrSteps; d++)
            {
                genHouseholderPete(ref Q, ref u, d, zeroThreshold);;
                                 
                for (int c = d; c < Q.N_Cols; c++) 
                {
                    fProxy dotProduct = 0;
                    for (int k = d; k < Q.M_Rows; k++)
                    {
                        dotProduct += u[k] * Q[k, c];
                    }

                    //dotProduct *= 2;
                    for (int r = d; r < Q.M_Rows; r++)
                    {
                        Q[r, c] -= u[r] * dotProduct;
                    }
                }

                // copy current Q diagonal element into R
                // it will be over-written in the next step
                R[d, d] = Q[d, d];

                // copy v into Q below diagonal, will be used to reconstruct Q
                for (int i = d; i < Q.M_Rows; i++)
                {
                    Q[i, d] = u[i];
                }
            }
            // End or R orthogonalization construction

            // Copy the upper triangular part of Q into R
            for (int r = 0; r < R.M_Rows; r++)
            for (int c = 0; c < R.N_Cols; c++)
            {
                if (c < r)
                {
                    // Below diagonal, set to 0
                    R[r, c] = 0;
                }
                else if (c > r)
                {
                    // above diagonal, copy from Q
                    R[r, c] = Q[r, c];
                }
            }

            /// Reconstruct Q from vectors stored inside Q columns

            // Initialize upper part of Q to identity matrix, including diagonals
            for (int r = 0; r < Q.M_Rows; r++)
            {
                for (int c = r; c < Q.N_Cols; c++)
                {
                    // On and above diagonal
                    if (c > r)
                    {   
                        Q[r, c] = 0;
                    }
                }
            }
            
            // Apply Householder transformations in reverse order
            // Reconstruct the Householder vector v from the original Q
            for (int d = Q.N_Cols - 1; d >= 0; d--)
            {               
                // includes diagonal elements
                for (int i = d; i < Q.M_Rows; i++)
                {
                    u[i] = Q[i, d];
                    Q[i, d] = i == d? 1 : 0;
                }

                // Apply the Householder transformation to Q
                for (int c = d; c < Q.N_Cols; c++) {

                    fProxy dotProduct = 0;
                    for (int k = d; k < Q.M_Rows; k++) {
                        dotProduct += u[k] * Q[k, c];
                    }
                    //dotProduct *= 2;
                    for (int r = d; r < Q.M_Rows; r++) {
                        Q[r, c] -= u[r] * dotProduct;
                    }
                }
            }

        }

        // Allocating wrapper: allocates the scratch vector u (Allocator.Temp) and delegates.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void qrDecomposition(ref fProxyMxN Q, ref fProxyMxN R)
        {
            var u = new fProxyN(Q.M_Rows, Allocator.Temp, false);
            qrDecomposition(ref Q, ref R, ref u);
            u.Dispose();
        }

        // Column-pivoted (rank-revealing) QR — Businger–Golub. Factorizes A·P = Q·R, where the
        // column permutation P is chosen greedily so the pivot at each step is the trailing column
        // of largest 2-norm. This forces the magnitudes of the R diagonal to be non-increasing
        // (|R[0,0]| >= |R[1,1]| >= ... >= |R[n-1,n-1]|), so trailing near-zero diagonal entries
        // reveal the numerical rank — the stable choice for rank-deficient least squares where the
        // plain (un-pivoted) qrDecomposition above requires full column rank.
        //
        //   Q  in:  A (m x n, m >= n)              out: orthogonal Q (m x n)
        //   R  out: upper triangular R (n x n)
        //   P  out: column Pivot, size n. Reset internally. Result column j is original column P[j];
        //           equivalently A[:, P[j]] == (Q*R)[:, j].
        //   u  scratch Householder vector, length EXACTLY Q.M_Rows.
        //
        // Partial column norms are recomputed exactly at each step (rows d..m-1) rather than
        // downdated. That is the same O(n^2 m) order as the reflector sweep itself, and it sidesteps
        // the catastrophic-cancellation failure mode of norm downdating (LAPACK xGEQPF needs a
        // recompute guard precisely because the cheap downdate loses all accuracy near rank
        // deficiency) — for the modest matrices this library targets, exact recompute is both
        // simpler and unconditionally robust.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void qrDecompositionColumnPivot(ref fProxyMxN Q, ref fProxyMxN R, ref Pivot P, ref fProxyN u)
        {
            if (Q.M_Rows < Q.N_Cols)
                throw new System.Exception("OrthoOP.qrDecompositionColumnPivot: Matrix must be square or tall (M_Rows >= N_Cols)");

            if (u.N != Q.M_Rows)
                throw new System.Exception("OrthoOP.qrDecompositionColumnPivot: scratch vector u.N must equal Q.M_Rows");

            if (P.N != Q.N_Cols)
                throw new System.Exception("OrthoOP.qrDecompositionColumnPivot: pivot P.N must equal Q.N_Cols");

            if (R.M_Rows != Q.N_Cols || R.N_Cols != Q.N_Cols)
                throw new System.Exception("OrthoOP.qrDecompositionColumnPivot: R must be N_Cols x N_Cols");

            P.Reset();

            int m = Q.M_Rows;
            int n = Q.N_Cols;

            // scale-relative zero-column threshold (see genHouseholderPete); LInf(Q) == max |entry|.
            fProxy zeroThreshold = Consts.fProxyZeroTreshold * fProxyNormsOP.LInf(in Q);

            for (int d = 0; d < n; d++)
            {
                // --- column pivot: among trailing columns d..n-1, pick the one whose partial 2-norm
                //     over rows d..m-1 is largest (recomputed exactly), and bring it to position d. ---

                // partial squared-norm of the incumbent column d.
                fProxy diagNorm2 = 0;
                for (int r = d; r < m; r++)
                    diagNorm2 += Q[r, d] * Q[r, d];

                int pivotCol = d;
                fProxy maxNorm2 = diagNorm2;
                for (int c = d + 1; c < n; c++)
                {
                    fProxy norm2 = 0;
                    for (int r = d; r < m; r++)
                        norm2 += Q[r, c] * Q[r, c];

                    if (norm2 > maxNorm2)
                    {
                        maxNorm2 = norm2;
                        pivotCol = c;
                    }
                }

                // Only pivot when the best column beats the incumbent by more than the accumulated
                // rounding noise of the norm sums (~ #terms * eps). This leaves numerically-tied
                // columns in place — notably the Kahan matrix, whose columns all have norm exactly 1
                // and which is provably invariant under column pivoting; a bare `>` would let a
                // ~1 ulp difference induce a spurious (and non-reproducible) permutation.
                fProxy pivotRelTol = (fProxy)(8 * m) * Consts.fProxyEpsilon;
                if (pivotCol != d && maxNorm2 > diagNorm2 * ((fProxy)1 + pivotRelTol))
                {
                    // Full-column swap (all rows): rows < d hold finished R entries that must travel
                    // with the column; rows >= d hold the live sub-matrix. Stored Householder vectors
                    // of earlier steps live in columns < d and are untouched (both indices are >= d).
                    SwapOP.Columns(ref Q, d, pivotCol);
                    P.Swap(d, pivotCol);
                }

                genHouseholderPete(ref Q, ref u, d, zeroThreshold);

                for (int c = d; c < n; c++)
                {
                    fProxy dotProduct = 0;
                    for (int k = d; k < m; k++)
                        dotProduct += u[k] * Q[k, c];

                    for (int r = d; r < m; r++)
                        Q[r, c] -= u[r] * dotProduct;
                }

                // copy current Q diagonal element into R (over-written in next step)
                R[d, d] = Q[d, d];

                // copy v into Q below diagonal, will be used to reconstruct Q
                for (int i = d; i < m; i++)
                    Q[i, d] = u[i];
            }

            // Copy the upper triangular part of Q into R
            for (int r = 0; r < R.M_Rows; r++)
            for (int c = 0; c < R.N_Cols; c++)
            {
                if (c < r)
                    R[r, c] = 0;
                else if (c > r)
                    R[r, c] = Q[r, c];
            }

            // Reconstruct Q from the Householder vectors stored in its columns (identical to the
            // un-pivoted qrDecomposition: pivoting only reordered the columns, not this step).
            for (int r = 0; r < m; r++)
                for (int c = r; c < n; c++)
                    if (c > r)
                        Q[r, c] = 0;

            for (int d = n - 1; d >= 0; d--)
            {
                for (int i = d; i < m; i++)
                {
                    u[i] = Q[i, d];
                    Q[i, d] = i == d ? 1 : 0;
                }

                for (int c = d; c < n; c++)
                {
                    fProxy dotProduct = 0;
                    for (int k = d; k < m; k++)
                        dotProduct += u[k] * Q[k, c];

                    for (int r = d; r < m; r++)
                        Q[r, c] -= u[r] * dotProduct;
                }
            }
        }

        // Allocating wrapper: allocates the scratch vector u (Allocator.Temp) and delegates.
        // The caller still owns P (its size carries the column count and it is reset internally).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void qrDecompositionColumnPivot(ref fProxyMxN Q, ref fProxyMxN R, ref Pivot P)
        {
            var u = new fProxyN(Q.M_Rows, Allocator.Temp, false);
            qrDecompositionColumnPivot(ref Q, ref R, ref P, ref u);
            u.Dispose();
        }

        // Q is original matrix A that will be turned into R (upper triangular) non square matrix
        // Q becomes R
        // b will be transformed into y, where y = Q^T b, and then solved for x
        // x is the solution
        // Q and b get modified (destroyed)
        // PRECONDITION: A has FULL COLUMN RANK. This un-pivoted solve back-substitutes through R's
        // diagonal; a rank-deficient A produces a zero on that diagonal and the result x is then
        // Inf/NaN (no guard). For rank-deficient / least-norm problems use the rank-revealing paths
        // instead: OrthoOP.qrDecompositionColumnPivot (QRCP), SVD.pinvSolve, or
        // Cholesky.choleskyPivotSolve.
        // Caller-provided scratch overload (zero-alloc): u is a workspace vector of length
        // EXACTLY A.M_Rows. Hoist u out of a hot loop to skip the per-call Allocator.Temp alloc.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void qrDirectSolve(ref fProxyMxN A, ref fProxyN b, ref fProxyN x, ref fProxyN u) {
            if (A.M_Rows < A.N_Cols)
                throw new System.Exception("OrthoOP.qrDirectSolve: Matrix A must be square or tall (more or equal rows than cols)");

            if (b.N != A.M_Rows)
                throw new System.Exception("OrthoOP.qrDirectSolve: b.N must equal A.M_Rows");

            if (x.N != A.N_Cols)
                throw new System.Exception("OrthoOP.qrDirectSolve: x.N must equal A.N_Cols");

            if (u.N != A.M_Rows)
                throw new System.Exception("OrthoOP.qrDirectSolve: scratch vector u.N must equal A.M_Rows");

            int qrSteps = A.N_Cols;

            // scale-relative zero-column threshold (see genHouseholderPete); LInf(A) == max |entry|.
            fProxy zeroThreshold = Consts.fProxyZeroTreshold * fProxyNormsOP.LInf(in A);

            fProxy dotProduct = 0;
            // forming R inside Q (will be copied into R later)
            // d = step and diagonal index
            for (int d = 0; d < qrSteps; d++) {

                genHouseholderPete(ref A, ref u, d, zeroThreshold);

                for (int c = d; c < A.N_Cols; c++) {

                    dotProduct = 0;
                    for (int r = d; r < A.M_Rows; r++)
                        dotProduct += u[r] * A[r, c];

                    //dotProduct *= 2;
                    for (int r = d; r < A.M_Rows; r++)
                        A[r, c] -= u[r] * dotProduct;
                }

                // apply same transformation to b vector
                dotProduct = 0;
                for (int r = d; r < A.M_Rows; r++)
                    dotProduct += u[r] * b[r];

                //dotProduct *= 2;
                for (int r = d; r < A.M_Rows; r++)
                    b[r] -= u[r] * dotProduct;
            }

            // copy b into x (x may be smaller dimension than b)
            for (int r = 0; r < A.N_Cols; r++)
                x[r] = b[r];

            // b was transformed to y, where y = Q^T b
            // Solve Rx = y

            Solvers.SolveUpperTriangular(ref A, ref x);
        }

        // Allocating wrapper: allocates the scratch vector u (Allocator.Temp) and delegates.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void qrDirectSolve(ref fProxyMxN A, ref fProxyN b, ref fProxyN x) {
            var u = new fProxyN(A.M_Rows, Allocator.Temp, false);
            qrDirectSolve(ref A, ref b, ref x, ref u);
            u.Dispose();
        }

        // QRCP-based rank-safe least-squares: basic (truncated) solution.
        //
        // Solves A x ≈ b (m >= n) for a possibly rank-deficient A. Uses column-pivoted QR
        // (Businger-Golub, A·P = Q·R) to expose the numerical rank r: the R diagonal is
        // non-increasing, so r = count of leading entries with |R[i,i]| > tol where
        //     tol = relTol * |R[0,0]|
        // and relTol defaults to max(m,n) * Consts.fProxyZeroTreshold (matching SVD.pinvSolve /
        // MatrixMetrics.rank, so rank detection agrees across the library). A negative relTol is
        // an "auto" sentinel that selects that same default.
        //
        // Only the leading r×r block of R is back-substituted; the remaining (n-r) free variables
        // are set to zero in the permuted ordering, then the column permutation P is un-applied to
        // recover x. This is the BASIC (truncated) solution: it minimises the residual ||Ax - b||
        // but is NOT the minimum-norm solution. For minimum-norm use SVD.pinvSolve. When A has
        // full column rank (r == n) the result is identical to ordinary QR least-squares.
        //
        //   A   in:  m x n matrix (m >= n). Not modified (copied into Q scratch).
        //   b   in:  right-hand side, length m. Must not alias x.
        //   x   out: solution, length n.
        //   Q   scratch: m x n (receives orthogonal factor; consumed).
        //   R   scratch: n x n (receives upper-triangular factor; consumed).
        //   P   scratch: column Pivot of size n (reset internally).
        //   u   scratch: length EXACTLY m (Householder workspace; first n entries are
        //                repurposed for the un-permute scatter after the decomposition).
        //   rank out: detected numerical rank r.
        //   relTol:  rank threshold ratio; tol = relTol * |R[0,0]|. Negative = auto default.
        /// <summary>
        /// QRCP-based rank-safe least-squares: basic (truncated) solution. Returns the least-squares
        /// solution x with at most r nonzeros in the permuted ordering, where r is the detected
        /// numerical rank. This is NOT the minimum-norm solution; see SVD.pinvSolve for that.
        /// The full-rank path (r == n) is divide-safe by construction (all used R diagonals exceed tol).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void qrcpDirectSolve(ref fProxyMxN A, ref fProxyN b, ref fProxyN x,
                                           ref fProxyMxN Q, ref fProxyMxN R, ref Pivot P,
                                           ref fProxyN u, out int rank, fProxy relTol)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("OrthoOP.qrcpDirectSolve: A must be square or tall (M_Rows >= N_Cols)");
            if (b.N != m)
                throw new ArgumentException("OrthoOP.qrcpDirectSolve: b.N must equal A.M_Rows");
            if (x.N != n)
                throw new ArgumentException("OrthoOP.qrcpDirectSolve: x.N must equal A.N_Cols");
            if (Q.M_Rows != m || Q.N_Cols != n)
                throw new ArgumentException("OrthoOP.qrcpDirectSolve: Q must be M_Rows x N_Cols");
            if (R.M_Rows != n || R.N_Cols != n)
                throw new ArgumentException("OrthoOP.qrcpDirectSolve: R must be N_Cols x N_Cols");
            if (P.N != n)
                throw new ArgumentException("OrthoOP.qrcpDirectSolve: P.N must equal A.N_Cols");
            if (u.N != m)
                throw new ArgumentException("OrthoOP.qrcpDirectSolve: u.N must equal A.M_Rows");

            // Negative relTol is an "auto" sentinel: use the library-standard rank threshold
            // (same default as SVD.pinvSolve / MatrixMetrics.rank). This also makes the threshold
            // divide-safe (tol >= 0), so a stray negative can never inflate rank into a divide-by-tiny.
            if (relTol < (fProxy)0)
                relTol = (fProxy)(math.max(m, n)) * Consts.fProxyZeroTreshold;

            // Degenerate: zero-column system.
            if (n == 0) { rank = 0; return; }

            // Step 1: copy A into Q (qrDecompositionColumnPivot destroys its input).
            Q.Data.CopyFrom(A.Data);

            // Step 2: QRCP — A·P = Q·R. P is reset and built inside this call.
            qrDecompositionColumnPivot(ref Q, ref R, ref P, ref u);

            // Step 3: determine numerical rank r from R's non-increasing diagonal.
            // tol = relTol * |R[0,0]|. When R[0,0] == 0 tol == 0, and |R[0,0]| > 0 is false
            // → rank stays 0. NaN in R[0,0] → tol = NaN → all comparisons false → rank = 0.
            fProxy tol = relTol * math.abs(R[0, 0]);
            rank = 0;
            for (int i = 0; i < n; i++)
            {
                if (math.abs(R[i, i]) > tol)
                    rank++;
                else
                    break;
            }

            // Step 4: zero matrix (rank == 0) → x = 0, done.
            if (rank == 0)
            {
                for (int j = 0; j < n; j++)
                    x[j] = (fProxy)0;
                return;
            }

            int r = rank;

            // Step 5: form c = Qᵀ b into x.
            // dot(in b, in Q, ref x) computes x[j] = Σ_i Q[i,j]·b[i] = (Qᵀb)[j].
            // dot zeroes x via MemClear before accumulating, so x needs no prior initialisation.
            // Guard: x must not alias b (enforced inside dot by pointer comparison).
            fProxyOP.dot(in b, in Q, ref x);

            // Step 6: back-solve the leading r×r block of R in place.
            // x holds c = Qᵀb; overwrite x[0..r-1] with the triangular solution.
            // Every R[i,i] for i < r satisfies |R[i,i]| > tol, so no divide-by-zero.
            for (int i = r - 1; i >= 0; i--)
            {
                fProxy sum = (fProxy)0;
                for (int j = i + 1; j < r; j++)
                    sum += R[i, j] * x[j];
                x[i] = (x[i] - sum) / R[i, i];
            }
            // Zero the free variables (columns beyond the numerical rank).
            for (int j = r; j < n; j++)
                x[j] = (fProxy)0;

            // Step 7: un-permute — scatter from permuted ordering back to original column ordering.
            // QRCP gives A·P = Q·R where P[j] = original column index promoted to position j.
            // The permuted solution z (in x) satisfies: x_final[P[j]] = z[j].
            // Borrow u[0..n-1] as scatter scratch (u is no longer needed after Step 2).
            for (int j = 0; j < n; j++)
                u[j] = x[j];
            for (int j = 0; j < n; j++)
                x[P[j]] = u[j];
        }

        // Default-tolerance overload: passes the auto sentinel (relTol < 0), so the primitive
        // uses max(m,n) * Consts.fProxyZeroTreshold (consistent with SVD.pinvSolve / MatrixMetrics.rank).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void qrcpDirectSolve(ref fProxyMxN A, ref fProxyN b, ref fProxyN x,
                                           ref fProxyMxN Q, ref fProxyMxN R, ref Pivot P,
                                           ref fProxyN u, out int rank)
        {
            qrcpDirectSolve(ref A, ref b, ref x, ref Q, ref R, ref P, ref u, out rank, (fProxy)(-1));
        }

        /// <summary>
        /// Allocating convenience wrapper: allocates Q (m×n), R (n×n), P (n-Pivot) and u (m)
        /// from Allocator.Temp and delegates to the zero-alloc primitive. Use the primitive in
        /// hot loops to avoid repeated Temp allocs.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void qrcpDirectSolve(ref fProxyMxN A, ref fProxyN b, ref fProxyN x,
                                           out int rank, fProxy relTol)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            var Q = new fProxyMxN(m, n, Allocator.Temp, false);
            var R = new fProxyMxN(n, n, Allocator.Temp, false);
            var P = new Pivot(n, Allocator.Temp);
            var u = new fProxyN(m, Allocator.Temp, false);
            qrcpDirectSolve(ref A, ref b, ref x, ref Q, ref R, ref P, ref u, out rank, relTol);
            u.Dispose();
            P.Dispose();
            R.Dispose();
            Q.Dispose();
        }

        /// <summary>
        /// Allocating convenience wrapper with default tolerance (max(m,n) * Consts.fProxyZeroTreshold,
        /// matching SVD.pinvSolve / MatrixMetrics.rank).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void qrcpDirectSolve(ref fProxyMxN A, ref fProxyN b, ref fProxyN x,
                                           out int rank)
        {
            qrcpDirectSolve(ref A, ref b, ref x, out rank, (fProxy)(-1));
        }
    }

}
