#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class SVD {

        /// <summary>
        /// Minimum-norm least-squares solve: x = argmin ||A x - b||2 (minimum ||x|| among minimizers).
        /// Works for any shape (m >= n and m &lt; n) and any rank, including rank 0.
        /// A is NOT modified (the Golub-Kahan path takes it as input). b is not modified. x (length
        /// N_Cols): output only; prior contents ignored; safe to allocate with uninit: true.
        /// relTol &lt; 0 selects auto tolerance: relTol = max(m, n) * Consts.fProxyZeroThreshold.
        /// Singular values S[j] &lt;= relTol * S[0] are treated as zero.
        /// Allocates temporaries from A's arena via fProxyTempVec/fProxyTempMat (not an InPlace op).
        /// Returns a <see cref="RankInfo"/>: <c>rank</c> is the numerical rank used (0 on a hard
        /// failure); <c>status</c> is <see cref="DirectSolveStatus.Success"/> (full rank),
        /// <see cref="DirectSolveStatus.RankDeficient"/> (still-usable, rank &lt; min(m,n)), or
        /// <see cref="DirectSolveStatus.NotConverged"/> if the inner SVD did not converge within
        /// maxSweeps — a HARD failure (<see cref="RankInfo.Solved"/> false); x is zeroed but NOT a
        /// valid solution in that case.
        /// </summary>
        // Caller-provided scratch overload (zero-alloc); scratch layout: see fProxySVDCache. Hoist these
        // out of a hot loop solving many same-shape systems to avoid per-call allocs.
        public static RankInfo pinvSolve(ref fProxyMxN A, in fProxyN b, ref fProxyN x,
                                    fProxy relTol, int maxSweeps,
                                    ref fProxyN S, ref fProxyMxN M, ref fProxyMxN U, ref fProxyMxN At)
        {
            if (b.N != A.M_Rows)
                throw new ArgumentException("pinvSolve: b.N must equal A.M_Rows");

            if (x.N != A.N_Cols)
                throw new ArgumentException("pinvSolve: x.N must equal A.N_Cols");

            if (maxSweeps < 1)
                throw new ArgumentException("pinvSolve: maxSweeps must be >= 1");

            int m = A.M_Rows;
            int n = A.N_Cols;
            int k = math.min(m, n);
            int big = math.max(m, n);

            if (S.N != k)
                throw new ArgumentException("pinvSolve: S scratch length must equal min(A.M_Rows, A.N_Cols)");

            if (M.M_Rows != k || M.N_Cols != k)
                throw new ArgumentException("pinvSolve: M scratch must be k x k, k = min(A.M_Rows, A.N_Cols)");

            if (U.M_Rows != big || U.N_Cols != k)
                throw new ArgumentException("pinvSolve: U scratch must be max(m,n) x min(m,n)");

            if (m >= n) {
                // Tall or square case: A = U * diag(S) * V^T; U receives the left factor, M = V.
                SVDInfo svdInfo = thin(in A, ref U, ref S, ref M, maxSweeps);

                // Zero x (prior contents ignored either way, per the doc comment).
                for (int kk = 0; kk < n; kk++)
                    x[kk] = (fProxy)0;

                if (!svdInfo)
                    return new RankInfo { status = DirectSolveStatus.NotConverged, rank = 0 };

                // Auto tolerance
                if (relTol < (fProxy)0)
                    relTol = (fProxy)math.max(m, n) * Consts.fProxyZeroThreshold;

                if (n == 0 || S[0] == (fProxy)0)
                    return new RankInfo { status = k == 0 ? DirectSolveStatus.Success : DirectSolveStatus.RankDeficient, rank = 0 };

                fProxy tol = relTol * S[0];
                int rank = 0;

                // x = V * diag(1/S_j) * U^T * b  (only for S[j] > tol)
                for (int j = 0; j < n; j++) {
                    if (S[j] <= tol)
                        continue;

                    // coeff = (U[:,j]^T * b) / S[j], U[:,j] is column j of the left factor
                    fProxy dot = (fProxy)0;
                    for (int i = 0; i < m; i++)
                        dot += U[i, j] * b[i];

                    fProxy coeff = dot / S[j];

                    for (int kk = 0; kk < n; kk++)
                        x[kk] += coeff * M[kk, j];

                    rank++;
                }

                return new RankInfo { status = rank == k ? DirectSolveStatus.Success : DirectSolveStatus.RankDeficient, rank = rank };
            }
            else {
                // Wide case: decompose A^T (n x m, tall). Right singular vectors of A are columns of
                // U (the left factor of A^T); left singular vectors of A are columns of M (= W).
                if (At.M_Rows != n || At.N_Cols != m)
                    throw new ArgumentException("pinvSolve: At scratch must be A.N_Cols x A.M_Rows for the wide (m < n) case");

                Blas.trans(in A, ref At);   // At = A^T (zero-alloc, ref-dest trans)

                SVDInfo svdInfo = thin(in At, ref U, ref S, ref M, maxSweeps);

                // Zero x (prior contents ignored either way, per the doc comment).
                for (int kk = 0; kk < n; kk++)
                    x[kk] = (fProxy)0;

                if (!svdInfo)
                    return new RankInfo { status = DirectSolveStatus.NotConverged, rank = 0 };

                // Auto tolerance
                if (relTol < (fProxy)0)
                    relTol = (fProxy)math.max(m, n) * Consts.fProxyZeroThreshold;

                if (m == 0 || S[0] == (fProxy)0)
                    return new RankInfo { status = k == 0 ? DirectSolveStatus.Success : DirectSolveStatus.RankDeficient, rank = 0 };

                fProxy tol = relTol * S[0];
                int rank = 0;

                // x = U * diag(1/S_j) * W^T * b  (only for S[j] > tol)
                // U columns (length n) are right singular vectors of A
                // M (= W) columns (length m) are left singular vectors of A
                for (int j = 0; j < m; j++) {
                    if (S[j] <= tol)
                        continue;

                    // coeff = (W[:,j]^T * b) / S[j]
                    fProxy dot = (fProxy)0;
                    for (int i = 0; i < m; i++)
                        dot += M[i, j] * b[i];

                    fProxy coeff = dot / S[j];

                    for (int kk = 0; kk < n; kk++)
                        x[kk] += coeff * U[kk, j];

                    rank++;
                }

                return new RankInfo { status = rank == k ? DirectSolveStatus.Success : DirectSolveStatus.RankDeficient, rank = rank };
            }
        }

        /// <summary>
        /// pinvSolve allocating wrapper: allocates the SVD scratch (S, k x k singular-vector matrix,
        /// and A^T for the wide case) from A's arena and delegates to the zero-alloc primitive.
        /// </summary>
        public static RankInfo pinvSolve(ref fProxyMxN A, in fProxyN b, ref fProxyN x,
                                    fProxy relTol, int maxSweeps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            int k = math.min(m, n);
            int big = math.max(m, n);

            fProxyN S = A.fProxyTempVec(k);
            fProxyMxN M = A.fProxyTempMat(k, k);
            fProxyMxN U = A.fProxyTempMat(big, k);
            fProxyMxN At = default;
            if (m < n)
                At = A.fProxyTempMat(n, m);

            return pinvSolve(ref A, in b, ref x, relTol, maxSweeps, ref S, ref M, ref U, ref At);
        }

        /// <summary>
        /// pinvSolve using a reusable workspace (Arena.fProxySVDCache(m, n)) — zero-alloc.
        /// The workspace must be sized for A's shape (k = min(A.M_Rows, A.N_Cols)); the guards in
        /// the underlying scratch primitive enforce this.
        /// </summary>
        public static RankInfo pinvSolve(ref fProxyMxN A, in fProxyN b, ref fProxyN x,
                                    ref fProxySVDCache ws, fProxy relTol, int maxSweeps)
            => pinvSolve(ref A, in b, ref x, relTol, maxSweeps, ref ws.S, ref ws.M, ref ws.U, ref ws.At);

        /// <summary>pinvSolve (workspace) with default maxSweeps (Consts.sweepBudget(min(A.M_Rows, A.N_Cols))).</summary>
        public static RankInfo pinvSolve(ref fProxyMxN A, in fProxyN b, ref fProxyN x,
                                    ref fProxySVDCache ws, fProxy relTol)
            => pinvSolve(ref A, in b, ref x, ref ws, relTol, Consts.sweepBudget(math.min(A.M_Rows, A.N_Cols)));

        /// <summary>pinvSolve (workspace) with default relTol (-1, auto) and maxSweeps (Consts.sweepBudget(min(A.M_Rows, A.N_Cols))).</summary>
        public static RankInfo pinvSolve(ref fProxyMxN A, in fProxyN b, ref fProxyN x,
                                    ref fProxySVDCache ws)
            => pinvSolve(ref A, in b, ref x, ref ws, (fProxy)(-1), Consts.sweepBudget(math.min(A.M_Rows, A.N_Cols)));

        /// <summary>pinvSolve with default maxSweeps (Consts.sweepBudget(min(A.M_Rows, A.N_Cols))).</summary>
        public static RankInfo pinvSolve(ref fProxyMxN A, in fProxyN b, ref fProxyN x,
                                    fProxy relTol)
            => pinvSolve(ref A, in b, ref x, relTol, Consts.sweepBudget(math.min(A.M_Rows, A.N_Cols)));

        /// <summary>pinvSolve with default relTol (-1, auto tolerance) and maxSweeps (Consts.sweepBudget(min(A.M_Rows, A.N_Cols))).</summary>
        public static RankInfo pinvSolve(ref fProxyMxN A, in fProxyN b, ref fProxyN x)
            => pinvSolve(ref A, in b, ref x, (fProxy)(-1), Consts.sweepBudget(math.min(A.M_Rows, A.N_Cols)));

        /// <summary>
        /// Moore-Penrose pseudo-inverse: Aplus (N_Cols x M_Rows, caller-allocated) = V diag(1/S_i, S_i > tol) U^T.
        /// A is NOT modified (the Golub-Kahan path takes it as input). Same tolerance/rank/return
        /// semantics as pinvSolve: a <see cref="RankInfo"/> whose <c>status</c> is
        /// <see cref="DirectSolveStatus.Success"/> (full rank), <see cref="DirectSolveStatus.RankDeficient"/>
        /// (still-usable, rank &lt; min(m,n)), or <see cref="DirectSolveStatus.NotConverged"/> if the
        /// inner SVD did not converge within maxSweeps — a HARD failure (Aplus is zeroed but NOT a
        /// valid pseudo-inverse in that case). Any shape.
        /// </summary>
        // Caller-provided scratch overload (zero-alloc); scratch layout: see fProxySVDCache.
        public static RankInfo pseudoInverse(ref fProxyMxN A, ref fProxyMxN Aplus,
                                        fProxy relTol, int maxSweeps,
                                        ref fProxyN S, ref fProxyMxN M, ref fProxyMxN U, ref fProxyMxN At)
        {
            if (Aplus.M_Rows != A.N_Cols)
                throw new ArgumentException("pseudoInverse: Aplus.M_Rows must equal A.N_Cols");

            if (Aplus.N_Cols != A.M_Rows)
                throw new ArgumentException("pseudoInverse: Aplus.N_Cols must equal A.M_Rows");

            if (maxSweeps < 1)
                throw new ArgumentException("pseudoInverse: maxSweeps must be >= 1");

            int m = A.M_Rows;
            int n = A.N_Cols;
            int k = math.min(m, n);
            int big = math.max(m, n);

            if (S.N != k)
                throw new ArgumentException("pseudoInverse: S scratch length must equal min(A.M_Rows, A.N_Cols)");

            if (M.M_Rows != k || M.N_Cols != k)
                throw new ArgumentException("pseudoInverse: M scratch must be k x k, k = min(A.M_Rows, A.N_Cols)");

            if (U.M_Rows != big || U.N_Cols != k)
                throw new ArgumentException("pseudoInverse: U scratch must be max(m,n) x min(m,n)");

            // Zero-initialize Aplus
            for (int r = 0; r < Aplus.M_Rows; r++)
                for (int c = 0; c < Aplus.N_Cols; c++)
                    Aplus[r, c] = (fProxy)0;

            if (m >= n) {
                // A = U * diag(S) * V^T; U receives the left factor, M = V
                SVDInfo svdInfo = thin(in A, ref U, ref S, ref M, maxSweeps);

                // Hard failure: thin's outputs are unwritten/partial — building Aplus from them
                // would silently return garbage. Aplus stays zeroed (see the zero-init above).
                if (!svdInfo)
                    return new RankInfo { status = DirectSolveStatus.NotConverged, rank = 0 };

                if (relTol < (fProxy)0)
                    relTol = (fProxy)math.max(m, n) * Consts.fProxyZeroThreshold;

                if (n == 0 || S[0] == (fProxy)0)
                    return new RankInfo { status = k == 0 ? DirectSolveStatus.Success : DirectSolveStatus.RankDeficient, rank = 0 };

                fProxy tol = relTol * S[0];
                int rank = 0;

                // Aplus[r, c] = sum_{j: S[j]>tol} V[r,j] * (1/S[j]) * U[c,j]
                // r in 0..n-1, c in 0..m-1
                for (int j = 0; j < n; j++) {
                    if (S[j] <= tol)
                        continue;

                    fProxy invS = (fProxy)1 / S[j];

                    for (int r = 0; r < n; r++) {
                        fProxy vr = M[r, j] * invS;
                        for (int c = 0; c < m; c++)
                            Aplus[r, c] += vr * U[c, j];
                    }

                    rank++;
                }

                return new RankInfo { status = rank == k ? DirectSolveStatus.Success : DirectSolveStatus.RankDeficient, rank = rank };
            }
            else {
                // Wide case: decompose A^T (n x m); U receives its left factor, M = W
                if (At.M_Rows != n || At.N_Cols != m)
                    throw new ArgumentException("pseudoInverse: At scratch must be A.N_Cols x A.M_Rows for the wide (m < n) case");

                Blas.trans(in A, ref At);   // At = A^T (zero-alloc, ref-dest trans)

                SVDInfo svdInfo = thin(in At, ref U, ref S, ref M, maxSweeps);

                // Hard failure — see the tall branch above.
                if (!svdInfo)
                    return new RankInfo { status = DirectSolveStatus.NotConverged, rank = 0 };

                if (relTol < (fProxy)0)
                    relTol = (fProxy)math.max(m, n) * Consts.fProxyZeroThreshold;

                if (m == 0 || S[0] == (fProxy)0)
                    return new RankInfo { status = k == 0 ? DirectSolveStatus.Success : DirectSolveStatus.RankDeficient, rank = 0 };

                fProxy tol = relTol * S[0];
                int rank = 0;

                // Aplus[r, c] = sum_{j: S[j]>tol} U[r,j] * (1/S[j]) * W[c,j]
                // r in 0..n-1, c in 0..m-1  (U holds the right singular vectors of A)
                for (int j = 0; j < m; j++) {
                    if (S[j] <= tol)
                        continue;

                    fProxy invS = (fProxy)1 / S[j];

                    for (int r = 0; r < n; r++) {
                        fProxy atr = U[r, j] * invS;
                        for (int c = 0; c < m; c++)
                            Aplus[r, c] += atr * M[c, j];
                    }

                    rank++;
                }

                return new RankInfo { status = rank == k ? DirectSolveStatus.Success : DirectSolveStatus.RankDeficient, rank = rank };
            }
        }

        /// <summary>
        /// pseudoInverse allocating wrapper: allocates the SVD scratch (S, k x k singular-vector
        /// matrix, and A^T for the wide case) from A's arena and delegates to the zero-alloc primitive.
        /// </summary>
        public static RankInfo pseudoInverse(ref fProxyMxN A, ref fProxyMxN Aplus,
                                        fProxy relTol, int maxSweeps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            int k = math.min(m, n);
            int big = math.max(m, n);

            fProxyN S = A.fProxyTempVec(k);
            fProxyMxN M = A.fProxyTempMat(k, k);
            fProxyMxN U = A.fProxyTempMat(big, k);
            fProxyMxN At = default;
            if (m < n)
                At = A.fProxyTempMat(n, m);

            return pseudoInverse(ref A, ref Aplus, relTol, maxSweeps, ref S, ref M, ref U, ref At);
        }

        /// <summary>
        /// pseudoInverse using a reusable workspace (Arena.fProxySVDCache(m, n)) — zero-alloc.
        /// The workspace must be sized for A's shape (k = min(A.M_Rows, A.N_Cols)).
        /// </summary>
        public static RankInfo pseudoInverse(ref fProxyMxN A, ref fProxyMxN Aplus,
                                        ref fProxySVDCache ws, fProxy relTol, int maxSweeps)
            => pseudoInverse(ref A, ref Aplus, relTol, maxSweeps, ref ws.S, ref ws.M, ref ws.U, ref ws.At);

        /// <summary>pseudoInverse (workspace) with default maxSweeps (Consts.sweepBudget(min(A.M_Rows, A.N_Cols))).</summary>
        public static RankInfo pseudoInverse(ref fProxyMxN A, ref fProxyMxN Aplus,
                                        ref fProxySVDCache ws, fProxy relTol)
            => pseudoInverse(ref A, ref Aplus, ref ws, relTol, Consts.sweepBudget(math.min(A.M_Rows, A.N_Cols)));

        /// <summary>pseudoInverse (workspace) with default relTol (-1, auto) and maxSweeps (Consts.sweepBudget(min(A.M_Rows, A.N_Cols))).</summary>
        public static RankInfo pseudoInverse(ref fProxyMxN A, ref fProxyMxN Aplus,
                                        ref fProxySVDCache ws)
            => pseudoInverse(ref A, ref Aplus, ref ws, (fProxy)(-1), Consts.sweepBudget(math.min(A.M_Rows, A.N_Cols)));

        /// <summary>pseudoInverse with default maxSweeps (Consts.sweepBudget(min(A.M_Rows, A.N_Cols))).</summary>
        public static RankInfo pseudoInverse(ref fProxyMxN A, ref fProxyMxN Aplus,
                                        fProxy relTol)
            => pseudoInverse(ref A, ref Aplus, relTol, Consts.sweepBudget(math.min(A.M_Rows, A.N_Cols)));

        /// <summary>pseudoInverse with default relTol (-1, auto tolerance) and maxSweeps (Consts.sweepBudget(min(A.M_Rows, A.N_Cols))).</summary>
        public static RankInfo pseudoInverse(ref fProxyMxN A, ref fProxyMxN Aplus)
            => pseudoInverse(ref A, ref Aplus, (fProxy)(-1), Consts.sweepBudget(math.min(A.M_Rows, A.N_Cols)));

        // ---- multi-RHS form: minimum-norm least-squares for a whole matrix of right-hand sides ----
        //
        // X = A⁺B, each RHS a COLUMN of B (m x nrhs) / X (n x nrhs). One SVD, reused across all
        // right-hand sides — the O(n³) factorization amortizes; the solve step (X += coeff·M) is
        // O(n²·nrhs) and is exactly the per-column vector pinvSolve, run for every column.

        /// <summary>
        /// Minimum-norm least-squares solve for a whole block of right-hand sides:
        /// X = argmin ‖A X - B‖ (minimum ‖X‖ among minimizers), each RHS a column of B (m x nrhs);
        /// X is n x nrhs. Any shape/rank. A and B are not modified. relTol &lt; 0 selects the auto
        /// tolerance (max(m,n)·Consts.fProxyZeroThreshold). Returns a <see cref="RankInfo"/> — see the
        /// vector pinvSolve for the identical rank/convergence semantics (X is zeroed on NotConverged).
        /// </summary>
        // Caller-provided scratch overload (zero-alloc); scratch layout: see fProxySVDCache.
        public static RankInfo pinvSolve(ref fProxyMxN A, in fProxyMxN B, ref fProxyMxN X,
                                    fProxy relTol, int maxSweeps,
                                    ref fProxyN S, ref fProxyMxN M, ref fProxyMxN U, ref fProxyMxN At)
        {
            if (B.M_Rows != A.M_Rows)
                throw new ArgumentException("pinvSolve: B.M_Rows must equal A.M_Rows");

            if (X.M_Rows != A.N_Cols)
                throw new ArgumentException("pinvSolve: X.M_Rows must equal A.N_Cols");

            if (X.N_Cols != B.N_Cols)
                throw new ArgumentException("pinvSolve: X.N_Cols must equal B.N_Cols");

            if (maxSweeps < 1)
                throw new ArgumentException("pinvSolve: maxSweeps must be >= 1");

            int m = A.M_Rows;
            int n = A.N_Cols;
            int k = math.min(m, n);
            int big = math.max(m, n);
            int nrhs = B.N_Cols;

            if (S.N != k)
                throw new ArgumentException("pinvSolve: S scratch length must equal min(A.M_Rows, A.N_Cols)");

            if (M.M_Rows != k || M.N_Cols != k)
                throw new ArgumentException("pinvSolve: M scratch must be k x k, k = min(A.M_Rows, A.N_Cols)");

            if (U.M_Rows != big || U.N_Cols != k)
                throw new ArgumentException("pinvSolve: U scratch must be max(m,n) x min(m,n)");

            // Zero X (prior contents ignored either way, per the doc comment).
            for (int r = 0; r < n; r++)
                for (int c = 0; c < nrhs; c++)
                    X[r, c] = (fProxy)0;

            if (m >= n) {
                // Tall/square: A = U diag(S) Vᵀ; U receives the left factor, M = V.
                SVDInfo svdInfo = thin(in A, ref U, ref S, ref M, maxSweeps);

                if (!svdInfo)
                    return new RankInfo { status = DirectSolveStatus.NotConverged, rank = 0 };

                if (relTol < (fProxy)0)
                    relTol = (fProxy)math.max(m, n) * Consts.fProxyZeroThreshold;

                if (n == 0 || S[0] == (fProxy)0)
                    return new RankInfo { status = k == 0 ? DirectSolveStatus.Success : DirectSolveStatus.RankDeficient, rank = 0 };

                fProxy tol = relTol * S[0];
                int rank = 0;

                // X = V diag(1/S_j) Uᵀ B  (only for S[j] > tol)
                for (int j = 0; j < n; j++) {
                    if (S[j] <= tol)
                        continue;

                    fProxy invS = (fProxy)1 / S[j];
                    for (int c = 0; c < nrhs; c++) {
                        // coeff = (U[:,j]ᵀ B[:,c]) / S[j]
                        fProxy dot = (fProxy)0;
                        for (int i = 0; i < m; i++)
                            dot += U[i, j] * B[i, c];
                        fProxy coeff = dot * invS;
                        for (int r = 0; r < n; r++)
                            X[r, c] += coeff * M[r, j];
                    }

                    rank++;
                }

                return new RankInfo { status = rank == k ? DirectSolveStatus.Success : DirectSolveStatus.RankDeficient, rank = rank };
            }
            else {
                // Wide: decompose Aᵀ (n x m, tall). Right singular vectors of A are columns of U (left
                // factor of Aᵀ); left singular vectors of A are columns of M (= W).
                if (At.M_Rows != n || At.N_Cols != m)
                    throw new ArgumentException("pinvSolve: At scratch must be A.N_Cols x A.M_Rows for the wide (m < n) case");

                Blas.trans(in A, ref At);   // At = Aᵀ (zero-alloc, ref-dest trans)

                SVDInfo svdInfo = thin(in At, ref U, ref S, ref M, maxSweeps);

                if (!svdInfo)
                    return new RankInfo { status = DirectSolveStatus.NotConverged, rank = 0 };

                if (relTol < (fProxy)0)
                    relTol = (fProxy)math.max(m, n) * Consts.fProxyZeroThreshold;

                if (m == 0 || S[0] == (fProxy)0)
                    return new RankInfo { status = k == 0 ? DirectSolveStatus.Success : DirectSolveStatus.RankDeficient, rank = 0 };

                fProxy tol = relTol * S[0];
                int rank = 0;

                // X = U diag(1/S_j) Wᵀ B  (only for S[j] > tol); U columns (length n) are right
                // singular vectors of A, M (= W) columns (length m) are left singular vectors of A.
                for (int j = 0; j < m; j++) {
                    if (S[j] <= tol)
                        continue;

                    fProxy invS = (fProxy)1 / S[j];
                    for (int c = 0; c < nrhs; c++) {
                        fProxy dot = (fProxy)0;
                        for (int i = 0; i < m; i++)
                            dot += M[i, j] * B[i, c];
                        fProxy coeff = dot * invS;
                        for (int r = 0; r < n; r++)
                            X[r, c] += coeff * U[r, j];
                    }

                    rank++;
                }

                return new RankInfo { status = rank == k ? DirectSolveStatus.Success : DirectSolveStatus.RankDeficient, rank = rank };
            }
        }

        /// <summary>pinvSolve (multi-RHS) allocating wrapper: allocates the SVD scratch from A's arena.</summary>
        public static RankInfo pinvSolve(ref fProxyMxN A, in fProxyMxN B, ref fProxyMxN X,
                                    fProxy relTol, int maxSweeps)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            int k = math.min(m, n);
            int big = math.max(m, n);

            fProxyN S = A.fProxyTempVec(k);
            fProxyMxN M = A.fProxyTempMat(k, k);
            fProxyMxN U = A.fProxyTempMat(big, k);
            fProxyMxN At = default;
            if (m < n)
                At = A.fProxyTempMat(n, m);

            return pinvSolve(ref A, in B, ref X, relTol, maxSweeps, ref S, ref M, ref U, ref At);
        }

        /// <summary>pinvSolve (multi-RHS) using a reusable workspace (Arena.fProxySVDCache(m, n)) — zero-alloc.</summary>
        public static RankInfo pinvSolve(ref fProxyMxN A, in fProxyMxN B, ref fProxyMxN X,
                                    ref fProxySVDCache ws, fProxy relTol, int maxSweeps)
            => pinvSolve(ref A, in B, ref X, relTol, maxSweeps, ref ws.S, ref ws.M, ref ws.U, ref ws.At);

        /// <summary>pinvSolve (multi-RHS, workspace) with default maxSweeps.</summary>
        public static RankInfo pinvSolve(ref fProxyMxN A, in fProxyMxN B, ref fProxyMxN X,
                                    ref fProxySVDCache ws, fProxy relTol)
            => pinvSolve(ref A, in B, ref X, ref ws, relTol, Consts.sweepBudget(math.min(A.M_Rows, A.N_Cols)));

        /// <summary>pinvSolve (multi-RHS, workspace) with default relTol (auto) and maxSweeps.</summary>
        public static RankInfo pinvSolve(ref fProxyMxN A, in fProxyMxN B, ref fProxyMxN X,
                                    ref fProxySVDCache ws)
            => pinvSolve(ref A, in B, ref X, ref ws, (fProxy)(-1), Consts.sweepBudget(math.min(A.M_Rows, A.N_Cols)));

        /// <summary>pinvSolve (multi-RHS) with default maxSweeps.</summary>
        public static RankInfo pinvSolve(ref fProxyMxN A, in fProxyMxN B, ref fProxyMxN X,
                                    fProxy relTol)
            => pinvSolve(ref A, in B, ref X, relTol, Consts.sweepBudget(math.min(A.M_Rows, A.N_Cols)));

        /// <summary>pinvSolve (multi-RHS) with default relTol (auto) and maxSweeps.</summary>
        public static RankInfo pinvSolve(ref fProxyMxN A, in fProxyMxN B, ref fProxyMxN X)
            => pinvSolve(ref A, in B, ref X, (fProxy)(-1), Consts.sweepBudget(math.min(A.M_Rows, A.N_Cols)));
    }
}
