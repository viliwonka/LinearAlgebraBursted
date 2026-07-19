using System;
using Unity.Collections;
using Unity.Mathematics;
using LinearAlgebra.Sparse;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    public static partial class Krylov
    {
        // ================= Block (multi-RHS) Krylov solvers =================
        // Block vectors are fProxyMxN with s ROWS x n COLS: row j is RHS/solution vector j (length
        // n = A.Rows). This matches IfProxyLinearOperator.ApplyBlock, which applies A to all s rows in
        // one streaming pass. The s x s block coefficients are formed and solved in the row space.

        // ---- block helpers (private) --------------------------------------------------------------

        // G = V * W^T  (s x s Gram of the block rows) via the optimized GEMM, then symmetrized
        // exactly (the Grams here -- P^T A P, R^T M^-1 R -- are symmetric by construction; forcing it
        // shields the s x s Cholesky from GEMM round-off asymmetry).
        static void BlockGram(in fProxyMxN V, in fProxyMxN W, ref fProxyMxN G, int s)
        {
            Blas.dot(in V, in W, ref G, false, true);       // G = V W^T
            for (int i = 0; i < s; i++)
                for (int j = i + 1; j < s; j++)
                {
                    fProxy avg = (fProxy)0.5 * (G[i, j] + G[j, i]);
                    G[i, j] = avg;
                    G[j, i] = avg;
                }
        }

        // dst = C^T * V  (s x n block) via the optimized GEMM. dst must be distinct from C and V.
        static void BlockCTV(in fProxyMxN C, in fProxyMxN V, ref fProxyMxN dst)
            => Blas.dot(in C, in V, ref dst, true, false);   // dst = C^T V

        // Y += sign * T over the whole (contiguous) s x n block.
        static unsafe void BlockAdd(ref fProxyMxN Y, in fProxyMxN T, fProxy sign)
        {
            fProxy* yp = Y.Data.Ptr; fProxy* tp = T.Data.Ptr;
            long len = (long)Y.M_Rows * Y.N_Cols;
            for (long i = 0; i < len; i++) yp[i] += sign * tp[i];
        }

        // dst = Z + T over the whole (contiguous) s x n block. dst may be Z? no -- dst distinct.
        static unsafe void BlockZplusT(in fProxyMxN Z, in fProxyMxN T, ref fProxyMxN dst)
        {
            fProxy* zp = Z.Data.Ptr; fProxy* tp = T.Data.Ptr; fProxy* dp = dst.Data.Ptr;
            long len = (long)dst.M_Rows * dst.N_Cols;
            for (long i = 0; i < len; i++) dp[i] = zp[i] + tp[i];
        }

        // Solve the s x s SPD system G * Xsol = RHS_to_X (each column an independent RHS), writing the
        // solution into RHS_to_X. Robust to a rank-deficient G (dependent RHS columns): retries an
        // escalating diagonal ridge scaled to G's own diagonal, mirroring LOBPCG's FactorGram. `work`
        // is s x s scratch (overwritten). Returns false only if even the largest ridge fails.
        static bool BlockSolveSPD(in fProxyMxN G, ref fProxyMxN RHS_to_X, ref fProxyMxN work, int s)
        {
            fProxy diagMax = (fProxy)0;
            for (int i = 0; i < s; i++) { fProxy d = G[i, i]; if (d > diagMax) diagMax = d; }
            if (diagMax <= (fProxy)0) diagMax = (fProxy)1;

            fProxy ridge = (fProxy)0;
            bool ok = false;
            for (int attempt = 0; attempt < 6; attempt++)
            {
                for (int i = 0; i < s; i++)
                    for (int j = 0; j < s; j++)
                        work[i, j] = G[i, j] + (i == j ? ridge : (fProxy)0);

                var info = CHO.decompInPlace(ref work);   // work -> L
                if (info.status == DirectSolveStatus.Success) { ok = true; break; }
                ridge = ridge == (fProxy)0 ? (fProxy)16 * Consts.fProxyEpsilon * diagMax : ridge * (fProxy)16;
            }
            if (!ok) return false;

            CHO.decompSolve(ref work, ref RHS_to_X);       // in-place forward/back substitution
            return true;
        }

        // Counts columns with ||R[j]||^2 <= thr[j]; also returns the worst ||R[j]||.
        static int CountConverged(in fProxyMxN R, in fProxyN thr, int s, int n, out double maxRnorm)
        {
            int conv = 0; double worst = 0;
            for (int j = 0; j < s; j++)
            {
                fProxy rr = (fProxy)0;
                for (int c = 0; c < n; c++) rr += R[j, c] * R[j, c];
                if (rr <= thr[j]) conv++;
                double rn = math.sqrt((double)rr);
                if (rn > worst) worst = rn;
            }
            maxRnorm = worst;
            return conv;
        }

        static void BlockApplyPre<TPre>(in TPre M, in fProxyMxN R, ref fProxyMxN Z, int s, int n,
                                        ref fProxyN rowIn, ref fProxyN rowOut)
            where TPre : struct, IfProxyPreconditioner
        {
            for (int i = 0; i < s; i++)
            {
                for (int c = 0; c < n; c++) rowIn[c] = R[i, c];
                M.Apply(in rowIn, ref rowOut);
                for (int c = 0; c < n; c++) Z[i, c] = rowOut[c];
            }
        }

        static void CopyBlock(in fProxyMxN src, ref fProxyMxN dst, int s, int n)
        {
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++) dst[i, c] = src[i, c];
        }

        static void CopyMat(in fProxyMxN src, ref fProxyMxN dst, int s)
        {
            for (int i = 0; i < s; i++)
                for (int j = 0; j < s; j++) dst[i, j] = src[i, j];
        }

        // ---- block-CG core -------------------------------------------------------------------------

        /// <summary>
        /// Zero-alloc block (multi-RHS) Conjugate Gradient for an SPD A and s simultaneous right-hand
        /// sides, generic over BOTH the operator (<see cref="IfProxyLinearOperator"/>) and the
        /// preconditioner (<see cref="IfProxyPreconditioner"/>). A TRUE block method: it builds ONE
        /// shared Krylov subspace from all s RHS, with s x s block coefficients, streaming A over the
        /// whole block once per iteration via <c>ApplyBlock</c> -- not s independent scalar solves. All
        /// s columns share the iteration count, so they converge together in ≤ the iterations the
        /// slowest single RHS would need alone (the block-Krylov advantage).
        ///
        /// B and X are s ROWS x n COLS (row j = the j-th RHS / solution, length n = A.Rows); X is
        /// warm-startable. R, P, Q, Z are s x n block scratch (Z UNUSED under the identity
        /// preconditioner -- pass <c>default</c>). Convergence is per column against tol²·‖B[j]‖²; a
        /// rank-deficient RHS block (linearly dependent columns) is handled by a ridge-regularized s x s
        /// solve rather than breaking down. Returns a <see cref="BlockSolveInfo"/>.
        /// </summary>
        public static BlockSolveInfo cg<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                        ref fProxyMxN R, ref fProxyMxN P, ref fProxyMxN Q, ref fProxyMxN Z,
                                        int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols) throw new ArgumentException("cg (block): A must be square");
            int n = A.Rows;
            int s = B.M_Rows;
            if (B.N_Cols != n) throw new ArgumentException("cg (block): B must be s x A.Rows");
            if (X.M_Rows != s || X.N_Cols != n) throw new ArgumentException("cg (block): X must match B");
            if (R.M_Rows != s || R.N_Cols != n) throw new ArgumentException("cg (block): R must match B");
            if (P.M_Rows != s || P.N_Cols != n) throw new ArgumentException("cg (block): P must match B");
            if (Q.M_Rows != s || Q.N_Cols != n) throw new ArgumentException("cg (block): Q must match B");
            if (!M.IsIdentity && (Z.M_Rows != s || Z.N_Cols != n))
                throw new ArgumentException("cg (block): Z must match B");
            if (maxIter < 1) throw new ArgumentException("cg (block): maxIter must be >= 1");

            // s x s coefficient scratch + per-column thresholds + row scratch for the preconditioner.
            var PQ    = new fProxyMxN(s, s, Allocator.Temp, true);
            var RZ    = new fProxyMxN(s, s, Allocator.Temp, true);
            var RZnew = new fProxyMxN(s, s, Allocator.Temp, true);
            var coef  = new fProxyMxN(s, s, Allocator.Temp, true);
            var work  = new fProxyMxN(s, s, Allocator.Temp, true);
            var T     = new fProxyMxN(s, n, Allocator.Temp, true);   // s x n GEMM target for block updates
            var thr   = new fProxyN(s);
            fProxyN rowIn = default, rowOut = default;
            if (!M.IsIdentity) { rowIn = new fProxyN(n); rowOut = new fProxyN(n); }

            IterativeSolveStatus status = IterativeSolveStatus.MaxIterations;
            int iters = maxIter;
            int converged = 0;
            double maxr = 0;

            // Per-column thresholds tol^2 ||B[j]||^2.
            for (int j = 0; j < s; j++)
            {
                fProxy bb = (fProxy)0;
                for (int c = 0; c < n; c++) bb += B[j, c] * B[j, c];
                thr[j] = tol * tol * bb;
            }

            // R = B - A X.
            A.ApplyBlock(in X, ref Q, s);                 // Q = A X (temp use of Q)
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++) R[i, c] = B[i, c] - Q[i, c];

            converged = CountConverged(in R, in thr, s, n, out maxr);
            if (converged == s) { status = IterativeSolveStatus.Converged; iters = 0; goto cleanup; }

            // Z = M^-1 R (identity: Z == R, used directly); P = Z; RZ = R^T Z.
            if (M.IsIdentity)
            {
                CopyBlock(in R, ref P, s, n);
                BlockGram(in R, in R, ref RZ, s);
            }
            else
            {
                BlockApplyPre(in M, in R, ref Z, s, n, ref rowIn, ref rowOut);
                CopyBlock(in Z, ref P, s, n);
                BlockGram(in R, in Z, ref RZ, s);
            }

            for (int k = 0; k < maxIter; k++)
            {
                A.ApplyBlock(in P, ref Q, s);             // Q = A P
                BlockGram(in P, in Q, ref PQ, s);         // PQ = P^T A P (s x s SPD)

                // alpha = (P^T A P)^-1 (R^T Z);   coef <- RZ, solved in place.
                CopyMat(in RZ, ref coef, s);
                if (!BlockSolveSPD(in PQ, ref coef, ref work, s))
                { status = IterativeSolveStatus.Breakdown; iters = k; goto cleanup; }

                BlockCTV(in coef, in P, ref T); BlockAdd(ref X, in T, (fProxy)1);    // X += alpha^T P
                BlockCTV(in coef, in Q, ref T); BlockAdd(ref R, in T, (fProxy)(-1)); // R -= alpha^T Q

                converged = CountConverged(in R, in thr, s, n, out maxr);
                if (converged == s) { status = IterativeSolveStatus.Converged; iters = k + 1; goto cleanup; }

                // Z = M^-1 R ; RZnew = R^T Z.
                if (M.IsIdentity)
                    BlockGram(in R, in R, ref RZnew, s);
                else
                {
                    BlockApplyPre(in M, in R, ref Z, s, n, ref rowIn, ref rowOut);
                    BlockGram(in R, in Z, ref RZnew, s);
                }

                // beta = (R^T Z)^-1 (R_new^T Z_new);   coef <- RZnew, solved in place.
                CopyMat(in RZnew, ref coef, s);
                if (!BlockSolveSPD(in RZ, ref coef, ref work, s))
                { status = IterativeSolveStatus.Breakdown; iters = k + 1; goto cleanup; }

                // P = Zeff + beta^T P: stage beta^T P into Q (free now), then P = Zeff + Q.
                BlockCTV(in coef, in P, ref Q);
                if (M.IsIdentity) BlockZplusT(in R, in Q, ref P);
                else              BlockZplusT(in Z, in Q, ref P);

                CopyMat(in RZnew, ref RZ, s);
            }

        cleanup:
            PQ.Dispose(); RZ.Dispose(); RZnew.Dispose(); coef.Dispose(); work.Dispose(); T.Dispose(); thr.Dispose();
            if (!M.IsIdentity) { rowIn.Dispose(); rowOut.Dispose(); }
            // Ridge version keeps the full block width every iteration (no column dropping) -> minActive = s.
            return new BlockSolveInfo { rhs = s, converged = converged, iterations = iters, maxRnorm = maxr, minActive = s, status = status };
        }

        // ---- unpreconditioned + concrete forwarders ------------------------------------------------

        /// <summary>Unpreconditioned block-CG -- forwards into the merged block
        /// <see cref="cg{TOp, TPre}(in TOp, in TPre, in fProxyMxN, ref fProxyMxN, ref fProxyMxN, ref fProxyMxN, ref fProxyMxN, ref fProxyMxN, int, fProxy)"/>
        /// with the identity preconditioner (needs no Z block).</summary>
        public static BlockSolveInfo cg<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X,
                                        ref fProxyMxN R, ref fProxyMxN P, ref fProxyMxN Q,
                                        int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            fProxyMxN Z = default;
            return cg(in A, default(fProxyIdentityPreconditioner), in B, ref X, ref R, ref P, ref Q, ref Z, maxIter, tol);
        }

        /// <summary>Block-CG over a dense SPD <see cref="fProxyMxN"/> A (n x n) with an s x n block B.
        /// Allocates block scratch from the arena.</summary>
        public static BlockSolveInfo cg(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
        {
            int s = B.M_Rows, n = A.M_Rows;
            fProxyMxN R = B.fProxyTempMat(s, n, true), P = B.fProxyTempMat(s, n, true), Q = B.fProxyTempMat(s, n, true);
            return cg(new fProxyDenseOperator(in A), in B, ref X, ref R, ref P, ref Q, maxIter, tol);
        }

        /// <summary>Block-CG over a dense SPD A with default maxIter (A.M_Rows) and tol (sqrtEps).</summary>
        public static BlockSolveInfo cg(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X)
            => cg(in A, in B, ref X, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Preconditioned block-CG over a dense SPD A. Allocates block scratch (incl. Z).</summary>
        public static BlockSolveInfo cg<TPre>(in fProxyMxN A, in TPre M, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
            where TPre : struct, IfProxyPreconditioner
        {
            int s = B.M_Rows, n = A.M_Rows;
            fProxyMxN R = B.fProxyTempMat(s, n, true), P = B.fProxyTempMat(s, n, true),
                      Q = B.fProxyTempMat(s, n, true), Z = B.fProxyTempMat(s, n, true);
            return cg(new fProxyDenseOperator(in A), in M, in B, ref X, ref R, ref P, ref Q, ref Z, maxIter, tol);
        }

        /// <summary>Block-CG over a block-sparse (BSR) SPD A with an s x n block B. Allocates block
        /// scratch from the arena.</summary>
        public static BlockSolveInfo cg(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
        {
            int s = B.M_Rows, n = A.M_Rows;
            fProxyMxN R = B.fProxyTempMat(s, n, true), P = B.fProxyTempMat(s, n, true), Q = B.fProxyTempMat(s, n, true);
            return cg(new fProxyBSROperator(in A), in B, ref X, ref R, ref P, ref Q, maxIter, tol);
        }

        /// <summary>Block-CG over a BSR SPD A with default maxIter (A.M_Rows) and tol (sqrtEps).</summary>
        public static BlockSolveInfo cg(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X)
            => cg(in A, in B, ref X, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Preconditioned block-CG over a BSR SPD A. Allocates block scratch (incl. Z).</summary>
        public static BlockSolveInfo cg<TPre>(in fProxyBSR A, in TPre M, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
            where TPre : struct, IfProxyPreconditioner
        {
            int s = B.M_Rows, n = A.M_Rows;
            fProxyMxN R = B.fProxyTempMat(s, n, true), P = B.fProxyTempMat(s, n, true),
                      Q = B.fProxyTempMat(s, n, true), Z = B.fProxyTempMat(s, n, true);
            return cg(new fProxyBSROperator(in A), in M, in B, ref X, ref R, ref P, ref Q, ref Z, maxIter, tol);
        }
    }
}
