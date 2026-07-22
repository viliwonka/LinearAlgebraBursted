using System;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    /// <summary>
    /// Augmented operator [A | damp·I] (Rows × (Cols+Rows)) behind the Tikhonov-DAMPED least-norm
    /// solvers. The minimum-norm solution of [A | damp·I]·(x,s) = b minimizes ‖x‖² + ‖s‖² subject to
    /// A x + damp·s = b, i.e. x = Aᵀ(A Aᵀ + damp²·I)⁻¹ b -- the ridge-regularized least-norm solution.
    /// So running an UNDAMPED least-norm solver (craig/craigmr/lnlq) over this operator yields the
    /// damped solve with NO recurrence change; the solver's solution vector is (x, s) and x is its
    /// first Cols entries. Needs one Cols-length scratch (the x-part copy / Aᵀx). Apply/ApplyT only
    /// (Golub-Kahan bidiagonalization uses no ApplyDot/ApplyBlock).
    /// </summary>
    public readonly struct fProxyDampedLeastNormOperator<TOp> : IfProxyLinearOperator
        where TOp : struct, IfProxyLinearOperator
    {
        readonly TOp _A;
        readonly fProxy _damp;
        readonly fProxyN _xScratch;   // length _A.Cols

        public fProxyDampedLeastNormOperator(in TOp A, fProxy damp, in fProxyN xScratch)
        {
            _A = A;
            _damp = damp;
            _xScratch = xScratch;
        }

        public int Rows => _A.Rows;
        public int Cols => _A.Cols + _A.Rows;

        // y = [A | damp·I] z = A·z[0:Ac] + damp·z[Ac:Ac+Ar]
        public void Apply(in fProxyN z, ref fProxyN y)
        {
            int ac = _A.Cols, ar = _A.Rows;
            var xp = _xScratch;
            for (int i = 0; i < ac; i++) xp[i] = z[i];
            _A.Apply(in xp, ref y);
            for (int i = 0; i < ar; i++) y[i] += _damp * z[ac + i];
        }

        // z = [A | damp·I]ᵀ x = (Aᵀx, damp·x)
        public void ApplyT(in fProxyN x, ref fProxyN z)
        {
            int ac = _A.Cols, ar = _A.Rows;
            var xp = _xScratch;
            _A.ApplyT(in x, ref xp);
            for (int i = 0; i < ac; i++) z[i] = xp[i];
            for (int i = 0; i < ar; i++) z[ac + i] = _damp * x[i];
        }

        public fProxy ApplyDot(in fProxyN x, ref fProxyN y)
            => throw new NotSupportedException("fProxyDampedLeastNormOperator: ApplyDot unused (Golub-Kahan uses Apply/ApplyT only)");

        public void ApplyBlock(in fProxyMxN Vrows, ref fProxyMxN AVrows, int rows)
            => throw new NotSupportedException("fProxyDampedLeastNormOperator: ApplyBlock unused");
    }

    public static partial class Krylov
    {
        // Extract x (first Ac entries of the augmented solution z) into caller x, then re-audit
        // diagnostics in ORIGINAL coordinates (like RightPreFinish / damped cgne): rnorm = ‖b-Ax‖
        // (= damp·‖s‖ at the optimum, legitimately nonzero), Arnorm = ‖Aᵀr - damp²x‖ (→0 -- the real
        // convergence cert), xnorm = ‖x‖; iterations/status from the augmented solve.
        static LstsqInfo DampedLeastNormFinish<TOp>(in TOp origOp, in fProxyN b, in fProxyN z, ref fProxyN x,
                                                    fProxy damp, int iterations, IterativeSolveStatus status,
                                                    ref fProxyN mScratch, ref fProxyN nScratch)
            where TOp : struct, IfProxyLinearOperator
        {
            int ac = origOp.Cols;
            for (int i = 0; i < ac; i++) x[i] = z[i];
            var info = lstsqResidual(in origOp, in b, in x, damp, ref mScratch, ref nScratch);
            info.iterations = iterations;
            info.status = status;
            return info;
        }

        // ===== craig: Tikhonov-damped least-norm =====

        /// <summary>
        /// Damped CRAIG: minimum-norm solve of the ridge-regularized underdetermined system,
        /// x = Aᵀ(A Aᵀ + damp²·I)⁻¹ b. Runs UNDAMPED <see cref="craig{TOp}"/> over the augmented
        /// operator [A | damp·I] (no recurrence change) and returns x. damp == 0 delegates to plain
        /// craig. The <see cref="LstsqInfo"/> is re-audited in original coords: rnorm = ‖b-Ax‖
        /// (nonzero at the damped optimum), Arnorm = ‖Aᵀr - damp²x‖ (→0), read status/Arnorm not rnorm.
        /// </summary>
        public static LstsqInfo craig(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol, fProxy damp)
        {
            if (damp == (fProxy)0) return craig(in A, in b, ref x, maxIter, tol);
            return CraigDampedCore(new fProxyDenseOperator(in A), in b, ref x, A.M_Rows, A.N_Cols, maxIter, tol, damp);
        }

        /// <summary>Damped CRAIG over a BSR matrix -- see the dense overload. damp == 0 delegates.</summary>
        public static LstsqInfo craig(in fProxyBSR A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol, fProxy damp)
        {
            if (damp == (fProxy)0) return craig(in A, in b, ref x, maxIter, tol);
            return CraigDampedCore(new fProxyBSROperator(in A), in b, ref x, A.M_Rows, A.N_Cols, maxIter, tol, damp);
        }

        static LstsqInfo CraigDampedCore<TOp>(in TOp origOp, in fProxyN b, ref fProxyN x, int ar, int ac,
                                              int maxIter, fProxy tol, fProxy damp)
            where TOp : struct, IfProxyLinearOperator
        {
            int augCols = ac + ar;
            var xScratch = b.fProxyTempVec(ac);
            var aug = new fProxyDampedLeastNormOperator<TOp>(in origOp, damp, in xScratch);

            var z    = b.fProxyTempVec(augCols);
            var u    = b.fProxyTempVec(ar);        // aug.Rows
            var v    = b.fProxyTempVec(augCols);   // aug.Cols
            var tmpM = b.fProxyTempVec(ar);
            var tmpN = b.fProxyTempVec(augCols);
            for (int i = 0; i < augCols; i++) z[i] = (fProxy)0;

            var ci = craig(in aug, in b, ref z, ref u, ref v, ref tmpM, ref tmpN, maxIter, tol);

            var mScratch = b.fProxyTempVec(ar);
            var nScratch = b.fProxyTempVec(ac);
            return DampedLeastNormFinish(in origOp, in b, in z, ref x, damp, ci.iterations, ci.status, ref mScratch, ref nScratch);
        }

        // ===== craigmr: Tikhonov-damped least-norm (monotonic-residual CRAIG) =====

        /// <summary>
        /// Damped CRAIGMR: ridge least-norm x = Aᵀ(A Aᵀ + damp²·I)⁻¹ b via UNDAMPED
        /// <see cref="craigmr{TOp}"/> over the augmented operator [A | damp·I]. damp == 0 delegates.
        /// See <see cref="craig(in fProxyMxN, in fProxyN, ref fProxyN, int, fProxy, fProxy)"/> for the
        /// original-coord diagnostics contract (rnorm = ‖b-Ax‖ nonzero, Arnorm = ‖Aᵀr-damp²x‖ →0).
        /// (lnlq is intentionally NOT damped this way: its certified forward-error bound applies to
        /// the augmented (x,s), not to x, so the augmentation defeats its distinctive feature.)
        /// </summary>
        public static LstsqInfo craigmr(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol, fProxy damp)
        {
            if (damp == (fProxy)0) return craigmr(in A, in b, ref x, maxIter, tol);
            return CraigmrDampedCore(new fProxyDenseOperator(in A), in b, ref x, A.M_Rows, A.N_Cols, maxIter, tol, damp);
        }

        /// <summary>Damped CRAIGMR over a BSR matrix -- see the dense overload. damp == 0 delegates.</summary>
        public static LstsqInfo craigmr(in fProxyBSR A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol, fProxy damp)
        {
            if (damp == (fProxy)0) return craigmr(in A, in b, ref x, maxIter, tol);
            return CraigmrDampedCore(new fProxyBSROperator(in A), in b, ref x, A.M_Rows, A.N_Cols, maxIter, tol, damp);
        }

        static LstsqInfo CraigmrDampedCore<TOp>(in TOp origOp, in fProxyN b, ref fProxyN x, int ar, int ac,
                                                int maxIter, fProxy tol, fProxy damp)
            where TOp : struct, IfProxyLinearOperator
        {
            int augCols = ac + ar;
            var xScratch = b.fProxyTempVec(ac);
            var aug = new fProxyDampedLeastNormOperator<TOp>(in origOp, damp, in xScratch);

            var z    = b.fProxyTempVec(augCols);
            var u    = b.fProxyTempVec(ar);
            var v    = b.fProxyTempVec(augCols);
            var d    = b.fProxyTempVec(augCols);
            var tmpM = b.fProxyTempVec(ar);
            var tmpN = b.fProxyTempVec(augCols);
            for (int i = 0; i < augCols; i++) z[i] = (fProxy)0;

            var ci = craigmr(in aug, in b, ref z, ref u, ref v, ref d, ref tmpM, ref tmpN, maxIter, tol);

            var mScratch = b.fProxyTempVec(ar);
            var nScratch = b.fProxyTempVec(ac);
            return DampedLeastNormFinish(in origOp, in b, in z, ref x, damp, ci.iterations, ci.status, ref mScratch, ref nScratch);
        }
    }
}
