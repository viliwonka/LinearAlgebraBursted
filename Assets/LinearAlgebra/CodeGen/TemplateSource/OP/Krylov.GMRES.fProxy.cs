using System;
using Unity.Mathematics;
using BULA.Sparse;

namespace BULA
{
    public static partial class Krylov
    {
        /// <summary>
        /// Restarted GMRES(m) for a general (nonsymmetric) square A x = b, generic over BOTH the
        /// operator (<see cref="IfProxyLinearOperator"/>) and the preconditioner
        /// (<see cref="IfProxyPreconditioner"/>). Thin façade over the shared
        /// <see cref="GmresCore{TOp,TPre}"/>: requires a CONSTANT preconditioner (rejected at entry
        /// otherwise -- use <see cref="fgmres{TOp,TPre}"/> for a per-step-varying M). With a real M
        /// it runs GMRES on A·M⁻¹ (right preconditioning); with
        /// <see cref="fProxyIdentityPreconditioner"/> the IsIdentity fold makes it plain GMRES
        /// bit-for-bit. x is a warm-startable initial guess, overwritten with the solution; tol is
        /// relative (‖b − Ax‖ ≤ tol·‖b‖); maxIter counts TOTAL inner iterations across restarts.
        ///
        /// Allocates its workspace (restart+1 basis vectors + a small Hessenberg / Givens set) from
        /// the Temp allocator. Returns the shared <see cref="SolveInfo"/>; rnorm is a freshly
        /// recomputed ‖b−Ax‖, verified before a Converged exit is reported. Status: Converged /
        /// MaxIterations; a happy-breakdown (exact Krylov solution) converges; an exact-zero
        /// Arnoldi/Givens pivot with no usable column reports Breakdown.
        /// </summary>
        public static SolveInfo gmres<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x, int restart, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols) throw new ArgumentException("gmres: A must be square");
            if (b.N != A.Rows) throw new ArgumentException("gmres: b.N must equal A.Rows");
            if (x.N != A.Rows) throw new ArgumentException("gmres: x.N must equal A.Rows");
            if (restart < 1) throw new ArgumentException("gmres: restart must be >= 1");
            if (maxIter < 1) throw new ArgumentException("gmres: maxIter must be >= 1");
            if (!M.IsConstant)
                throw new ArgumentException("Krylov.gmres: requires a constant (non-flexible) preconditioner (M.IsConstant == false — e.g. an AMG K-cycle). Use the flexible variant (fcg / fgmres).");

            return GmresCore(in A, in M, in b, ref x, restart, maxIter, tol);
        }

        /// <summary>
        /// Unpreconditioned restarted GMRES(m) -- forwards into the merged
        /// <see cref="gmres{TOp, TPre}(in TOp, in TPre, in fProxyN, ref fProxyN, int, int, fProxy)"/>
        /// with the identity preconditioner (whose IsIdentity fold strips the M⁻¹ applies and the
        /// preconditioned workspace).
        /// </summary>
        public static SolveInfo gmres<TOp>(in TOp A, in fProxyN b, ref fProxyN x, int restart, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            return gmres(in A, default(fProxyIdentityPreconditioner), in b, ref x, restart, maxIter, tol);
        }

        /// <summary>GMRES(m) over a dense <see cref="fProxyMxN"/>. Forwards via fProxyDenseOperator.</summary>
        public static SolveInfo gmres(in fProxyMxN A, in fProxyN b, ref fProxyN x, int restart, int maxIter, fProxy tol)
            => gmres(new fProxyDenseOperator(in A), in b, ref x, restart, maxIter, tol);

        /// <summary>GMRES over a dense matrix with defaults (restart = min(30, N), maxIter = N, tol = sqrtEps).</summary>
        public static SolveInfo gmres(in fProxyMxN A, in fProxyN b, ref fProxyN x)
            => gmres(new fProxyDenseOperator(in A), in b, ref x, math.min(30, A.M_Rows), A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>GMRES(m) over a block-sparse (BSR) matrix. Forwards via fProxyBSROperator.</summary>
        public static SolveInfo gmres(in fProxyBSR A, in fProxyN b, ref fProxyN x, int restart, int maxIter, fProxy tol)
            => gmres(new fProxyBSROperator(in A), in b, ref x, restart, maxIter, tol);

        /// <summary>GMRES over a BSR matrix with defaults (restart = min(30, N), maxIter = N, tol = sqrtEps).</summary>
        public static SolveInfo gmres(in fProxyBSR A, in fProxyN b, ref fProxyN x)
            => gmres(new fProxyBSROperator(in A), in b, ref x, math.min(30, A.M_Rows), A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Right-preconditioned GMRES(m) over a BSR matrix with ANY <see cref="IfProxyPreconditioner"/> (ILU0).</summary>
        public static SolveInfo gmres<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x, int restart, int maxIter, fProxy tol)
            where TPre : struct, IfProxyPreconditioner
            => gmres(new fProxyBSROperator(in A), in M, in b, ref x, restart, maxIter, tol);

        /// <summary>Right-preconditioned GMRES over a BSR matrix with ANY <see cref="IfProxyPreconditioner"/> (ILU0), with defaults (restart = min(30, N)).</summary>
        public static SolveInfo gmres<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x)
            where TPre : struct, IfProxyPreconditioner
            => gmres(new fProxyBSROperator(in A), in M, in b, ref x, math.min(30, A.M_Rows), A.M_Rows, Consts.fProxySqrtEps);
    }
}
