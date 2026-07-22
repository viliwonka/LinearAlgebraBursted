using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov
    {
        /// <summary>
        /// Restarted GMRES(m) for a general (nonsymmetric) square A x = b, generic over BOTH the
        /// operator (<see cref="IfProxyLinearOperator"/>) and the preconditioner
        /// (<see cref="IfProxyPreconditioner"/>) -- the SINGLE body behind the plain and the
        /// RIGHT-preconditioned entry points. Builds an orthonormal Krylov basis by Arnoldi with
        /// modified Gram–Schmidt, minimizes the residual over that space via an incrementally
        /// Givens-rotated least-squares, and restarts every <paramref name="restart"/> inner steps to
        /// bound memory. x is a warm-startable initial guess, overwritten with the solution; tol is
        /// relative (‖b − Ax‖ ≤ tol·‖b‖); maxIter counts TOTAL inner iterations across restarts.
        ///
        /// With <see cref="fProxyIdentityPreconditioner"/> the IsIdentity fold makes this plain
        /// GMRES bit-for-bit (no M⁻¹ apply, no zt workspace, solution accumulated straight into x).
        /// With a real M it runs GMRES on A·M⁻¹ (right preconditioning): the Arnoldi residual stays
        /// equal to the true ‖b − Ax‖ so the convergence test is unchanged, at the cost of one M⁻¹
        /// apply per inner step plus one per restart for the solution update.
        ///
        /// Allocates its workspace (restart+1 basis vectors + a small Hessenberg / Givens set) from
        /// the Temp allocator — heavier than the cg/biCGStab primitives, the nature of GMRES. Returns
        /// the shared <see cref="SolveInfo"/>; rnorm is a freshly recomputed ‖b−Ax‖, not the raw
        /// Arnoldi/Givens estimate -- a Converged exit is verified before being reported, falling
        /// through to another restart cycle if the estimate turned out optimistic. Status: Converged
        /// / MaxIterations; a happy-breakdown (exact Krylov solution) converges. An exact-zero
        /// Arnoldi/Givens pivot with no usable column produced this cycle reports Breakdown instead
        /// of dividing by zero.
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

            int n = A.Rows;
            int m = restart;

            fProxy bb = Blas.dot(b, b);
            if (bb == (fProxy)0)
            {
                x.CopyFrom(in b);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }
            fProxy bnorm = math.sqrt(bb);
            fProxy thresh = tol * bnorm;

            // Workspace (Temp): basis V (m+1 x n as rows), Hessenberg H (m+1 x m), Givens cs/sn,
            // rotated rhs g, least-squares solution y, and an Arnoldi work vector w. zt (the M⁻¹
            // apply target) is allocated only for a real preconditioner.
            var V = new UnsafeList<fProxyN>(m + 1, Allocator.Temp);
            for (int i = 0; i <= m; i++) V.Add(new fProxyN(n));
            var H = new fProxyMxN(m + 1, m, Allocator.Temp, false);   // cleared: read only written entries
            var cs = new fProxyN(m);
            var sn = new fProxyN(m);
            var g = new fProxyN(m + 1);
            var y = new fProxyN(m);
            var w = new fProxyN(n);
            fProxyN zt = default;
            if (!M.IsIdentity) zt = new fProxyN(n);

            int total = 0;
            fProxy resnorm = bnorm;
            bool converged = false;
            // Set only when an Arnoldi/Givens step breaks down (H[j,j] and the Arnoldi subdiagonal
            // both exactly zero) with ZERO usable columns produced this cycle -- x is then
            // unchanged, so retrying would reproduce an identical residual forever. Forces the
            // outer loop to stop instead of spinning.
            bool deadEnd = false;

            while (total < maxIter && !converged && !deadEnd)
            {
                fProxyN v0 = V[0];
                A.Apply(in x, ref w);                       // w = A x
                v0.CopyFrom(in b);
                v0.addScaledInPlace((fProxy)(-1), w);        // v0 = b - A x
                fProxy beta = math.sqrt(Blas.dot(v0, v0));
                resnorm = beta;
                if (beta <= thresh) { converged = true; break; }

                fProxy invBeta = (fProxy)1 / beta;
                for (int i = 0; i < n; i++) v0[i] *= invBeta;   // v0 /= beta
                for (int i = 0; i <= m; i++) g[i] = (fProxy)0;
                g[0] = beta;

                int k = 0;
                bool breakdown = false;
                for (int j = 0; j < m && total < maxIter; j++)
                {
                    fProxyN vj = V[j];
                    // w = A (M⁻¹ v_j). Identity: M⁻¹ v_j = v_j, so w = A v_j directly (zt untouched).
                    if (M.IsIdentity)
                    {
                        A.Apply(in vj, ref w);
                    }
                    else
                    {
                        M.Apply(in vj, ref zt);              // zt = M⁻¹ v_j
                        A.Apply(in zt, ref w);              // w  = A M⁻¹ v_j
                    }

                    // Arnoldi step: orthogonalize w against V[0..j], normalize into V[j+1].
                    ArnoldiMGSStep(w, in V, ref H, j, n);
                    total++;   // one Arnoldi/matvec step consumed, whether or not it yields a pivot

                    // Apply/generate the Givens rotations, rotating H's column j and rhs g. False
                    // means H[j,j] and the Arnoldi subdiagonal are both exactly zero -- column j has
                    // no usable pivot (would divide by zero in HessenbergBackSolve). Exclude it: k
                    // stays at the last valid column count, and this cycle stops here.
                    if (!GivensApplyAndGenerate(ref H, ref cs, ref sn, ref g, j))
                    {
                        breakdown = true;
                        break;
                    }

                    resnorm = math.abs(g[j + 1]);
                    k = j + 1;
                    if (resnorm <= thresh) { converged = true; break; }
                }

                // Back-substitute H[0..k-1,0..k-1] y = g[0..k-1] (k == 0 only on an immediate,
                // zero-column breakdown -- both loops below are then no-ops).
                if (k > 0)
                {
                    HessenbergBackSolve(in H, in g, ref y, k);
                    // Identity: x += sum y_i v_i (accumulated straight into x -- matches plain GMRES).
                    // Preconditioned: x += M⁻¹(sum y_i v_i) -- accumulate into w, apply M⁻¹ once.
                    if (M.IsIdentity)
                    {
                        for (int i = 0; i < k; i++)
                        {
                            fProxyN vi = V[i];
                            x.addScaledInPlace(y[i], vi);
                        }
                    }
                    else
                    {
                        for (int i = 0; i < n; i++) w[i] = (fProxy)0;
                        for (int i = 0; i < k; i++)
                        {
                            fProxyN vi = V[i];
                            w.addScaledInPlace(y[i], vi);
                        }
                        M.Apply(in w, ref zt);
                        x.addScaledInPlace((fProxy)1, zt);
                    }
                }

                // Verify-at-exit (converged path) OR final honesty check on a dead-end breakdown
                // (zero columns produced -- x unchanged, further restarts are futile): the
                // rotated-rhs estimate |g[j+1]| can drift from the true ‖b-Ax‖ once MGS loses
                // orthogonality. w and V[0] are both about to be fully overwritten at the top of the
                // next cycle regardless, so they're free scratch here; a failed verify on the
                // converged path falls through to another restart cycle instead of a false Converged.
                if (converged || (breakdown && k == 0))
                {
                    fProxyN v0v = V[0];
                    fProxy trueRR = VerifyTrueResidual(in A, in b, in x, ref w, ref v0v);
                    resnorm = math.sqrt(trueRR);
                    converged = resnorm <= thresh;
                    if (breakdown && k == 0) deadEnd = true;
                }
            }

            for (int i = 0; i <= m; i++) V[i].Dispose();
            V.Dispose();
            H.Dispose(); cs.Dispose(); sn.Dispose(); g.Dispose(); y.Dispose(); w.Dispose();
            if (!M.IsIdentity) zt.Dispose();

            var status = converged ? IterativeSolveStatus.Converged
                       : deadEnd ? IterativeSolveStatus.Breakdown
                       : IterativeSolveStatus.MaxIterations;
            return MakeSolveInfo(status, total, resnorm);
        }

        /// <summary>
        /// Unpreconditioned restarted GMRES(m) -- forwards into the merged
        /// <see cref="gmres{TOp, TPre}(in TOp, in TPre, in fProxyN, ref fProxyN, int, int, fProxy)"/>
        /// with the identity preconditioner (whose IsIdentity fold strips the M⁻¹ applies and the zt
        /// workspace).
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
