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
        /// Restarted Flexible GMRES(m) (Saad 1993) for a general (nonsymmetric) square A x = b,
        /// generic over both the operator (<see cref="IfProxyLinearOperator"/>) and the
        /// preconditioner (<see cref="IfProxyPreconditioner"/>). Unlike
        /// <see cref="gmres{TOp, TPre}(in TOp, in TPre, in fProxyN, ref fProxyN, int, int, fProxy)"/>,
        /// the preconditioner M is allowed to CHANGE every inner step (a nonlinear / inner-iterative
        /// M -- AMG, an inner Krylov solve, ...). Each Arnoldi step preconditions the CURRENT basis
        /// vector, z_j = M⁻¹ v_j, then advances with w = A z_j; the preconditioned basis
        /// Z = [z_1 … z_j] is stored so the solution update reads x = x0 + Z y, the SAME
        /// least-squares y produced by gmres's Hessenberg/Givens/restart machinery (gmres instead
        /// applies M once to Σ y_i v_i, which is only valid when M is fixed across the cycle).
        ///
        /// With <see cref="fProxyIdentityPreconditioner"/> the IsIdentity fold makes z_j == v_j: no Z
        /// workspace is allocated and the solution accumulates straight into x via V exactly as plain
        /// GMRES, so this is bit-identical to <see cref="gmres{TOp,TPre}"/> under identity.
        ///
        /// x is a warm-startable initial guess, overwritten with the solution; tol is relative
        /// (‖b − Ax‖ ≤ tol·‖b‖); maxIter counts TOTAL inner iterations across restarts. Allocates its
        /// workspace (restart+1 basis vectors V, restart preconditioned vectors Z when M is not the
        /// identity, and a small Hessenberg / Givens set) from the Temp allocator. Returns the shared
        /// <see cref="SolveInfo"/>; rnorm is a freshly recomputed ‖b−Ax‖, not the raw Arnoldi/Givens
        /// estimate -- a Converged exit is verified before being reported, falling through to
        /// another restart cycle if the estimate turned out optimistic. Status: Converged /
        /// MaxIterations; a happy-breakdown (exact Krylov solution) converges. An exact-zero
        /// Arnoldi/Givens pivot with no usable column produced this cycle reports Breakdown instead
        /// of dividing by zero.
        /// </summary>
        public static SolveInfo fgmres<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x, int restart, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols) throw new ArgumentException("fgmres: A must be square");
            if (b.N != A.Rows) throw new ArgumentException("fgmres: b.N must equal A.Rows");
            if (x.N != A.Rows) throw new ArgumentException("fgmres: x.N must equal A.Rows");
            if (restart < 1) throw new ArgumentException("fgmres: restart must be >= 1");
            if (maxIter < 1) throw new ArgumentException("fgmres: maxIter must be >= 1");

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
            // rotated rhs g, least-squares solution y, and an Arnoldi work vector w. Z (m x n, the
            // flexible preconditioned basis) is allocated only for a real preconditioner -- under
            // identity z_j == v_j so V alone carries the solution update.
            var V = new UnsafeList<fProxyN>(m + 1, Allocator.Temp);
            for (int i = 0; i <= m; i++) V.Add(new fProxyN(n));
            UnsafeList<fProxyN> Z = default;
            if (!M.IsIdentity)
            {
                Z = new UnsafeList<fProxyN>(m, Allocator.Temp);
                for (int i = 0; i < m; i++) Z.Add(new fProxyN(n));
            }
            var H = new fProxyMxN(m + 1, m, Allocator.Temp, false);   // cleared: read only written entries
            var cs = new fProxyN(m);
            var sn = new fProxyN(m);
            var g = new fProxyN(m + 1);
            var y = new fProxyN(m);
            var w = new fProxyN(n);

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
                    // w = A (M⁻¹ v_j), preconditioning the CURRENT basis vector (M may vary per
                    // step). z_j is stored in Z for the flexible solution update below. Identity:
                    // M⁻¹ v_j = v_j, so w = A v_j directly (no Z workspace, no M.Apply).
                    if (M.IsIdentity)
                    {
                        A.Apply(in vj, ref w);
                    }
                    else
                    {
                        fProxyN zj = Z[j];
                        M.Apply(in vj, ref zj);              // z_j = M⁻¹ v_j
                        A.Apply(in zj, ref w);               // w   = A z_j
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
                    // x += Z y (identity: Z aliases V, accumulated straight into x -- matches plain
                    // GMRES exactly). Preconditioned: x += sum y_i z_i using the STORED per-step
                    // preconditioned vectors, valid even when M varied across j = 0..k-1.
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
                        for (int i = 0; i < k; i++)
                        {
                            fProxyN zi = Z[i];
                            x.addScaledInPlace(y[i], zi);
                        }
                    }
                }

                // Verify-at-exit (converged path) OR final honesty check on a dead-end breakdown
                // (zero columns produced -- x unchanged, further restarts are futile): the
                // rotated-rhs estimate |g[j+1]| can drift from the true ‖b-Ax‖ once MGS loses
                // orthogonality (an aggressive inner-iterative M widens the gap further). w and V[0]
                // are both about to be fully overwritten at the top of the next cycle regardless, so
                // they're free scratch here; a failed verify on the converged path falls through to
                // another restart cycle instead of a false Converged.
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
            if (!M.IsIdentity)
            {
                for (int i = 0; i < m; i++) Z[i].Dispose();
                Z.Dispose();
            }
            H.Dispose(); cs.Dispose(); sn.Dispose(); g.Dispose(); y.Dispose(); w.Dispose();

            var status = converged ? IterativeSolveStatus.Converged
                       : deadEnd ? IterativeSolveStatus.Breakdown
                       : IterativeSolveStatus.MaxIterations;
            return MakeSolveInfo(status, total, resnorm);
        }

        /// <summary>
        /// Unpreconditioned restarted flexible GMRES(m) -- forwards into the merged
        /// <see cref="fgmres{TOp, TPre}(in TOp, in TPre, in fProxyN, ref fProxyN, int, int, fProxy)"/>
        /// with the identity preconditioner (whose IsIdentity fold strips the M⁻¹ applies and the Z
        /// workspace), making this numerically identical to
        /// <see cref="gmres{TOp}(in TOp, in fProxyN, ref fProxyN, int, int, fProxy)"/>.
        /// </summary>
        public static SolveInfo fgmres<TOp>(in TOp A, in fProxyN b, ref fProxyN x, int restart, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            return fgmres(in A, default(fProxyIdentityPreconditioner), in b, ref x, restart, maxIter, tol);
        }

        /// <summary>Flexible GMRES(m) over a dense <see cref="fProxyMxN"/>. Forwards via fProxyDenseOperator.</summary>
        public static SolveInfo fgmres(in fProxyMxN A, in fProxyN b, ref fProxyN x, int restart, int maxIter, fProxy tol)
            => fgmres(new fProxyDenseOperator(in A), in b, ref x, restart, maxIter, tol);

        /// <summary>Flexible GMRES over a dense matrix with defaults (restart = min(30, N), maxIter = N, tol = sqrtEps).</summary>
        public static SolveInfo fgmres(in fProxyMxN A, in fProxyN b, ref fProxyN x)
            => fgmres(new fProxyDenseOperator(in A), in b, ref x, math.min(30, A.M_Rows), A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Flexible GMRES(m) over a block-sparse (BSR) matrix. Forwards via fProxyBSROperator.</summary>
        public static SolveInfo fgmres(in fProxyBSR A, in fProxyN b, ref fProxyN x, int restart, int maxIter, fProxy tol)
            => fgmres(new fProxyBSROperator(in A), in b, ref x, restart, maxIter, tol);

        /// <summary>Flexible GMRES over a BSR matrix with defaults (restart = min(30, N), maxIter = N, tol = sqrtEps).</summary>
        public static SolveInfo fgmres(in fProxyBSR A, in fProxyN b, ref fProxyN x)
            => fgmres(new fProxyBSROperator(in A), in b, ref x, math.min(30, A.M_Rows), A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Right-preconditioned flexible GMRES(m) over a BSR matrix with an ILU(0) preconditioner.</summary>
        public static SolveInfo fgmres(in fProxyBSR A, in fProxyILU0 M, in fProxyN b, ref fProxyN x, int restart, int maxIter, fProxy tol)
            => fgmres(new fProxyBSROperator(in A), in M, in b, ref x, restart, maxIter, tol);

        /// <summary>ILU(0)-right-preconditioned flexible GMRES over a BSR matrix with defaults (restart = min(30, N)).</summary>
        public static SolveInfo fgmres(in fProxyBSR A, in fProxyILU0 M, in fProxyN b, ref fProxyN x)
            => fgmres(new fProxyBSROperator(in A), in M, in b, ref x, math.min(30, A.M_Rows), A.M_Rows, Consts.fProxySqrtEps);
    }
}
