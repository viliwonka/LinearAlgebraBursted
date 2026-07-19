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
        /// <see cref="SolveInfo"/> (rnorm from the Arnoldi residual estimate). Status: Converged /
        /// MaxIterations; a happy-breakdown (exact Krylov solution) converges.
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

            while (total < maxIter && !converged)
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

                    // Modified Gram–Schmidt against v_0..v_j.
                    for (int i = 0; i <= j; i++)
                    {
                        fProxyN vi = V[i];
                        fProxy hij = Blas.dot(w, vi);
                        H[i, j] = hij;
                        w.addScaledInPlace(-hij, vi);
                    }
                    fProxy hj1 = math.sqrt(Blas.dot(w, w));
                    H[j + 1, j] = hj1;
                    if (hj1 > (fProxy)0)
                    {
                        fProxyN vj1 = V[j + 1];
                        fProxy invh = (fProxy)1 / hj1;
                        vj1.CopyFrom(in w);
                        for (int i = 0; i < n; i++) vj1[i] *= invh;
                    }

                    // Apply previous Givens rotations to column j of H.
                    for (int i = 0; i < j; i++)
                    {
                        fProxy t0 = cs[i] * H[i, j] + sn[i] * H[i + 1, j];
                        H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j];
                        H[i, j] = t0;
                    }

                    // New Givens rotation zeroing H[j+1,j].
                    fProxy a = H[j, j], bb2 = H[j + 1, j];
                    fProxy rr = math.sqrt(a * a + bb2 * bb2);
                    fProxy c, s;
                    if (rr > (fProxy)0) { c = a / rr; s = bb2 / rr; }
                    else { c = (fProxy)1; s = (fProxy)0; }
                    cs[j] = c; sn[j] = s;
                    H[j, j] = rr;
                    H[j + 1, j] = (fProxy)0;

                    fProxy gj = g[j];
                    g[j] = c * gj;
                    g[j + 1] = -s * gj;

                    resnorm = math.abs(g[j + 1]);
                    total++;
                    k = j + 1;
                    if (resnorm <= thresh) { converged = true; break; }
                }

                // Back-substitute H[0..k-1,0..k-1] y = g[0..k-1].
                for (int i = k - 1; i >= 0; i--)
                {
                    fProxy sum = g[i];
                    for (int l = i + 1; l < k; l++) sum -= H[i, l] * y[l];
                    y[i] = sum / H[i, i];
                }
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

            for (int i = 0; i <= m; i++) V[i].Dispose();
            V.Dispose();
            if (!M.IsIdentity)
            {
                for (int i = 0; i < m; i++) Z[i].Dispose();
                Z.Dispose();
            }
            H.Dispose(); cs.Dispose(); sn.Dispose(); g.Dispose(); y.Dispose(); w.Dispose();

            return MakeSolveInfo(converged ? IterativeSolveStatus.Converged : IterativeSolveStatus.MaxIterations,
                                 total, resnorm);
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
