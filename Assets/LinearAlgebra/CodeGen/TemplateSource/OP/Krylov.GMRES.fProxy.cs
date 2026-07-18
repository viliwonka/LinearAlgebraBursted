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
        /// Restarted GMRES(m) for a general (nonsymmetric) square A x = b, generic over any
        /// <see cref="IfProxyLinearOperator"/>. Builds an orthonormal Krylov basis by Arnoldi with
        /// modified Gram–Schmidt, minimizes the residual over that space via an incrementally
        /// Givens-rotated least-squares, and restarts every <paramref name="restart"/> inner steps to
        /// bound memory. x is a warm-startable initial guess, overwritten with the solution; tol is
        /// relative (‖b − Ax‖ ≤ tol·‖b‖); maxIter counts TOTAL inner iterations across restarts.
        ///
        /// Allocates its workspace (restart+1 basis vectors + a small Hessenberg / Givens set) from
        /// the Temp allocator — heavier than the cg/biCGStab primitives, the nature of GMRES. Returns
        /// the shared <see cref="SolveInfo"/> (rnorm from the Arnoldi residual estimate). Status:
        /// Converged / MaxIterations; a happy-breakdown (exact Krylov solution) converges.
        /// </summary>
        public static SolveInfo gmres<TOp>(in TOp A, in fProxyN b, ref fProxyN x, int restart, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            if (A.Rows != A.Cols) throw new ArgumentException("gmres: A must be square");
            if (b.N != A.Rows) throw new ArgumentException("gmres: b.N must equal A.Rows");
            if (x.N != A.Rows) throw new ArgumentException("gmres: x.N must equal A.Rows");
            if (restart < 1) throw new ArgumentException("gmres: restart must be >= 1");
            if (maxIter < 1) throw new ArgumentException("gmres: maxIter must be >= 1");

            int n = A.Rows;
            int m = restart;

            fProxy bb = Blas.dot(b, b);
            if (bb == (fProxy)0)
            {
                x.Data.CopyFrom(b.Data);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }
            fProxy bnorm = math.sqrt(bb);
            fProxy thresh = tol * bnorm;

            // Workspace (Temp): basis V (m+1 x n as rows), Hessenberg H (m+1 x m), Givens cs/sn,
            // rotated rhs g, least-squares solution y, and an Arnoldi work vector w.
            var V = new UnsafeList<fProxyN>(m + 1, Allocator.Temp);
            for (int i = 0; i <= m; i++) V.Add(new fProxyN(n));
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
                v0.Data.CopyFrom(b.Data);
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
                    A.Apply(in vj, ref w);                   // w = A v_j

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
                        vj1.Data.CopyFrom(w.Data);
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

                // Back-substitute H[0..k-1,0..k-1] y = g[0..k-1], then x += sum y_i v_i.
                for (int i = k - 1; i >= 0; i--)
                {
                    fProxy sum = g[i];
                    for (int l = i + 1; l < k; l++) sum -= H[i, l] * y[l];
                    y[i] = sum / H[i, i];
                }
                for (int i = 0; i < k; i++)
                {
                    fProxyN vi = V[i];
                    x.addScaledInPlace(y[i], vi);
                }
            }

            for (int i = 0; i <= m; i++) V[i].Dispose();
            V.Dispose();
            H.Dispose(); cs.Dispose(); sn.Dispose(); g.Dispose(); y.Dispose(); w.Dispose();

            return MakeSolveInfo(converged ? IterativeSolveStatus.Converged : IterativeSolveStatus.MaxIterations,
                                 total, resnorm);
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

        /// <summary>
        /// RIGHT-preconditioned restarted GMRES(m): solves A x = b for general A with preconditioner
        /// M ≈ A (M⁻¹ applied via <typeparamref name="TPre"/>) by running GMRES on A·M⁻¹. Right
        /// preconditioning keeps the Arnoldi residual equal to the true residual ‖b − Ax‖, so the
        /// convergence test is unchanged; the solution update is one extra M⁻¹ apply per restart. Same
        /// contract, workspace, and restart semantics as <see cref="gmres{TOp}"/>.
        /// </summary>
        public static SolveInfo pgmres<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x, int restart, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols) throw new ArgumentException("pgmres: A must be square");
            if (b.N != A.Rows) throw new ArgumentException("pgmres: b.N must equal A.Rows");
            if (x.N != A.Rows) throw new ArgumentException("pgmres: x.N must equal A.Rows");
            if (restart < 1) throw new ArgumentException("pgmres: restart must be >= 1");
            if (maxIter < 1) throw new ArgumentException("pgmres: maxIter must be >= 1");

            int n = A.Rows;
            int m = restart;

            fProxy bb = Blas.dot(b, b);
            if (bb == (fProxy)0)
            {
                x.Data.CopyFrom(b.Data);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }
            fProxy bnorm = math.sqrt(bb);
            fProxy thresh = tol * bnorm;

            var V = new UnsafeList<fProxyN>(m + 1, Allocator.Temp);
            for (int i = 0; i <= m; i++) V.Add(new fProxyN(n));
            var H = new fProxyMxN(m + 1, m, Allocator.Temp, false);
            var cs = new fProxyN(m);
            var sn = new fProxyN(m);
            var g = new fProxyN(m + 1);
            var y = new fProxyN(m);
            var w = new fProxyN(n);
            var zt = new fProxyN(n);                          // M⁻¹ apply target

            int total = 0;
            fProxy resnorm = bnorm;
            bool converged = false;

            while (total < maxIter && !converged)
            {
                fProxyN v0 = V[0];
                A.Apply(in x, ref w);
                v0.Data.CopyFrom(b.Data);
                v0.addScaledInPlace((fProxy)(-1), w);
                fProxy beta = math.sqrt(Blas.dot(v0, v0));
                resnorm = beta;
                if (beta <= thresh) { converged = true; break; }

                fProxy invBeta = (fProxy)1 / beta;
                for (int i = 0; i < n; i++) v0[i] *= invBeta;
                for (int i = 0; i <= m; i++) g[i] = (fProxy)0;
                g[0] = beta;

                int k = 0;
                for (int j = 0; j < m && total < maxIter; j++)
                {
                    fProxyN vj = V[j];
                    M.Apply(in vj, ref zt);                   // zt = M⁻¹ v_j
                    A.Apply(in zt, ref w);                    // w  = A M⁻¹ v_j

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
                        vj1.Data.CopyFrom(w.Data);
                        for (int i = 0; i < n; i++) vj1[i] *= invh;
                    }

                    for (int i = 0; i < j; i++)
                    {
                        fProxy t0 = cs[i] * H[i, j] + sn[i] * H[i + 1, j];
                        H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j];
                        H[i, j] = t0;
                    }

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

                for (int i = k - 1; i >= 0; i--)
                {
                    fProxy sum = g[i];
                    for (int l = i + 1; l < k; l++) sum -= H[i, l] * y[l];
                    y[i] = sum / H[i, i];
                }
                // x += M⁻¹ (sum_i y_i v_i): accumulate the v-space combination into w, apply M⁻¹ once.
                for (int i = 0; i < n; i++) w[i] = (fProxy)0;
                for (int i = 0; i < k; i++)
                {
                    fProxyN vi = V[i];
                    w.addScaledInPlace(y[i], vi);
                }
                M.Apply(in w, ref zt);
                x.addScaledInPlace((fProxy)1, zt);
            }

            for (int i = 0; i <= m; i++) V[i].Dispose();
            V.Dispose();
            H.Dispose(); cs.Dispose(); sn.Dispose(); g.Dispose(); y.Dispose(); w.Dispose(); zt.Dispose();

            return MakeSolveInfo(converged ? IterativeSolveStatus.Converged : IterativeSolveStatus.MaxIterations,
                                 total, resnorm);
        }

        /// <summary>Right-preconditioned GMRES(m) over a BSR matrix with an ILU(0) preconditioner.</summary>
        public static SolveInfo pgmres(in fProxyBSR A, in fProxyILU0 M, in fProxyN b, ref fProxyN x, int restart, int maxIter, fProxy tol)
            => pgmres(new fProxyBSROperator(in A), in M, in b, ref x, restart, maxIter, tol);

        /// <summary>ILU(0)-right-preconditioned GMRES over a BSR matrix with defaults (restart = min(30, N)).</summary>
        public static SolveInfo pgmres(in fProxyBSR A, in fProxyILU0 M, in fProxyN b, ref fProxyN x)
            => pgmres(new fProxyBSROperator(in A), in M, in b, ref x, math.min(30, A.M_Rows), A.M_Rows, Consts.fProxySqrtEps);
    }
}
