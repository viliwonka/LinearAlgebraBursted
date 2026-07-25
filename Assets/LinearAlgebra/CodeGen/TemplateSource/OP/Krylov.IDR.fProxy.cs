using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using BULA.Sparse;

namespace BULA
{
    public static partial class Krylov {

        /// <summary>
        /// IDR(s) (Induced Dimension Reduction, Sonneveld &amp; van Gijzen 2008) for a general
        /// (nonsymmetric) square system A x = b, generic over BOTH the operator (<see
        /// cref="IfProxyLinearOperator"/>) and the preconditioner (<see cref="IfProxyPreconditioner"/>)
        /// -- the SINGLE body behind the plain and the right-preconditioned entry points. x is a
        /// warm-startable initial guess, overwritten with the solution. tol is relative (‖b − Ax‖ ≤
        /// tol·‖b‖); maxIter bounds the total number of A-applies (one per per-sweep step, one more
        /// per end-of-sweep step). s is the shadow-space dimension; the s-dimensional shadow space P
        /// is generated deterministically from <paramref name="seed"/> (same seed ⇒ bit-identical x
        /// on every run and architecture).
        ///
        /// With <see cref="fProxyIdentityPreconditioner"/> the IsIdentity fold makes this plain
        /// IDR(s) (no M⁻¹ apply, no VHat workspace). With a real M every direction fed to A is first
        /// passed through M⁻¹ (right preconditioning).
        ///
        /// Allocates its own workspace (P, G, U: s vectors of length A.Rows each; Q, V, and -- only
        /// for a real preconditioner -- VHat; an s×s system with its f/c vectors) from Allocator.Temp,
        /// disposed before every return. Returns a <see cref="SolveInfo"/>; status is Breakdown on a
        /// zero/NaN pivot in the s×s lower-triangular solve, a zero/NaN M[k,k], or a zero/NaN
        /// end-of-sweep omega -- never NaN, never throws from the recurrence itself.
        /// </summary>
        public static SolveInfo idr<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x,
                                         int s, int maxIter, fProxy tol, uint seed = 0x9E3779B1u)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols) throw new ArgumentException("idr: A must be square");
            if (b.N != A.Rows) throw new ArgumentException("idr: b.N must equal A.Rows");
            if (x.N != A.Rows) throw new ArgumentException("idr: x.N must equal A.Rows");
            if (s < 1) throw new ArgumentException("idr: s must be >= 1");
            if (maxIter < 1) throw new ArgumentException("idr: maxIter must be >= 1");
            if (!M.IsConstant)
                throw new ArgumentException("Krylov.idr: requires a constant (non-flexible) preconditioner (M.IsConstant == false — e.g. an AMG K-cycle). Use the flexible variant (fcg / fgmres).");

            int n = A.Rows;

            fProxy bb = Blas.dot(b, b);
            if (bb == (fProxy)0)
            {
                x.CopyFrom(in b);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }
            fProxy threshold = tol * tol * bb;

            // Workspace (Temp). P is the fixed shadow space (deterministic, from seed); G/U are the s
            // search-space vectors (start at zero); Q/V (and VHat, only for a real preconditioner) are
            // scratch; Msys/f/c are the small s×s system and its right-hand side / solution.
            var P = new UnsafeList<fProxyN>(s, Allocator.Temp);
            var G = new UnsafeList<fProxyN>(s, Allocator.Temp);
            var U = new UnsafeList<fProxyN>(s, Allocator.Temp);
            for (int i = 0; i < s; i++)
            {
                P.Add(new fProxyN(n, Allocator.Temp, true));
                G.Add(new fProxyN(n));
                U.Add(new fProxyN(n));
            }

            var rng = new Unity.Mathematics.Random(seed == 0 ? 0x9E3779B1u : seed);
            for (int i = 0; i < s; i++)
            {
                fProxyN Pi = P[i];
                for (int col = 0; col < n; col++) Pi[col] = rng.NextFProxy();
            }

            var R = new fProxyN(n);
            var V = new fProxyN(n);
            var Q = new fProxyN(n);
            fProxyN VHat = default;
            if (!M.IsIdentity) VHat = new fProxyN(n);

            var Msys = new fProxyMxN(s, s, Allocator.Temp, false);
            for (int i = 0; i < s; i++) Msys[i, i] = (fProxy)1;
            var f = new fProxyN(s);
            var c = new fProxyN(s);

            // R = b - A x
            A.Apply(in x, ref V);
            R.CopyFrom(in b);
            R.addScaledInPlace((fProxy)(-1), V);

            fProxy rr = Blas.dot(R, R);
            fProxy om = (fProxy)1;
            int iter = 0;

            IterativeSolveStatus status = IterativeSolveStatus.MaxIterations;
            bool done = false;
            if (rr <= threshold) { status = IterativeSolveStatus.Converged; done = true; }

            while (iter < maxIter && !done)
            {
                for (int i = 0; i < s; i++) f[i] = Blas.dot(P[i], R);

                for (int k = 0; k < s && iter < maxIter; k++)
                {
                    fProxyN Gk = G[k];
                    fProxyN Uk = U[k];

                    // Forward-substitute LowerTriangular(Msys[k..s-1,k..s-1]) c[k..s-1] = f[k..s-1].
                    bool breakdown = false;
                    for (int i = k; i < s && !breakdown; i++)
                    {
                        fProxy sum = f[i];
                        for (int j = k; j < i; j++) sum -= Msys[i, j] * c[j];
                        fProxy dii = Msys[i, i];
                        if (dii == (fProxy)0 || math.isnan(dii)) { breakdown = true; break; }
                        c[i] = sum / dii;
                    }
                    if (breakdown) { status = IterativeSolveStatus.Breakdown; done = true; break; }

                    // V = sum_{i=k}^{s-1} c[i] G[i] ; Q = sum_{i=k}^{s-1} c[i] U[i] (last sweep's G/U).
                    Blas.scaledCopy(c[k], G[k], ref V);
                    Blas.scaledCopy(c[k], U[k], ref Q);
                    for (int i = k + 1; i < s; i++)
                    {
                        V.addScaledInPlace(c[i], G[i]);
                        Q.addScaledInPlace(c[i], U[i]);
                    }

                    V.scaleAddInPlace((fProxy)(-1), R);   // V = R - V

                    if (M.IsIdentity)
                    {
                        Uk.CopyFrom(in Q);
                        Uk.addScaledInPlace(om, V);
                        A.Apply(in Uk, ref Gk);
                    }
                    else
                    {
                        M.Apply(in V, ref VHat);           // VHat = M⁻¹ V
                        Uk.CopyFrom(in Q);
                        Uk.addScaledInPlace(om, VHat);
                        A.Apply(in Uk, ref Gk);
                    }

                    // Bi-orthogonalise against the columns already refreshed this sweep.
                    for (int i = 0; i < k && !breakdown; i++)
                    {
                        fProxy dii = Msys[i, i];
                        if (dii == (fProxy)0 || math.isnan(dii)) { breakdown = true; break; }
                        fProxy alpha = Blas.dot(P[i], Gk) / dii;
                        Gk.addScaledInPlace(-alpha, G[i]);
                        Uk.addScaledInPlace(-alpha, U[i]);
                    }
                    if (breakdown) { status = IterativeSolveStatus.Breakdown; done = true; break; }

                    // New column of Msys = P^T G (rows above k stay untouched -- always zero, never read).
                    for (int i = k; i < s; i++) Msys[i, k] = Blas.dot(P[i], Gk);

                    fProxy dkk = Msys[k, k];
                    if (dkk == (fProxy)0 || math.isnan(dkk))
                    {
                        status = IterativeSolveStatus.Breakdown; done = true; break;
                    }
                    fProxy beta = f[k] / dkk;

                    rr = Blas.axpyNormSq(-beta, Gk, ref R);
                    x.addScaledInPlace(beta, Uk);
                    iter++;

                    if (rr <= threshold)
                    {
                        // Verify-at-exit: V/Q are idle here (both idle from right after they feed Uk
                        // above until the next k-step's forward substitution reuses them). On a
                        // failed verify, R is left holding the fresh residual (correct sign) so
                        // subsequent P[i]-dot-R work stays correct.
                        fProxy trueRR = VerifyTrueResidual(in A, in b, in x, ref V, ref Q);
                        R.CopyFrom(in Q);
                        rr = trueRR;
                        if (trueRR <= threshold) { status = IterativeSolveStatus.Converged; done = true; break; }
                    }

                    if (k < s - 1)
                        for (int i = k + 1; i < s; i++) f[i] -= beta * Msys[i, k];
                }

                if (done || iter >= maxIter) break;

                // End-of-sweep step: R is already orthogonal to P, so v = r; refine one level deeper.
                V.CopyFrom(in R);
                if (M.IsIdentity)
                {
                    A.Apply(in V, ref Q);
                }
                else
                {
                    M.Apply(in V, ref VHat);
                    A.Apply(in VHat, ref Q);
                }

                fProxy nt2 = Blas.dot(Q, Q);
                if (!(nt2 > (fProxy)0) || math.isnan(nt2))
                {
                    status = IterativeSolveStatus.Breakdown; break;
                }

                fProxy ts = Blas.dot(Q, R);
                fProxy ns2 = Blas.dot(R, R);
                fProxy nt = math.sqrt(nt2);
                fProxy ns = math.sqrt(ns2);
                fProxy rho = math.abs(ts / (nt * ns));
                om = ts / nt2;
                if (rho > (fProxy)0 && rho < (fProxy)0.7) om = om * (fProxy)0.7 / rho;

                if (om == (fProxy)0 || math.isnan(om))
                {
                    status = IterativeSolveStatus.Breakdown; break;
                }

                rr = Blas.axpyNormSq(-om, Q, ref R);
                if (M.IsIdentity) x.addScaledInPlace(om, V);
                else              x.addScaledInPlace(om, VHat);
                iter++;

                if (rr <= threshold)
                {
                    // Verify-at-exit: V/Q are idle here (V: last read by the x update above; Q: last
                    // read forming rr above), same buffer-reuse shape as the in-sweep site. On a
                    // failed verify, R is left holding the fresh residual (correct sign).
                    fProxy trueRR = VerifyTrueResidual(in A, in b, in x, ref V, ref Q);
                    R.CopyFrom(in Q);
                    rr = trueRR;
                    if (trueRR <= threshold) { status = IterativeSolveStatus.Converged; break; }
                }
            }

            for (int i = 0; i < s; i++) { P[i].Dispose(); G[i].Dispose(); U[i].Dispose(); }
            P.Dispose(); G.Dispose(); U.Dispose();
            R.Dispose(); V.Dispose(); Q.Dispose();
            if (!M.IsIdentity) VHat.Dispose();
            Msys.Dispose(); f.Dispose(); c.Dispose();

            return MakeSolveInfo(status, iter, math.sqrt(rr));
        }

        /// <summary>
        /// Unpreconditioned IDR(s) -- forwards into the merged
        /// <see cref="idr{TOp, TPre}(in TOp, in TPre, in fProxyN, ref fProxyN, int, int, fProxy, uint)"/>
        /// with the identity preconditioner (whose IsIdentity fold strips the M⁻¹ applies and the
        /// VHat workspace).
        /// </summary>
        public static SolveInfo idr<TOp>(in TOp A, in fProxyN b, ref fProxyN x, int s, int maxIter, fProxy tol, uint seed = 0x9E3779B1u)
            where TOp : struct, IfProxyLinearOperator
        {
            return idr(in A, default(fProxyIdentityPreconditioner), in b, ref x, s, maxIter, tol, seed);
        }

        /// <summary>IDR(s) over a dense <see cref="fProxyMxN"/>. Forwards via <see cref="fProxyDenseOperator"/>.</summary>
        public static SolveInfo idr(in fProxyMxN A, in fProxyN b, ref fProxyN x, int s, int maxIter, fProxy tol, uint seed = 0x9E3779B1u)
            => idr(new fProxyDenseOperator(in A), in b, ref x, s, maxIter, tol, seed);

        /// <summary>IDR(s) over a dense matrix with defaults (s = 4, maxIter = A.M_Rows, tol = Consts.fProxySqrtEps, seed = default).</summary>
        public static SolveInfo idr(in fProxyMxN A, in fProxyN b, ref fProxyN x)
            => idr(in A, in b, ref x, 4, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>IDR(s) over a block-sparse (BSR) matrix. Forwards via <c>fProxyBSROperator</c>.</summary>
        public static SolveInfo idr(in fProxyBSR A, in fProxyN b, ref fProxyN x, int s, int maxIter, fProxy tol, uint seed = 0x9E3779B1u)
            => idr(new fProxyBSROperator(in A), in b, ref x, s, maxIter, tol, seed);

        /// <summary>IDR(s) over a BSR matrix with defaults (s = 4, maxIter = A.M_Rows, tol = Consts.fProxySqrtEps, seed = default).</summary>
        public static SolveInfo idr(in fProxyBSR A, in fProxyN b, ref fProxyN x)
            => idr(in A, in b, ref x, 4, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Right-preconditioned IDR(s) over a BSR matrix with ANY <see cref="IfProxyPreconditioner"/>
        /// (ILU0/block-Jacobi). Forwards via <c>fProxyBSROperator</c>.</summary>
        public static SolveInfo idr<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x, int s, int maxIter, fProxy tol, uint seed = 0x9E3779B1u)
            where TPre : struct, IfProxyPreconditioner
            => idr(new fProxyBSROperator(in A), in M, in b, ref x, s, maxIter, tol, seed);

        /// <summary>Right-preconditioned IDR(s) over BSR with ANY <see cref="IfProxyPreconditioner"/>
        /// (ILU0/block-Jacobi), with defaults (s = 4, maxIter = A.M_Rows, tol = Consts.fProxySqrtEps, seed = default).</summary>
        public static SolveInfo idr<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x)
            where TPre : struct, IfProxyPreconditioner
            => idr(in A, in M, in b, ref x, 4, A.M_Rows, Consts.fProxySqrtEps);
    }
}
