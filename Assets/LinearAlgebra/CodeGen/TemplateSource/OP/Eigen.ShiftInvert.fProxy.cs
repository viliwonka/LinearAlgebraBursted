using System;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    /// <summary>
    /// Shift-invert operator: Apply(x) = (A - shift*I)⁻¹ x, computed by an inner MINRES-QLP solve
    /// with the eigenvalue shift (see <see cref="Krylov.minresQLP{TOp,TPre}"/>). A MUST be symmetric,
    /// so (A - shift*I)⁻¹ is symmetric and ApplyT == Apply. Each Apply is a full iterative solve --
    /// expensive by nature; this is the standard shift-invert matvec that turns a Lanczos eigensolver
    /// into an interior-eigenvalue solver (eigenvalues of A nearest shift ↔ extreme eigenvalues of
    /// this operator, which Lanczos converges first). The inner solve is cold-started each call.
    /// </summary>
    public readonly struct fProxyShiftInvertOperator<TOp> : IfProxyLinearOperator
        where TOp : struct, IfProxyLinearOperator
    {
        readonly TOp _A;
        readonly fProxy _shift;
        readonly int _innerMaxIter;
        readonly fProxy _innerTol;

        public fProxyShiftInvertOperator(in TOp A, fProxy shift, int innerMaxIter, fProxy innerTol)
        {
            _A = A;
            _shift = shift;
            _innerMaxIter = innerMaxIter;
            _innerTol = innerTol;
        }

        public int Rows => _A.Rows;
        public int Cols => _A.Cols;

        public void Apply(in fProxyN x, ref fProxyN y)
        {
            for (int i = 0; i < y.N; i++) y[i] = (fProxy)0;   // cold start: solve (A - shift*I) y = x
            Krylov.minresQLP(in _A, default(fProxyIdentityPreconditioner), in x, ref y, _innerMaxIter, _innerTol, _shift);
        }

        public void ApplyT(in fProxyN x, ref fProxyN y) => Apply(in x, ref y);

        public fProxy ApplyDot(in fProxyN x, ref fProxyN y) { Apply(in x, ref y); return Blas.dot(x, y); }

        public void ApplyBlock(in fProxyMxN Vrows, ref fProxyMxN AVrows, int rows)
            => throw new NotSupportedException("fProxyShiftInvertOperator: ApplyBlock not supported (shift-invert Lanczos uses Apply only)");
    }

    public static partial class Eigen
    {
        /// <summary>
        /// Shift-and-invert eigensolver: finds the <paramref name="k"/> eigenpairs of the SYMMETRIC
        /// operator <paramref name="A"/> whose eigenvalues are NEAREST <paramref name="shift"/> (the
        /// interior-eigenvalue problem). Runs symmetric Lanczos (<see cref="lanczosVectors{TOp}"/>)
        /// over the shift-invert operator (A - shift*I)⁻¹ (each matvec = an inner MINRES-QLP solve),
        /// whose extreme Ritz values -- the ones Lanczos converges first, on BOTH sides of shift --
        /// correspond to A's eigenvalues nearest shift. Each returned eigenvalue is recovered by a
        /// Rayleigh quotient vᵀA v / vᵀv against the ORIGINAL A (robust to an inexact inner solve --
        /// far better than trusting shift + 1/theta), then the produced Ritz pairs are sorted by
        /// |lambda - shift| and the k nearest returned.
        ///
        /// <paramref name="steps"/> Lanczos steps must exceed k for the extreme Ritz values to
        /// converge (a few extra is usually enough for well-separated interior modes). Allocates
        /// eigenvalues (length min(k, produced)) and eigenvectors (min(k, produced) x A.Rows, row i =
        /// eigenvector i) from the arena. innerTol / innerMaxIter bound the inner MINRES-QLP solve.
        /// Returns the underlying <see cref="LanczosInfo"/> (its `produced` bounds how many pairs are
        /// meaningful; status reflects the tridiagonal QL convergence). A must be symmetric.
        /// </summary>
        public static LanczosInfo eigNearShift<TOp>(ref Arena arena, in TOp A, fProxy shift, int k, int steps,
                                                    out fProxyN eigenvalues, out fProxyMxN eigenvectors,
                                                    fProxy innerTol, int innerMaxIter)
            where TOp : struct, IfProxyLinearOperator
        {
            if (A.Rows != A.Cols) throw new ArgumentException("eigNearShift: A must be square");
            if (k < 1) throw new ArgumentException("eigNearShift: k must be >= 1");
            if (steps < k) throw new ArgumentException("eigNearShift: steps must be >= k");
            if (steps > A.Rows) steps = A.Rows;

            int n = A.Rows;

            var siOp = new fProxyShiftInvertOperator<TOp>(in A, shift, innerMaxIter, innerTol);

            var cache = arena.fProxyLanczosCache(n, steps);
            var Yt = arena.fProxyMat(steps, steps);
            var theta = arena.fProxyVec(steps);      // Ritz values of the shift-invert operator (unused after recovery)
            var ritz = arena.fProxyMat(steps, n);    // ritz row j = Ritz vector j

            var info = lanczosVectors(in siOp, ref cache, ref Yt, ref theta, ref ritz, steps, Consts.fProxyEpsilon);
            int produced = info.produced;
            int outK = math.min(k, produced);

            // Rayleigh quotient lambda_j = (v_jᵀ A v_j)/(v_jᵀ v_j) against the ORIGINAL A, per produced
            // Ritz vector -- accurate λ recovery even from an inexact inner solve. The SELECTION key is
            // NOT |lambda - shift| (an UNconverged Ritz vector can have a Rayleigh quotient that lands
            // near shift by accident, poisoning the result); it is |theta_j| = |Ritz value of
            // (A-shift I)⁻¹| = 1/|lambda_j - shift|, which is LARGE exactly for the modes nearest shift
            // AND is the extreme end of T's spectrum that Lanczos actually converges first. sel[j]
            // = -|theta_j| so "smallest sel" = "largest |theta|" = nearest+converged.
            var lam = arena.fProxyVec(produced);
            var sel = arena.fProxyVec(produced);
            var v = arena.fProxyVec(n);
            var Av = arena.fProxyVec(n);
            for (int j = 0; j < produced; j++)
            {
                for (int i = 0; i < n; i++) v[i] = ritz[j, i];
                A.Apply(in v, ref Av);
                fProxy vv = Blas.dot(v, v);
                fProxy vAv = Blas.dot(v, Av);
                lam[j] = vv > (fProxy)0 ? vAv / vv : shift;
                sel[j] = -math.abs(theta[j]);
            }

            eigenvalues = arena.fProxyVec(outK);
            eigenvectors = arena.fProxyMat(outK, n);

            // Emit the outK modes with largest |theta| (nearest shift, best-converged), nearest first.
            for (int slot = 0; slot < outK; slot++)
            {
                int best = 0;
                for (int j = 1; j < produced; j++) if (sel[j] < sel[best]) best = j;
                eigenvalues[slot] = lam[best];
                for (int i = 0; i < n; i++) eigenvectors[slot, i] = ritz[best, i];
                sel[best] = fProxy.MaxValue;
            }

            return info;
        }

        /// <summary>
        /// Shift-and-invert eigensolver over a dense symmetric <see cref="fProxyMxN"/> -- forwards via
        /// <see cref="fProxyDenseOperator"/>. steps defaults to min(A.Rows, 2*k + 20); inner solve
        /// tol = sqrtEps, maxIter = A.Rows.
        /// </summary>
        public static LanczosInfo eigNearShift(ref Arena arena, in fProxyMxN A, fProxy shift, int k,
                                               out fProxyN eigenvalues, out fProxyMxN eigenvectors)
        {
            int steps = math.min(A.M_Rows, 2 * k + 20);
            return eigNearShift(ref arena, new fProxyDenseOperator(in A), shift, k, steps,
                                out eigenvalues, out eigenvectors, Consts.fProxySqrtEps, A.M_Rows);
        }

        /// <summary>
        /// Shift-and-invert eigensolver over a symmetric block-sparse (BSR) matrix -- forwards via
        /// <see cref="fProxyBSROperator"/>. steps defaults to min(A.Rows, 2*k + 20); inner solve
        /// tol = sqrtEps, maxIter = A.Rows.
        /// </summary>
        public static LanczosInfo eigNearShift(ref Arena arena, in fProxyBSR A, fProxy shift, int k,
                                               out fProxyN eigenvalues, out fProxyMxN eigenvectors)
        {
            int steps = math.min(A.M_Rows, 2 * k + 20);
            return eigNearShift(ref arena, new fProxyBSROperator(in A), shift, k, steps,
                                out eigenvalues, out eigenvectors, Consts.fProxySqrtEps, A.M_Rows);
        }
    }
}
