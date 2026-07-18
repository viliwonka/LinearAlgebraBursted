using System;
using LinearAlgebra;
using Unity.Mathematics;

namespace LinearAlgebra.Sparse
{
    /// <summary>Setup/Apply knobs for <see cref="fProxyChebyshev"/>.</summary>
    public struct fProxyChebyshevOptions
    {
        /// <summary>spMVs per Apply; must be &gt;= 1.</summary>
        public int degree;

        /// <summary>Lo = Hi / kappa; must be &gt; 1.</summary>
        public fProxy kappa;

        /// <summary>Pinned Lanczos steps used to estimate Hi; must be &gt;= 1.</summary>
        public int eigSteps;

        /// <summary>Hi = safety * (Lanczos estimate of the scaled operator's largest eigenvalue); must be &gt;= 1.</summary>
        public fProxy safety;

        /// <summary>degree=3, kappa=30, eigSteps=10, safety=1.1.</summary>
        public static fProxyChebyshevOptions Default => new fProxyChebyshevOptions
        {
            degree = 3,
            kappa = (fProxy)30,
            eigSteps = 10,
            safety = (fProxy)1.1,
        };
    }

    /// <summary>
    /// Symmetric Jacobi-scaled wrapper S·A·S (S = D^(-1/2), D = diag(A)) -- same spectrum as
    /// D⁻¹A but symmetric, so a symmetric Lanczos tridiagonalization applies to it. Used only by
    /// <see cref="fProxyChebyshev"/>'s setup; not part of the public API.
    /// </summary>
    internal readonly struct fProxyJacobiScaledBSROperator : IfProxyLinearOperator
    {
        public readonly fProxyBSR A;
        public readonly fProxyN InvSqrtD;   // D^(-1/2), length n
        public readonly fProxyN Scratch;    // length n

        public fProxyJacobiScaledBSROperator(in fProxyBSR a, in fProxyN invSqrtD, in fProxyN scratch)
        {
            A = a;
            InvSqrtD = invSqrtD;
            Scratch = scratch;
        }

        public int Rows => A.M_Rows;
        public int Cols => A.N_Cols;

        // y = S A S x: s = InvSqrtD . x ; y = A s ; y = InvSqrtD . y.
        public void Apply(in fProxyN x, ref fProxyN y)
        {
            int n = InvSqrtD.N;
            for (int i = 0; i < n; i++) Scratch[i] = InvSqrtD[i] * x[i];
            BSR.spMV(in A, in Scratch, ref y);
            for (int i = 0; i < n; i++) y[i] = InvSqrtD[i] * y[i];
        }

        public void ApplyT(in fProxyN x, ref fProxyN y) => Apply(in x, ref y);   // symmetric

        // Composes Apply + a plain dot; no fused kernel here.
        public fProxy ApplyDot(in fProxyN x, ref fProxyN y)
        {
            Apply(in x, ref y);
            return Blas.dot(x, y);
        }

        // Per-row fallback via Apply (fProxyColScaledOperator's Temp-buffer pattern). Lanczos
        // never calls this -- present only to satisfy the interface.
        public void ApplyBlock(in fProxyMxN Vrows, ref fProxyMxN AVrows, int rows)
        {
            int n = InvSqrtD.N;
            var rin = new fProxyN(n, Unity.Collections.Allocator.Temp, false);
            var rout = new fProxyN(n, Unity.Collections.Allocator.Temp, false);
            for (int i = 0; i < rows; i++)
            {
                for (int c = 0; c < n; c++) rin[c] = Vrows[i, c];
                Apply(in rin, ref rout);
                for (int c = 0; c < n; c++) AVrows[i, c] = rout[c];
            }
            rout.Dispose();
            rin.Dispose();
        }
    }

    /// <summary>
    /// Chebyshev polynomial preconditioner over a square SPD BSR: z = q(D⁻¹A)·D⁻¹r, q a degree-
    /// <see cref="Degree"/> Chebyshev polynomial on [<see cref="Lo"/>, <see cref="Hi"/>] bracketing
    /// the upper spectrum of the point-Jacobi-scaled operator D⁻¹A (D = diag(A), scalar diagonal).
    /// Apply is dot-free: exactly <see cref="Degree"/> spMVs, Degree+1 diagonal scales, ~2·Degree
    /// axpys, zero triangular solves.
    ///
    /// Requires A square (BlockRows==BlockCols, BR==BC), every diagonal block stored, and every
    /// scalar diagonal entry A[i,i] &gt; 0. The induced M⁻¹ is SPD as long as no eigenvalue of
    /// D⁻¹A exceeds <see cref="Hi"/>: setup estimates Hi from a pinned-eigSteps Lanczos run on the
    /// symmetrically-scaled operator D^(-1/2)·A·D^(-1/2), scaled by
    /// <see cref="fProxyChebyshevOptions.safety"/>; Lo = Hi / <see cref="fProxyChebyshevOptions.kappa"/>.
    /// Underestimating Hi breaks the SPD guarantee; overestimating only weakens the preconditioner.
    ///
    /// Accepts either Symmetric (lower-block-only) or full storage -- <see cref="BSR.spMV"/>
    /// handles both natively, so unlike <see cref="fProxySSOR"/> no mirror-to-full copy is needed.
    ///
    /// Composed entirely of arena-tracked pieces -- no record table of its own, no Dispose(). All
    /// fields are readonly, set once at construction: IJob-struct-copy-safe.
    /// </summary>
    public readonly struct fProxyChebyshev : IfProxyPreconditioner
    {
        public readonly fProxyBSR A;

        /// <summary>1 / diag(A), the scalar (point) Jacobi diagonal. Length Rows.</summary>
        public readonly fProxyN InvDiag;

        /// <summary>Final interval bracketing the upper spectrum of D⁻¹A (diagnostics -- already
        /// folded into Theta/Delta/Sigma below).</summary>
        public readonly fProxy Lo, Hi;

        /// <summary>Chebyshev recurrence coefficients: Theta=(Hi+Lo)/2, Delta=(Hi-Lo)/2, Sigma=Theta/Delta.</summary>
        public readonly fProxy Theta, Delta, Sigma;

        /// <summary>spMVs per Apply.</summary>
        public readonly int Degree;

        /// <summary>Apply's owned scratch (d, rk, t of the recurrence). Length Rows.</summary>
        public readonly fProxyN Scratch1, Scratch2, Scratch3;

        public int Rows => A.M_Rows;

        /// <summary>
        /// Builds InvDiag, then Hi/Lo from a Lanczos run on the symmetrically-scaled operator (the
        /// step count is clamped to A.Rows, so a system smaller than opt.eigSteps builds fine), then
        /// the Chebyshev recurrence coefficients. Throws ArgumentException if A is not square
        /// (BlockRows==BlockCols, BR==BC), a diagonal block is missing, any scalar diagonal entry
        /// A[i,i] &lt;= 0, the Lanczos eigen-estimate fails to converge or yields a non-positive
        /// largest eigenvalue ("is A symmetric positive definite?"), opt.degree &lt; 1, opt.kappa
        /// &lt;= 1, opt.eigSteps &lt; 1, or opt.safety &lt; 1.
        /// </summary>
        public fProxyChebyshev(in fProxyBSR a, in fProxyChebyshevOptions opt, ref Arena arena)
        {
            if (a.BlockRows != a.BlockCols || a.BR != a.BC)
                throw new ArgumentException("fProxyChebyshev: A must be square (BlockRows==BlockCols, BR==BC)");
            if (opt.degree < 1)
                throw new ArgumentException("fProxyChebyshev: opt.degree must be >= 1");
            if (!(opt.kappa > (fProxy)1))
                throw new ArgumentException("fProxyChebyshev: opt.kappa must be > 1");
            if (opt.eigSteps < 1)
                throw new ArgumentException("fProxyChebyshev: opt.eigSteps must be >= 1");
            if (!(opt.safety >= (fProxy)1))
                throw new ArgumentException("fProxyChebyshev: opt.safety must be >= 1");

            A = a;
            int n = a.M_Rows;
            int BR = a.BR;
            int blockLen = BR * BR;

            // ---- InvDiag: 1 / A[i,i] over the stored scalar diagonal (both storage modes store
            // the diagonal block) -- same block-row scan fProxySSOR's ctor uses to find it. ----
            var invDiag = arena.fProxyVec(n);
            for (int i = 0; i < a.BlockRows; i++)
            {
                int s = a.RowPtr[i], e = a.RowPtr[i + 1];
                int found = -1;
                for (int k = s; k < e; k++)
                {
                    int col = a.ColInd[k];
                    if (col == i) { found = k; break; }
                    if (col > i) break;
                }
                if (found < 0)
                    throw new ArgumentException("fProxyChebyshev: missing diagonal block in A");

                int off = found * blockLen;
                int rowBase = i * BR;
                for (int r = 0; r < BR; r++)
                {
                    fProxy dv = a.Values[off + r * BR + r];
                    if (!(dv > (fProxy)0))
                        throw new ArgumentException("fProxyChebyshev: A has a non-positive diagonal entry -- is A symmetric positive definite?");
                    invDiag[rowBase + r] = (fProxy)1 / dv;
                }
            }
            InvDiag = invDiag;

            // ---- Hi/Lo: safety * lambdaMax(D^-1 A), via a pinned Lanczos run on the
            // symmetrically-scaled S.A.S, S = D^(-1/2) (same spectrum as D^-1 A, symmetric). ----
            var invSqrtD = arena.fProxyVec(n);
            for (int i = 0; i < n; i++)
                invSqrtD[i] = math.sqrt(invDiag[i]);
            var scaledScratch = arena.fProxyVec(n);
            var scaledOp = new fProxyJacobiScaledBSROperator(in A, in invSqrtD, in scaledScratch);

            // Lanczos needs steps in [1, n]; clamp so a system smaller than opt.eigSteps still
            // builds (fewer steps only coarsens the estimate, never invalidates it).
            int eigSteps = math.min(opt.eigSteps, n);
            var ws = arena.fProxyLanczosCache(n, eigSteps);
            var ritz = arena.fProxyVec(eigSteps);
            var lInfo = Eigen.lanczos(in scaledOp, ref ws, ref ritz, eigSteps);

            fProxy lambdaMax = ritz[0];
            for (int i = 1; i < lInfo.produced; i++)
                if (ritz[i] > lambdaMax) lambdaMax = ritz[i];

            // A failed eigen-estimate (non-converged Lanczos, or a non-positive largest eigenvalue)
            // makes Hi/Sigma garbage and the induced M^-1 indefinite/NaN -- signal a bad SPD build.
            if (!(lInfo.status == IterativeSolveStatus.Converged) || !(lambdaMax > (fProxy)0))
                throw new ArgumentException("fProxyChebyshev: Lanczos produced no positive largest-eigenvalue estimate for D^-1 A -- is A symmetric positive definite? (raise opt.eigSteps or check A)");

            fProxy hi = opt.safety * lambdaMax;
            fProxy lo = hi / opt.kappa;
            Hi = hi;
            Lo = lo;

            Theta = (hi + lo) / (fProxy)2;
            Delta = (hi - lo) / (fProxy)2;
            Sigma = Theta / Delta;

            Degree = opt.degree;

            Scratch1 = arena.fProxyVec(n);
            Scratch2 = arena.fProxyVec(n);
            Scratch3 = arena.fProxyVec(n);
        }

        /// <summary>fProxyChebyshev with fProxyChebyshevOptions.Default (degree=3, kappa=30, eigSteps=10, safety=1.1).</summary>
        public fProxyChebyshev(in fProxyBSR a, ref Arena arena) : this(in a, fProxyChebyshevOptions.Default, ref arena) { }

        /// <summary>z = q(D⁻¹A)·D⁻¹r via the degree-Degree Chebyshev recurrence
        /// (<see cref="BSR.chebyApply"/>) over the struct's owned scratch. z must not alias r.</summary>
        public unsafe void Apply(in fProxyN r, ref fProxyN z)
        {
            int n = Rows;
            if (r.N != n)
                throw new ArgumentException("fProxyChebyshev.Apply: r.N must equal Rows");
            if (z.N != n)
                throw new ArgumentException("fProxyChebyshev.Apply: z.N must equal Rows");
            if (z.Data.Ptr == r.Data.Ptr)
                throw new ArgumentException("fProxyChebyshev.Apply: z must not alias r");

            var d = Scratch1;
            var rk = Scratch2;
            var t = Scratch3;
            BSR.chebyApply(in A, in InvDiag, Theta, Delta, Sigma, Degree, in r, ref z, ref d, ref rk, ref t);
        }
    }

    public static partial class BSR
    {
        /// <summary>
        /// Zero-initial-guess degree-<paramref name="degree"/> Chebyshev recurrence on D⁻¹A over
        /// [lo, hi] (theta=(hi+lo)/2, delta=(hi-lo)/2, sigma=theta/delta -- caller-supplied):
        /// z = q(D⁻¹A)·(invDiag ∘ r). Explicit scratch d/rk/t (length A.M_Rows, distinct from
        /// r/z/each other) so callers (e.g. an AMG smoother) can supply their own buffers. Exactly
        /// <paramref name="degree"/> spMVs; no reductions.
        /// </summary>
        public static unsafe void chebyApply(in fProxyBSR A, in fProxyN invDiag, fProxy theta, fProxy delta, fProxy sigma,
            int degree, in fProxyN r, ref fProxyN z, ref fProxyN d, ref fProxyN rk, ref fProxyN t)
        {
            int n = A.M_Rows;
            fProxy invTheta = (fProxy)1 / theta;

            fProxy* rp = r.Data.Ptr;
            fProxy* invDp = invDiag.Data.Ptr;
            fProxy* dp = d.Data.Ptr;
            fProxy* zp = z.Data.Ptr;
            fProxy* rkp = rk.Data.Ptr;

            for (int i = 0; i < n; i++)
            {
                fProxy di = invTheta * (invDp[i] * rp[i]);
                dp[i] = di;
                zp[i] = di;
                rkp[i] = rp[i];
            }

            fProxy rhoPrev = (fProxy)1 / sigma;

            for (int k = 0; k < degree; k++)
            {
                spMV(in A, in d, ref t);              // t = A d
                rk.addScaledInPlace((fProxy)(-1), t); // rk -= t

                fProxy rho = (fProxy)1 / ((fProxy)2 * sigma - rhoPrev);
                fProxy a = rho * rhoPrev;
                fProxy b = ((fProxy)2 * rho) / delta;

                for (int i = 0; i < n; i++)
                {
                    fProxy dNew = a * dp[i] + b * (invDp[i] * rkp[i]);
                    dp[i] = dNew;
                    zp[i] += dNew;
                }

                rhoPrev = rho;
            }
        }
    }
}
