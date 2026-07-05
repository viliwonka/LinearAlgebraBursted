using System;

namespace LinearAlgebra
{
    public static partial class LOBPCG
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an n-dimensional symmetric operator
        /// run for a <paramref name="k"/>-eigenpair LOBPCG solve -- the layout produced by
        /// <c>Arena.floatLOBPCGCache(n, k)</c>.
        /// </summary>
        static void RequireLOBPCGWorkspace(in floatLOBPCGCache ws, int n, int k)
        {
            if (ws.X.M_Rows != k || ws.X.N_Cols != n)
                throw new ArgumentException("LOBPCG: workspace X must be k x n (use Arena.floatLOBPCGCache(n, k))");
            if (ws.AX.M_Rows != k || ws.AX.N_Cols != n)
                throw new ArgumentException("LOBPCG: workspace AX must be k x n (use Arena.floatLOBPCGCache(n, k))");
            if (ws.W.M_Rows != k || ws.W.N_Cols != n)
                throw new ArgumentException("LOBPCG: workspace W must be k x n (use Arena.floatLOBPCGCache(n, k))");
            if (ws.AW.M_Rows != k || ws.AW.N_Cols != n)
                throw new ArgumentException("LOBPCG: workspace AW must be k x n (use Arena.floatLOBPCGCache(n, k))");
            if (ws.P.M_Rows != k || ws.P.N_Cols != n)
                throw new ArgumentException("LOBPCG: workspace P must be k x n (use Arena.floatLOBPCGCache(n, k))");
            if (ws.AP.M_Rows != k || ws.AP.N_Cols != n)
                throw new ArgumentException("LOBPCG: workspace AP must be k x n (use Arena.floatLOBPCGCache(n, k))");
            if (ws.R.M_Rows != k || ws.R.N_Cols != n)
                throw new ArgumentException("LOBPCG: workspace R must be k x n (use Arena.floatLOBPCGCache(n, k))");
            if (ws.Xnext.M_Rows != k || ws.Xnext.N_Cols != n)
                throw new ArgumentException("LOBPCG: workspace Xnext must be k x n (use Arena.floatLOBPCGCache(n, k))");
            if (ws.AXnext.M_Rows != k || ws.AXnext.N_Cols != n)
                throw new ArgumentException("LOBPCG: workspace AXnext must be k x n (use Arena.floatLOBPCGCache(n, k))");
            if (ws.Pnext.M_Rows != k || ws.Pnext.N_Cols != n)
                throw new ArgumentException("LOBPCG: workspace Pnext must be k x n (use Arena.floatLOBPCGCache(n, k))");
            if (ws.APnext.M_Rows != k || ws.APnext.N_Cols != n)
                throw new ArgumentException("LOBPCG: workspace APnext must be k x n (use Arena.floatLOBPCGCache(n, k))");

            if (ws.BX.M_Rows != k || ws.BX.N_Cols != n)
                throw new ArgumentException("LOBPCG: workspace BX must be k x n (use Arena.floatLOBPCGCache(n, k))");
            if (ws.BW.M_Rows != k || ws.BW.N_Cols != n)
                throw new ArgumentException("LOBPCG: workspace BW must be k x n (use Arena.floatLOBPCGCache(n, k))");
            if (ws.BP.M_Rows != k || ws.BP.N_Cols != n)
                throw new ArgumentException("LOBPCG: workspace BP must be k x n (use Arena.floatLOBPCGCache(n, k))");

            if (ws.lambda.N != k)
                throw new ArgumentException("LOBPCG: workspace lambda must have length k (use Arena.floatLOBPCGCache(n, k))");
            if (ws.residual.N != k)
                throw new ArgumentException("LOBPCG: workspace residual must have length k (use Arena.floatLOBPCGCache(n, k))");

            if (ws.rowIn.N != n || ws.rowOut.N != n)
                throw new ArgumentException("LOBPCG: workspace rowIn/rowOut must have length n (use Arena.floatLOBPCGCache(n, k))");
            if (ws.rowAux.N != n)
                throw new ArgumentException("LOBPCG: workspace rowAux must have length n (use Arena.floatLOBPCGCache(n, k))");

            int cap = 3 * k;
            if (!ws.Gram.IsSquare || ws.Gram.M_Rows != cap)
                throw new ArgumentException("LOBPCG: workspace Gram must be 3k x 3k (use Arena.floatLOBPCGCache(n, k))");
            if (!ws.H.IsSquare || ws.H.M_Rows != cap)
                throw new ArgumentException("LOBPCG: workspace H must be 3k x 3k (use Arena.floatLOBPCGCache(n, k))");
            if (!ws.L.IsSquare || ws.L.M_Rows != cap)
                throw new ArgumentException("LOBPCG: workspace L must be 3k x 3k (use Arena.floatLOBPCGCache(n, k))");
            if (!ws.Atrans.IsSquare || ws.Atrans.M_Rows != cap)
                throw new ArgumentException("LOBPCG: workspace Atrans must be 3k x 3k (use Arena.floatLOBPCGCache(n, k))");
            if (!ws.Y.IsSquare || ws.Y.M_Rows != cap)
                throw new ArgumentException("LOBPCG: workspace Y must be 3k x 3k (use Arena.floatLOBPCGCache(n, k))");
            if (!ws.C.IsSquare || ws.C.M_Rows != cap)
                throw new ArgumentException("LOBPCG: workspace C must be 3k x 3k (use Arena.floatLOBPCGCache(n, k))");
        }
    }

    /// <summary>
    /// Reusable scratch for <see cref="LOBPCG.lobpcg{TOp,TBOp,TPre}"/> (blocked LOBPCG for the k
    /// smallest eigenpairs of an n-dimensional symmetric operator, or the k smallest of the
    /// GENERALIZED pencil (A, B) -- the SAME cache and its SAME shape serve both, see
    /// <see cref="BX"/>/<see cref="BW"/>/<see cref="BP"/>). Sized for k eigenpairs over an
    /// n-dimensional operator. Allocate ONCE via <c>Arena.floatLOBPCGCache(n, k)</c> and reuse it
    /// across same-shape calls so repeated solves are zero-alloc at the O(n) scale (see the
    /// class doc comment on <see cref="LOBPCG"/> for the one exception: the tiny O(k)-sized
    /// Rayleigh-Ritz eigensolve still uses a few small, bounded <c>Allocator.Temp</c> scratch
    /// vectors internally, exactly like <see cref="Eigen.eigenSymmetric(ref floatMxN, ref floatN, ref floatMxN)"/>
    /// already does for its own callers).
    /// </summary>
    public struct floatLOBPCGCache
    {
        /// <summary>k x n. Current eigenvector estimates (rows), ascending-sorted only at the
        /// final return -- during iteration rows [0, numActive) are the still-iterating pairs and
        /// rows [numActive, k) are locked (converged, frozen) pairs, in lock order.</summary>
        public floatMxN X;

        /// <summary>k x n. A applied to each row of <see cref="X"/>; recomputed via a FRESH
        /// <c>A.Apply</c> every iteration for the active rows (see the <c>LOBPCG</c> class doc
        /// comment's "AX/AP freshness" note -- an earlier version maintained this purely via
        /// linearity, which compounded rounding error into a slow convergence stall).</summary>
        public floatMxN AX;

        /// <summary>k x n. Preconditioned residual directions; only rows [0, numActive) are
        /// meaningful in any given iteration.</summary>
        public floatMxN W;

        /// <summary>k x n. A applied to each row of <see cref="W"/> (one matvec batch per
        /// iteration, over the active rows only).</summary>
        public floatMxN AW;

        /// <summary>k x n. Previous search directions (the "conjugate" part of the basis); only
        /// rows [0, numActive) are meaningful, and only once at least one iteration has run.</summary>
        public floatMxN P;

        /// <summary>k x n. A applied to each row of <see cref="P"/>; recomputed via a FRESH
        /// <c>A.Apply</c> every iteration for the active rows, same rationale as <see cref="AX"/>
        /// (this one mattered even more in practice -- an inaccurate AP fed directly into the
        /// NEXT iteration's Rayleigh-Ritz energy matrix, not just the residual check).</summary>
        public floatMxN AP;

        /// <summary>k x n. Raw (unpreconditioned) residuals A x - lambda x for the active rows;
        /// scratch feeding the preconditioner Apply that produces <see cref="W"/>.</summary>
        public floatMxN R;

        /// <summary>k x n each. <see cref="Xnext"/>/<see cref="Pnext"/> are the ping-pong
        /// destination buffers for the new X/P block computed each iteration (the combination
        /// reads the CURRENT X/W/P, so it cannot safely write in place) -- swapped into
        /// <see cref="X"/>/<see cref="P"/> at the end of every iteration (a cheap struct-handle
        /// swap, not a buffer copy). <c>AXnext</c>/<c>APnext</c> are allocated but UNUSED: an
        /// earlier version mirror-combined AX/AP the same way, but that work was always
        /// immediately discarded (the caller unconditionally recomputes AX/AP via a fresh
        /// <c>A.Apply</c> right after -- see <see cref="AX"/>/<see cref="AP"/>), so they were
        /// removed from the hot path; the fields remain (rather than reshaping this struct and
        /// every codegen'd caller) but are dead weight -- do not rely on their contents.</summary>
        public floatMxN Xnext, AXnext, Pnext, APnext;

        /// <summary>k x n each. GENERALIZED-eigenproblem B-images of <see cref="X"/>/<see cref="W"/>/
        /// <see cref="P"/> (B applied row-wise) -- see the <c>LOBPCG</c> class doc comment's
        /// "Generalized eigenproblem" / "fresh-matvec principle extends to B" notes. <c>BX</c>/
        /// <c>BP</c> are recomputed via a FRESH <c>B.Apply</c> at the same points <see cref="AX"/>/
        /// <see cref="AP"/> are (never mirror-combined across an iteration boundary); <c>BW</c> gets
        /// ONE fresh <c>B.Apply</c> per iteration (mirroring <see cref="AW"/>) and is then carried
        /// through that SAME iteration's deflation/internal-orthonormalization via linearity. For
        /// the standard (B=I) forwarding path these are exact bit-copies of X/W/P respectively --
        /// see the class doc's "B=I strategy" note for why that keeps the standard path
        /// bit-identical to the pre-generalization implementation despite the extra buffers.</summary>
        public floatMxN BX, BW, BP;

        /// <summary>Length k. Current Ritz value estimates (in/out; sorted ascending only at the
        /// final return).</summary>
        public floatN lambda;

        /// <summary>Length k. Per-pair 2-norm residual ‖A x_i - lambda_i B x_i‖ (B=I reduces this
        /// to ‖A x_i - lambda_i x_i‖), latest computed value (frozen at its locking-time value for
        /// already-converged pairs).</summary>
        public floatN residual;

        /// <summary>Length n each. Scratch row buffers used whenever a single operator/
        /// preconditioner Apply call needs to read from or write into one row of a k x n block
        /// (Apply operates on <see cref="floatN"/>, not a matrix row).</summary>
        public floatN rowIn, rowOut;

        /// <summary>Length n. Third row-combination scratch used only by
        /// <c>LOBPCG.OrthonormalizeBlockB</c> (the B-aware sibling of <c>OrthonormalizeBlock</c>) to
        /// carry a block's B-image (BW or BP) through the SAME Cholesky-QR row combination applied
        /// to that block itself and its A-image -- <see cref="rowIn"/>/<see cref="rowOut"/> already
        /// serve as the other two (V/AV) scratch slots there, so a third distinct buffer is needed
        /// for BV.</summary>
        public floatN rowAux;

        /// <summary>3k x 3k each. Backing store for the small dense Rayleigh-Ritz sub-problem
        /// (Gram = S^T S, H = S^T A S, L = Cholesky factor of Gram, Atrans = the transformed
        /// standard-form matrix L^-1 H L^-T, Y = Atrans's eigenvectors, C = the recovered
        /// combination coefficients L^-T Y). Only the leading m x m block is used in any given
        /// iteration (m = 2*numActive or 3*numActive &lt;= 3k) via a same-buffer, smaller-shaped
        /// LOGICAL view (see <c>LOBPCG.View</c>) -- never a fresh allocation.</summary>
        public floatMxN Gram, H, L, Atrans, Y, C;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an LOBPCG workspace for a k-eigenpair solve over an n-dimensional symmetric
        /// operator (standard B=I or generalized pencil (A, B) -- the SAME layout serves both, see
        /// <see cref="floatLOBPCGCache"/>'s BX/BW/BP fields). See <see cref="floatLOBPCGCache"/>
        /// for reuse guidance.
        /// </summary>
        public static floatLOBPCGCache floatLOBPCGCache(this ref Arena arena, int n, int k)
        {
            int cap = 3 * k;
            return new floatLOBPCGCache
            {
                X = arena.floatMat(k, n),
                AX = arena.floatMat(k, n),
                W = arena.floatMat(k, n),
                AW = arena.floatMat(k, n),
                P = arena.floatMat(k, n),
                AP = arena.floatMat(k, n),
                R = arena.floatMat(k, n),
                Xnext = arena.floatMat(k, n),
                AXnext = arena.floatMat(k, n),
                Pnext = arena.floatMat(k, n),
                APnext = arena.floatMat(k, n),
                BX = arena.floatMat(k, n),
                BW = arena.floatMat(k, n),
                BP = arena.floatMat(k, n),
                lambda = arena.floatVec(k),
                residual = arena.floatVec(k),
                rowIn = arena.floatVec(n),
                rowOut = arena.floatVec(n),
                rowAux = arena.floatVec(n),
                Gram = arena.floatMat(cap, cap),
                H = arena.floatMat(cap, cap),
                L = arena.floatMat(cap, cap),
                Atrans = arena.floatMat(cap, cap),
                Y = arena.floatMat(cap, cap),
                C = arena.floatMat(cap, cap),
            };
        }
    }
}
