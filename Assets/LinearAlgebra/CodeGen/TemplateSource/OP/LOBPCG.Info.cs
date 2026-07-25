//singularFile//
using Unity.Collections;

namespace BULA
{
    /// <summary>
    /// Result of a blocked LOBPCG solve (<c>Eigen.lobpcg</c>). Every overload RETURNS this by
    /// value; an implicit <c>bool</c> conversion (== <see cref="Solved"/>) means the same
    /// success-test call shape used by every other iterative solver in this library compiles:
    /// <code>
    ///   if (Eigen.lobpcg(in A, ref ws, k, tol, maxIter)) { ... }
    ///   var info = Eigen.lobpcg(in A, in M, ref ws, k);
    ///   if (info.Solved) Debug.Log(info.converged);
    /// </code>
    ///
    /// Reuses <see cref="IterativeSolveStatus"/> (the same enum every other Krylov/eigen solver in
    /// this library uses) rather than a dedicated LOBPCG enum: Converged means all k pairs locked
    /// within tolerance; MaxIterations means the iteration budget ran out with some pairs still active;
    /// Degenerate means at least one requested pair was numerically degenerate at exit (its B-norm
    /// below the certification floor -- a collapsed Rayleigh-Ritz basis), so the returned pairs are
    /// NOT certified and must be treated as non-converged; Breakdown means an unrecoverable
    /// numerical condition was hit (the initial X seed could not be orthonormalized -- e.g. k &gt;
    /// n -- or a non-finite residual leaked into the loop), in which case X/lambda are undefined.
    ///
    /// Type-agnostic (no per-precision prefix) on purpose, matching <see cref="EigenSolveInfo"/> /
    /// <see cref="LanczosInfo"/> / <see cref="SolveInfo"/>: it lives in a non-templated file so
    /// codegen does not emit a duplicate definition into both the float and double partials
    /// (CS0102).
    /// </summary>
    public struct LOBPCGInfo
    {
        /// <summary>Outer iterations actually performed (0 if every pair was already converged/
        /// locked before the first iteration -- e.g. the caller warm-started with the exact
        /// eigenvectors).</summary>
        public int iterations;

        /// <summary>Number of eigenpairs that are CERTIFIED: within the residual tolerance AND not
        /// numerically degenerate (B-norm at or above the certification floor). 0..k; equals k iff
        /// <see cref="status"/> is Converged.</summary>
        public int converged;

        /// <summary>Worst-case (maximum) scale-invariant relative residual
        /// ‖A x_i - lambda_i B x_i‖ / scale_i over all k returned pairs, with scale_i =
        /// min(normAEst·‖x_i‖ + |lambda_i|·normBEst·‖x_i‖_B, max(|lambda_i|, 1)·‖x_i‖_B) --
        /// the solver's own convergence-test denominator (normAEst/normBEst are its Frobenius
        /// operator-norm estimates of A and B; normBEst is 1 for B=I), widened to <c>double</c> regardless
        /// of the solve's precision (matching every other *_Info.residual/rnorm convention in this
        /// library).
        /// Filled from the per-pair residual norms the solver already tracks (locked pairs keep
        /// their locking-time value) -- never a fresh matvec. <see cref="double.NaN"/> on a
        /// Breakdown return, where X/lambda are undefined.</summary>
        public double maxResidual;

        /// <summary>Why the solve stopped -- see <see cref="IterativeSolveStatus"/>.</summary>
        public IterativeSolveStatus status;

        /// <summary>True iff every requested eigenpair converged (<c>status ==
        /// IterativeSolveStatus.Converged</c>).</summary>
        public bool Solved => status == IterativeSolveStatus.Converged;

        /// <summary>Same as <see cref="Solved"/>, so <c>if (lobpcg(...))</c> / <c>bool ok =
        /// lobpcg(...)</c> read the same way every other solver in this library does.</summary>
        public static implicit operator bool(LOBPCGInfo i) => i.status == IterativeSolveStatus.Converged;

        /// <summary>Burst-safe compact summary, e.g. <c>LOBPCGInfo(Converged, iters=12,
        /// converged=4, maxResidual=1.23E-08)</c>. Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "LOBPCGInfo(";
            str.Append(status.Name());
            FixedString128Bytes tail = $", iters={iterations}, converged={converged}, maxResidual={maxResidual:G3})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }
}
