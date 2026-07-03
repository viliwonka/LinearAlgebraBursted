using LinearAlgebra;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Thin <see cref="IfProxyLinearOperator"/> wrapper over a compressed <see cref="fProxyBSR"/>.
    /// Forwards Apply straight to <see cref="BSR.spMV"/>. ApplyT forwards either to the
    /// on-the-fly, scatter-traversal <see cref="BSR.spMVT"/> (one-arg ctor, the default --
    /// no transpose materialized) or to a cache-friendly FORWARD <see cref="BSR.spMV"/>
    /// over a precomputed transpose AT (two-arg ctor -- see <see cref="Arena.fProxyBSRTranspose"/>),
    /// depending on which constructor built this operator.
    /// -- this is the wrapper the Phase-1 SparseOP.fProxy.cs header comment anticipated. Lets the
    /// generic Krylov solvers (<c>Solvers.cg&lt;TOp&gt;</c>, <c>Solvers.pcg&lt;TOp,TPre&gt;</c>) run
    /// over a BSR with zero-cost Burst static dispatch, no vtable.
    /// Readonly: a value copy of this struct only copies the fProxyBSR/AT headers (a handful of
    /// UnsafeList headers + ints), not the underlying buffers -- cheap and safe to pass through
    /// `in` parameters in generic constrained calls.
    /// </summary>
    public readonly struct fProxyBSROperator : IfProxyLinearOperator
    {
        public readonly fProxyBSR A;

        /// <summary>
        /// Optional precomputed transpose of A (see <see cref="Arena.fProxyBSRTranspose"/>).
        /// Default/unset (one-arg ctor) when <see cref="_hasT"/> is false -- ApplyT then falls
        /// back to the on-the-fly <see cref="BSR.spMVT"/>.
        /// </summary>
        public readonly fProxyBSR AT;
        private readonly bool _hasT;

        /// <summary>
        /// No precomputed transpose: ApplyT runs the on-the-fly scatter-traversal spMVT every
        /// call. Keeps today's behavior for callers that only ever do a one-shot ApplyT (or a
        /// few), where materializing AT up front wouldn't pay for itself.
        /// </summary>
        public fProxyBSROperator(in fProxyBSR a)
        {
            A = a;
            AT = default;
            _hasT = false;
        }

        /// <summary>
        /// Carries a precomputed transpose aT (typically <c>arena.fProxyBSRTranspose(in a)</c>,
        /// built ONCE per solve). ApplyT then forwards to <see cref="BSR.spMV(in fProxyBSR,
        /// in fProxyN, ref fProxyN)"/> over aT -- a forward, cache-friendly block-CSR traversal --
        /// instead of the scatter-heavy <see cref="BSR.spMVT"/> over a. The one-time O(nnz)
        /// transpose build is amortized over every iteration a solver (e.g. cgls/lsqr) calls
        /// ApplyT. Caller is responsible for aT actually being a's transpose -- this ctor does not
        /// verify it (that would defeat the point of precomputing it once).
        /// </summary>
        public fProxyBSROperator(in fProxyBSR a, in fProxyBSR aT)
        {
            A = a;
            AT = aT;
            _hasT = true;
        }

        public int Rows => A.M_Rows;
        public int Cols => A.N_Cols;

        public void Apply(in fProxyN x, ref fProxyN y) => BSR.spMV(in A, in x, ref y);

        public void ApplyT(in fProxyN x, ref fProxyN y)
        {
            if (_hasT)
                BSR.spMV(in AT, in x, ref y);
            else
                BSR.spMVT(in A, in x, ref y);
        }
    }
}
