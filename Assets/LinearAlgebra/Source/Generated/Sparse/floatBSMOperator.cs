using LinearAlgebra;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Thin <see cref="IfloatLinearOperator"/> wrapper over a compressed <see cref="floatBSM"/>.
    /// Forwards Apply straight to <see cref="Sparse_OP.spMV"/>. ApplyT forwards either to the
    /// on-the-fly, scatter-traversal <see cref="Sparse_OP.spMVT"/> (one-arg ctor, the default --
    /// no transpose materialized) or to a cache-friendly FORWARD <see cref="Sparse_OP.spMV"/>
    /// over a precomputed transpose AT (two-arg ctor -- see <see cref="Arena.floatBSMTranspose"/>),
    /// depending on which constructor built this operator.
    /// -- this is the wrapper the Phase-1 SparseOP.float.cs header comment anticipated. Lets the
    /// generic Krylov solvers (<c>Solvers.cg&lt;TOp&gt;</c>, <c>Solvers.pcg&lt;TOp,TPre&gt;</c>) run
    /// over a BSM with zero-cost Burst static dispatch, no vtable.
    /// Readonly: a value copy of this struct only copies the floatBSM/AT headers (a handful of
    /// UnsafeList headers + ints), not the underlying buffers -- cheap and safe to pass through
    /// `in` parameters in generic constrained calls.
    /// </summary>
    public readonly struct floatBSMOperator : IfloatLinearOperator
    {
        public readonly floatBSM A;

        /// <summary>
        /// Optional precomputed transpose of A (see <see cref="Arena.floatBSMTranspose"/>).
        /// Default/unset (one-arg ctor) when <see cref="_hasT"/> is false -- ApplyT then falls
        /// back to the on-the-fly <see cref="Sparse_OP.spMVT"/>.
        /// </summary>
        public readonly floatBSM AT;
        private readonly bool _hasT;

        /// <summary>
        /// No precomputed transpose: ApplyT runs the on-the-fly scatter-traversal spMVT every
        /// call. Keeps today's behavior for callers that only ever do a one-shot ApplyT (or a
        /// few), where materializing AT up front wouldn't pay for itself.
        /// </summary>
        public floatBSMOperator(in floatBSM a)
        {
            A = a;
            AT = default;
            _hasT = false;
        }

        /// <summary>
        /// Carries a precomputed transpose aT (typically <c>arena.floatBSMTranspose(in a)</c>,
        /// built ONCE per solve). ApplyT then forwards to <see cref="Sparse_OP.spMV(in floatBSM,
        /// in floatN, ref floatN)"/> over aT -- a forward, cache-friendly block-CSR traversal --
        /// instead of the scatter-heavy <see cref="Sparse_OP.spMVT"/> over a. The one-time O(nnz)
        /// transpose build is amortized over every iteration a solver (e.g. cgls/lsqr) calls
        /// ApplyT. Caller is responsible for aT actually being a's transpose -- this ctor does not
        /// verify it (that would defeat the point of precomputing it once).
        /// </summary>
        public floatBSMOperator(in floatBSM a, in floatBSM aT)
        {
            A = a;
            AT = aT;
            _hasT = true;
        }

        public int Rows => A.M_Rows;
        public int Cols => A.N_Cols;

        public void Apply(in floatN x, ref floatN y) => Sparse_OP.spMV(in A, in x, ref y);

        public void ApplyT(in floatN x, ref floatN y)
        {
            if (_hasT)
                Sparse_OP.spMV(in AT, in x, ref y);
            else
                Sparse_OP.spMVT(in A, in x, ref y);
        }
    }
}
