using LinearAlgebra;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Thin <see cref="IfloatLinearOperator"/> wrapper over a compressed <see cref="floatBSR"/>.
    /// Forwards Apply straight to <see cref="BSR.spMV"/>. ApplyT forwards either to the
    /// on-the-fly, scatter-traversal <see cref="BSR.spMVT"/> (one-arg ctor, the default --
    /// no transpose materialized) or to a cache-friendly FORWARD <see cref="BSR.spMV"/>
    /// over a precomputed transpose AT (two-arg ctor -- see <see cref="Arena.floatBSRTranspose"/>),
    /// depending on which constructor built this operator.
    /// Lets the generic Krylov solvers (<c>Krylov.cg&lt;TOp&gt;</c>,
    /// <c>Krylov.pcg&lt;TOp,TPre&gt;</c>) run over a BSR matrix.
    /// Readonly: a value copy of this struct only copies the floatBSR/AT headers (a handful of
    /// UnsafeList headers + ints), not the underlying buffers -- cheap and safe to pass through
    /// `in` parameters in generic constrained calls.
    /// </summary>
    public readonly struct floatBSROperator : IfloatLinearOperator
    {
        public readonly floatBSR A;

        /// <summary>
        /// Optional precomputed transpose of A (see <see cref="Arena.floatBSRTranspose"/>).
        /// Default/unset (one-arg ctor) when <see cref="_hasT"/> is false -- ApplyT then falls
        /// back to the on-the-fly <see cref="BSR.spMVT"/>.
        /// </summary>
        public readonly floatBSR AT;
        private readonly bool _hasT;

        /// <summary>
        /// No precomputed transpose: ApplyT runs the on-the-fly scatter-traversal spMVT every
        /// call. Keeps today's behavior for callers that only ever do a one-shot ApplyT (or a
        /// few), where materializing AT up front wouldn't pay for itself.
        /// </summary>
        public floatBSROperator(in floatBSR a)
        {
            A = a;
            AT = default;
            _hasT = false;
        }

        /// <summary>
        /// Carries a precomputed transpose aT (typically <c>arena.floatBSRTranspose(in a)</c>,
        /// built ONCE per solve). ApplyT then forwards to <see cref="BSR.spMV(in floatBSR,
        /// in floatN, ref floatN)"/> over aT -- a forward, cache-friendly block-CSR traversal --
        /// instead of the scatter-heavy <see cref="BSR.spMVT"/> over a. The one-time O(nnz)
        /// transpose build is amortized over every iteration a solver (e.g. cgls/lsqr) calls
        /// ApplyT. Caller is responsible for aT actually being a's transpose -- this ctor does not
        /// verify it (that would defeat the point of precomputing it once).
        /// </summary>
        public floatBSROperator(in floatBSR a, in floatBSR aT)
        {
            A = a;
            AT = aT;
            _hasT = true;
        }

        public int Rows => A.M_Rows;
        public int Cols => A.N_Cols;

        public void Apply(in floatN x, ref floatN y) => BSR.spMV(in A, in x, ref y);

        public void ApplyT(in floatN x, ref floatN y)
        {
            if (_hasT)
                BSR.spMV(in AT, in x, ref y);
            else
                BSR.spMVT(in A, in x, ref y);
        }

        // Forwards to BSR.spMVDot, which composes (spMV then a plain dot) -- see that method's doc
        // comment for why a genuinely-fused kernel was tried and measured slower.
        public float ApplyDot(in floatN x, ref floatN y) => BSR.spMVDot(in A, in x, ref y);

        // Krylov R5 (docs/draft-spec-krylov-optimization.md): a real BSR SpMM kernel -- streams A's
        // stored blocks ONCE and applies to all `rows` row-vectors together, no Allocator.Temp churn.
        // See BSR.spMM (SparseOP.float.cs) for the dispatch and bsrMatMat*/bsrMatMatSym* (UnsafeOP.
        // Sparse.float.cs) for the kernels -- each is bit-identical, row by row, to `rows` separate
        // Apply calls.
        public void ApplyBlock(in floatMxN Vrows, ref floatMxN AVrows, int rows) => BSR.spMM(in A, in Vrows, ref AVrows, rows);
    }
}
