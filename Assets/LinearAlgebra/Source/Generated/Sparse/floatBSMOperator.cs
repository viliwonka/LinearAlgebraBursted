using LinearAlgebra;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Thin <see cref="IfloatLinearOperator"/> wrapper over a compressed <see cref="floatBSM"/>.
    /// Forwards Apply/ApplyT straight to <see cref="Sparse_OP.spMV"/>/<see cref="Sparse_OP.spMVT"/>
    /// -- this is the wrapper the Phase-1 SparseOP.float.cs header comment anticipated. Lets the
    /// generic Krylov solvers (<c>Solvers.cg&lt;TOp&gt;</c>, <c>Solvers.pcg&lt;TOp,TPre&gt;</c>) run
    /// over a BSM with zero-cost Burst static dispatch, no vtable.
    /// Readonly: a value copy of this struct only copies the floatBSM header (a handful of
    /// UnsafeList headers + ints), not the underlying buffers -- cheap and safe to pass through
    /// `in` parameters in generic constrained calls.
    /// </summary>
    public readonly struct floatBSMOperator : IfloatLinearOperator
    {
        public readonly floatBSM A;

        public floatBSMOperator(in floatBSM a)
        {
            A = a;
        }

        public int Rows => A.M_Rows;
        public int Cols => A.N_Cols;

        public void Apply(in floatN x, ref floatN y) => Sparse_OP.spMV(in A, in x, ref y);

        public void ApplyT(in floatN x, ref floatN y) => Sparse_OP.spMVT(in A, in x, ref y);
    }
}
