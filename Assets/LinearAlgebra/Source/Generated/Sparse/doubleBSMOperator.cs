using LinearAlgebra;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Thin <see cref="IdoubleLinearOperator"/> wrapper over a compressed <see cref="doubleBSM"/>.
    /// Forwards Apply/ApplyT straight to <see cref="Sparse_OP.spMV"/>/<see cref="Sparse_OP.spMVT"/>
    /// -- this is the wrapper the Phase-1 SparseOP.double.cs header comment anticipated. Lets the
    /// generic Krylov solvers (<c>Solvers.cg&lt;TOp&gt;</c>, <c>Solvers.pcg&lt;TOp,TPre&gt;</c>) run
    /// over a BSM with zero-cost Burst static dispatch, no vtable.
    /// Readonly: a value copy of this struct only copies the doubleBSM header (a handful of
    /// UnsafeList headers + ints), not the underlying buffers -- cheap and safe to pass through
    /// `in` parameters in generic constrained calls.
    /// </summary>
    public readonly struct doubleBSMOperator : IdoubleLinearOperator
    {
        public readonly doubleBSM A;

        public doubleBSMOperator(in doubleBSM a)
        {
            A = a;
        }

        public int Rows => A.M_Rows;
        public int Cols => A.N_Cols;

        public void Apply(in doubleN x, ref doubleN y) => Sparse_OP.spMV(in A, in x, ref y);

        public void ApplyT(in doubleN x, ref doubleN y) => Sparse_OP.spMVT(in A, in x, ref y);
    }
}
