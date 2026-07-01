namespace LinearAlgebra
{
    /// <summary>
    /// A linear operator y = A x, abstracted behind a Burst-friendly generic struct
    /// constraint (NOT a managed interface dispatch/vtable -- solvers are generic over
    /// <c>TOp : struct, IdoubleLinearOperator</c>, so Burst monomorphizes each concrete
    /// operator into its own zero-cost specialization). Lets Krylov solvers (CG, PCG, MINRES,
    /// BiCGSTAB, CGLS, LSQR -- see <c>Solvers</c>) be written ONCE and reused over both dense
    /// (<see cref="doubleDenseOperator"/>) and block-sparse
    /// (<c>LinearAlgebra.Sparse.doubleBSMOperator</c>) matrices without duplicating the solver
    /// loop.
    /// Implement on a small, ideally-readonly struct holding only blittable fields (a value
    /// copy of the wrapped matrix/BSM struct) -- same Burst-functor contract as
    /// <see cref="IdoubleScalarFunction"/> / <see cref="IdoubleSampler"/>.
    /// </summary>
    public interface IdoubleLinearOperator
    {
        int Rows { get; }
        int Cols { get; }

        /// <summary>y = A x. y must be distinct from x (see each implementation's aliasing guard).</summary>
        void Apply(in doubleN x, ref doubleN y);

        /// <summary>y = Aᵀ x. y must be distinct from x. Needed by CGLS/LSQR/BiCGSTAB (Phase 3).</summary>
        void ApplyT(in doubleN x, ref doubleN y);
    }

    /// <summary>
    /// A preconditioner z = M⁻¹ r, the same Burst-friendly generic-struct-constraint shape as
    /// <see cref="IdoubleLinearOperator"/>. Solvers are generic over <c>TPre : struct,
    /// IdoublePreconditioner</c>. No Rows/Cols on this interface -- the operand's own dimension
    /// is the source of truth (M and A must agree; implementations validate against their own
    /// stored shape).
    /// </summary>
    public interface IdoublePreconditioner
    {
        /// <summary>z = M⁻¹ r. z must be distinct from r (see each implementation's aliasing guard).</summary>
        void Apply(in doubleN r, ref doubleN z);
    }

    /// <summary>
    /// Thin <see cref="IdoubleLinearOperator"/> wrapper over a dense <see cref="doubleMxN"/>.
    /// Forwards Apply/ApplyT to the existing dense matVec/vecMat dot kernels -- this is what
    /// the concrete <c>Solvers.conjugateGradient(in doubleMxN, ...)</c> overloads wrap
    /// internally so the generic <c>Solvers.cg&lt;TOp&gt;</c> loop is the single source of truth.
    /// Readonly: a value copy of this struct is cheap (holds only the doubleMxN header, no
    /// buffer copy) and safe to pass through `in` parameters in generic constrained calls
    /// without observable mutation.
    /// </summary>
    public readonly struct doubleDenseOperator : IdoubleLinearOperator
    {
        public readonly doubleMxN A;

        public doubleDenseOperator(in doubleMxN a)
        {
            A = a;
        }

        public int Rows => A.M_Rows;
        public int Cols => A.N_Cols;

        public void Apply(in doubleN x, ref doubleN y) => Linear_OP.dot(in A, in x, ref y);

        // Aᵀx via the existing vector*matrix dot: result[j] = sum_i x[i]*A[i,j] == (Aᵀx)[j].
        public void ApplyT(in doubleN x, ref doubleN y) => Linear_OP.dot(in x, in A, ref y);
    }
}
