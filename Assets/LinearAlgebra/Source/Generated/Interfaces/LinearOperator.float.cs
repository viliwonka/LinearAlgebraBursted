namespace LinearAlgebra
{
    /// <summary>
    /// A linear operator y = A x, abstracted behind a Burst-friendly generic struct
    /// constraint (NOT a managed interface dispatch/vtable -- solvers are generic over
    /// <c>TOp : struct, IfloatLinearOperator</c>, so Burst monomorphizes each concrete
    /// operator into its own zero-cost specialization). Lets Krylov solvers (CG, PCG, MINRES,
    /// BiCGSTAB, CGLS, LSQR -- see <c>Solvers</c>) be written ONCE and reused over both dense
    /// (<see cref="floatDenseOperator"/>) and block-sparse
    /// (<c>LinearAlgebra.Sparse.floatBSMOperator</c>) matrices without duplicating the solver
    /// loop.
    /// Implement on a small, ideally-readonly struct holding only blittable fields (a value
    /// copy of the wrapped matrix/BSM struct) -- same Burst-functor contract as
    /// <see cref="IfloatScalarFunction"/> / <see cref="IfloatSampler"/>.
    /// </summary>
    public interface IfloatLinearOperator
    {
        int Rows { get; }
        int Cols { get; }

        /// <summary>y = A x. y must be distinct from x (see each implementation's aliasing guard).</summary>
        void Apply(in floatN x, ref floatN y);

        /// <summary>y = Aᵀ x. y must be distinct from x. Needed by CGLS/LSQR/BiCGSTAB (Phase 3).</summary>
        void ApplyT(in floatN x, ref floatN y);
    }

    /// <summary>
    /// A preconditioner z = M⁻¹ r, the same Burst-friendly generic-struct-constraint shape as
    /// <see cref="IfloatLinearOperator"/>. Solvers are generic over <c>TPre : struct,
    /// IfloatPreconditioner</c>. No Rows/Cols on this interface -- the operand's own dimension
    /// is the source of truth (M and A must agree; implementations validate against their own
    /// stored shape).
    /// </summary>
    public interface IfloatPreconditioner
    {
        /// <summary>z = M⁻¹ r. z must be distinct from r (see each implementation's aliasing guard).</summary>
        void Apply(in floatN r, ref floatN z);
    }

    /// <summary>
    /// Thin <see cref="IfloatLinearOperator"/> wrapper over a dense <see cref="floatMxN"/>.
    /// Forwards Apply/ApplyT to the existing dense matVec/vecMat dot kernels -- this is what
    /// the concrete <c>Solvers.conjugateGradient(in floatMxN, ...)</c> overloads wrap
    /// internally so the generic <c>Solvers.cg&lt;TOp&gt;</c> loop is the single source of truth.
    /// Readonly: a value copy of this struct is cheap (holds only the floatMxN header, no
    /// buffer copy) and safe to pass through `in` parameters in generic constrained calls
    /// without observable mutation.
    /// </summary>
    public readonly struct floatDenseOperator : IfloatLinearOperator
    {
        public readonly floatMxN A;

        public floatDenseOperator(in floatMxN a)
        {
            A = a;
        }

        public int Rows => A.M_Rows;
        public int Cols => A.N_Cols;

        public void Apply(in floatN x, ref floatN y) => Linear_OP.dot(in A, in x, ref y);

        // Aᵀx via the existing vector*matrix dot: result[j] = sum_i x[i]*A[i,j] == (Aᵀx)[j].
        public void ApplyT(in floatN x, ref floatN y) => Linear_OP.dot(in x, in A, ref y);
    }
}
