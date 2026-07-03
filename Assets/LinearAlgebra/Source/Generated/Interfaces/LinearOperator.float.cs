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
    /// the concrete <c>Solvers.cg(in floatMxN, ...)</c> overloads wrap
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

    /// <summary>
    /// Right column-scaling wrapper: presents the operator A·D where D = diag(d) is a diagonal
    /// scaling of the INPUT (column) space, over any inner <typeparamref name="TInner"/> operator.
    /// Composes with every generic solver with NO solver change (they are already generic over
    /// <c>TOp</c>), so passing <c>floatColScaledOperator&lt;floatDenseOperator&gt;</c> (or the BSM
    /// operator) turns cgls/lsqr/lsmr into their column-preconditioned variants: with
    /// d[j] = 1/‖A_:,j‖₂ (an AᵀA-Jacobi / column-equilibration preconditioner, built via
    /// <c>Linear_OP.columnNormsSquared</c> + <c>Linear_OP.buildJacobiScale</c>) the scaled operator
    /// A·D has a unit-diagonal normal matrix, so an ill-conditioned least-squares problem converges
    /// in fewer iterations. Solve (A·D) y = b for y, then recover x = D·y (x[j] = d[j]·y[j]).
    ///
    /// Holds the inner operator, the diagonal <c>d</c> (length inner.Cols), and one owned
    /// <c>scratch</c> buffer (length inner.Cols) that <see cref="Apply"/> uses to form d.*x without
    /// touching the caller's x. <c>d</c> and <c>scratch</c> must be distinct from each other and
    /// from every vector the solver passes to Apply/ApplyT (the arena convenience overloads
    /// guarantee this by allocating fresh). Readonly, like <see cref="floatDenseOperator"/>. NOTE:
    /// with Tikhonov damping, damping the SCALED system penalizes ‖y‖ = ‖D⁻¹x‖ (a column-weighted
    /// ridge on x), NOT ‖x‖ -- a different regularizer; use the composable path if you need that control.
    /// </summary>
    public readonly struct floatColScaledOperator<TInner> : IfloatLinearOperator
        where TInner : struct, IfloatLinearOperator
    {
        public readonly TInner Inner;
        public readonly floatN D;        // length Inner.Cols: the column scale
        public readonly floatN Scratch;  // length Inner.Cols: workspace for Apply (holds d .* x)

        public floatColScaledOperator(in TInner inner, in floatN d, in floatN scratch)
        {
            if (d.N != inner.Cols)
                throw new System.ArgumentException("floatColScaledOperator: d.N must equal inner.Cols");
            if (scratch.N != inner.Cols)
                throw new System.ArgumentException("floatColScaledOperator: scratch.N must equal inner.Cols");

            Inner = inner;
            D = d;
            Scratch = scratch;
        }

        public int Rows => Inner.Rows;
        public int Cols => Inner.Cols;

        // (A D) x = A (d .* x). Scales into the owned Scratch so the caller's x is untouched.
        public void Apply(in floatN x, ref floatN y)
        {
            for (int j = 0; j < D.N; j++) Scratch[j] = D[j] * x[j];
            Inner.Apply(in Scratch, ref y);
        }

        // (A D)ᵀ y = D Aᵀ y = d .* (Aᵀ y). Scales y in place after the inner transpose -- no extra
        // scratch needed (y is length Cols, the output).
        public void ApplyT(in floatN x, ref floatN y)
        {
            Inner.ApplyT(in x, ref y);
            for (int j = 0; j < D.N; j++) y[j] *= D[j];
        }
    }
}
