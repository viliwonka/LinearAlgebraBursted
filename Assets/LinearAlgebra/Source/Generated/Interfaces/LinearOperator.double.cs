namespace LinearAlgebra
{
    /// <summary>
    /// A linear operator y = A x, abstracted behind a Burst-friendly generic struct
    /// constraint (NOT a managed interface dispatch/vtable -- solvers are generic over
    /// <c>TOp : struct, IdoubleLinearOperator</c>, so Burst monomorphizes each concrete
    /// operator into its own zero-cost specialization). Lets Krylov solvers (CG, PCG, MINRES,
    /// BiCGSTAB, CGLS, LSQR -- see <c>Solvers</c>) be written ONCE and reused over both dense
    /// (<see cref="doubleDenseOperator"/>) and block-sparse
    /// (<c>LinearAlgebra.Sparse.doubleBSROperator</c>) matrices without duplicating the solver
    /// loop.
    /// Implement on a small, ideally-readonly struct holding only blittable fields (a value
    /// copy of the wrapped matrix/BSR struct) -- same Burst-functor contract as
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
    /// the concrete <c>Solvers.cg(in doubleMxN, ...)</c> overloads wrap
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

        public void Apply(in doubleN x, ref doubleN y) => Blas.dot(in A, in x, ref y);

        // Aᵀx via the existing vector*matrix dot: result[j] = sum_i x[i]*A[i,j] == (Aᵀx)[j].
        public void ApplyT(in doubleN x, ref doubleN y) => Blas.dot(in x, in A, ref y);
    }

    /// <summary>
    /// Right column-scaling wrapper: presents the operator A·D where D = diag(d) is a diagonal
    /// scaling of the INPUT (column) space, over any inner <typeparamref name="TInner"/> operator.
    /// Composes with every generic solver with NO solver change (they are already generic over
    /// <c>TOp</c>), so passing <c>doubleColScaledOperator&lt;doubleDenseOperator&gt;</c> (or the BSR
    /// operator) turns cgls/lsqr/lsmr into their column-preconditioned variants: with
    /// d[j] = 1/‖A_:,j‖₂ (an AᵀA-Jacobi / column-equilibration preconditioner, built via
    /// <c>Blas.columnNormsSquared</c> + <c>Blas.buildJacobiScale</c>) the scaled operator
    /// A·D has a unit-diagonal normal matrix, so an ill-conditioned least-squares problem converges
    /// in fewer iterations. Solve (A·D) y = b for y, then recover x = D·y (x[j] = d[j]·y[j]).
    ///
    /// Holds the inner operator, the diagonal <c>d</c> (length inner.Cols), and one owned
    /// <c>scratch</c> buffer (length inner.Cols) that <see cref="Apply"/> uses to form d.*x without
    /// touching the caller's x. <c>d</c> and <c>scratch</c> must be distinct from each other and
    /// from every vector the solver passes to Apply/ApplyT (the arena convenience overloads
    /// guarantee this by allocating fresh). Readonly, like <see cref="doubleDenseOperator"/>. NOTE:
    /// with Tikhonov damping, damping the SCALED system penalizes ‖y‖ = ‖D⁻¹x‖ (a column-weighted
    /// ridge on x), NOT ‖x‖ -- a different regularizer; use the composable path if you need that control.
    /// </summary>
    public readonly struct doubleColScaledOperator<TInner> : IdoubleLinearOperator
        where TInner : struct, IdoubleLinearOperator
    {
        public readonly TInner Inner;
        public readonly doubleN D;        // length Inner.Cols: the column scale
        public readonly doubleN Scratch;  // length Inner.Cols: workspace for Apply (holds d .* x)

        public doubleColScaledOperator(in TInner inner, in doubleN d, in doubleN scratch)
        {
            if (d.N != inner.Cols)
                throw new System.ArgumentException("doubleColScaledOperator: d.N must equal inner.Cols");
            if (scratch.N != inner.Cols)
                throw new System.ArgumentException("doubleColScaledOperator: scratch.N must equal inner.Cols");

            Inner = inner;
            D = d;
            Scratch = scratch;
        }

        public int Rows => Inner.Rows;
        public int Cols => Inner.Cols;

        // (A D) x = A (d .* x). Scales into the owned Scratch so the caller's x is untouched.
        public void Apply(in doubleN x, ref doubleN y)
        {
            for (int j = 0; j < D.N; j++) Scratch[j] = D[j] * x[j];
            Inner.Apply(in Scratch, ref y);
        }

        // (A D)ᵀ y = D Aᵀ y = d .* (Aᵀ y). Scales y in place after the inner transpose -- no extra
        // scratch needed (y is length Cols, the output).
        public void ApplyT(in doubleN x, ref doubleN y)
        {
            Inner.ApplyT(in x, ref y);
            for (int j = 0; j < D.N; j++) y[j] *= D[j];
        }
    }
}
