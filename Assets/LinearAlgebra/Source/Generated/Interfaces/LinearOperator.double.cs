namespace LinearAlgebra
{
    /// <summary>
    /// A linear operator y = A x, abstracted behind a generic struct constraint: solvers are
    /// generic over <c>TOp : struct, IdoubleLinearOperator</c>, so each Apply compiles to a direct
    /// call (no virtual dispatch). Lets Krylov solvers (CG, PCG, MINRES, BiCGSTAB, CGLS, LSQR --
    /// see <c>Krylov</c>) be written ONCE and reused over both dense
    /// (<see cref="doubleDenseOperator"/>) and block-sparse
    /// (<c>LinearAlgebra.Sparse.doubleBSROperator</c>) matrices without duplicating the solver
    /// loop.
    /// Implement on a small, ideally-readonly struct holding only blittable fields (a value
    /// copy of the wrapped matrix/BSR struct) -- same struct-functor contract as
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

        /// <summary>
        /// y = A x (same contract as <see cref="Apply"/>), and also returns dot(x, y) -- a single
        /// call site for cg/pcg's <c>pAp = dot(p, Ap)</c>. Only meaningful when x and y are the
        /// same length (A square, Rows == Cols). Every implementation composes Apply then
        /// <c>Blas.dot</c>; opt-in per call site, not a replacement for Apply.
        /// </summary>
        double ApplyDot(in doubleN x, ref doubleN y);

        /// <summary>
        /// Applies the operator to a BLOCK of row-vectors at once: for i in [0, rows),
        /// AVrows[i,:] = A · Vrows[i,:]. Vrows/AVrows are row-major (at least rows × Cols); only the
        /// first <paramref name="rows"/> rows are read/written, and AVrows must not alias Vrows. Lets
        /// a caller holding many simultaneous vectors (e.g. LOBPCG's k-wide X/W/P blocks) stream the
        /// operator's data ONCE instead of once per vector. Intended for SYMMETRIC operators (A = Aᵀ):
        /// the dense fast path exploits symmetry, and the only caller — the symmetric-eigenproblem
        /// LOBPCG — always passes symmetric A and B.
        /// </summary>
        void ApplyBlock(in doubleMxN Vrows, ref doubleMxN AVrows, int rows);
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
    /// the concrete <c>Krylov.cg(in doubleMxN, ...)</c> overloads wrap
    /// internally so the generic <c>Krylov.cg&lt;TOp&gt;</c> loop is the single source of truth.
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

        // Composes (Apply, then a plain dot pass) via Blas.dotSelf.
        public double ApplyDot(in doubleN x, ref doubleN y) => Blas.dotSelf(in A, in x, ref y);

        // Block apply: AVrows[i,:] = A · Vrows[i,:] for the first `rows` rows, as ONE matMatDot that
        // streams A once (vs `rows` separate GEMVs). Blas.dotRows(Vrows, A)[r,j] = Σ_i Vrows[r,i]·A[i,j]
        // = (A · Vrows[r])[j] when A is symmetric — the invariant every ApplyBlock caller holds. Only
        // the first `rows` rows of AVrows are written; the rest are preserved (locked-pair data).
        public void ApplyBlock(in doubleMxN Vrows, ref doubleMxN AVrows, int rows)
            => Blas.dotRows(in Vrows, in A, ref AVrows, rows);
    }

    /// <summary>
    /// Identity preconditioner: z = r (no-op copy). Gives a <c>TPre</c>-generic solver (e.g.
    /// <see cref="Eigen.lobpcg{TOp,TPre}"/>) a concrete, zero-cost "no preconditioner" instance so
    /// its UNPRECONDITIONED entry point can forward into the single preconditioned generic
    /// implementation with a one-line call instead of duplicating a large loop body -- the same
    /// role <see cref="doubleDenseOperator"/> plays for dense callers of a <c>TOp</c>-generic
    /// solver. Readonly and stateless: a value copy costs nothing, and the `M.Apply` call compiles
    /// straight to the plain copy below.
    /// </summary>
    public readonly struct doubleIdentityPreconditioner : IdoublePreconditioner
    {
        public void Apply(in doubleN r, ref doubleN z) => z.Data.CopyFrom(r.Data);
    }

    /// <summary>
    /// Identity linear operator: y = x (an exact bit-copy), Rows == Cols == the size fixed at
    /// construction. Lets B=I callers (e.g. <see cref="Eigen.lobpcg{TOp,TPre}"/>) forward the
    /// standard eigenproblem into the generalized <see cref="Eigen.lobpcg{TOp,TBOp,TPre}"/> core
    /// with B played by this identity, without duplicating a Euclidean-only implementation.
    /// </summary>
    public readonly struct doubleIdentityOperator : IdoubleLinearOperator
    {
        public readonly int N;

        public doubleIdentityOperator(int n)
        {
            N = n;
        }

        public int Rows => N;
        public int Cols => N;

        public void Apply(in doubleN x, ref doubleN y) => y.Data.CopyFrom(x.Data);
        public void ApplyT(in doubleN x, ref doubleN y) => y.Data.CopyFrom(x.Data);

        // y == x exactly (an exact bit-copy), so dot(x,y) == dot(x,x) == ||x||^2. Nothing to
        // fuse beyond the copy itself -- this composes (Apply, then Blas.dot), which for the
        // identity operator is already the cheapest possible ApplyDot.
        public double ApplyDot(in doubleN x, ref doubleN y)
        {
            Apply(in x, ref y);
            return Blas.dot(x, y);
        }

        // Identity block apply: copy the first `rows` rows (an exact bit-copy, like Apply).
        public void ApplyBlock(in doubleMxN Vrows, ref doubleMxN AVrows, int rows)
        {
            int cols = Vrows.N_Cols;
            for (int i = 0; i < rows; i++)
                for (int c = 0; c < cols; c++)
                    AVrows[i, c] = Vrows[i, c];
        }
    }

    /// <summary>
    /// Wraps <typeparamref name="TInner"/> with a diagonal column scale D = diag(d), presenting the
    /// operator A·D. <see cref="Apply"/> forms A(d.*x) via an owned scratch buffer; solve (A·D) y = b
    /// for y, then recover x = D·y (x[j] = d[j]·y[j]). <c>d</c>/<c>scratch</c> must not alias each
    /// other or any vector passed to Apply/ApplyT. With Tikhonov damping, damping the scaled system
    /// penalizes ‖y‖ = ‖D⁻¹x‖ (a column-weighted ridge on x), not ‖x‖ -- a different regularizer.
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

        // Composes: Apply, then a separate dot pass. This wrapper is RECTANGULAR in its usual
        // callers (cgls/lsqr column-preconditioning, x length Cols, y length Rows) -- dot(x,y)
        // isn't even well-formed there, so ApplyDot exists only to satisfy the interface; no
        // solver calls it on this operator today (cgls/lsqr don't use ApplyDot).
        public double ApplyDot(in doubleN x, ref doubleN y)
        {
            Apply(in x, ref y);
            return Blas.dot(x, y);
        }

        // No block specialization (this wrapper composes over an arbitrary inner operator): apply per
        // row through the scalar Apply, into two bounded Temp scratch vectors.
        public void ApplyBlock(in doubleMxN Vrows, ref doubleMxN AVrows, int rows)
        {
            int cols = Vrows.N_Cols;
            var rin = new doubleN(cols, Unity.Collections.Allocator.Temp, false);
            var rout = new doubleN(cols, Unity.Collections.Allocator.Temp, false);
            for (int i = 0; i < rows; i++)
            {
                for (int c = 0; c < cols; c++) rin[c] = Vrows[i, c];
                Apply(in rin, ref rout);
                for (int c = 0; c < cols; c++) AVrows[i, c] = rout[c];
            }
            rout.Dispose();
            rin.Dispose();
        }
    }
}
