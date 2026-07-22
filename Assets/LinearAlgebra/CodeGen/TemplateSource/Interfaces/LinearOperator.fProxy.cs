namespace LinearAlgebra
{
    /// <summary>
    /// A linear operator y = A x, abstracted behind a generic struct constraint: solvers are
    /// generic over <c>TOp : struct, IfProxyLinearOperator</c>, so each Apply compiles to a direct
    /// call (no virtual dispatch). Lets Krylov solvers (CG, PCG, MINRES, BiCGSTAB, LSQR, LSMR --
    /// see <c>Krylov</c>) be written ONCE and reused over both dense
    /// (<see cref="fProxyDenseOperator"/>) and block-sparse
    /// (<c>LinearAlgebra.Sparse.fProxyBSROperator</c>) matrices without duplicating the solver
    /// loop.
    /// Implement on a small, ideally-readonly struct holding only blittable fields (a value
    /// copy of the wrapped matrix/BSR struct) -- same struct-functor contract as
    /// <see cref="IfProxyScalarFunction"/> / <see cref="IfProxySampler"/>.
    /// </summary>
    public interface IfProxyLinearOperator
    {
        int Rows { get; }
        int Cols { get; }

        /// <summary>y = A x. y must be distinct from x (see each implementation's aliasing guard).</summary>
        void Apply(in fProxyN x, ref fProxyN y);

        /// <summary>y = Aᵀ x. y must be distinct from x. Needed by transpose-using solvers
        /// (LSQR/LSMR) and residual audits; biCGStab is transpose-free and never calls this.</summary>
        void ApplyT(in fProxyN x, ref fProxyN y);

        /// <summary>
        /// y = A x (same contract as <see cref="Apply"/>), and also returns dot(x, y) -- a single
        /// call site for cg's <c>pAp = dot(p, Ap)</c>. Only meaningful when x and y are the
        /// same length (A square, Rows == Cols). Every implementation composes Apply then
        /// <c>Blas.dot</c>; opt-in per call site, not a replacement for Apply.
        /// </summary>
        fProxy ApplyDot(in fProxyN x, ref fProxyN y);

        /// <summary>
        /// Applies the operator to a BLOCK of row-vectors at once: for i in [0, rows),
        /// AVrows[i,:] = A · Vrows[i,:]. Vrows/AVrows are row-major (at least rows × Cols); only the
        /// first <paramref name="rows"/> rows are read/written, and AVrows must not alias Vrows. Lets
        /// a caller holding many simultaneous vectors (e.g. LOBPCG's k-wide X/W/P blocks) stream the
        /// operator's data ONCE instead of once per vector. Intended for SYMMETRIC operators (A = Aᵀ):
        /// the dense fast path exploits symmetry, and the only caller — the symmetric-eigenproblem
        /// LOBPCG — always passes symmetric A and B.
        /// </summary>
        void ApplyBlock(in fProxyMxN Vrows, ref fProxyMxN AVrows, int rows);
    }

    /// <summary>
    /// A preconditioner z = M⁻¹ r, the same Burst-friendly generic-struct-constraint shape as
    /// <see cref="IfProxyLinearOperator"/>. Solvers are generic over <c>TPre : struct,
    /// IfProxyPreconditioner</c>. No Rows/Cols on this interface -- the operand's own dimension
    /// is the source of truth (M and A must agree; implementations validate against their own
    /// stored shape).
    /// </summary>
    public interface IfProxyPreconditioner
    {
        /// <summary>z = M⁻¹ r. z must be distinct from r (see each implementation's aliasing guard).</summary>
        void Apply(in fProxyN r, ref fProxyN z);

        /// <summary>Compile-time literal: true ONLY for <see cref="fProxyIdentityPreconditioner"/>,
        /// false for every real preconditioner. A <c>TPre</c>-generic solver branches on it
        /// (<c>if (!M.IsIdentity) M.Apply(...)</c>); because TPre is a struct type parameter the
        /// branch constant-folds per specialization, so the identity case compiles down to the
        /// unpreconditioned solver (no <c>Apply</c>, no z traffic) from the single body.</summary>
        bool IsIdentity { get; }

        /// <summary>True iff M is symmetric positive-definite. Required by solvers that measure
        /// convergence in the M-inner-product (cg, minres, minresQLP, fcg, lobpcg, and their block
        /// variants) -- an indefinite or non-symmetric M breaks that inner product. On the static
        /// preconditioners this is a compile-time literal, so a <c>TPre</c>-generic solver's check
        /// constant-folds to zero cost per specialization; only a runtime-variable case (AMG) pays
        /// for the check.</summary>
        bool IsSpd { get; }

        /// <summary>True iff M is the same fixed linear operator on every iteration (false for a
        /// variable one, e.g. an AMG K-cycle that runs an inner Krylov acceleration). Required by
        /// non-flexible solvers (gmres, biCGStab, idr, tfqmr, gcrodr, and their block variants,
        /// plus cg/minres/minresQLP which also require <see cref="IsSpd"/>); a variable
        /// preconditioner needs the flexible variant (fcg / fgmres). Compile-time literal on the
        /// static preconditioners, same zero-cost constant-folding as <see cref="IsSpd"/>.</summary>
        bool IsConstant { get; }
    }

    /// <summary>
    /// Thin <see cref="IfProxyLinearOperator"/> wrapper over a dense <see cref="fProxyMxN"/>.
    /// Forwards Apply/ApplyT to the existing dense matVec/vecMat dot kernels -- this is what
    /// the concrete <c>Krylov.cg(in fProxyMxN, ...)</c> overloads wrap
    /// internally so the generic <c>Krylov.cg&lt;TOp&gt;</c> loop is the single source of truth.
    /// Readonly: a value copy of this struct is cheap (holds only the fProxyMxN header, no
    /// buffer copy) and safe to pass through `in` parameters in generic constrained calls
    /// without observable mutation.
    /// </summary>
    public readonly struct fProxyDenseOperator : IfProxyLinearOperator
    {
        public readonly fProxyMxN A;

        public fProxyDenseOperator(in fProxyMxN a)
        {
            A = a;
        }

        public int Rows => A.M_Rows;
        public int Cols => A.N_Cols;

        public void Apply(in fProxyN x, ref fProxyN y) => Blas.dot(in A, in x, ref y);

        // Aᵀx via the existing vector*matrix dot: result[j] = sum_i x[i]*A[i,j] == (Aᵀx)[j].
        public void ApplyT(in fProxyN x, ref fProxyN y) => Blas.dot(in x, in A, ref y);

        // Composes (Apply, then a plain dot pass) via Blas.dotSelf.
        public fProxy ApplyDot(in fProxyN x, ref fProxyN y) => Blas.dotSelf(in A, in x, ref y);

        // Block apply: AVrows[i,:] = A · Vrows[i,:] for the first `rows` rows, as ONE matMatDot that
        // streams A once (vs `rows` separate GEMVs). Blas.dotRows(Vrows, A)[r,j] = Σ_i Vrows[r,i]·A[i,j]
        // = (A · Vrows[r])[j] when A is symmetric — the invariant every ApplyBlock caller holds. Only
        // the first `rows` rows of AVrows are written; the rest are preserved (locked-pair data).
        public void ApplyBlock(in fProxyMxN Vrows, ref fProxyMxN AVrows, int rows)
            => Blas.dotRows(in Vrows, in A, ref AVrows, rows);
    }

    /// <summary>
    /// Identity preconditioner: z = r (no-op copy). Gives a <c>TPre</c>-generic solver (e.g.
    /// <see cref="Eigen.lobpcg{TOp,TPre}"/>) a concrete, zero-cost "no preconditioner" instance so
    /// its UNPRECONDITIONED entry point can forward into the single preconditioned generic
    /// implementation with a one-line call instead of duplicating a large loop body -- the same
    /// role <see cref="fProxyDenseOperator"/> plays for dense callers of a <c>TOp</c>-generic
    /// solver. Readonly and stateless: a value copy costs nothing, and the `M.Apply` call compiles
    /// straight to the plain copy below.
    /// </summary>
    public readonly struct fProxyIdentityPreconditioner : IfProxyPreconditioner
    {
        public void Apply(in fProxyN r, ref fProxyN z) => z.CopyFrom(in r);

        public bool IsIdentity => true;
        public bool IsSpd => true;
        public bool IsConstant => true;
    }

    /// <summary>
    /// Symmetric diagonal preconditioner N = diag(d): <see cref="Apply"/> gives z = d .* r.
    /// The natural <see cref="IfProxyPreconditioner"/> for least-squares right (column)
    /// preconditioning via <see cref="fProxyRightPreconditionedOperator{TInner,TPre}"/> (d built
    /// e.g. from <c>Blas.columnNormsSquared</c> + <c>Blas.buildJacobiScale</c>). z must be
    /// distinct from r. Holds the <c>D</c> handle (length n); no buffer copy.
    /// </summary>
    public readonly struct fProxyDiagonalPreconditioner : IfProxyPreconditioner
    {
        public readonly fProxyN D;   // length n: the diagonal of N

        public fProxyDiagonalPreconditioner(in fProxyN d)
        {
            D = d;
        }

        public bool IsIdentity => false;

        /// <summary>Symmetric by construction (diag), and SPD assuming a positive diagonal --
        /// the caller's contract for its Jacobi/column-scale uses. Only ever used on the
        /// least-squares path (<see cref="fProxyRightPreconditionedOperator{TInner,TPre}"/>),
        /// which checks neither flag.</summary>
        public bool IsSpd => true;
        public bool IsConstant => true;

        public void Apply(in fProxyN r, ref fProxyN z)
        {
            if (z.N != r.N || D.N != r.N)
                throw new System.ArgumentException("fProxyDiagonalPreconditioner.Apply: r, z and D lengths must match");
            for (int j = 0; j < r.N; j++) z[j] = D[j] * r[j];
        }
    }

    /// <summary>
    /// Identity linear operator: y = x (an exact bit-copy), Rows == Cols == the size fixed at
    /// construction. Lets B=I callers (e.g. <see cref="Eigen.lobpcg{TOp,TPre}"/>) forward the
    /// standard eigenproblem into the generalized <see cref="Eigen.lobpcg{TOp,TBOp,TPre}"/> core
    /// with B played by this identity, without duplicating a Euclidean-only implementation.
    /// </summary>
    public readonly struct fProxyIdentityOperator : IfProxyLinearOperator
    {
        public readonly int N;

        public fProxyIdentityOperator(int n)
        {
            N = n;
        }

        public int Rows => N;
        public int Cols => N;

        public void Apply(in fProxyN x, ref fProxyN y) => y.CopyFrom(in x);
        public void ApplyT(in fProxyN x, ref fProxyN y) => y.CopyFrom(in x);

        // y == x exactly (an exact bit-copy), so dot(x,y) == dot(x,x) == ||x||^2. Nothing to
        // fuse beyond the copy itself -- this composes (Apply, then Blas.dot), which for the
        // identity operator is already the cheapest possible ApplyDot.
        public fProxy ApplyDot(in fProxyN x, ref fProxyN y)
        {
            Apply(in x, ref y);
            return Blas.dot(x, y);
        }

        // Identity block apply: copy the first `rows` rows (an exact bit-copy, like Apply).
        public void ApplyBlock(in fProxyMxN Vrows, ref fProxyMxN AVrows, int rows)
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
    public readonly struct fProxyColScaledOperator<TInner> : IfProxyLinearOperator
        where TInner : struct, IfProxyLinearOperator
    {
        public readonly TInner Inner;
        public readonly fProxyN D;        // length Inner.Cols: the column scale
        public readonly fProxyN Scratch;  // length Inner.Cols: workspace for Apply (holds d .* x)

        public fProxyColScaledOperator(in TInner inner, in fProxyN d, in fProxyN scratch)
        {
            if (d.N != inner.Cols)
                throw new System.ArgumentException("fProxyColScaledOperator: d.N must equal inner.Cols");
            if (scratch.N != inner.Cols)
                throw new System.ArgumentException("fProxyColScaledOperator: scratch.N must equal inner.Cols");

            Inner = inner;
            D = d;
            Scratch = scratch;
        }

        public int Rows => Inner.Rows;
        public int Cols => Inner.Cols;

        // (A D) x = A (d .* x). Scales into the owned Scratch so the caller's x is untouched.
        public void Apply(in fProxyN x, ref fProxyN y)
        {
            for (int j = 0; j < D.N; j++) Scratch[j] = D[j] * x[j];
            Inner.Apply(in Scratch, ref y);
        }

        // (A D)ᵀ y = D Aᵀ y = d .* (Aᵀ y). Scales y in place after the inner transpose -- no extra
        // scratch needed (y is length Cols, the output).
        public void ApplyT(in fProxyN x, ref fProxyN y)
        {
            Inner.ApplyT(in x, ref y);
            for (int j = 0; j < D.N; j++) y[j] *= D[j];
        }

        // Composes: Apply, then a separate dot pass. This wrapper is RECTANGULAR in its usual
        // callers (lsqr/lsmr column-preconditioning, x length Cols, y length Rows) -- dot(x,y)
        // isn't even well-formed there, so ApplyDot exists only to satisfy the interface; no
        // solver calls it on this operator today (lsqr/lsmr don't use ApplyDot).
        public fProxy ApplyDot(in fProxyN x, ref fProxyN y)
        {
            Apply(in x, ref y);
            return Blas.dot(x, y);
        }

        // No block specialization (this wrapper composes over an arbitrary inner operator): apply per
        // row through the scalar Apply, into two bounded Temp scratch vectors.
        public void ApplyBlock(in fProxyMxN Vrows, ref fProxyMxN AVrows, int rows)
        {
            int cols = Vrows.N_Cols;
            int outCols = Inner.Rows;
            var rin = new fProxyN(cols, Unity.Collections.Allocator.Temp, false);
            var rout = new fProxyN(outCols, Unity.Collections.Allocator.Temp, false);
            for (int i = 0; i < rows; i++)
            {
                for (int c = 0; c < cols; c++) rin[c] = Vrows[i, c];
                Apply(in rin, ref rout);
                for (int c = 0; c < outCols; c++) AVrows[i, c] = rout[c];
            }
            rout.Dispose();
            rin.Dispose();
        }
    }

    /// <summary>
    /// Wraps <typeparamref name="TInner"/> with a SYMMETRIC right (column) preconditioner N
    /// (n×n, n = Inner.Cols), presenting the operator A·N. <see cref="Apply"/> forms A(N·x) via an
    /// owned scratch buffer; <see cref="ApplyT"/> forms N(Aᵀx) -- valid only because N = Nᵀ, which
    /// the caller must guarantee. Solve (A·N) y = b for y, then recover x = N·y. <c>scratch</c>
    /// must not alias any vector passed to Apply/ApplyT. With Tikhonov damping, damping the
    /// preconditioned system penalizes ‖y‖ = ‖N⁻¹x‖ (an N-weighted ridge on x), not ‖x‖ -- a
    /// different regularizer.
    /// </summary>
    public readonly struct fProxyRightPreconditionedOperator<TInner, TPre> : IfProxyLinearOperator
        where TInner : struct, IfProxyLinearOperator
        where TPre : struct, IfProxyPreconditioner
    {
        public readonly TInner Inner;
        public readonly TPre N;           // symmetric right preconditioner
        public readonly fProxyN Scratch;  // length Inner.Cols: workspace (holds N·x / Aᵀx)

        public fProxyRightPreconditionedOperator(in TInner inner, in TPre n, in fProxyN scratch)
        {
            if (scratch.N != inner.Cols)
                throw new System.ArgumentException("fProxyRightPreconditionedOperator: scratch.N must equal inner.Cols");

            Inner = inner;
            N = n;
            Scratch = scratch;
        }

        public int Rows => Inner.Rows;
        public int Cols => Inner.Cols;

        // (A N) x = A (N x). Preconditions into the owned Scratch so the caller's x is untouched.
        public void Apply(in fProxyN x, ref fProxyN y)
        {
            fProxyN s = Scratch;
            N.Apply(in x, ref s);
            Inner.Apply(in s, ref y);
        }

        // (A N)ᵀ x = N Aᵀ x (N symmetric). Inner transpose into the owned Scratch, then N into y.
        public void ApplyT(in fProxyN x, ref fProxyN y)
        {
            fProxyN s = Scratch;
            Inner.ApplyT(in x, ref s);
            N.Apply(in s, ref y);
        }

        // Composes: Apply, then a separate dot pass. This wrapper is RECTANGULAR in its usual
        // callers (lsqr/lsmr right preconditioning, x length Cols, y length Rows) -- dot(x,y)
        // isn't even well-formed there, so ApplyDot exists only to satisfy the interface; no
        // solver calls it on this operator today (lsqr/lsmr don't use ApplyDot).
        public fProxy ApplyDot(in fProxyN x, ref fProxyN y)
        {
            Apply(in x, ref y);
            return Blas.dot(x, y);
        }

        // No block specialization (this wrapper composes over an arbitrary inner operator): apply per
        // row through the scalar Apply, into two bounded Temp scratch vectors.
        public void ApplyBlock(in fProxyMxN Vrows, ref fProxyMxN AVrows, int rows)
        {
            int cols = Vrows.N_Cols;
            int outCols = Inner.Rows;
            var rin = new fProxyN(cols, Unity.Collections.Allocator.Temp, false);
            var rout = new fProxyN(outCols, Unity.Collections.Allocator.Temp, false);
            for (int i = 0; i < rows; i++)
            {
                for (int c = 0; c < cols; c++) rin[c] = Vrows[i, c];
                Apply(in rin, ref rout);
                for (int c = 0; c < outCols; c++) AVrows[i, c] = rout[c];
            }
            rout.Dispose();
            rin.Dispose();
        }
    }

    /// <summary>
    /// Wraps <typeparamref name="TInner"/> with a GENERAL (not necessarily symmetric) right (column)
    /// preconditioner N (n×n, n = Inner.Cols) supplied as an <see cref="IfProxyLinearOperator"/>,
    /// presenting the operator A·N. <see cref="Apply"/> forms A(N·x); <see cref="ApplyT"/> forms
    /// Nᵀ(Aᵀx) = (A·N)ᵀx via N's OWN transpose -- so N may be non-symmetric, unlike
    /// <see cref="fProxyRightPreconditionedOperator{TInner,TPre}"/> which requires N = Nᵀ. Solve
    /// (A·N) y = b for y, then recover x = N·y. This is the shape of the strong least-squares right
    /// preconditioners: N = R⁻¹ from a QR or randomized sketch of A (Blendenpik/LSRN), for which
    /// A·N ≈ Q (orthonormal columns) and lsqr/lsmr converge in a handful of iterations.
    /// <c>scratch</c> (length n) must not alias any vector passed to Apply/ApplyT. With Tikhonov
    /// damping, the penalty acts on ‖y‖ = ‖N⁻¹x‖, an N-weighted ridge on x, not ‖x‖.
    /// </summary>
    public readonly struct fProxyGeneralRightPreconditionedOperator<TInner, TPreN> : IfProxyLinearOperator
        where TInner : struct, IfProxyLinearOperator
        where TPreN : struct, IfProxyLinearOperator
    {
        public readonly TInner Inner;
        public readonly TPreN N;          // general (n×n) right preconditioner, applied as N·x / Nᵀ·x
        public readonly fProxyN Scratch;  // length Inner.Cols: workspace (holds N·x / Aᵀx)

        public fProxyGeneralRightPreconditionedOperator(in TInner inner, in TPreN n, in fProxyN scratch)
        {
            if (n.Rows != inner.Cols || n.Cols != inner.Cols)
                throw new System.ArgumentException("fProxyGeneralRightPreconditionedOperator: N must be square with N.Rows == N.Cols == inner.Cols");
            if (scratch.N != inner.Cols)
                throw new System.ArgumentException("fProxyGeneralRightPreconditionedOperator: scratch.N must equal inner.Cols");

            Inner = inner;
            N = n;
            Scratch = scratch;
        }

        public int Rows => Inner.Rows;
        public int Cols => Inner.Cols;

        // (A N) x = A (N x). Preconditions into the owned Scratch so the caller's x is untouched.
        public void Apply(in fProxyN x, ref fProxyN y)
        {
            fProxyN s = Scratch;
            N.Apply(in x, ref s);
            Inner.Apply(in s, ref y);
        }

        // (A N)ᵀ x = Nᵀ (Aᵀ x). Inner transpose into the owned Scratch, then N's transpose into y --
        // uses ApplyT on N, so N need not be symmetric.
        public void ApplyT(in fProxyN x, ref fProxyN y)
        {
            fProxyN s = Scratch;
            Inner.ApplyT(in x, ref s);
            N.ApplyT(in s, ref y);
        }

        // Composes: Apply, then a separate dot pass. RECTANGULAR in its usual callers (lsqr/lsmr
        // right preconditioning) -- dot(x,y) isn't well-formed there, so ApplyDot exists only to
        // satisfy the interface; no solver calls it on this operator today.
        public fProxy ApplyDot(in fProxyN x, ref fProxyN y)
        {
            Apply(in x, ref y);
            return Blas.dot(x, y);
        }

        // No block specialization (composes over an arbitrary inner operator): apply per row through
        // the scalar Apply, into two bounded Temp scratch vectors.
        public void ApplyBlock(in fProxyMxN Vrows, ref fProxyMxN AVrows, int rows)
        {
            int cols = Vrows.N_Cols;
            int outCols = Inner.Rows;
            var rin = new fProxyN(cols, Unity.Collections.Allocator.Temp, false);
            var rout = new fProxyN(outCols, Unity.Collections.Allocator.Temp, false);
            for (int i = 0; i < rows; i++)
            {
                for (int c = 0; c < cols; c++) rin[c] = Vrows[i, c];
                Apply(in rin, ref rout);
                for (int c = 0; c < outCols; c++) AVrows[i, c] = rout[c];
            }
            rout.Dispose();
            rin.Dispose();
        }
    }
}
