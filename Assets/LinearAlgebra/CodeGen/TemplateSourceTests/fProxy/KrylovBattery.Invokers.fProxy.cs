namespace LinearAlgebra
{
    /// <summary>
    /// Struct-functor entry point the Krylov battery drives a square (single-RHS) solver family
    /// through -- test infrastructure only (lives under TemplateSourceTests, never ships in the
    /// UPM package), mirroring the <see cref="IfProxyLinearOperator"/> struct-functor idiom.
    /// A concrete implementation (e.g. fProxyCgInvoker) owns its own scratch vectors and forwards
    /// Solve/SolveWithPrecond to the matching Krylov entry point.
    /// </summary>
    public interface IfProxySquareSolverInvoker
    {
        MatrixProfile Requires { get; }
        MatrixProfile Forbids { get; }
        PreconditionerKind PrecondKind { get; }   // for the Sparse-only preconditioned check
        fProxy Tol { get; }
        int MaxIter(int n);

        /// Allocate/resize any caller-owned scratch vectors for an n x n system. No-op for
        /// solvers whose production entry point self-allocates from Allocator.Temp (gmres,
        /// fgmres, idr). Called once per gallery matrix, before any Solve* call.
        void Init(ref Arena arena, int n);

        SolveInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator;

        SolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner;
    }

    /// <summary>
    /// Struct-functor entry point the Krylov battery drives a block (multi-RHS) solver family
    /// through -- same role as <see cref="IfProxySquareSolverInvoker"/> for the block family. Test
    /// infrastructure only.
    /// </summary>
    public interface IfProxyBlockSolverInvoker
    {
        MatrixProfile Requires { get; }
        MatrixProfile Forbids { get; }
        PreconditionerKind PrecondKind { get; }
        fProxy Tol { get; }
        int MaxIter(int n);

        /// True for solvers whose Requires includes Nonsymmetric: the dense gallery path must
        /// wrap A in fProxyDenseOperatorGeneral, NOT fProxyDenseOperator (fProxyDenseOperator's
        /// ApplyBlock is symmetric-only and would silently solve A^Tx=b otherwise). BSR entries
        /// are unaffected (fProxyBSROperator.ApplyBlock -> BSR.spMM is general).
        bool NeedsGeneralDenseOperator { get; }

        void Init(ref Arena arena, int n, int s);   // s = block width (RHS count)

        BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator;

        BlockSolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner;

        /// The scalar solver this block family reduces to at s=1 / is compared per-column
        /// against. E.g. a bcg invoker's ScalarCounterpart() returns a fProxyCgInvoker.
        IfProxySquareSolverInvoker ScalarCounterpart();
    }

    /// <summary>
    /// Struct-functor entry point the Krylov battery drives a least-squares solver family
    /// through -- same role as <see cref="IfProxySquareSolverInvoker"/> for the rectangular
    /// (lsqr/lsmr) family. Test infrastructure only. No TPre-generic overload: lsqr/lsmr's only
    /// "preconditioning" is column (Jacobi) scaling, wired as a SEPARATE invoker implementing
    /// this same interface, not a second method here.
    /// </summary>
    public interface IfProxyLstsqSolverInvoker
    {
        MatrixProfile Requires { get; }
        MatrixProfile Forbids { get; }
        fProxy Tol { get; }
        int MaxIter(int rows, int cols);

        void Init(ref Arena arena, int rows, int cols);

        /// damp: 0 for the plain-solve checks; the damped-path check calls this a second time
        /// with damp > 0.
        LstsqInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x, fProxy damp)
            where TOp : struct, IfProxyLinearOperator;
    }

    /// <summary>
    /// Struct-functor entry point the Krylov battery drives a block (multi-RHS) least-squares solver
    /// family through -- same role as <see cref="IfProxyLstsqSolverInvoker"/>, block-shaped like
    /// <see cref="IfProxyBlockSolverInvoker"/>. Test infrastructure only. Spans both rectangular
    /// regimes: OVERDETERMINED (blsmr/bcgls, tall A, min-RESIDUAL) and UNDERDETERMINED (bcraig/bcraigmr,
    /// wide A, min-NORM). No TPre-generic overload (mirrors <see cref="IfProxyLstsqSolverInvoker"/>: neither
    /// production solver has a preconditioned entry point) and no damp parameter (mirrors
    /// <see cref="IfProxyBlockSolverInvoker"/>'s NeedsGeneralDenseOperator-free shape -- blsmr/bcgls have
    /// no Tikhonov-damped production entry point). <see cref="Solve{TOp}"/> takes an explicit maxIter
    /// (rather than deriving it from <see cref="MaxIter"/> internally) so the battery's tiny-maxIter
    /// no-NaN check can force a single iteration without a second invoker configuration.
    /// </summary>
    public interface IfProxyBlockLstsqSolverInvoker
    {
        MatrixProfile Requires { get; }
        MatrixProfile Forbids { get; }
        fProxy Tol { get; }
        int MaxIter(int rows, int cols);

        void Init(ref Arena arena, int rows, int cols, int s);

        BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X, int maxIter)
            where TOp : struct, IfProxyLinearOperator;

        /// The scalar solver this block family reduces to at s=1 / is compared per-column against.
        /// Both blsmr and bcgls compare against plain <see cref="Krylov.lsmr{TOp}"/> -- their shared
        /// least-squares solution, not a distinct per-solver counterpart.
        IfProxyLstsqSolverInvoker ScalarCounterpart();
    }

    /// <summary>
    /// <see cref="IfProxySquareSolverInvoker"/> for <see cref="Krylov.cg{TOp}"/> -- the square-SPD
    /// spike solver for the Krylov battery (see KrylovSquareBatteryTests). Owns the four scratch
    /// vectors cg's zero-alloc primitive needs; r/p/Ap/z are sized to the current gallery matrix by
    /// Init. Named with the fProxy prefix (unlike the spec's illustrative "CgInvoker") because,
    /// unlike the nested per-family TestJob types, this is a top-level type: the float and double
    /// generated copies land in the SAME assembly, and only names containing the fProxy token get
    /// disambiguated by codegen (CS0101 otherwise).
    /// </summary>
    public struct fProxyCgInvoker : IfProxySquareSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyN r, p, Ap, z;

        public MatrixProfile Requires => MatrixProfile.SPD;
        public MatrixProfile Forbids => MatrixProfile.None;
        public PreconditionerKind PrecondKind => PreconditionerKind.SymmetricBSR;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n)
        {
            r = arena.fProxyVec(n);
            p = arena.fProxyVec(n);
            Ap = arena.fProxyVec(n);
            z = arena.fProxyVec(n);
        }

        public SolveInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.cg(in A, in b, ref x, ref r, ref p, ref Ap, MaxIter(A.Rows), Tol);

        public SolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, MaxIter(A.Rows), Tol);
    }

    /// <summary>
    /// <see cref="IfProxySquareSolverInvoker"/> for <see cref="Krylov.fcg{TOp,TPre}"/> -- flexible
    /// CG (varying preconditioner). fcg has no unpreconditioned entry point of its own, so
    /// <see cref="Solve{TOp}"/> forwards with an explicit identity preconditioner.
    /// </summary>
    public struct fProxyFcgInvoker : IfProxySquareSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyN r, p, Ap, z, rOld;

        public MatrixProfile Requires => MatrixProfile.SPD;
        public MatrixProfile Forbids => MatrixProfile.None;
        public PreconditionerKind PrecondKind => PreconditionerKind.SymmetricBSR;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n)
        {
            r = arena.fProxyVec(n);
            p = arena.fProxyVec(n);
            Ap = arena.fProxyVec(n);
            z = arena.fProxyVec(n);
            rOld = arena.fProxyVec(n);
        }

        public SolveInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.fcg(in A, default(fProxyIdentityPreconditioner), in b, ref x, ref r, ref p, ref Ap, ref z, ref rOld, MaxIter(A.Rows), Tol);

        public SolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.fcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, ref rOld, MaxIter(A.Rows), Tol);
    }

    /// <summary>
    /// <see cref="IfProxySquareSolverInvoker"/> for <see cref="Krylov.minres{TOp,TPre}"/> -- symmetric
    /// (possibly indefinite) systems. Forbids Nonsymmetric rather than requiring a single KIND flag,
    /// since MatrixProfile's KIND group (SPD/SymmetricIndefinite/Nonsymmetric) is mutually exclusive
    /// per matrix and MINRES accepts either symmetric KIND. Also forbids IllConditioned: this
    /// unregularized Lanczos recurrence has no guard against a near-zero Givens denominator, which
    /// the gallery's one IllConditioned symmetric-indefinite entry (Rosser, clustered near-degenerate
    /// spectrum) trips into unbounded divergence, not mere slow convergence -- see the folder DEVLOG.
    /// </summary>
    public struct fProxyMinresInvoker : IfProxySquareSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyN y, r1, r2, v, w, w1, w2, z;

        public MatrixProfile Requires => MatrixProfile.Square;
        public MatrixProfile Forbids => MatrixProfile.Nonsymmetric | MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.SymmetricBSR;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n)
        {
            y = arena.fProxyVec(n);
            r1 = arena.fProxyVec(n);
            r2 = arena.fProxyVec(n);
            v = arena.fProxyVec(n);
            w = arena.fProxyVec(n);
            w1 = arena.fProxyVec(n);
            w2 = arena.fProxyVec(n);
            z = arena.fProxyVec(n);
        }

        public SolveInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, MaxIter(A.Rows), Tol);

        public SolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.minres(in A, in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, MaxIter(A.Rows), Tol);
    }

    /// <summary>
    /// <see cref="IfProxySquareSolverInvoker"/> for <see cref="Krylov.minresQLP{TOp,TPre}"/> --
    /// structural sibling of <see cref="fProxyMinresInvoker"/> (same Requires/Forbids, including the
    /// IllConditioned exclusion): the xnorm/Acond safety guards stop it from diverging the way plain
    /// MINRES does on the gallery's clustered-spectrum Rosser entry, but its own relres criterion
    /// still exits well short of this battery's fresh-residual bound there -- see the folder DEVLOG.
    /// Genuinely singular/rank-deficient coverage (this solver's actual purpose beyond plain MINRES)
    /// belongs in a dedicated special-case file with its own Singular-tagged matrix, not this battery.
    /// <see cref="Tol"/> (reported to the check, drives its bound) and the tolerance actually handed
    /// to the solver are deliberately different: minresQLP's own stopping test (rnorm / (Anorm*xnorm +
    /// beta1)) is normalized by the solution/matrix scale, looser than this battery's raw
    /// ‖b-Ax‖/‖b‖ check by roughly that same scale factor -- so the internal target is driven well
    /// past <see cref="Tol"/> to land the fresh residual inside the check's bound. See DEVLOG.
    /// </summary>
    public struct fProxyMinresQLPInvoker : IfProxySquareSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyN v, r1, r2, r3, w, wl, wl2, xl2, t1;

        public MatrixProfile Requires => MatrixProfile.Square;
        public MatrixProfile Forbids => MatrixProfile.Nonsymmetric | MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.SymmetricBSR;
        public fProxy Tol => TolValue;
        fProxy SolveTol => TolValue * (fProxy)0.02;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n)
        {
            v = arena.fProxyVec(n);
            r1 = arena.fProxyVec(n);
            r2 = arena.fProxyVec(n);
            r3 = arena.fProxyVec(n);
            w = arena.fProxyVec(n);
            wl = arena.fProxyVec(n);
            wl2 = arena.fProxyVec(n);
            xl2 = arena.fProxyVec(n);
            t1 = arena.fProxyVec(n);
        }

        public SolveInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.minresQLP(in A, in b, ref x, ref v, ref r1, ref r2, ref r3, ref w, ref wl, ref wl2, ref xl2, ref t1, MaxIter(A.Rows), SolveTol);

        public SolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.minresQLP(in A, in M, in b, ref x, ref v, ref r1, ref r2, ref r3, ref w, ref wl, ref wl2, ref xl2, ref t1, MaxIter(A.Rows), SolveTol);
    }

    /// <summary>
    /// <see cref="IfProxySquareSolverInvoker"/> for <see cref="Krylov.biCGStab{TOp,TPre}"/> --
    /// general (nonsymmetric) square systems; usable on any square matrix kind, so Requires is just
    /// Square (same breadth as <see cref="fProxyIdrInvoker"/>). Forbids IllConditioned: the gallery's
    /// one IllConditioned symmetric-indefinite entry (Rosser, clustered near-degenerate spectrum)
    /// trips this short two-term recurrence into unbounded divergence rather than mere slow
    /// convergence, while its WellConditioned/other-IllConditioned entries (Hilbert4, Pascal5,
    /// Grcar8) all converge cleanly -- see the folder DEVLOG.
    /// </summary>
    public struct fProxyBiCGStabInvoker : IfProxySquareSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyN r, rHat0, p, v, t, pHat, sHat;

        public MatrixProfile Requires => MatrixProfile.Square;
        public MatrixProfile Forbids => MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.NonsymmetricBSR;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n)
        {
            r = arena.fProxyVec(n);
            rHat0 = arena.fProxyVec(n);
            p = arena.fProxyVec(n);
            v = arena.fProxyVec(n);
            t = arena.fProxyVec(n);
            pHat = arena.fProxyVec(n);
            sHat = arena.fProxyVec(n);
        }

        public SolveInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, MaxIter(A.Rows), Tol);

        public SolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.biCGStab(in A, in M, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, ref pHat, ref sHat, MaxIter(A.Rows), Tol);
    }

    /// <summary>
    /// <see cref="IfProxySquareSolverInvoker"/> for <see cref="Krylov.gmres{TOp,TPre}"/> -- restarted
    /// GMRES(m); usable on any square matrix kind (Requires = Square only). Self-allocates its
    /// Arnoldi workspace from Allocator.Temp, so <see cref="Init"/> is a no-op. Forbids IllConditioned:
    /// the Hessenberg back-substitution has no guard against a near-zero pivot, which the gallery's
    /// clustered-near-degenerate-spectrum entry (Rosser) trips into an unbounded y -- see DEVLOG.
    /// </summary>
    public struct fProxyGmresInvoker : IfProxySquareSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;
        public int Restart;

        public MatrixProfile Requires => MatrixProfile.Square;
        public MatrixProfile Forbids => MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.NonsymmetricBSR;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n) { }

        public SolveInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.gmres(in A, in b, ref x, Restart, MaxIter(A.Rows), Tol);

        public SolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.gmres(in A, in M, in b, ref x, Restart, MaxIter(A.Rows), Tol);
    }

    /// <summary>
    /// <see cref="IfProxySquareSolverInvoker"/> for <see cref="Krylov.fgmres{TOp,TPre}"/> -- restarted
    /// flexible GMRES(m) (per-step-varying preconditioner). Same profile and no-op <see cref="Init"/>
    /// as <see cref="fProxyGmresInvoker"/> (including the IllConditioned exclusion -- it shares
    /// gmres's unguarded Hessenberg back-substitution); SolveWithPrecond's TPre slots in cleanly
    /// since a single battery call only ever passes one (possibly internally-iterative)
    /// preconditioner instance.
    /// </summary>
    public struct fProxyFgmresInvoker : IfProxySquareSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;
        public int Restart;

        public MatrixProfile Requires => MatrixProfile.Square;
        public MatrixProfile Forbids => MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.NonsymmetricBSR;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n) { }

        public SolveInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.fgmres(in A, in b, ref x, Restart, MaxIter(A.Rows), Tol);

        public SolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.fgmres(in A, in M, in b, ref x, Restart, MaxIter(A.Rows), Tol);
    }

    /// <summary>
    /// <see cref="IfProxySquareSolverInvoker"/> for <see cref="Krylov.gcrodr{TOp,TPre}"/> -- restarted
    /// GMRES(m) with a recycled harmonic-Ritz subspace; usable on any square matrix kind (Requires =
    /// Square only). Self-allocates its Arnoldi + recycle workspace from Allocator.Temp, so
    /// <see cref="Init"/> is a no-op. Forbids IllConditioned: shares gmres's Hessenberg
    /// back-substitution shape on the gallery's clustered-near-degenerate-spectrum entry (Rosser) --
    /// gcrodr's own pivot guard turns that into a clean Breakdown rather than gmres's unbounded
    /// divergence, but RunStandardChecks treats Breakdown as a failing status like any other solver
    /// here, so the exclusion stays for consistency with gmres/biCGStab/idr (task #53).
    /// </summary>
    public struct fProxyGcrodrInvoker : IfProxySquareSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;
        public int Restart;
        public int Recycle;

        public MatrixProfile Requires => MatrixProfile.Square;
        public MatrixProfile Forbids => MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.NonsymmetricBSR;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n) { }

        public SolveInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.gcrodr(in A, in b, ref x, Restart, Recycle, MaxIter(A.Rows), Tol);

        public SolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.gcrodr(in A, in M, in b, ref x, Restart, Recycle, MaxIter(A.Rows), Tol);
    }

    /// <summary>
    /// <see cref="IfProxySquareSolverInvoker"/> for <see cref="Krylov.idr{TOp,TPre}"/> -- IDR(s);
    /// usable on any square matrix kind (Requires = Square only). Self-allocates its shadow-space
    /// workspace from Allocator.Temp, so <see cref="Init"/> is a no-op. Forbids IllConditioned: the
    /// s x s in-sweep system has no guard against a near-singular pivot beyond a zero/NaN check,
    /// which the gallery's clustered-near-degenerate-spectrum entry (Rosser) trips into unbounded
    /// divergence -- see DEVLOG.
    /// </summary>
    public struct fProxyIdrInvoker : IfProxySquareSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;
        public int S;
        public uint Seed;

        public MatrixProfile Requires => MatrixProfile.Square;
        public MatrixProfile Forbids => MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.NonsymmetricBSR;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n) { }

        public SolveInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.idr(in A, in b, ref x, S, MaxIter(A.Rows), Tol, Seed);

        public SolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.idr(in A, in M, in b, ref x, S, MaxIter(A.Rows), Tol, Seed);
    }

    /// <summary>
    /// <see cref="IfProxySquareSolverInvoker"/> for <see cref="Krylov.tfqmr{TOp,TPre}"/> -- TFQMR
    /// (transpose-free QMR) for general (nonsymmetric) square systems; usable on any square matrix
    /// kind, so Requires is just Square (same breadth as <see cref="fProxyBiCGStabInvoker"/>). Owns
    /// the seven scratch vectors the zero-alloc primitive needs (uHat unused under the identity
    /// fold of the plain path). Forbids IllConditioned: like every transpose-free nonsymmetric
    /// method here, the gallery's clustered-near-degenerate-spectrum entry (Rosser) trips it past a
    /// near-breakdown rather than mere slow convergence -- same rationale as biCGStab/gmres/idr, see
    /// DEVLOG. MaxIterMul is in HALF-steps (~one A-apply each), so ~40 matches biCGStab's 20
    /// two-matvec-per-pass budget.
    /// </summary>
    public struct fProxyTfqmrInvoker : IfProxySquareSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyN rHat0, u, w, v, au, d, uHat;

        public MatrixProfile Requires => MatrixProfile.Square;
        public MatrixProfile Forbids => MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.NonsymmetricBSR;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n)
        {
            rHat0 = arena.fProxyVec(n);
            u = arena.fProxyVec(n);
            w = arena.fProxyVec(n);
            v = arena.fProxyVec(n);
            au = arena.fProxyVec(n);
            d = arena.fProxyVec(n);
            uHat = arena.fProxyVec(n);
        }

        public SolveInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.tfqmr(in A, in b, ref x, ref rHat0, ref u, ref w, ref v, ref au, ref d, MaxIter(A.Rows), Tol);

        public SolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.tfqmr(in A, in M, in b, ref x, ref rHat0, ref u, ref w, ref v, ref au, ref d, ref uHat, MaxIter(A.Rows), Tol);
    }

    /// <summary>
    /// <see cref="IfProxyBlockSolverInvoker"/> for <see cref="Krylov.bcg{TOp, TPre}"/> -- ridge-
    /// regularized block CG for an SPD A and s simultaneous right-hand sides. Owns the four s x n
    /// scratch blocks bcg's zero-alloc primitive needs (Z unused under the identity preconditioner);
    /// R/P/Q/Z are sized to the current gallery matrix and block width by Init.
    /// <see cref="ScalarCounterpart"/> is a plain <see cref="fProxyCgInvoker"/> -- bcg solves the
    /// same SPD system CG does, just s columns at once. Forbids IllConditioned: the ridge-regularized
    /// s x s solve only guards RANK-deficient search blocks, not the s == n corner (the battery's
    /// block width equals the gallery's smallest matrix dimension, Hilbert4) combined with extreme
    /// conditioning, which diverges outright rather than converging slowly (bcgrq/bfbcg's rank-
    /// revealing-LQ search basis does not share this failure mode and keep IllConditioned).
    /// </summary>
    public struct fProxyBcgInvoker : IfProxyBlockSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyMxN R, P, Q, Z;

        public MatrixProfile Requires => MatrixProfile.SPD;
        public MatrixProfile Forbids => MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.SymmetricBSR;
        public bool NeedsGeneralDenseOperator => false;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n, int s)
        {
            R = arena.fProxyMat(s, n);
            P = arena.fProxyMat(s, n);
            Q = arena.fProxyMat(s, n);
            Z = arena.fProxyMat(s, n);
        }

        public BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.bcg(in A, in B, ref X, ref R, ref P, ref Q, MaxIter(A.Rows), Tol);

        public BlockSolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.bcg(in A, in M, in B, ref X, ref R, ref P, ref Q, ref Z, MaxIter(A.Rows), Tol);

        public IfProxySquareSolverInvoker ScalarCounterpart() => new fProxyCgInvoker { TolValue = TolValue, MaxIterMul = MaxIterMul };
    }

    /// <summary>
    /// <see cref="IfProxyBlockSolverInvoker"/> for <see cref="Krylov.bcgrq{TOp, TPre}"/> -- rank-
    /// revealing-LQ block CG (deflates dependent search directions instead of ridge-patching) for an
    /// SPD A and s simultaneous right-hand sides. Owns the five s x n scratch blocks bcgrq's zero-
    /// alloc primitive needs (Z unused under the identity preconditioner).
    /// <see cref="ScalarCounterpart"/> is <see cref="fProxyCgInvoker"/> -- same SPD system as bcg.
    /// </summary>
    public struct fProxyBcgrqInvoker : IfProxyBlockSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyMxN R, P, AP, Pa, Z;

        public MatrixProfile Requires => MatrixProfile.SPD;
        public MatrixProfile Forbids => MatrixProfile.None;
        public PreconditionerKind PrecondKind => PreconditionerKind.SymmetricBSR;
        public bool NeedsGeneralDenseOperator => false;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n, int s)
        {
            R = arena.fProxyMat(s, n);
            P = arena.fProxyMat(s, n);
            AP = arena.fProxyMat(s, n);
            Pa = arena.fProxyMat(s, n);
            Z = arena.fProxyMat(s, n);
        }

        public BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.bcgrq(in A, in B, ref X, ref R, ref P, ref AP, ref Pa, MaxIter(A.Rows), Tol);

        public BlockSolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.bcgrq(in A, in M, in B, ref X, ref R, ref P, ref AP, ref Pa, ref Z, MaxIter(A.Rows), Tol);

        public IfProxySquareSolverInvoker ScalarCounterpart() => new fProxyCgInvoker { TolValue = TolValue, MaxIterMul = MaxIterMul };
    }

    /// <summary>
    /// <see cref="IfProxyBlockSolverInvoker"/> for <see cref="Krylov.bfbcg{TOp, TPre}"/> -- breakdown-
    /// free block CG (Ji &amp; Li 2017; every iteration re-orthonormalizes the search block via a
    /// rank-revealing LQ) for an SPD A and s simultaneous right-hand sides. Same scratch shape as
    /// <see cref="fProxyBcgrqInvoker"/> (R/P/AP/Pa/Z, Z unused under the identity preconditioner).
    /// <see cref="ScalarCounterpart"/> is <see cref="fProxyCgInvoker"/> -- same SPD system as bcg.
    /// </summary>
    public struct fProxyBfbcgInvoker : IfProxyBlockSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyMxN R, P, AP, Pa, Z;

        public MatrixProfile Requires => MatrixProfile.SPD;
        public MatrixProfile Forbids => MatrixProfile.None;
        public PreconditionerKind PrecondKind => PreconditionerKind.SymmetricBSR;
        public bool NeedsGeneralDenseOperator => false;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n, int s)
        {
            R = arena.fProxyMat(s, n);
            P = arena.fProxyMat(s, n);
            AP = arena.fProxyMat(s, n);
            Pa = arena.fProxyMat(s, n);
            Z = arena.fProxyMat(s, n);
        }

        public BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.bfbcg(in A, in B, ref X, ref R, ref P, ref AP, ref Pa, MaxIter(A.Rows), Tol);

        public BlockSolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.bfbcg(in A, in M, in B, ref X, ref R, ref P, ref AP, ref Pa, ref Z, MaxIter(A.Rows), Tol);

        public IfProxySquareSolverInvoker ScalarCounterpart() => new fProxyCgInvoker { TolValue = TolValue, MaxIterMul = MaxIterMul };
    }

    /// <summary>
    /// <see cref="IfProxyBlockSolverInvoker"/> for <see cref="Krylov.bminres{TOp, TPre}"/> -- block
    /// MINRES for a SYMMETRIC (possibly indefinite) A and s simultaneous right-hand sides. Same
    /// Requires/Forbids as <see cref="fProxyMinresInvoker"/> (Forbids Nonsymmetric and IllConditioned
    /// -- the gallery's clustered-spectrum entry, see that invoker's own doc). <see cref="PrecondKind"/>
    /// is SymmetricBSR: bminres's preconditioned path is the r-space block-Lanczos recurrence over
    /// the unpreconditioned residual blocks, so the battery's preconditioned-convergence check (#5)
    /// exercises it with a real BlockJacobi M.
    /// </summary>
    public struct fProxyBminresInvoker : IfProxyBlockSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyMxN Vprev, Vcur, Wk, W, W1, W2, Z;

        public MatrixProfile Requires => MatrixProfile.Square;
        public MatrixProfile Forbids => MatrixProfile.Nonsymmetric | MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.SymmetricBSR;
        public bool NeedsGeneralDenseOperator => false;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n, int s)
        {
            Vprev = arena.fProxyMat(s, n);
            Vcur = arena.fProxyMat(s, n);
            Wk = arena.fProxyMat(s, n);
            W = arena.fProxyMat(s, n);
            W1 = arena.fProxyMat(s, n);
            W2 = arena.fProxyMat(s, n);
            Z = arena.fProxyMat(s, n);
        }

        public BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.bminres(in A, in B, ref X, ref Vprev, ref Vcur, ref Wk, ref W, ref W1, ref W2, MaxIter(A.Rows), Tol);

        public BlockSolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.bminres(in A, in M, in B, ref X, ref Vprev, ref Vcur, ref Wk, ref W, ref W1, ref W2, ref Z, MaxIter(A.Rows), Tol);

        public IfProxySquareSolverInvoker ScalarCounterpart() => new fProxyMinresInvoker { TolValue = TolValue, MaxIterMul = MaxIterMul };
    }

    /// <summary>
    /// <see cref="IfProxyBlockSolverInvoker"/> for <see cref="Krylov.bbiCGStab{TOp, TPre}"/> -- block
    /// BiCGSTAB for a NON-symmetric (general) square A and s simultaneous right-hand sides.
    /// <see cref="NeedsGeneralDenseOperator"/> is true: the dense gallery path must wrap A in
    /// <see cref="fProxyDenseOperatorGeneral"/>, not <see cref="fProxyDenseOperator"/> (SS4's
    /// symmetric-only ApplyBlock landmine). Owns the seven s x n scratch blocks the zero-alloc
    /// primitive needs (Phat/Shat unused under the identity preconditioner).
    /// </summary>
    public struct fProxyBbiCGStabInvoker : IfProxyBlockSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyMxN R, Rhat0, P, V, T, Phat, Shat;

        public MatrixProfile Requires => MatrixProfile.Nonsymmetric;
        public MatrixProfile Forbids => MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.NonsymmetricBSR;
        public bool NeedsGeneralDenseOperator => true;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n, int s)
        {
            R = arena.fProxyMat(s, n);
            Rhat0 = arena.fProxyMat(s, n);
            P = arena.fProxyMat(s, n);
            V = arena.fProxyMat(s, n);
            T = arena.fProxyMat(s, n);
            Phat = arena.fProxyMat(s, n);
            Shat = arena.fProxyMat(s, n);
        }

        public BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.bbiCGStab(in A, in B, ref X, ref R, ref Rhat0, ref P, ref V, ref T, MaxIter(A.Rows), Tol);

        public BlockSolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.bbiCGStab(in A, in M, in B, ref X, ref R, ref Rhat0, ref P, ref V, ref T, ref Phat, ref Shat, MaxIter(A.Rows), Tol);

        public IfProxySquareSolverInvoker ScalarCounterpart() => new fProxyBiCGStabInvoker { TolValue = TolValue, MaxIterMul = MaxIterMul };
    }

    /// <summary>
    /// <see cref="IfProxyBlockSolverInvoker"/> for <see cref="Krylov.bidr{TOp, TPre}"/> -- true block
    /// IDR(s) for a NON-symmetric (general) square A and s simultaneous right-hand sides.
    /// <see cref="NeedsGeneralDenseOperator"/> is true (same landmine as <see cref="fProxyBbiCGStabInvoker"/>).
    /// Self-allocates its whole shadow-space/history workspace from Allocator.Temp, so <see
    /// cref="Init"/> is a no-op (mirrors <see cref="fProxyIdrInvoker"/>). <see cref="S"/>/<see
    /// cref="Seed"/> are the IDR shadow-space DEPTH and its deterministic RNG seed -- unrelated to the
    /// CheckFlags block width (RHS count) the battery drives this invoker through.
    /// </summary>
    public struct fProxyBidrInvoker : IfProxyBlockSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;
        public int S;
        public uint Seed;

        public MatrixProfile Requires => MatrixProfile.Nonsymmetric;
        public MatrixProfile Forbids => MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.NonsymmetricBSR;
        public bool NeedsGeneralDenseOperator => true;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n, int s) { }

        public BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.bidr(in A, in B, ref X, S, MaxIter(A.Rows), Tol, Seed);

        public BlockSolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.bidr(in A, in M, in B, ref X, S, MaxIter(A.Rows), Tol, Seed);

        public IfProxySquareSolverInvoker ScalarCounterpart() => new fProxyIdrInvoker { TolValue = TolValue, MaxIterMul = MaxIterMul, S = S, Seed = Seed };
    }

    /// <summary>
    /// <see cref="IfProxyBlockSolverInvoker"/> for <see cref="Krylov.bgmres{TOp, TPre}"/> -- restarted
    /// block GMRES(m) for a NON-symmetric (general) square A and s simultaneous right-hand sides.
    /// <see cref="NeedsGeneralDenseOperator"/> is true (same landmine as <see cref="fProxyBbiCGStabInvoker"/>).
    /// Self-allocates its whole Arnoldi workspace from Allocator.Temp, so <see cref="Init"/> is a no-op
    /// (mirrors <see cref="fProxyGmresInvoker"/>).
    /// </summary>
    public struct fProxyBgmresInvoker : IfProxyBlockSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;
        public int Restart;

        public MatrixProfile Requires => MatrixProfile.Nonsymmetric;
        public MatrixProfile Forbids => MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.NonsymmetricBSR;
        public bool NeedsGeneralDenseOperator => true;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n, int s) { }

        public BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.bgmres(in A, in B, ref X, Restart, MaxIter(A.Rows), Tol);

        public BlockSolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.bgmres(in A, in M, in B, ref X, Restart, MaxIter(A.Rows), Tol);

        public IfProxySquareSolverInvoker ScalarCounterpart() => new fProxyGmresInvoker { TolValue = TolValue, MaxIterMul = MaxIterMul, Restart = Restart };
    }

    /// <summary>
    /// <see cref="IfProxyBlockSolverInvoker"/> for <see cref="Krylov.bgcrodr{TOp, TPre}"/> -- block
    /// GCRO-DR (restarted block GMRES(m) with a recycled harmonic-Ritz subspace) for a NON-symmetric
    /// (general) square A and s simultaneous right-hand sides. Same profile and no-op <see cref="Init"/>
    /// as <see cref="fProxyBgmresInvoker"/> (including <see cref="NeedsGeneralDenseOperator"/> and the
    /// IllConditioned exclusion -- it shares bgmres's block Arnoldi/Hessenberg machinery, same
    /// rationale). <see cref="ScalarCounterpart"/> is <see cref="fProxyGcrodrInvoker"/> (same recycled-
    /// subspace family, not plain gmres).
    /// </summary>
    public struct fProxyBgcrodrInvoker : IfProxyBlockSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;
        public int Restart;
        public int Recycle;

        public MatrixProfile Requires => MatrixProfile.Nonsymmetric;
        public MatrixProfile Forbids => MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.NonsymmetricBSR;
        public bool NeedsGeneralDenseOperator => true;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n, int s) { }

        public BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.bgcrodr(in A, in B, ref X, Restart, Recycle, MaxIter(A.Rows), Tol);

        public BlockSolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.bgcrodr(in A, in M, in B, ref X, Restart, Recycle, MaxIter(A.Rows), Tol);

        public IfProxySquareSolverInvoker ScalarCounterpart() => new fProxyGcrodrInvoker { TolValue = TolValue, MaxIterMul = MaxIterMul, Restart = Restart, Recycle = Recycle };
    }

    /// <summary>
    /// <see cref="IfProxyBlockSolverInvoker"/> for <see cref="Krylov.bfgmres{TOp, TPre}"/> -- restarted
    /// block FLEXIBLE GMRES(m) (per-step-varying preconditioner) for a NON-symmetric (general) square A
    /// and s simultaneous right-hand sides. Same profile and no-op <see cref="Init"/> as
    /// <see cref="fProxyBgmresInvoker"/> (including <see cref="NeedsGeneralDenseOperator"/> and the
    /// IllConditioned exclusion -- it shares bgmres's block Arnoldi/Hessenberg machinery and has no
    /// monotone block-advantage bound either, same rationale as bgmres's own KrylovBlockBatteryTests
    /// case); SolveWithPrecond's TPre slots in cleanly since a single battery call only ever passes one
    /// (possibly internally-iterative) preconditioner instance.
    /// </summary>
    public struct fProxyBfgmresInvoker : IfProxyBlockSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;
        public int Restart;

        public MatrixProfile Requires => MatrixProfile.Nonsymmetric;
        public MatrixProfile Forbids => MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.NonsymmetricBSR;
        public bool NeedsGeneralDenseOperator => true;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n, int s) { }

        public BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.bfgmres(in A, in B, ref X, Restart, MaxIter(A.Rows), Tol);

        public BlockSolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.bfgmres(in A, in M, in B, ref X, Restart, MaxIter(A.Rows), Tol);

        public IfProxySquareSolverInvoker ScalarCounterpart() => new fProxyFgmresInvoker { TolValue = TolValue, MaxIterMul = MaxIterMul, Restart = Restart };
    }

    /// <summary>
    /// <see cref="IfProxyBlockSolverInvoker"/> for <see cref="Krylov.btfqmr{TOp, TPre}"/> -- PSEUDO-block
    /// TFQMR (s independent scalar-TFQMR recurrences batched over one shared ApplyBlock per half-step,
    /// NOT a subspace-mixing true block method -- see OP/DEVLOG.md "Krylov.Block.TFQMR") for a
    /// NON-symmetric (general) square A and s simultaneous right-hand sides.
    /// <see cref="NeedsGeneralDenseOperator"/> is true (same landmine as <see cref="fProxyBbiCGStabInvoker"/>).
    /// Owns the seven s x n scratch blocks the zero-alloc primitive needs (UHat unused under the identity
    /// preconditioner). MaxIterMul is in HALF-steps per row (mirrors <see cref="fProxyTfqmrInvoker"/>).
    /// </summary>
    public struct fProxyBtfqmrInvoker : IfProxyBlockSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyMxN Rhat0, U, W, V, AU, D, UHat;

        public MatrixProfile Requires => MatrixProfile.Nonsymmetric;
        public MatrixProfile Forbids => MatrixProfile.IllConditioned;
        public PreconditionerKind PrecondKind => PreconditionerKind.NonsymmetricBSR;
        public bool NeedsGeneralDenseOperator => true;
        public fProxy Tol => TolValue;
        public int MaxIter(int n) => MaxIterMul * n;

        public void Init(ref Arena arena, int n, int s)
        {
            Rhat0 = arena.fProxyMat(s, n);
            U = arena.fProxyMat(s, n);
            W = arena.fProxyMat(s, n);
            V = arena.fProxyMat(s, n);
            AU = arena.fProxyMat(s, n);
            D = arena.fProxyMat(s, n);
            UHat = arena.fProxyMat(s, n);
        }

        public BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.btfqmr(in A, in B, ref X, ref Rhat0, ref U, ref W, ref V, ref AU, ref D, MaxIter(A.Rows), Tol);

        public BlockSolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => Krylov.btfqmr(in A, in M, in B, ref X, ref Rhat0, ref U, ref W, ref V, ref AU, ref D, ref UHat, MaxIter(A.Rows), Tol);

        public IfProxySquareSolverInvoker ScalarCounterpart() => new fProxyTfqmrInvoker { TolValue = TolValue, MaxIterMul = MaxIterMul };
    }

    /// <summary>
    /// <see cref="IfProxyLstsqSolverInvoker"/> for <see cref="Krylov.lsqr{TOp}"/> -- least-squares
    /// (Paige-Saunders) solver for OVERDETERMINED (Rows &gt;= Cols, full column rank) systems:
    /// minimizes ‖Ax-b‖. Owns the five scratch vectors lsqr's zero-alloc primitive needs. Forbids
    /// RankDeficient: the battery's min-residual oracle (normal-equations optimality vs. a direct
    /// QR solve) assumes the unique least-squares solution a rank-deficient A does not have.
    /// </summary>
    public struct fProxyLsqrInvoker : IfProxyLstsqSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyN u, v, w, tmpM, tmpN;

        public MatrixProfile Requires => MatrixProfile.Overdetermined;
        public MatrixProfile Forbids => MatrixProfile.RankDeficient;
        public fProxy Tol => TolValue;
        public int MaxIter(int rows, int cols) => MaxIterMul * (rows < cols ? rows : cols);

        public void Init(ref Arena arena, int rows, int cols)
        {
            u = arena.fProxyVec(rows);
            v = arena.fProxyVec(cols);
            w = arena.fProxyVec(cols);
            tmpM = arena.fProxyVec(rows);
            tmpN = arena.fProxyVec(cols);
        }

        public LstsqInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x, fProxy damp)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, MaxIter(A.Rows, A.Cols), Tol, damp);
    }

    /// <summary>
    /// <see cref="IfProxyLstsqSolverInvoker"/> for <see cref="Krylov.lsmr{TOp}"/> -- LSMR
    /// (Fong-Saunders; monotone normal-equation residual) solver for OVERDETERMINED (Rows &gt;=
    /// Cols, full column rank) systems: minimizes ‖Ax-b‖. Owns the six scratch vectors lsmr's
    /// zero-alloc primitive needs (one more than <see cref="fProxyLsqrInvoker"/> -- the
    /// MINRES-folded direction hbar). Same Requires/Forbids as fProxyLsqrInvoker.
    /// </summary>
    public struct fProxyLsmrInvoker : IfProxyLstsqSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyN u, v, h, hbar, tmpM, tmpN;

        public MatrixProfile Requires => MatrixProfile.Overdetermined;
        public MatrixProfile Forbids => MatrixProfile.RankDeficient;
        public fProxy Tol => TolValue;
        public int MaxIter(int rows, int cols) => MaxIterMul * (rows < cols ? rows : cols);

        public void Init(ref Arena arena, int rows, int cols)
        {
            u = arena.fProxyVec(rows);
            v = arena.fProxyVec(cols);
            h = arena.fProxyVec(cols);
            hbar = arena.fProxyVec(cols);
            tmpM = arena.fProxyVec(rows);
            tmpN = arena.fProxyVec(cols);
        }

        public LstsqInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x, fProxy damp)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.lsmr(in A, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, MaxIter(A.Rows, A.Cols), Tol, damp);
    }

    /// <summary>
    /// <see cref="IfProxyLstsqSolverInvoker"/> for <see cref="Krylov.craig{TOp}"/> -- least-NORM
    /// (Craig 1955 / Paige-Saunders) solver for UNDERDETERMINED (Rows &lt;= Cols, full row rank)
    /// CONSISTENT systems: among all x with Ax=b, finds the minimum-‖x‖ one. Owns the four
    /// scratch vectors craig's zero-alloc primitive needs. Forbids RankDeficient (no
    /// Underdetermined gallery entry currently carries it; kept for symmetry with
    /// <see cref="fProxyLsqrInvoker"/>).
    /// </summary>
    public struct fProxyCraigInvoker : IfProxyLstsqSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyN u, v, tmpM, tmpN;

        public MatrixProfile Requires => MatrixProfile.Underdetermined;
        public MatrixProfile Forbids => MatrixProfile.RankDeficient;
        public fProxy Tol => TolValue;
        public int MaxIter(int rows, int cols) => MaxIterMul * (rows < cols ? rows : cols);

        public void Init(ref Arena arena, int rows, int cols)
        {
            u = arena.fProxyVec(rows);
            v = arena.fProxyVec(cols);
            tmpM = arena.fProxyVec(rows);
            tmpN = arena.fProxyVec(cols);
        }

        /// damp is ignored: craig has no Tikhonov-damped production entry point (a consistent
        /// min-norm system has no residual/norm trade-off to regularize). The battery only
        /// exercises the damped-path check on Overdetermined invokers.
        public LstsqInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x, fProxy damp)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.craig(in A, in b, ref x, ref u, ref v, ref tmpM, ref tmpN, MaxIter(A.Rows, A.Cols), Tol);
    }

    /// <summary>
    /// <see cref="IfProxyLstsqSolverInvoker"/> for <see cref="Krylov.lnlq{TOp}"/> -- least-NORM
    /// (Estrin-Orban-Saunders) solver for UNDERDETERMINED (Rows &lt;= Cols, full row rank) CONSISTENT
    /// systems. Returns the SAME minimum-‖x‖ iterate as <see cref="fProxyCraigInvoker"/> (LNLQ's
    /// transferred CRAIG point), folding the same Golub-Kahan bidiagonalization through an LQ
    /// factorization. Owns the four scratch vectors lnlq's zero-alloc primitive needs; same
    /// Requires/Forbids and tol regime as craig (Golub-Kahan cond(A) conditioning, not CGNE's κ²).
    /// lnlq returns <see cref="LnlqInfo"/>; only its status is consumed here (check #11 recomputes the
    /// residual and compares against LQ.minNormSolve on its own), so the adapter maps status through
    /// and leaves the least-squares-only Arnorm NaN.
    /// </summary>
    public struct fProxyLnlqInvoker : IfProxyLstsqSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyN u, v, tmpM, tmpN;

        public MatrixProfile Requires => MatrixProfile.Underdetermined;
        public MatrixProfile Forbids => MatrixProfile.RankDeficient;
        public fProxy Tol => TolValue;
        public int MaxIter(int rows, int cols) => MaxIterMul * (rows < cols ? rows : cols);

        public void Init(ref Arena arena, int rows, int cols)
        {
            u = arena.fProxyVec(rows);
            v = arena.fProxyVec(cols);
            tmpM = arena.fProxyVec(rows);
            tmpN = arena.fProxyVec(cols);
        }

        /// damp is ignored: lnlq, like craig, has no Tikhonov-damped entry point (a consistent
        /// min-norm system has no residual/norm trade-off to regularize).
        public LstsqInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x, fProxy damp)
            where TOp : struct, IfProxyLinearOperator
        {
            var info = Krylov.lnlq(in A, in b, ref x, ref u, ref v, ref tmpM, ref tmpN, MaxIter(A.Rows, A.Cols), Tol);
            return new LstsqInfo { rnorm = info.rnorm, Arnorm = double.NaN, xnorm = info.xnorm, iterations = info.iterations, status = info.status };
        }
    }

    /// <summary>
    /// <see cref="IfProxyLstsqSolverInvoker"/> for <see cref="Krylov.cgne{TOp}"/> -- least-NORM
    /// CGNE (CG on the normal equations of the second kind, AAᵀy=b, x=Aᵀy, matrix-free) for
    /// UNDERDETERMINED (Rows &lt;= Cols, full row rank) CONSISTENT systems: among all x with Ax=b,
    /// finds the minimum-‖x‖ one -- the SAME x <see cref="fProxyCraigInvoker"/> computes, but via a
    /// direct CG recurrence on the (never-formed) AAᵀ rather than Golub-Kahan bidiagonalization.
    /// Owns the four scratch vectors cgne's zero-alloc primitive needs (r,Ap sized to Rows; p,tmpN
    /// to Cols). Requires Underdetermined; Forbids RankDeficient -- CGNE runs CG directly on AAᵀ, so
    /// its effective conditioning is cond(A)² and it cannot handle a rank-deficient (row-rank-
    /// deficient) A the way a rank-revealing method could.
    ///
    /// TolValue is set an order of magnitude TIGHTER than <see cref="fProxyCraigInvoker"/>'s in the
    /// battery: CGNE's κ² sensitivity means its solution error scales as cond(A)²·(residual tol),
    /// vs craig's cond(A)·(residual tol), so to land x inside the battery's shared element-agreement
    /// band (TolBand = 50·sqrtEps for the well-conditioned WideRandom10x30, the only underdetermined
    /// full-rank gallery entry) the residual must be driven ~10x lower than craig needs. The
    /// battery's residual threshold (10·Tol·‖b‖) scales with Tol in lockstep, so the residual check
    /// stays satisfied at 0.1·(that scale) regardless. See the KrylovLstsqBatteryTests switch.
    /// </summary>
    public struct fProxyCgneInvoker : IfProxyLstsqSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyN r, p, Ap, tmpN;

        public MatrixProfile Requires => MatrixProfile.Underdetermined;
        public MatrixProfile Forbids => MatrixProfile.RankDeficient;
        public fProxy Tol => TolValue;
        public int MaxIter(int rows, int cols) => MaxIterMul * (rows < cols ? rows : cols);

        public void Init(ref Arena arena, int rows, int cols)
        {
            r = arena.fProxyVec(rows);
            p = arena.fProxyVec(cols);
            Ap = arena.fProxyVec(rows);
            tmpN = arena.fProxyVec(cols);
        }

        /// damp is ignored: cgne has no Tikhonov-damped production entry point (a consistent
        /// min-norm system has no residual/norm trade-off to regularize), same as craig.
        public LstsqInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x, fProxy damp)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.cgne(in A, in b, ref x, ref r, ref p, ref Ap, ref tmpN, MaxIter(A.Rows, A.Cols), Tol);
    }

    /// <summary>
    /// <see cref="IfProxyLstsqSolverInvoker"/> for <see cref="Krylov.craigmr{TOp}"/> -- MINRES-
    /// flavored CRAIG (monotone residual) for UNDERDETERMINED (Rows &lt;= Cols, full row rank)
    /// CONSISTENT systems: among all x with Ax=b, finds the minimum-‖x‖ one. Owns the five
    /// scratch vectors craigmr's zero-alloc primitive needs (one more than
    /// <see cref="fProxyCraigInvoker"/> -- the running-QR direction d). damp is ignored, same
    /// rationale as fProxyCraigInvoker.
    /// </summary>
    public struct fProxyCraigmrInvoker : IfProxyLstsqSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        fProxyN u, v, d, tmpM, tmpN;

        public MatrixProfile Requires => MatrixProfile.Underdetermined;
        public MatrixProfile Forbids => MatrixProfile.RankDeficient;
        public fProxy Tol => TolValue;
        public int MaxIter(int rows, int cols) => MaxIterMul * (rows < cols ? rows : cols);

        public void Init(ref Arena arena, int rows, int cols)
        {
            u = arena.fProxyVec(rows);
            v = arena.fProxyVec(cols);
            d = arena.fProxyVec(cols);
            tmpM = arena.fProxyVec(rows);
            tmpN = arena.fProxyVec(cols);
        }

        public LstsqInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x, fProxy damp)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.craigmr(in A, in b, ref x, ref u, ref v, ref d, ref tmpM, ref tmpN, MaxIter(A.Rows, A.Cols), Tol);
    }

    /// <summary>
    /// <see cref="IfProxyBlockLstsqSolverInvoker"/> for <see cref="Krylov.blsmr{TOp}"/> -- block LSMR
    /// (Mojarrab &amp; Toutounian 2015) for a TALL/overdetermined operator A and s simultaneous
    /// right-hand sides: minimizes ‖A X - B‖_F. Owns no scratch (blsmr allocates its whole workspace
    /// from Allocator.Temp, mirroring <see cref="fProxyGmresInvoker"/>), so <see cref="Init"/> is a
    /// no-op. <see cref="ScalarCounterpart"/> is <see cref="fProxyLsmrInvoker"/> -- blsmr solves the
    /// same least-squares problem lsmr does, one column at a time.
    /// </summary>
    public struct fProxyBlsmrInvoker : IfProxyBlockLstsqSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        public MatrixProfile Requires => MatrixProfile.Overdetermined;
        public MatrixProfile Forbids => MatrixProfile.RankDeficient;
        public fProxy Tol => TolValue;
        public int MaxIter(int rows, int cols) => MaxIterMul * cols;

        public void Init(ref Arena arena, int rows, int cols, int s) { }

        public BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X, int maxIter)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.blsmr(in A, in B, ref X, maxIter, Tol);

        public IfProxyLstsqSolverInvoker ScalarCounterpart() => new fProxyLsmrInvoker { TolValue = TolValue, MaxIterMul = MaxIterMul };
    }

    /// <summary>
    /// <see cref="IfProxyBlockLstsqSolverInvoker"/> for <see cref="Krylov.bcgls{TOp}"/> -- block CGLS
    /// (block conjugate gradients on the normal equations AᵀA X = AᵀB, never forming AᵀA) for a TALL/
    /// overdetermined operator A and s simultaneous right-hand sides: minimizes ‖A X - B‖_F. Owns no
    /// scratch (bcgls allocates its whole workspace from Allocator.Temp), so <see cref="Init"/> is a
    /// no-op. Same <see cref="ScalarCounterpart"/> as <see cref="fProxyBlsmrInvoker"/> -- same
    /// least-squares problem, same lsmr comparison target. Forbids IllConditioned (unlike blsmr):
    /// squaring the condition number via the normal equations makes the gallery's IllConditioned
    /// overdetermined entry (Lauchli, eps=1e-3) genuinely too hard for this s x s Gram-based
    /// coefficient solve at a small block width -- see the folder DEVLOG.
    /// </summary>
    public struct fProxyBcglsInvoker : IfProxyBlockLstsqSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        public MatrixProfile Requires => MatrixProfile.Overdetermined;
        public MatrixProfile Forbids => MatrixProfile.RankDeficient | MatrixProfile.IllConditioned;
        public fProxy Tol => TolValue;
        public int MaxIter(int rows, int cols) => MaxIterMul * cols;

        public void Init(ref Arena arena, int rows, int cols, int s) { }

        public BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X, int maxIter)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.bcgls(in A, in B, ref X, maxIter, Tol);

        public IfProxyLstsqSolverInvoker ScalarCounterpart() => new fProxyLsmrInvoker { TolValue = TolValue, MaxIterMul = MaxIterMul };
    }

    /// <summary>
    /// <see cref="IfProxyBlockLstsqSolverInvoker"/> for <see cref="Krylov.bcraig{TOp}"/> -- block CRAIG
    /// for a WIDE/underdetermined operator A (A.Rows &lt;= A.Cols, full row rank) and s simultaneous
    /// CONSISTENT right-hand sides: among all X with A X_j = B_j (per row j), finds the minimum-
    /// Euclidean-norm one. Block min-NORM counterpart to <see cref="fProxyBlsmrInvoker"/>/
    /// <see cref="fProxyBcglsInvoker"/> (which are OVERDETERMINED / min-RESIDUAL) -- exactly the block
    /// analog of what <see cref="fProxyCraigInvoker"/> is to <see cref="fProxyLsmrInvoker"/>. Owns no
    /// scratch (bcraig allocates its whole workspace from Allocator.Temp, mirroring
    /// <see cref="fProxyBlsmrInvoker"/>), so <see cref="Init"/> is a no-op. <see cref="ScalarCounterpart"/>
    /// is <see cref="fProxyCraigInvoker"/> -- bcraig finds the same minimum-norm solution craig does, one
    /// column at a time. Requires Underdetermined and Forbids RankDeficient, mirroring
    /// <see cref="fProxyCraigInvoker"/>.
    /// </summary>
    public struct fProxyBcraigInvoker : IfProxyBlockLstsqSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        public MatrixProfile Requires => MatrixProfile.Underdetermined;
        public MatrixProfile Forbids => MatrixProfile.RankDeficient;
        public fProxy Tol => TolValue;
        public int MaxIter(int rows, int cols) => MaxIterMul * cols;

        public void Init(ref Arena arena, int rows, int cols, int s) { }

        public BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X, int maxIter)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.bcraig(in A, in B, ref X, maxIter, Tol);

        public IfProxyLstsqSolverInvoker ScalarCounterpart() => new fProxyCraigInvoker { TolValue = TolValue, MaxIterMul = MaxIterMul };
    }

    /// <summary>
    /// <see cref="IfProxyBlockLstsqSolverInvoker"/> for <see cref="Krylov.bcraigmr{TOp}"/> -- block
    /// CRAIGMR (MINRES-flavored block CRAIG, monotonic residual) for a WIDE/underdetermined operator A
    /// (A.Rows &lt;= A.Cols, full row rank) and s simultaneous CONSISTENT right-hand sides: among all X
    /// with A X_j = B_j, finds the minimum-Euclidean-norm one. Same min-NORM regime as <see
    /// cref="fProxyBcraigInvoker"/> (Requires Underdetermined, Forbids RankDeficient) -- bcraigmr finds
    /// the identical minimum-norm solution bcraig does, via a running block-QR continuation instead of
    /// block forward substitution. Owns no scratch (bcraigmr allocates its whole workspace from
    /// Allocator.Temp, mirroring <see cref="fProxyBcraigInvoker"/>), so <see cref="Init"/> is a no-op.
    /// <see cref="ScalarCounterpart"/> is <see cref="fProxyCraigmrInvoker"/> -- bcraigmr reduces to
    /// scalar craigmr one column at a time, not to plain craig.
    /// </summary>
    public struct fProxyBcraigmrInvoker : IfProxyBlockLstsqSolverInvoker
    {
        public fProxy TolValue;
        public int MaxIterMul;

        public MatrixProfile Requires => MatrixProfile.Underdetermined;
        public MatrixProfile Forbids => MatrixProfile.RankDeficient;
        public fProxy Tol => TolValue;
        public int MaxIter(int rows, int cols) => MaxIterMul * cols;

        public void Init(ref Arena arena, int rows, int cols, int s) { }

        public BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X, int maxIter)
            where TOp : struct, IfProxyLinearOperator
            => Krylov.bcraigmr(in A, in B, ref X, maxIter, Tol);

        public IfProxyLstsqSolverInvoker ScalarCounterpart() => new fProxyCraigmrInvoker { TolValue = TolValue, MaxIterMul = MaxIterMul };
    }
}
