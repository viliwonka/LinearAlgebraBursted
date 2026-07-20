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
}
