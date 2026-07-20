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
}
