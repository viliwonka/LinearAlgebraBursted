using System;
using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Flexible CG (Krylov.fcg): the variable-preconditioner CG needed by the AMG K-cycle.
// Cases run inside a [BurstCompile] IJob (matches the other Krylov suites). Coverage:
//   - solves an SPD BSR with a fixed SPD preconditioner (block-Jacobi), residual-based;
//   - agrees with pcg when the preconditioner is CONSTANT (fcg's reduction property);
//   - converges with a genuinely VARIABLE preconditioner (an inner-CG apply, whose k-step
//     iterate is a data-dependent polynomial in r) -- the case plain pcg is not built for;
//   - zero-rhs short-circuit.
public class fProxyFlexibleCGTests
{
    // Variable preconditioner: z = (a few CG steps on A z = r from z = 0). k fixed steps of CG
    // is a NONLINEAR (data-dependent) operator in r, so the effective M changes per outer
    // iteration -- exactly what fcg is designed to tolerate and pcg is not. Scratch vectors are
    // arena-owned (built on the main thread), so Apply allocates nothing.
    readonly struct InnerCgPreconditioner : IfProxyPreconditioner
    {
        readonly fProxyBSR A;
        readonly fProxyN sr, sp, sAp;
        readonly int steps;

        public InnerCgPreconditioner(in fProxyBSR a, fProxyN sr, fProxyN sp, fProxyN sAp, int steps)
        { A = a; this.sr = sr; this.sp = sp; this.sAp = sAp; this.steps = steps; }

        public int Rows => A.M_Rows;

        public bool IsIdentity => false;

        public void Apply(in fProxyN r, ref fProxyN z)
        {
            for (int i = 0; i < z.N; i++) z[i] = (fProxy)0;
            // fProxyN handles copied to locals: cg takes them by ref (scratch), and a readonly
            // struct field cannot be passed by ref -- the local aliases the same buffer.
            fProxyN lr = sr, lp = sp, lAp = sAp;
            Krylov.cg(in A, in r, ref z, ref lr, ref lp, ref lAp, steps, (fProxy)0);
        }
    }

    [BurstCompile(CompileSynchronously = true)]
    public struct FcgTestJob : IJob
    {
        public enum TestType
        {
            SolvesSpdBlockJacobi,
            MatchesPcgConstantM,
            VariableInnerCgConverges,
            ZeroRhsConverges,
        }

        public TestType Type;

        static fProxy Tol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.SolvesSpdBlockJacobi:     SolvesSpdBlockJacobi(); break;
                case TestType.MatchesPcgConstantM:      MatchesPcgConstantM(); break;
                case TestType.VariableInnerCgConverges: VariableInnerCgConverges(); break;
                case TestType.ZeroRhsConverges:         ZeroRhsConverges(); break;
            }
        }

        // Relative true residual ‖b − Ax‖ / ‖b‖.
        static fProxy RelResidual(in fProxyBSR A, in fProxyN x, in fProxyN b)
        {
            var Ax = BSR.spMV(in A, in x);
            fProxy num = 0, den = 0;
            for (int i = 0; i < b.N; i++)
            {
                fProxy d = Ax[i] - b[i];
                num += d * d;
                den += b[i] * b[i];
            }
            return math.sqrt(num) / math.sqrt(math.max(den, (fProxy)1e-30));
        }

        void SolvesSpdBlockJacobi()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyLaplacian2D(16, 16);
            int n = A.M_Rows;
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0xF01u);
            var M = arena.fProxyBlockJacobi(in A);
            var op = new fProxyBSROperator(in A);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;

            var info = Krylov.fcg(in op, in M, in b, ref x, 4 * n, Tol());
            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(RelResidual(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        // Constant SPD preconditioner: fcg's Polak–Ribière beta reduces to pcg's Fletcher–Reeves
        // (the <z_new, r_old> cross term is zero for constant M), so the two track the same
        // trajectory. Asserted via iteration-count agreement -- robust, unlike an element-wise
        // solution compare whose error scales with cond(A)·residual. Both must also converge.
        void MatchesPcgConstantM()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyLaplacian2D(12, 12);
            int n = A.M_Rows;
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0xF02u);
            var M = arena.fProxyBlockJacobi(in A);
            var op = new fProxyBSROperator(in A);

            var xF = arena.fProxyVec(n);
            var xP = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) { xF[i] = (fProxy)0; xP[i] = (fProxy)0; }

            var infoF = Krylov.fcg(in op, in M, in b, ref xF, 4 * n, Tol());
            var infoP = Krylov.pcg(in op, in M, in b, ref xP, 4 * n, Tol());
            Assert.IsTrue(infoF.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(infoP.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(RelResidual(in A, in xF, in b) <= Tol());

            // Same convergence trajectory for constant M (allow a small floating-point slack from
            // fcg's extra cross-term dot).
            Assert.IsTrue(math.abs(infoF.iterations - infoP.iterations) <= 2);

            arena.Dispose();
        }

        void VariableInnerCgConverges()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyLaplacian2D(16, 16);
            int n = A.M_Rows;
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0xF03u);
            var op = new fProxyBSROperator(in A);

            var sr = arena.fProxyVec(n); var sp = arena.fProxyVec(n); var sAp = arena.fProxyVec(n);
            var M = new InnerCgPreconditioner(in A, sr, sp, sAp, 3);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;

            var info = Krylov.fcg(in op, in M, in b, ref x, 4 * n, Tol());
            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(RelResidual(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        void ZeroRhsConverges()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyLaplacian2D(8, 8);
            int n = A.M_Rows;
            var b = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) b[i] = (fProxy)0;
            var M = arena.fProxyBlockJacobi(in A);
            var op = new fProxyBSROperator(in A);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)7;   // nonzero start -> must be driven to 0

            var info = Krylov.fcg(in op, in M, in b, ref x, 4 * n, Tol());
            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(info.iterations == 0);
            for (int i = 0; i < n; i++) Assert.IsTrue(x[i] == (fProxy)0);

            arena.Dispose();
        }
    }

    [Test]
    public void SolvesSpdBlockJacobiTest()
        => new FcgTestJob { Type = FcgTestJob.TestType.SolvesSpdBlockJacobi }.Run();

    [Test]
    public void MatchesPcgConstantMTest()
        => new FcgTestJob { Type = FcgTestJob.TestType.MatchesPcgConstantM }.Run();

    [Test]
    public void VariableInnerCgConvergesTest()
        => new FcgTestJob { Type = FcgTestJob.TestType.VariableInnerCgConverges }.Run();

    [Test]
    public void ZeroRhsConvergesTest()
        => new FcgTestJob { Type = FcgTestJob.TestType.ZeroRhsConverges }.Run();
}
