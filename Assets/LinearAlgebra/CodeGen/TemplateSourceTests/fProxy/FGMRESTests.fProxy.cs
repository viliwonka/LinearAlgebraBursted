using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Flexible GMRES(m) (Krylov.fgmres): GMRES(m) whose preconditioner may vary every inner step
// (AMG, an inner Krylov solve, ...). Cases run inside a [BurstCompile] IJob (matches the other
// Krylov suites). Coverage:
//   - agrees with right-preconditioned gmres for a FIXED preconditioner (fgmres's reduction
//     property: for constant M the stored-Z update equals gmres's apply-M-once update);
//   - unpreconditioned fgmres matches plain gmres bit-for-bit (the IsIdentity fold);
//   - known-solution recovery on a nonsymmetric BSR system;
//   - restart correctness (restart << n forces multiple restart cycles, still converges);
//   - converges with a genuinely VARIABLE preconditioner (a fixed-step-count inner GMRES apply,
//     whose map r -> z is data-dependent/nonlinear) -- the case plain gmres is not built for.
public class fProxyFGMRESTests
{
    // Variable preconditioner: z = (a few unpreconditioned GMRES steps on A z = r from z = 0).
    // k fixed inner steps is a NONLINEAR (data-dependent) operator in r, so the effective M
    // changes per outer step -- exactly what fgmres is designed to tolerate.
    readonly struct InnerGmresPreconditioner : IfProxyPreconditioner
    {
        readonly fProxyBSR A;
        readonly int innerRestart, steps;

        public InnerGmresPreconditioner(in fProxyBSR a, int innerRestart, int steps)
        { A = a; this.innerRestart = innerRestart; this.steps = steps; }

        public bool IsIdentity => false;
        // Fixed-step inner GMRES from z=0 is a nonlinear (data-dependent) map r -> z -- exactly the
        // VARIABLE case fgmres tolerates; never passed to a non-flexible solver.
        public bool IsSpd => false;
        public bool IsConstant => false;

        public void Apply(in fProxyN r, ref fProxyN z)
        {
            for (int i = 0; i < z.N; i++) z[i] = (fProxy)0;
            Krylov.gmres(in A, in r, ref z, innerRestart, steps, (fProxy)0);
        }
    }

    [BurstCompile(CompileSynchronously = true)]
    public struct FgmresTestJob : IJob
    {
        public enum TestType
        {
            MatchesGmresConstantPrecond,
            MatchesGmresIdentity,
            KnownSolutionRecovery,
            RestartConverges,
            VariableInnerGmresConverges,
        }

        public TestType Type;

        static fProxy Tol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;

        // Dense nonsymmetric, diagonally dominant (well-conditioned, nonsingular): random entries +
        // a heavy diagonal. Not symmetric (random off-diagonals differ across the diagonal).
        static fProxyMxN DenseNonsym(ref Arena arena, int n, uint seed)
        {
            var A = arena.fProxyRandomMat(n, n, -1f, 1f, seed);
            for (int i = 0; i < n; i++) A[i, i] += (fProxy)(2 * n);
            return A;
        }

        // Scalar 1D convection-diffusion (BR=1): diagonal 6, super -1, sub -3 -- nonsymmetric,
        // diagonally dominant. Full storage.
        static fProxyBSR ConvDiff1D(ref Arena arena, int n)
        {
            var b = arena.fProxyBSRBuilder(n, n, 1, 1, 3 * n);
            for (int i = 0; i < n; i++)
            {
                b.AddValue(i, i, (fProxy)6);
                if (i > 0) b.AddValue(i, i - 1, (fProxy)(-3));
                if (i < n - 1) b.AddValue(i, i + 1, (fProxy)(-1));
            }
            return b.ToBSR(ref arena);
        }

        public void Execute()
        {
            switch (Type)
            {
                case TestType.MatchesGmresConstantPrecond: MatchesGmresConstantPrecond(); break;
                case TestType.MatchesGmresIdentity:         MatchesGmresIdentity(); break;
                case TestType.KnownSolutionRecovery:        KnownSolutionRecovery(); break;
                case TestType.RestartConverges:             RestartConverges(); break;
                case TestType.VariableInnerGmresConverges:  VariableInnerGmresConverges(); break;
            }
        }

        // Constant preconditioner (block-Jacobi, IsConstant == true): post-GmresCore-merge, a
        // CONSTANT M makes fgmres take the SAME standard (single-zt) path as gmres -- not the Z
        // basis -- so fgmres(constant M) is now BIT-IDENTICAL to gmres(constant M), not merely
        // close. Assert exact equality on iterations, rnorm, and every x component.
        void MatchesGmresConstantPrecond()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 150;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0xFA01u);
            var M = arena.fProxyBlockJacobi(in A);
            var op = new fProxyBSROperator(in A);
            fProxy tol = Tol();

            var xG = arena.fProxyVec(n);
            var xF = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) { xG[i] = (fProxy)0; xF[i] = (fProxy)0; }

            var gi = Krylov.gmres(in op, in M, in b, ref xG, 20, 8 * n, tol);
            var fi = Krylov.fgmres(in op, in M, in b, ref xF, 20, 8 * n, tol);

            Assert.IsTrue(gi.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(fi.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(fProxyKrylovBatteryOracles.RelResidualBSR(in A, in xF, in b) <= tol);
            Assert.IsTrue(fi.iterations == gi.iterations);
            Assert.IsTrue(fi.rnorm == gi.rnorm);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(xF[i] == xG[i]);

            arena.Dispose();
        }

        // Unpreconditioned fgmres reduces to plain gmres under the IsIdentity fold: identical code
        // path (no Z workspace, solution accumulated straight into V), so bit-for-bit equal.
        void MatchesGmresIdentity()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 40;
            var A = DenseNonsym(ref arena, n, 0xFA11u);
            var xTrue = arena.fProxyRandomVec(n, -1f, 1f, 0xFA12u);
            var b = Blas.dot(A, xTrue);

            var xG = arena.fProxyVec(n);
            var xF = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) { xG[i] = (fProxy)0; xF[i] = (fProxy)0; }

            var gi = Krylov.gmres(in A, in b, ref xG, n, 4 * n, Tol());
            var fi = Krylov.fgmres(in A, in b, ref xF, n, 4 * n, Tol());

            Assert.IsTrue(gi.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(fi.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(fProxyKrylovBatteryOracles.RelResidualDense(in A, in xF, in b) <= Tol());
            Assert.AreEqual(gi.iterations, fi.iterations);
            Assert.AreEqual(gi.rnorm, fi.rnorm);
            for (int i = 0; i < n; i++) Assert.IsTrue(xG[i] == xF[i]);

            arena.Dispose();
        }

        void KnownSolutionRecovery()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 120;
            var A = ConvDiff1D(ref arena, n);
            var xTrue = arena.fProxyRandomVec(n, -1f, 1f, 0xFA21u);
            var b = BSR.spMV(in A, in xTrue);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.fgmres(in A, in b, ref x, 40, 4 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(fProxyKrylovBatteryOracles.RelResidualBSR(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        // A restart well below n must still converge (multiple restart cycles).
        void RestartConverges()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 120;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0xFA31u);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.fgmres(in A, in b, ref x, 10, 20 * n, Tol());   // restart 10 << n

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(fProxyKrylovBatteryOracles.RelResidualBSR(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        void VariableInnerGmresConverges()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 150;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0xFA41u);
            var op = new fProxyBSROperator(in A);
            var M = new InnerGmresPreconditioner(in A, 5, 3);   // 3 fixed unpreconditioned-GMRES steps

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;

            var info = Krylov.fgmres(in op, in M, in b, ref x, 20, 8 * n, Tol());
            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(fProxyKrylovBatteryOracles.RelResidualBSR(in A, in x, in b) <= Tol());

            arena.Dispose();
        }
    }

    [Test] public void MatchesGmresConstantPrecondTest() => new FgmresTestJob { Type = FgmresTestJob.TestType.MatchesGmresConstantPrecond }.Run();
    [Test] public void MatchesGmresIdentityTest() => new FgmresTestJob { Type = FgmresTestJob.TestType.MatchesGmresIdentity }.Run();
    [Test] public void KnownSolutionRecoveryTest() => new FgmresTestJob { Type = FgmresTestJob.TestType.KnownSolutionRecovery }.Run();
    [Test] public void RestartConvergesTest() => new FgmresTestJob { Type = FgmresTestJob.TestType.RestartConverges }.Run();
    [Test] public void VariableInnerGmresConvergesTest() => new FgmresTestJob { Type = FgmresTestJob.TestType.VariableInnerGmresConverges }.Run();
}
