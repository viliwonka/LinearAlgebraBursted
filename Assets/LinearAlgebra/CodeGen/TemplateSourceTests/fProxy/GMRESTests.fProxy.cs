using System;
using BULA;
using BULA.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Restarted GMRES(m) for general (nonsymmetric) systems. Cases run inside a [BurstCompile] IJob:
// dense + BSR nonsymmetric solves, a restart smaller than n, agreement with biCGStab, and zero-rhs.
public class fProxyGMRESTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct GmresTestJob : IJob
    {
        public enum TestType
        {
            SolvesDenseNonsym,
            SolvesBSRNonsym,
            RestartConverges,
            MatchesBiCGStab,
            ZeroRhs,
            PreconditionedFewerIters,
        }

        public TestType Type;

        static fProxy Tol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;
        static fProxy MatchTol() => /*+choose[2e-3f|1e-7]*/2e-3f/*-choose*/;

        // Dense nonsymmetric, diagonally dominant (well-conditioned, nonsingular): random entries + a
        // heavy diagonal. Not symmetric (random off-diagonals differ across the diagonal).
        static fProxyMxN DenseNonsym(int n, uint seed)
        {
            var A = GenerateOP.fProxyRandomMat(n, n, -1f, 1f, seed);
            for (int i = 0; i < n; i++) A[i, i] += (fProxy)(2 * n);
            return A;
        }

        // Scalar 1D convection-diffusion (BR=1): diagonal 6, super -1, sub -3 — nonsymmetric,
        // diagonally dominant. Full storage.
        static fProxyBSR ConvDiff1D(int n)
        {
            var b = new fProxyBSRBuilder(n, n, 1, 1, Allocator.Temp, 3 * n);
            for (int i = 0; i < n; i++)
            {
                b.AddValue(i, i, (fProxy)6);
                if (i > 0) b.AddValue(i, i - 1, (fProxy)(-3));
                if (i < n - 1) b.AddValue(i, i + 1, (fProxy)(-1));
            }
            return b.ToBSR(Allocator.Temp);
        }

        public void Execute()
        {
            switch (Type)
            {
                case TestType.SolvesDenseNonsym: SolvesDenseNonsym(); break;
                case TestType.SolvesBSRNonsym:   SolvesBSRNonsym(); break;
                case TestType.RestartConverges:  RestartConverges(); break;
                case TestType.MatchesBiCGStab:   MatchesBiCGStab(); break;
                case TestType.ZeroRhs:           ZeroRhs(); break;
                case TestType.PreconditionedFewerIters: PreconditionedFewerIters(); break;
            }
        }

        void SolvesDenseNonsym()
        {
            int n = 40;
            var A = DenseNonsym(n, 0x9E01u);
            var xTrue = GenerateOP.fProxyRandomVec(n, -1f, 1f, 0x9E02u);
            var b = Blas.dot(A, xTrue);

            var x = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.gmres(in A, in b, ref x, n, 4 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(fProxyKrylovBatteryOracles.RelResidualDense(in A, in x, in b) <= Tol());
        }

        void SolvesBSRNonsym()
        {
            int n = 120;
            var A = ConvDiff1D(n);
            var xTrue = GenerateOP.fProxyRandomVec(n, -1f, 1f, 0x9E12u);
            var b = BSR.spMV(in A, in xTrue);

            var x = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.gmres(in A, in b, ref x, 40, 4 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(fProxyKrylovBatteryOracles.RelResidualBSR(in A, in x, in b) <= Tol());
        }

        // A restart well below n must still converge (multiple restart cycles).
        void RestartConverges()
        {
            int n = 120;
            var A = ConvDiff1D(n);
            var b = GenerateOP.fProxyRandomVec(n, -1f, 1f, 0x9E22u);

            var x = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.gmres(in A, in b, ref x, 10, 20 * n, Tol());   // restart 10 << n

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(fProxyKrylovBatteryOracles.RelResidualBSR(in A, in x, in b) <= Tol());
        }

        void MatchesBiCGStab()
        {
            int n = 100;
            var A = ConvDiff1D(n);
            var b = GenerateOP.fProxyRandomVec(n, -1f, 1f, 0x9E32u);

            var xG = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) xG[i] = (fProxy)0;
            var gi = Krylov.gmres(in A, in b, ref xG, n, 4 * n, Tol());

            var xB = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) xB[i] = (fProxy)0;
            var bi = Krylov.biCGStab(in A, in b, ref xB, 8 * n, Tol());

            Assert.IsTrue(gi.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(bi.status == IterativeSolveStatus.Converged);
            // Both solve the same well-conditioned system -> solutions agree.
            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(xG[i] - xB[i]) <= MatchTol() * ((fProxy)1 + math.abs(xB[i])));
        }

        // ILU(0)-right-preconditioned GMRES converges AND in fewer inner iterations than plain GMRES.
        void PreconditionedFewerIters()
        {
            int n = 200;
            var A = ConvDiff1D(n);
            var b = GenerateOP.fProxyRandomVec(n, -1f, 1f, 0x9E42u);
            fProxy tol = Tol();

            var xG = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) xG[i] = (fProxy)0;
            var gi = Krylov.gmres(in A, in b, ref xG, 20, 8 * n, tol);

            var M = new fProxyILU0(in A, Allocator.Temp);
            var xP = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) xP[i] = (fProxy)0;
            var pi = Krylov.gmres(in A, in M, in b, ref xP, 20, 8 * n, tol);

            Assert.IsTrue(gi.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(pi.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(fProxyKrylovBatteryOracles.RelResidualBSR(in A, in xP, in b) <= tol);
            Assert.IsTrue(pi.iterations < gi.iterations);
        }

        void ZeroRhs()
        {
            int n = 30;
            var A = ConvDiff1D(n);
            var b = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) b[i] = (fProxy)0;

            var x = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) x[i] = (fProxy)5;
            var info = Krylov.gmres(in A, in b, ref x, 20, 4 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(info.iterations == 0);
            for (int i = 0; i < n; i++) Assert.IsTrue(x[i] == (fProxy)0);
        }
    }

    [Test] public void SolvesDenseNonsymTest() => new GmresTestJob { Type = GmresTestJob.TestType.SolvesDenseNonsym }.Run();
    [Test] public void SolvesBSRNonsymTest() => new GmresTestJob { Type = GmresTestJob.TestType.SolvesBSRNonsym }.Run();
    [Test] public void RestartConvergesTest() => new GmresTestJob { Type = GmresTestJob.TestType.RestartConverges }.Run();
    [Test] public void MatchesBiCGStabTest() => new GmresTestJob { Type = GmresTestJob.TestType.MatchesBiCGStab }.Run();
    [Test] public void ZeroRhsTest() => new GmresTestJob { Type = GmresTestJob.TestType.ZeroRhs }.Run();
    [Test] public void PreconditionedFewerItersTest() => new GmresTestJob { Type = GmresTestJob.TestType.PreconditionedFewerIters }.Run();
}
