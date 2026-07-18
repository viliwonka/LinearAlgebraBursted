using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
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
        }

        public TestType Type;

        static fProxy Tol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;
        static fProxy MatchTol() => /*+choose[2e-3f|1e-7]*/2e-3f/*-choose*/;

        // Dense nonsymmetric, diagonally dominant (well-conditioned, nonsingular): random entries + a
        // heavy diagonal. Not symmetric (random off-diagonals differ across the diagonal).
        static fProxyMxN DenseNonsym(ref Arena arena, int n, uint seed)
        {
            var A = arena.fProxyRandomMat(n, n, -1f, 1f, seed);
            for (int i = 0; i < n; i++) A[i, i] += (fProxy)(2 * n);
            return A;
        }

        // Scalar 1D convection-diffusion (BR=1): diagonal 6, super -1, sub -3 — nonsymmetric,
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

        static fProxy RelResidualDense(in fProxyMxN A, in fProxyN x, in fProxyN b)
        {
            var Ax = Blas.dot(A, x);
            fProxy num = 0, den = 0;
            for (int i = 0; i < b.N; i++) { fProxy d = Ax[i] - b[i]; num += d * d; den += b[i] * b[i]; }
            return math.sqrt(num) / math.sqrt(math.max(den, (fProxy)1e-30));
        }

        static fProxy RelResidualBSR(in fProxyBSR A, in fProxyN x, in fProxyN b)
        {
            var Ax = BSR.spMV(in A, in x);
            fProxy num = 0, den = 0;
            for (int i = 0; i < b.N; i++) { fProxy d = Ax[i] - b[i]; num += d * d; den += b[i] * b[i]; }
            return math.sqrt(num) / math.sqrt(math.max(den, (fProxy)1e-30));
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
            }
        }

        void SolvesDenseNonsym()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 40;
            var A = DenseNonsym(ref arena, n, 0x9E01u);
            var xTrue = arena.fProxyRandomVec(n, -1f, 1f, 0x9E02u);
            var b = Blas.dot(A, xTrue);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.gmres(in A, in b, ref x, n, 4 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(RelResidualDense(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        void SolvesBSRNonsym()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 120;
            var A = ConvDiff1D(ref arena, n);
            var xTrue = arena.fProxyRandomVec(n, -1f, 1f, 0x9E12u);
            var b = BSR.spMV(in A, in xTrue);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.gmres(in A, in b, ref x, 40, 4 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(RelResidualBSR(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        // A restart well below n must still converge (multiple restart cycles).
        void RestartConverges()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 120;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x9E22u);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.gmres(in A, in b, ref x, 10, 20 * n, Tol());   // restart 10 << n

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(RelResidualBSR(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        void MatchesBiCGStab()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 100;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x9E32u);

            var xG = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) xG[i] = (fProxy)0;
            var gi = Krylov.gmres(in A, in b, ref xG, n, 4 * n, Tol());

            var xB = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) xB[i] = (fProxy)0;
            var bi = Krylov.biCGStab(in A, in b, ref xB, 8 * n, Tol());

            Assert.IsTrue(gi.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(bi.status == IterativeSolveStatus.Converged);
            // Both solve the same well-conditioned system -> solutions agree.
            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(xG[i] - xB[i]) <= MatchTol() * ((fProxy)1 + math.abs(xB[i])));

            arena.Dispose();
        }

        void ZeroRhs()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 30;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) b[i] = (fProxy)0;

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)5;
            var info = Krylov.gmres(in A, in b, ref x, 20, 4 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(info.iterations == 0);
            for (int i = 0; i < n; i++) Assert.IsTrue(x[i] == (fProxy)0);

            arena.Dispose();
        }
    }

    [Test] public void SolvesDenseNonsymTest() => new GmresTestJob { Type = GmresTestJob.TestType.SolvesDenseNonsym }.Run();
    [Test] public void SolvesBSRNonsymTest() => new GmresTestJob { Type = GmresTestJob.TestType.SolvesBSRNonsym }.Run();
    [Test] public void RestartConvergesTest() => new GmresTestJob { Type = GmresTestJob.TestType.RestartConverges }.Run();
    [Test] public void MatchesBiCGStabTest() => new GmresTestJob { Type = GmresTestJob.TestType.MatchesBiCGStab }.Run();
    [Test] public void ZeroRhsTest() => new GmresTestJob { Type = GmresTestJob.TestType.ZeroRhs }.Run();
}
