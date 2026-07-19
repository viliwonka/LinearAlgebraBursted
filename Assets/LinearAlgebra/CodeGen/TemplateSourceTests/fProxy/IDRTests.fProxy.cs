using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// IDR(s) (Induced Dimension Reduction, Sonneveld & van Gijzen 2008) for general (nonsymmetric)
// square systems. Every case runs inside a [BurstCompile] IJob (.Run()): dense + BSR nonsymmetric
// solves, known-solution recovery, cross-check vs gmres, identity-fold (unpreconditioned) rungs,
// ILU0/BlockJacobi-preconditioned BSR convergence, s=1 degenerate, zero-rhs, and — the key new-
// feature invariant — BIT-IDENTICAL determinism from the seeded shadow space (explicit + default seed).
public class fProxyIDRTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct IdrTestJob : IJob
    {
        public enum TestType
        {
            SolvesDenseNonsym,
            KnownSolution,
            MatchesGmres,
            IdentityFoldDense,
            IdentityFoldBSR,
            PreconditionedILU0,
            PreconditionedBlockJacobi,
            DeterminismExplicitSeed,
            DeterminismDefaultSeed,
            SEqualsOne,
            ZeroRhs,
        }

        public TestType Type;

        static fProxy Tol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;
        static fProxy MatchTol() => /*+choose[2e-3f|1e-6]*/2e-3f/*-choose*/;
        static fProxy SolTol() => /*+choose[5e-3f|1e-6]*/5e-3f/*-choose*/;

        // Dense nonsymmetric, diagonally dominant (well-conditioned, nonsingular): random entries +
        // a heavy diagonal. Not symmetric (random off-diagonals differ across the diagonal).
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
                case TestType.SolvesDenseNonsym:         SolvesDenseNonsym(); break;
                case TestType.KnownSolution:             KnownSolution(); break;
                case TestType.MatchesGmres:              MatchesGmres(); break;
                case TestType.IdentityFoldDense:         IdentityFoldDense(); break;
                case TestType.IdentityFoldBSR:           IdentityFoldBSR(); break;
                case TestType.PreconditionedILU0:        PreconditionedILU0(); break;
                case TestType.PreconditionedBlockJacobi: PreconditionedBlockJacobi(); break;
                case TestType.DeterminismExplicitSeed:   DeterminismExplicitSeed(); break;
                case TestType.DeterminismDefaultSeed:    DeterminismDefaultSeed(); break;
                case TestType.SEqualsOne:                SEqualsOne(); break;
                case TestType.ZeroRhs:                   ZeroRhs(); break;
            }
        }

        // (1) Basic convergence on a nonsymmetric square system.
        void SolvesDenseNonsym()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 40;
            var A = DenseNonsym(ref arena, n, 0x1D01u);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x1D02u);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.idr(in A, in b, ref x, 4, 20 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(info.Solved);
            Assert.IsTrue(RelResidualDense(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        // (2) Known-solution recovery: b = A*xTrue -> recovered x ~ xTrue.
        void KnownSolution()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 32;
            var A = DenseNonsym(ref arena, n, 0x1D11u);
            var xTrue = arena.fProxyRandomVec(n, -1f, 1f, 0x1D12u);
            var b = Blas.dot(A, xTrue);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.idr(in A, in b, ref x, 4, 20 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(x[i] - xTrue[i]) <= SolTol() * ((fProxy)1 + math.abs(xTrue[i])));

            arena.Dispose();
        }

        // (1, cont.) Cross-check the IDR solution against a gmres reference on the SAME system.
        void MatchesGmres()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 100;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x1D22u);

            var xI = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) xI[i] = (fProxy)0;
            var ii = Krylov.idr(in A, in b, ref xI, 4, 20 * n, Tol());

            var xG = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) xG[i] = (fProxy)0;
            var gi = Krylov.gmres(in A, in b, ref xG, n, 4 * n, Tol());

            Assert.IsTrue(ii.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(gi.status == IterativeSolveStatus.Converged);
            // Both solve the same well-conditioned system -> solutions agree.
            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(xI[i] - xG[i]) <= MatchTol() * ((fProxy)1 + math.abs(xG[i])));

            arena.Dispose();
        }

        // (5) Identity-fold (unpreconditioned) dense rung.
        void IdentityFoldDense()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 36;
            var A = DenseNonsym(ref arena, n, 0x1D31u);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x1D32u);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.idr(in A, in b, ref x, 4, 20 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(RelResidualDense(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        // (5) Identity-fold (unpreconditioned) BSR rung.
        void IdentityFoldBSR()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 120;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x1D42u);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.idr(in A, in b, ref x, 4, 20 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(RelResidualBSR(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        // (6) ILU0-right-preconditioned BSR converges.
        void PreconditionedILU0()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 150;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x1D52u);
            var M = arena.fProxyILU0(in A);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.idr(in A, in M, in b, ref x, 4, 20 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(RelResidualBSR(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        // (6) BlockJacobi-right-preconditioned BSR converges.
        void PreconditionedBlockJacobi()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 150;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x1D62u);
            var M = arena.fProxyBlockJacobi(in A);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.idr(in A, in M, in b, ref x, 4, 20 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(RelResidualBSR(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        // (3) Determinism with an explicit seed: two independent solves from the same initial x
        // must produce a BIT-IDENTICAL x (the seeded shadow space is the only randomness).
        void DeterminismExplicitSeed()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 60;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x1D72u);
            uint seed = 0x1234ABCDu;

            var x1 = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x1[i] = (fProxy)0;
            var i1 = Krylov.idr(in A, in b, ref x1, 4, 20 * n, Tol(), seed);

            var x2 = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x2[i] = (fProxy)0;
            var i2 = Krylov.idr(in A, in b, ref x2, 4, 20 * n, Tol(), seed);

            Assert.IsTrue(i1.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(i2.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(i1.iterations == i2.iterations);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(x1[i] == x2[i]);   // EXACT, bit-identical

            arena.Dispose();
        }

        // (3) Determinism with the DEFAULT seed (omitted): two independent solves must still
        // produce a bit-identical x, via the zero-arg convenience overload.
        void DeterminismDefaultSeed()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 60;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x1D82u);

            var x1 = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x1[i] = (fProxy)0;
            Krylov.idr(in A, in b, ref x1);

            var x2 = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x2[i] = (fProxy)0;
            Krylov.idr(in A, in b, ref x2);

            for (int i = 0; i < n; i++)
                Assert.IsTrue(x1[i] == x2[i]);   // EXACT, bit-identical

            arena.Dispose();
        }

        // Edge: s = 1 (legal degenerate shadow-space dimension) still solves.
        void SEqualsOne()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 48;
            var A = DenseNonsym(ref arena, n, 0x1D91u);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x1D92u);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.idr(in A, in b, ref x, 1, 40 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(RelResidualDense(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        // Edge: zero rhs -> immediate converged, x set to zero, no iterations.
        void ZeroRhs()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 30;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) b[i] = (fProxy)0;

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)5;
            var info = Krylov.idr(in A, in b, ref x, 4, 20 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(info.iterations == 0);
            for (int i = 0; i < n; i++) Assert.IsTrue(x[i] == (fProxy)0);

            arena.Dispose();
        }
    }

    [Test] public void SolvesDenseNonsymTest() => new IdrTestJob { Type = IdrTestJob.TestType.SolvesDenseNonsym }.Run();
    [Test] public void KnownSolutionTest() => new IdrTestJob { Type = IdrTestJob.TestType.KnownSolution }.Run();
    [Test] public void MatchesGmresTest() => new IdrTestJob { Type = IdrTestJob.TestType.MatchesGmres }.Run();
    [Test] public void IdentityFoldDenseTest() => new IdrTestJob { Type = IdrTestJob.TestType.IdentityFoldDense }.Run();
    [Test] public void IdentityFoldBSRTest() => new IdrTestJob { Type = IdrTestJob.TestType.IdentityFoldBSR }.Run();
    [Test] public void PreconditionedILU0Test() => new IdrTestJob { Type = IdrTestJob.TestType.PreconditionedILU0 }.Run();
    [Test] public void PreconditionedBlockJacobiTest() => new IdrTestJob { Type = IdrTestJob.TestType.PreconditionedBlockJacobi }.Run();
    [Test] public void DeterminismExplicitSeedTest() => new IdrTestJob { Type = IdrTestJob.TestType.DeterminismExplicitSeed }.Run();
    [Test] public void DeterminismDefaultSeedTest() => new IdrTestJob { Type = IdrTestJob.TestType.DeterminismDefaultSeed }.Run();
    [Test] public void SEqualsOneTest() => new IdrTestJob { Type = IdrTestJob.TestType.SEqualsOne }.Run();
    [Test] public void ZeroRhsTest() => new IdrTestJob { Type = IdrTestJob.TestType.ZeroRhs }.Run();
}
