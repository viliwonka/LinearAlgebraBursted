using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// TFQMR (Transpose-Free QMR, Freund 1993) for general (nonsymmetric) square systems A x = b.
// Every case runs inside a [BurstCompile] IJob (.Run()): dense + BSR nonsymmetric solves,
// known-solution recovery, cross-check vs the direct LU solver, the identity-fold BIT-IDENTICAL
// invariant (unpreconditioned entry == generic entry with an explicit identity preconditioner),
// job-struct-copy safety (two solves inside one Execute produce bit-identical output), an
// ILU0-right-preconditioned BSR solve, and the zero-rhs edge case.
public class fProxyTFQMRTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct TfqmrTestJob : IJob
    {
        public enum TestType
        {
            SolvesDenseNonsym,
            KnownSolution,
            MatchesDirectSolve,
            SolvesBSRNonsym,
            PreconditionedILU0,
            IdentityFold,
            Determinism,
            ZeroRhs,
        }

        public TestType Type;

        // TFQMR's Converged bound guarantees the TRUE residual is within tol*||b||, so the same
        // tolerance band works for the freshly-recomputed relative residual checks below.
        static fProxy Tol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;
        // Cross-solver / known-solution agreement is one convergence tolerance looser than the
        // solve target on well-conditioned systems.
        static fProxy MatchTol() => /*+choose[2e-3f|1e-6]*/2e-3f/*-choose*/;
        static fProxy SolTol() => /*+choose[5e-3f|1e-6]*/5e-3f/*-choose*/;

        // Generous half-step budget (TFQMR's maxIter counts half-steps, ~one A-apply each; ~40n
        // for parity with a 20n two-matvec-per-pass method, padded here since this is a test).
        static int MaxIter(int n) => 50 * n;

        // Dense nonsymmetric, diagonally dominant (well-conditioned, nonsingular): random entries +
        // a heavy diagonal. Not symmetric (random off-diagonals differ across the diagonal).
        static fProxyMxN DenseNonsym(ref Arena arena, int n, uint seed)
        {
            var A = arena.fProxyRandomMat(n, n, -1f, 1f, seed);
            for (int i = 0; i < n; i++) A[i, i] += (fProxy)(2 * n);
            return A;
        }

        // Scalar 1D convection-diffusion: diagonal 6, super -1, sub -3 — nonsymmetric, diagonally
        // dominant. Full storage.
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
                case TestType.SolvesDenseNonsym:  SolvesDenseNonsym(); break;
                case TestType.KnownSolution:      KnownSolution(); break;
                case TestType.MatchesDirectSolve: MatchesDirectSolve(); break;
                case TestType.SolvesBSRNonsym:    SolvesBSRNonsym(); break;
                case TestType.PreconditionedILU0: PreconditionedILU0(); break;
                case TestType.IdentityFold:       IdentityFold(); break;
                case TestType.Determinism:        Determinism(); break;
                case TestType.ZeroRhs:            ZeroRhs(); break;
            }
        }

        // Basic convergence on a nonsymmetric dense square system + fresh residual check.
        void SolvesDenseNonsym()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 40;
            var A = DenseNonsym(ref arena, n, 0x7F01u);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x7F02u);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.tfqmr(in A, in b, ref x, MaxIter(n), Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(info.Solved);
            Assert.IsTrue(RelResidualDense(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        // Required #1: known-solution recovery. b = A*xTrue -> recovered x ~ xTrue elementwise.
        void KnownSolution()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 32;
            var A = DenseNonsym(ref arena, n, 0x7F11u);
            var xTrue = arena.fProxyRandomVec(n, -1f, 1f, 0x7F12u);
            var b = Blas.dot(A, xTrue);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.tfqmr(in A, in b, ref x, MaxIter(n), Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(x[i] - xTrue[i]) <= SolTol() * ((fProxy)1 + math.abs(xTrue[i])));

            arena.Dispose();
        }

        // Required #2: agreement with an independent direct LU solve on the SAME random (A, b).
        void MatchesDirectSolve()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 30;
            var A = DenseNonsym(ref arena, n, 0x7F21u);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x7F22u);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.tfqmr(in A, in b, ref x, MaxIter(n), Tol());

            // Independent direct solve via the library's own LU (mirrors the battery's
            // ReferenceSolveDense Nonsymmetric branch).
            var xRef = b.Copy();
            var LUm = A.Copy();
            var P = new Pivot(n, Allocator.Temp);
            LU.decompInPlace(ref LUm, ref P);
            LU.decompSolve(ref LUm, in P, ref xRef);
            P.Dispose();

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(x[i] - xRef[i]) <= MatchTol() * ((fProxy)1 + math.abs(xRef[i])));

            arena.Dispose();
        }

        // BSR nonsymmetric convergence + fresh residual check.
        void SolvesBSRNonsym()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 120;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x7F32u);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.tfqmr(in A, in b, ref x, MaxIter(n), Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(RelResidualBSR(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        // ILU(0)-right-preconditioned BSR converges.
        void PreconditionedILU0()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 150;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x7F42u);
            var M = arena.fProxyILU0(in A);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)0;
            var info = Krylov.tfqmr(in A, in M, in b, ref x, MaxIter(n), Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(RelResidualBSR(in A, in x, in b) <= Tol());

            arena.Dispose();
        }

        // Required #3: identity-fold BIT-IDENTICAL. The unpreconditioned generic entry point and the
        // merged generic entry point with an explicit default(fProxyIdentityPreconditioner) must
        // produce bit-for-bit identical x (and equal iteration counts) on the SAME (A, b, x0=0). uHat
        // is allocated but never read/written under the identity fold.
        void IdentityFold()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 36;
            var A = DenseNonsym(ref arena, n, 0x7F51u);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x7F52u);
            var Aop = new fProxyDenseOperator(in A);
            int maxIter = MaxIter(n);
            fProxy tol = Tol();

            // Path A: unpreconditioned generic entry (six buffers, no uHat).
            var xa = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) xa[i] = (fProxy)0;
            var rHat0a = arena.fProxyVec(n);
            var ua = arena.fProxyVec(n);
            var wa = arena.fProxyVec(n);
            var va = arena.fProxyVec(n);
            var aua = arena.fProxyVec(n);
            var da = arena.fProxyVec(n);
            var infoA = Krylov.tfqmr(in Aop, in b, ref xa,
                ref rHat0a, ref ua, ref wa, ref va, ref aua, ref da, maxIter, tol);

            // Path B: merged generic entry with an explicit identity preconditioner (+ uHat, unused).
            var xb = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) xb[i] = (fProxy)0;
            var rHat0b = arena.fProxyVec(n);
            var ub = arena.fProxyVec(n);
            var wb = arena.fProxyVec(n);
            var vb = arena.fProxyVec(n);
            var aub = arena.fProxyVec(n);
            var db = arena.fProxyVec(n);
            var uHat = arena.fProxyVec(n);
            var infoB = Krylov.tfqmr(in Aop, default(fProxyIdentityPreconditioner), in b, ref xb,
                ref rHat0b, ref ub, ref wb, ref vb, ref aub, ref db, ref uHat, maxIter, tol);

            Assert.IsTrue(infoA.iterations == infoB.iterations);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(xa[i] == xb[i]);   // EXACT, bit-identical

            arena.Dispose();
        }

        // Required #4: job-struct-copy safety / no hidden state between calls. Two independent solves
        // of the identical (A, b) from x0=0 inside this ONE Execute must produce bit-identical x AND
        // identical iteration counts (TFQMR keeps all state in caller-supplied buffers).
        void Determinism()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 60;
            var A = ConvDiff1D(ref arena, n);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x7F62u);

            var x1 = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x1[i] = (fProxy)0;
            var i1 = Krylov.tfqmr(in A, in b, ref x1, MaxIter(n), Tol());

            var x2 = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x2[i] = (fProxy)0;
            var i2 = Krylov.tfqmr(in A, in b, ref x2, MaxIter(n), Tol());

            Assert.IsTrue(i1.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(i2.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(i1.iterations == i2.iterations);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(x1[i] == x2[i]);   // EXACT, bit-identical

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
            var info = Krylov.tfqmr(in A, in b, ref x, MaxIter(n), Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(info.iterations == 0);
            for (int i = 0; i < n; i++) Assert.IsTrue(x[i] == (fProxy)0);

            arena.Dispose();
        }
    }

    [Test] public void SolvesDenseNonsymTest() => new TfqmrTestJob { Type = TfqmrTestJob.TestType.SolvesDenseNonsym }.Run();
    [Test] public void KnownSolutionTest() => new TfqmrTestJob { Type = TfqmrTestJob.TestType.KnownSolution }.Run();
    [Test] public void MatchesDirectSolveTest() => new TfqmrTestJob { Type = TfqmrTestJob.TestType.MatchesDirectSolve }.Run();
    [Test] public void SolvesBSRNonsymTest() => new TfqmrTestJob { Type = TfqmrTestJob.TestType.SolvesBSRNonsym }.Run();
    [Test] public void PreconditionedILU0Test() => new TfqmrTestJob { Type = TfqmrTestJob.TestType.PreconditionedILU0 }.Run();
    [Test] public void IdentityFoldTest() => new TfqmrTestJob { Type = TfqmrTestJob.TestType.IdentityFold }.Run();
    [Test] public void DeterminismTest() => new TfqmrTestJob { Type = TfqmrTestJob.TestType.Determinism }.Run();
    [Test] public void ZeroRhsTest() => new TfqmrTestJob { Type = TfqmrTestJob.TestType.ZeroRhs }.Run();
}
