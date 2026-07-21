using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// GCRO-DR (Krylov.gcrodr, Morgan 2002): restarted GMRES(m) that RECYCLES a k-dimensional
// approximate invariant subspace (harmonic Ritz vectors) across restart cycles, for a general
// (nonsymmetric) square A x = b. recycle = 0 disables recycling and is bit-identical to gmres(m).
//
// All cases run inside a [BurstCompile] IJob driven through .Run() (matches the other Krylov
// suites and guards against the IJob struct-copy / ping-pong-buffer bug class). Coverage:
//   - RecyclingBeatsGmres  : THE distinguishing test -- on a spectrum with a few small isolated
//                            eigenvalues, deflation converges in far fewer inner iterations than
//                            plain gmres(m), which stalls across many restart cycles.
//   - MatchesLUOracle      : agreement with a direct LU solve on a well-conditioned nonsym system.
//   - IdentityFoldBitExact : gcrodr(A,b,x) == gcrodr(A, identity, b, x), recycle > 0.
//   - RecycleZeroMatchesGmres : gcrodr(...,recycle=0,...) == gmres(...) bit-for-bit.
//   - Deterministic        : two runs from x0=0 on identical (A,b) are bit-identical.
//   - ZeroRhs              : b=0 -> Converged, iterations=0, x=0 exactly.
//   - SingularBreakdown    : 0-matrix operator (no solution) -> honest Breakdown, no NaN.
//   - SmallScaleWellConditioned : A = c*I for a tiny c -> must Converge (pivotGuard is ||A||-scaled,
//                                 not ||b||-scaled -- a well-conditioned small-magnitude A must not
//                                 trip a spurious Breakdown).
public class fProxyGCRODRTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct GcrodrTestJob : IJob
    {
        public enum TestType
        {
            RecyclingBeatsGmres,
            MatchesLUOracle,
            IdentityFoldBitExact,
            RecycleZeroMatchesGmres,
            Deterministic,
            ZeroRhs,
            SingularBreakdown,
            SmallScaleWellConditioned,
        }

        public TestType Type;

        // Residual / element comparison band, per numeric type (float looser). Mirrors GMRESTests.
        static fProxy Tol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;
        // Tighter convergence tolerance for the distinguishing test: loose enough that gcrodr's
        // deflated spectrum reaches it, tight enough that plain gmres(m) must grind many restarts.
        static fProxy TightTol() => /*+choose[1e-4f|1e-8]*/1e-4f/*-choose*/;

        // Dense nonsymmetric, diagonally dominant (well-conditioned, nonsingular): random entries + a
        // heavy diagonal. Not symmetric. Same construction GMRESTests/FGMRESTests use.
        static fProxyMxN DenseNonsym(ref Arena arena, int n, uint seed)
        {
            var A = arena.fProxyRandomMat(n, n, -1f, 1f, seed);
            for (int i = 0; i < n; i++) A[i, i] += (fProxy)(2 * n);
            return A;
        }

        // Upper-triangular nonsymmetric matrix whose eigenvalues are EXACTLY its diagonal: a handful
        // of small, well-isolated eigenvalues below a well-separated O(1) cluster, plus a mild
        // strictly-upper perturbation (keeps eigenvalues on the diagonal, makes A nonsymmetric).
        // This is the textbook GCRO-DR win: restarted gmres(m) stalls on the isolated small
        // eigenvalues (degree-m residual polynomial cannot resolve them AND the cluster each cycle),
        // while gcrodr recycles their harmonic-Ritz vectors and deflates them permanently.
        static fProxyMxN SmallIsolatedEig(ref Arena arena, int n, uint seed)
        {
            var A = arena.fProxyMat(n, n);   // zero
            A[0, 0] = (fProxy)0.01;
            A[1, 1] = (fProxy)0.03;
            A[2, 2] = (fProxy)0.06;
            for (int i = 3; i < n; i++)
                A[i, i] = (fProxy)(4.0 + 6.0 * (i - 3) / (double)(n - 4));   // cluster in [4, 10]

            var rnd = new Unity.Mathematics.Random(seed);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    A[i, j] = (fProxy)(rnd.NextFloat(-1f, 1f) * 0.02f);   // mild nonsymmetry, near-normal
            return A;
        }

        static fProxy RelResidualDense(in fProxyMxN A, in fProxyN x, in fProxyN b)
        {
            var Ax = Blas.dot(A, x);
            fProxy num = 0, den = 0;
            for (int i = 0; i < b.N; i++) { fProxy d = Ax[i] - b[i]; num += d * d; den += b[i] * b[i]; }
            return math.sqrt(num) / math.sqrt(math.max(den, (fProxy)1e-30));
        }

        public void Execute()
        {
            switch (Type)
            {
                case TestType.RecyclingBeatsGmres:     RecyclingBeatsGmres();     break;
                case TestType.MatchesLUOracle:         MatchesLUOracle();         break;
                case TestType.IdentityFoldBitExact:    IdentityFoldBitExact();    break;
                case TestType.RecycleZeroMatchesGmres: RecycleZeroMatchesGmres(); break;
                case TestType.Deterministic:           Deterministic();           break;
                case TestType.ZeroRhs:                 ZeroRhs();                 break;
                case TestType.SingularBreakdown:       SingularBreakdown();       break;
                case TestType.SmallScaleWellConditioned: SmallScaleWellConditioned(); break;
            }
        }

        // ---- THE distinguishing test: recycling beats plain gmres(m) on a small-isolated-eigenvalue
        // spectrum. Same system, same restart, same tol, same maxIter budget for both solvers. ----
        void RecyclingBeatsGmres()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 60;
            int m = 12, k = 3;               // recycle exactly the 3 isolated small eigenvalues
            fProxy tol = TightTol();
            int maxIter = 60 * n;

            var A = SmallIsolatedEig(ref arena, n, 0x6C0Du);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x6C0Eu);   // excites all eigen-components

            var xG = arena.fProxyVec(n);     // plain gmres(m)
            var giG = Krylov.gmres(in A, in b, ref xG, m, maxIter, tol);

            var xR = arena.fProxyVec(n);     // gcrodr(m, recycle=k)
            var giR = Krylov.gcrodr(in A, in b, ref xR, m, k, maxIter, tol);

            // gcrodr must actually solve it.
            Assert.IsTrue(giR.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(RelResidualDense(in A, in xR, in b) <= tol);

            // Non-tautology guard: plain gmres(m) genuinely grinds many restart cycles here (or
            // exhausts the budget) -- not a one-cycle solve. m inner iters == one restart cycle.
            Assert.IsTrue(giG.iterations > 3 * m);

            // THE POINT: deflation reaches the tolerance in strictly fewer total inner iterations.
            Assert.IsTrue(giR.iterations < giG.iterations);

            arena.Dispose();
        }

        // ---- Agreement with a direct LU solve on an ordinary well-conditioned nonsym system. ----
        void MatchesLUOracle()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 32;
            var A = DenseNonsym(ref arena, n, 0x6C11u);
            var xTrue = arena.fProxyRandomVec(n, -1f, 1f, 0x6C12u);
            var b = Blas.dot(A, xTrue);

            // Dense LU oracle on copies (decompInPlace/decompSolve are destructive).
            var LUcopy = A.Copy();
            var pivot = new Pivot(n, Allocator.Temp);
            bool okLU = LU.decompInPlace(ref LUcopy, ref pivot);
            Assert.IsTrue(okLU);
            var xLU = b.Copy();
            LU.decompSolve(ref LUcopy, in pivot, ref xLU);
            pivot.Dispose();

            var x = arena.fProxyVec(n);
            var info = Krylov.gcrodr(in A, in b, ref x, 16, 4, 8 * n, Tol());
            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(RelResidualDense(in A, in x, in b) <= Tol());

            // Agrees with the LU oracle and recovers the planted solution (unique on a nonsingular A).
            for (int i = 0; i < n; i++)
            {
                Assert.IsTrue(math.abs(x[i] - xLU[i]) <= Tol() * ((fProxy)1 + math.abs(xLU[i])));
                Assert.IsTrue(math.abs(x[i] - xTrue[i]) <= Tol() * ((fProxy)1 + math.abs(xTrue[i])));
            }

            arena.Dispose();
        }

        // ---- Identity fold, recycle > 0: the no-preconditioner entry point and the explicit
        // fProxyIdentityPreconditioner path share the exact IsIdentity-folded body, so x, iteration
        // count and rnorm must be bit-for-bit identical (== on floats, not a tolerance). ----
        void IdentityFoldBitExact()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 40;
            int m = 12, k = 4;
            var A = DenseNonsym(ref arena, n, 0x6C21u);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x6C22u);
            var op = new fProxyDenseOperator(in A);

            var xImplicit = arena.fProxyVec(n);
            var i0 = Krylov.gcrodr(in op, in b, ref xImplicit, m, k, 8 * n, Tol());

            var xExplicit = arena.fProxyVec(n);
            var i1 = Krylov.gcrodr(in op, default(fProxyIdentityPreconditioner), in b, ref xExplicit, m, k, 8 * n, Tol());

            Assert.IsTrue(i0.status == IterativeSolveStatus.Converged);
            Assert.AreEqual(i0.iterations, i1.iterations);
            Assert.AreEqual(i0.rnorm, i1.rnorm);
            for (int i = 0; i < n; i++) Assert.IsTrue(xImplicit[i] == xExplicit[i]);

            arena.Dispose();
        }

        // ---- recycle = 0 disables the whole recycling code path -> must equal plain gmres(m)
        // bit-for-bit (same restart, same everything). Strong internal-consistency check. ----
        void RecycleZeroMatchesGmres()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 40;
            int m = 12;
            var A = DenseNonsym(ref arena, n, 0x6C31u);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x6C32u);

            var xR = arena.fProxyVec(n);
            var iR = Krylov.gcrodr(in A, in b, ref xR, m, 0, 8 * n, Tol());

            var xG = arena.fProxyVec(n);
            var iG = Krylov.gmres(in A, in b, ref xG, m, 8 * n, Tol());

            Assert.IsTrue(iR.status == IterativeSolveStatus.Converged);
            Assert.AreEqual(iG.iterations, iR.iterations);
            Assert.AreEqual(iG.rnorm, iR.rnorm);
            for (int i = 0; i < n; i++) Assert.IsTrue(xR[i] == xG[i]);

            arena.Dispose();
        }

        // ---- Determinism: two runs from x0 = 0 on identical (A, b) with recycle > 0 give bit-identical
        // solution, iteration count and rnorm (the recycling deflation path included). ----
        void Deterministic()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 44;
            int m = 12, k = 3;
            var A = SmallIsolatedEig(ref arena, n, 0x6C41u);
            var b = arena.fProxyRandomVec(n, -1f, 1f, 0x6C42u);

            var x1 = arena.fProxyVec(n);
            var i1 = Krylov.gcrodr(in A, in b, ref x1, m, k, 40 * n, Tol());

            var x2 = arena.fProxyVec(n);
            var i2 = Krylov.gcrodr(in A, in b, ref x2, m, k, 40 * n, Tol());

            Assert.IsTrue(i1.status == IterativeSolveStatus.Converged);
            Assert.AreEqual(i1.iterations, i2.iterations);
            Assert.AreEqual(i1.rnorm, i2.rnorm);
            for (int i = 0; i < n; i++) Assert.IsTrue(x1[i] == x2[i]);

            arena.Dispose();
        }

        // ---- Zero RHS: exact early-out. x = b = 0, Converged, zero iterations, EXACT (not approx). ----
        void ZeroRhs()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 30;
            var A = DenseNonsym(ref arena, n, 0x6C51u);
            var b = arena.fProxyVec(n);   // all zeros

            // Seed x with garbage to prove gcrodr overwrites it with b on the early-out path.
            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = (fProxy)7;

            var info = Krylov.gcrodr(in A, in b, ref x, 10, 3, 4 * n, Tol());
            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(info.iterations == 0);
            for (int i = 0; i < n; i++) Assert.IsTrue(x[i] == (fProxy)0);

            arena.Dispose();
        }

        // ---- Singular operator (all-zero A): 0 x = b has NO solution for b != 0, so the Hessenberg
        // pivot collapses -> honest Breakdown, never a false Converged, never a NaN. ----
        void SingularBreakdown()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 8;
            var A = arena.fProxyMat(n, n);   // all zeros -> singular, A x == 0 for every x
            var b = arena.fProxyRandomVec(n, 1f, 2f, 0x6C61u);   // nonzero, not in range(A)

            var x = arena.fProxyVec(n);
            var info = Krylov.gcrodr(in A, in b, ref x, 5, 2, 4 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Breakdown);
            Assert.IsFalse(double.IsNaN((double)info.rnorm));

            arena.Dispose();
        }

        // ---- pivotGuard must scale with ||A||, not ||b||: a well-conditioned but uniformly tiny-
        // magnitude diagonal system (A = c*I) is exactly as solvable as its O(1) counterpart -- a
        // ||b||-scaled guard clamps the (legitimately tiny) ||A||-scaled Hessenberg pivot and reports
        // a spurious Breakdown with x left untouched. ----
        void SmallScaleWellConditioned()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 8;
            fProxy c = /*+choose[1e-6f|1e-30]*/1e-6f/*-choose*/;
            var A = arena.fProxyMat(n, n);
            for (int i = 0; i < n; i++) A[i, i] = c;
            var b = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) b[i] = (fProxy)1;

            var x = arena.fProxyVec(n);
            var info = Krylov.gcrodr(in A, in b, ref x, 5, 2, 4 * n, Tol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            fProxy expected = (fProxy)1 / c;
            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(x[i] - expected) <= Tol() * expected);

            arena.Dispose();
        }
    }

    [Test] public void RecyclingBeatsGmresTest()     => new GcrodrTestJob { Type = GcrodrTestJob.TestType.RecyclingBeatsGmres }.Run();
    [Test] public void MatchesLUOracleTest()         => new GcrodrTestJob { Type = GcrodrTestJob.TestType.MatchesLUOracle }.Run();
    [Test] public void IdentityFoldBitExactTest()    => new GcrodrTestJob { Type = GcrodrTestJob.TestType.IdentityFoldBitExact }.Run();
    [Test] public void RecycleZeroMatchesGmresTest() => new GcrodrTestJob { Type = GcrodrTestJob.TestType.RecycleZeroMatchesGmres }.Run();
    [Test] public void DeterministicTest()           => new GcrodrTestJob { Type = GcrodrTestJob.TestType.Deterministic }.Run();
    [Test] public void ZeroRhsTest()                 => new GcrodrTestJob { Type = GcrodrTestJob.TestType.ZeroRhs }.Run();
    [Test] public void SingularBreakdownTest()       => new GcrodrTestJob { Type = GcrodrTestJob.TestType.SingularBreakdown }.Run();
    [Test] public void SmallScaleWellConditionedTest() => new GcrodrTestJob { Type = GcrodrTestJob.TestType.SmallScaleWellConditioned }.Run();
}
