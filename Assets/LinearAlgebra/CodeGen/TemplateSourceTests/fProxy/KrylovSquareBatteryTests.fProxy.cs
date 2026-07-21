using System;

using LinearAlgebra;
using LinearAlgebra.Sparse;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Krylov square-solver battery -- solver x matrix-regime cross-coverage for the square (single-RHS)
// Krylov family, driven through the IfProxySquareSolverInvoker struct-functor shape (see
// KrylovBattery.Invokers.fProxy.cs). Every single-RHS square solver in the codebase is wired here:
// cg, fcg, minres, minresQLP, biCGStab, gmres, fgmres, idr. Add a new solver as a new SolverKind case
// + RunStandardChecks call, nothing else in this file changes.
//
// Every case runs the same 4 standard checks (SS5.2 #1-4 of the battery spec) across every gallery
// matrix whose tags satisfy the invoker's Requires/Forbids (MatrixProfileMatch.Applicable), plus a
// 5th Sparse-only preconditioned-convergence check on the BSR gallery, a 6th warm-start-correctness
// check (nonzero initial guess must be carried into the solution, not discarded), and a 7th
// verify-at-exit-honesty check folded into every solve site in the loop (any Converged return must
// have a fresh true residual within bound). Grouped BY SOLVER (one NUnit case per SolverKind); the
// failing (matrix, check, got, expected) is surfaced via Fail.
public class fProxyKrylovSquareBatteryTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum SolverKind { Cg, Fcg, Minres, MinresQLP, BiCGStab, Gmres, Fgmres, Idr, Tfqmr, Gcrodr }

        public SolverKind Kind;

        // [0] flag (1 = failure recorded) [1] matrix-enum-as-int [2] check-id [3] got [4] expected
        public NativeArray<fProxy> Fail;

        const int DenseCount = (int)GalleryDenseMatrix.RandSPDIllCond20 + 1;
        const int BSRCount = (int)GalleryBSRMatrix.RandomSparseNonsym_80 + 1;

        public void Execute()
        {
            BurstProbe.RequireBursted();
            switch (Kind)
            {
                case SolverKind.Cg:        RunStandardChecks(new fProxyCgInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 }); break;
                case SolverKind.Fcg:       RunStandardChecks(new fProxyFcgInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 }); break;
                case SolverKind.Minres:    RunStandardChecks(new fProxyMinresInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 }); break;
                case SolverKind.MinresQLP: RunStandardChecks(new fProxyMinresQLPInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 }); break;
                case SolverKind.BiCGStab:  RunStandardChecks(new fProxyBiCGStabInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 }); break;
                case SolverKind.Gmres:     RunStandardChecks(new fProxyGmresInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 4, Restart = 30 }); break;
                case SolverKind.Fgmres:    RunStandardChecks(new fProxyFgmresInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 4, Restart = 30 }); break;
                case SolverKind.Idr:       RunStandardChecks(new fProxyIdrInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20, S = 4, Seed = 0x9E3779B1u }); break;
                case SolverKind.Tfqmr:     RunStandardChecks(new fProxyTfqmrInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 40 }); break;
                case SolverKind.Gcrodr:    RunStandardChecks(new fProxyGcrodrInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 4, Restart = 30, Recycle = 10 }); break;
            }
        }

        void RunStandardChecks<TInvoker>(TInvoker inv) where TInvoker : struct, IfProxySquareSolverInvoker
        {
            for (int i = 0; i < DenseCount; i++)
            {
                var gm = (GalleryDenseMatrix)i;
                if (MatrixProfileMatch.Applicable(inv.Requires, inv.Forbids, GalleryProfiles.Of(gm)))
                    CheckDense(inv, gm);
            }

            for (int i = 0; i < BSRCount; i++)
            {
                var gm = (GalleryBSRMatrix)i;
                if (MatrixProfileMatch.Applicable(inv.Requires, inv.Forbids, GalleryProfiles.Of(gm)))
                    CheckBSR(inv, gm);
            }
        }

        // Checks #1-4 (SS5.2) on one dense literature-gallery matrix.
        void CheckDense<TInvoker>(TInvoker inv, GalleryDenseMatrix gm) where TInvoker : struct, IfProxySquareSolverInvoker
        {
            var arena = new Arena(Allocator.Persistent);

            var A = fProxyKrylovBatteryGallery.Build(ref arena, gm);
            int n = A.M_Rows;
            var Aop = new fProxyDenseOperator(in A);
            MatrixProfile tags = GalleryProfiles.Of(gm);
            fProxy tolBand = TolBand(tags);

            var b = arena.fProxyRandomVec(n, (fProxy)(-1), (fProxy)1, 0xD000u + (uint)gm);

            inv.Init(ref arena, n);

            // 1. Converges: Solved or MaxIterations, AND the fresh residual bound holds either way.
            var x1 = arena.fProxyVec(n);
            SolveInfo info1 = inv.Solve(in Aop, in b, ref x1);
            bool statusOk1 = info1.status == IterativeSolveStatus.Converged || info1.status == IterativeSolveStatus.MaxIterations;
            Record(statusOk1, (int)gm, 1, (fProxy)(int)info1.status, (fProxy)0);
            fProxy relRes1 = fProxyKrylovBatteryOracles.RelResidualDense(in A, in x1, in b);
            Record(relRes1 <= (fProxy)10 * inv.Tol, (int)gm, 1, relRes1, (fProxy)10 * inv.Tol);
            VerifyHonestDense(info1.status, in A, in x1, in b, inv.Tol, (int)gm);

            // 2. Correctness vs. direct-solve reference (same A, b, and the x1 solved in #1).
            var xRef = ReferenceSolveDense(in A, in b, tags);
            for (int i = 0; i < n; i++)
                Record(math.abs(x1[i] - xRef[i]) <= tolBand * ((fProxy)1 + math.abs(xRef[i])), (int)gm, 2, x1[i], xRef[i]);

            // 3. Determinism: two independent solves from x0=0 on the identical (A, b) match bit-for-bit.
            var x3a = arena.fProxyVec(n);
            var x3b = arena.fProxyVec(n);
            SolveInfo info3a = inv.Solve(in Aop, in b, ref x3a);
            SolveInfo info3b = inv.Solve(in Aop, in b, ref x3b);
            for (int i = 0; i < n; i++)
                Record(x3a[i] == x3b[i], (int)gm, 3, x3a[i], x3b[i]);
            Record(info3a.iterations == info3b.iterations, (int)gm, 3, (fProxy)info3a.iterations, (fProxy)info3b.iterations);
            VerifyHonestDense(info3a.status, in A, in x3a, in b, inv.Tol, (int)gm);
            VerifyHonestDense(info3b.status, in A, in x3b, in b, inv.Tol, (int)gm);

            // 4. Identity-fold: the unpreconditioned path == the generic path with an explicit identity.
            var x4a = arena.fProxyVec(n);
            var x4b = arena.fProxyVec(n);
            SolveInfo info4a = inv.Solve(in Aop, in b, ref x4a);
            SolveInfo info4b = inv.SolveWithPrecond(in Aop, default(fProxyIdentityPreconditioner), in b, ref x4b);
            for (int i = 0; i < n; i++)
                Record(x4a[i] == x4b[i], (int)gm, 4, x4a[i], x4b[i]);
            Record(info4a.iterations == info4b.iterations, (int)gm, 4, (fProxy)info4a.iterations, (fProxy)info4b.iterations);
            VerifyHonestDense(info4a.status, in A, in x4a, in b, inv.Tol, (int)gm);
            VerifyHonestDense(info4b.status, in A, in x4b, in b, inv.Tol, (int)gm);

            // 6. Warm-start correctness: a NONZERO initial guess must be carried into the solution
            // (x0 + dx), not silently discarded -- known x*, b = A x*, x seeded to a fixed nonzero
            // vector unrelated to x*. Applies unconditionally: every square solver takes ref x.
            var xStar6 = arena.fProxyRandomVec(n, (fProxy)(-1), (fProxy)1, 0xE000u + (uint)gm);
            var bWarm6 = arena.fProxyVec(n);
            Blas.dot(in A, in xStar6, ref bWarm6);
            var xWarm6 = arena.fProxyRandomVec(n, (fProxy)(-1), (fProxy)1, 0xE100u + (uint)gm);
            SolveInfo infoWarm6 = inv.Solve(in Aop, in bWarm6, ref xWarm6);
            Record(infoWarm6.status == IterativeSolveStatus.Converged, (int)gm, 6, (fProxy)(int)infoWarm6.status, (fProxy)(int)IterativeSolveStatus.Converged);
            fProxy relResWarm6 = fProxyKrylovBatteryOracles.RelResidualDense(in A, in xWarm6, in bWarm6);
            Record(relResWarm6 <= (fProxy)100 * inv.Tol, (int)gm, 6, relResWarm6, (fProxy)100 * inv.Tol);
            VerifyHonestDense(infoWarm6.status, in A, in xWarm6, in bWarm6, inv.Tol, (int)gm);

            arena.Dispose();
        }

        // Checks #1-5 (SS5.2) on one BSR gallery matrix -- #5 (preconditioned convergence) always
        // applies here since every BSR gallery entry carries the Sparse tag.
        void CheckBSR<TInvoker>(TInvoker inv, GalleryBSRMatrix gm) where TInvoker : struct, IfProxySquareSolverInvoker
        {
            var arena = new Arena(Allocator.Persistent);

            var A = fProxyKrylovBatteryGallery.Build(ref arena, gm);
            int n = A.M_Rows;
            var Aop = new fProxyBSROperator(in A);
            MatrixProfile tags = GalleryProfiles.Of(gm);
            fProxy tolBand = TolBand(tags);

            var b = arena.fProxyRandomVec(n, (fProxy)(-1), (fProxy)1, 0xB000u + (uint)gm);

            inv.Init(ref arena, n);

            // 1. Converges.
            var x1 = arena.fProxyVec(n);
            SolveInfo info1 = inv.Solve(in Aop, in b, ref x1);
            bool statusOk1 = info1.status == IterativeSolveStatus.Converged || info1.status == IterativeSolveStatus.MaxIterations;
            Record(statusOk1, (int)gm, 1, (fProxy)(int)info1.status, (fProxy)0);
            fProxy relRes1 = fProxyKrylovBatteryOracles.RelResidualBSR(in A, in x1, in b);
            Record(relRes1 <= (fProxy)10 * inv.Tol, (int)gm, 1, relRes1, (fProxy)10 * inv.Tol);
            VerifyHonestBSR(info1.status, in A, in x1, in b, inv.Tol, (int)gm);

            // 2. Correctness vs. direct-solve reference (densify A -- no direct BSR factorization).
            var Adense = A.ToDense(ref arena);
            var xRef = ReferenceSolveDense(in Adense, in b, tags);
            for (int i = 0; i < n; i++)
                Record(math.abs(x1[i] - xRef[i]) <= tolBand * ((fProxy)1 + math.abs(xRef[i])), (int)gm, 2, x1[i], xRef[i]);

            // 3. Determinism.
            var x3a = arena.fProxyVec(n);
            var x3b = arena.fProxyVec(n);
            SolveInfo info3a = inv.Solve(in Aop, in b, ref x3a);
            SolveInfo info3b = inv.Solve(in Aop, in b, ref x3b);
            for (int i = 0; i < n; i++)
                Record(x3a[i] == x3b[i], (int)gm, 3, x3a[i], x3b[i]);
            Record(info3a.iterations == info3b.iterations, (int)gm, 3, (fProxy)info3a.iterations, (fProxy)info3b.iterations);
            VerifyHonestBSR(info3a.status, in A, in x3a, in b, inv.Tol, (int)gm);
            VerifyHonestBSR(info3b.status, in A, in x3b, in b, inv.Tol, (int)gm);

            // 4. Identity-fold.
            var x4a = arena.fProxyVec(n);
            var x4b = arena.fProxyVec(n);
            SolveInfo info4a = inv.Solve(in Aop, in b, ref x4a);
            SolveInfo info4b = inv.SolveWithPrecond(in Aop, default(fProxyIdentityPreconditioner), in b, ref x4b);
            for (int i = 0; i < n; i++)
                Record(x4a[i] == x4b[i], (int)gm, 4, x4a[i], x4b[i]);
            Record(info4a.iterations == info4b.iterations, (int)gm, 4, (fProxy)info4a.iterations, (fProxy)info4b.iterations);
            VerifyHonestBSR(info4a.status, in A, in x4a, in b, inv.Tol, (int)gm);
            VerifyHonestBSR(info4b.status, in A, in x4b, in b, inv.Tol, (int)gm);

            // 5. Preconditioned convergence (Sparse-only), M built per inv.PrecondKind.
            var x5 = arena.fProxyVec(n);
            SolveInfo info5;
            if (inv.PrecondKind == PreconditionerKind.SymmetricBSR)
            {
                var M = arena.fProxyBlockJacobi(in A);
                info5 = inv.SolveWithPrecond(in Aop, in M, in b, ref x5);
            }
            else if (inv.PrecondKind == PreconditionerKind.NonsymmetricBSR)
            {
                var M = arena.fProxyILU0(in A);
                info5 = inv.SolveWithPrecond(in Aop, in M, in b, ref x5);
            }
            else
            {
                info5 = inv.Solve(in Aop, in b, ref x5);
            }
            bool statusOk5 = info5.status == IterativeSolveStatus.Converged || info5.status == IterativeSolveStatus.MaxIterations;
            Record(statusOk5, (int)gm, 5, (fProxy)(int)info5.status, (fProxy)0);
            fProxy relRes5 = fProxyKrylovBatteryOracles.RelResidualBSR(in A, in x5, in b);
            Record(relRes5 <= (fProxy)10 * inv.Tol, (int)gm, 5, relRes5, (fProxy)10 * inv.Tol);
            VerifyHonestBSR(info5.status, in A, in x5, in b, inv.Tol, (int)gm);

            // 6. Warm-start correctness (mirrors CheckDense).
            var xStar6 = arena.fProxyRandomVec(n, (fProxy)(-1), (fProxy)1, 0xE000u + (uint)gm);
            var bWarm6 = BSR.spMV(in A, in xStar6);
            var xWarm6 = arena.fProxyRandomVec(n, (fProxy)(-1), (fProxy)1, 0xE100u + (uint)gm);
            SolveInfo infoWarm6 = inv.Solve(in Aop, in bWarm6, ref xWarm6);
            Record(infoWarm6.status == IterativeSolveStatus.Converged, (int)gm, 6, (fProxy)(int)infoWarm6.status, (fProxy)(int)IterativeSolveStatus.Converged);
            fProxy relResWarm6 = fProxyKrylovBatteryOracles.RelResidualBSR(in A, in xWarm6, in bWarm6);
            Record(relResWarm6 <= (fProxy)100 * inv.Tol, (int)gm, 6, relResWarm6, (fProxy)100 * inv.Tol);
            VerifyHonestBSR(infoWarm6.status, in A, in xWarm6, in bWarm6, inv.Tol, (int)gm);

            arena.Dispose();
        }

        // Reference solution via the library's own direct solver for A's KIND: CHO for SPD,
        // LU for SymmetricIndefinite/Nonsymmetric.
        fProxyN ReferenceSolveDense(in fProxyMxN A, in fProxyN b, MatrixProfile tags)
        {
            var xRef = b.Copy();
            if ((tags & MatrixProfile.SPD) != 0)
            {
                var L = A.Copy();
                CHO.decompInPlace(ref L);
                CHO.decompSolve(ref L, ref xRef);
            }
            else
            {
                var LUm = A.Copy();
                var P = new Pivot(A.M_Rows, Allocator.Temp);
                LU.decompInPlace(ref LUm, ref P);
                LU.decompSolve(ref LUm, in P, ref xRef);
                P.Dispose();
            }
            return xRef;
        }

        // WellConditioned -> 50*sqrtEps, IllConditioned -> 5E-2 (mirrors SolverBatteryTests' hardcoded
        // per-matrix bands; not a live Analysis.cond() call, to keep the battery cheap).
        static fProxy TolBand(MatrixProfile tags)
            => (tags & MatrixProfile.IllConditioned) != 0 ? (fProxy)5E-2 : (fProxy)50 * Consts.fProxySqrtEps;

        // 7. Verify-at-exit honesty (universal invariant, folded into every solve call above rather
        // than a standalone check block): whenever a solve claims Converged, the fresh true residual
        // (recomputed from x/A/b, independent of whatever the solver tracked internally) must
        // actually be within a generous bound. Guards the whole silent-false-Converged family across
        // every check site in the loop, not only the "Converges" check.
        void VerifyHonestDense(IterativeSolveStatus status, in fProxyMxN A, in fProxyN x, in fProxyN b, fProxy tol, int gmIdx)
        {
            if (status != IterativeSolveStatus.Converged) return;
            fProxy relRes = fProxyKrylovBatteryOracles.RelResidualDense(in A, in x, in b);
            Record(relRes <= (fProxy)100 * tol, gmIdx, 7, relRes, (fProxy)100 * tol);
        }

        void VerifyHonestBSR(IterativeSolveStatus status, in fProxyBSR A, in fProxyN x, in fProxyN b, fProxy tol, int gmIdx)
        {
            if (status != IterativeSolveStatus.Converged) return;
            fProxy relRes = fProxyKrylovBatteryOracles.RelResidualBSR(in A, in x, in b);
            Record(relRes <= (fProxy)100 * tol, gmIdx, 7, relRes, (fProxy)100 * tol);
        }

        void Record(bool ok, int matrixIdx, int checkId, fProxy got, fProxy expected)
        {
            if (!ok && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = (fProxy)matrixIdx;
                Fail[2] = (fProxy)checkId;
                Fail[3] = got;
                Fail[4] = expected;
            }
            Assert.IsTrue(ok);
        }
    }

    static Array GetKinds() => Enum.GetValues(typeof(TestJob.SolverKind));

    // Generous timeout: the first case run in a session pays one cold Burst compile of the whole
    // Execute() body, and checks #6/#7 add a warm-start solve (plus an honesty re-verify) per
    // gallery matrix on top of checks #1-5 (mirrors KrylovLstsqBatteryTests' own [Timeout], see DEVLOG).
    [Timeout(600000)]
    [TestCaseSource(nameof(GetKinds))]
    public void SquareBattery(TestJob.SolverKind kind)
    {
        var fail = new NativeArray<fProxy>(5, Allocator.TempJob);
        try
        {
            new TestJob { Kind = kind, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{kind}: matrix={fail[1]} check={fail[2]} got={fail[3]} expected={fail[4]}");
        }
        finally { fail.Dispose(); }
    }
}
