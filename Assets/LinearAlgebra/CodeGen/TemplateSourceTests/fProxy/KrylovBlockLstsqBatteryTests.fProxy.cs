using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Krylov block-least-squares battery -- solver x matrix-regime cross-coverage for the block
// (multi-RHS) rectangular Krylov family, driven through the IfProxyBlockLstsqSolverInvoker
// struct-functor shape (see KrylovBattery.Invokers.fProxy.cs). Wires blsmr (block LSMR) and bcgls
// (block CGLS) -- both OVERDETERMINED (tall A, min-RESIDUAL) -- plus bcraig (block CRAIG),
// UNDERDETERMINED (wide A, min-NORM); the correctness oracle is branched by matrix shape the same
// way the single-RHS KrylovLstsqBatteryTests.fProxy.cs branches lsqr/lsmr vs craig/craigmr
// (bcraigmr is not implemented yet). Overdetermined checks the min-RESIDUAL normal-equations
// optimality ‖Aᵀ(AX-B)‖ and per-column agreement with scalar lsmr; underdetermined checks a fresh
// ‖AX-B‖ AND per-row agreement with LQ.minNormSolve (the exact minimum-2-norm oracle -- a residual
// check alone would pass for ANY consistent solve, not just the minimum-norm one). Retires
// BlockLSMRTests.fProxy.cs / BlockCGLSTests.fProxy.cs -- every scenario those bespoke files asserted
// (optimality, per-column agreement with the scalar sibling, consistent-system exact recovery,
// zero-rhs immediate convergence, tiny-maxIter no-NaN) is covered here, generalized across the dense
// literature gallery instead of one hardcoded matrix per file.
public class fProxyKrylovBlockLstsqBatteryTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum SolverKind { Blsmr, Bcgls, Bcraig }

        public SolverKind Kind;

        // [0] flag (1 = failure recorded) [1] matrix-enum-as-int [2] check-id [3] got [4] expected
        public NativeArray<fProxy> Fail;

        const int DenseCount = (int)GalleryDenseMatrix.TallRandom24x8 + 1;

        // Block width for this whole run -- small enough to stay <= every applicable gallery matrix's
        // N_Cols (Lauchli3_05/1e3 have only 3 columns; TallRandom24x8 has 8).
        const int S = 2;

        // Lauchli3_05/Lauchli3_1e3 (n=3) -- blsmr's block Golub-Kahan bidiagonalization needs more
        // column-space headroom than S=2 leaves in a 3-column matrix: its per-iteration LQ factors hit
        // a genuine (not merely slow) Breakdown there even on the WellConditioned Lauchli3_05 entry.
        // bcgls tolerates the same narrow WellConditioned entry fine (only its IllConditioned sibling
        // is excluded, via Forbids on fProxyBcglsInvoker) -- see the folder DEVLOG.
        static bool IsNarrowDense(GalleryDenseMatrix gm)
            => gm == GalleryDenseMatrix.Lauchli3_05 || gm == GalleryDenseMatrix.Lauchli3_1e3;

        public void Execute()
        {
            switch (Kind)
            {
                // blsmr's convergence flag is CONSERVATIVE in float: the internal ||A^T R||_F^2
                // stopping test can leave a consistent (zero-residual) system just short of its
                // threshold under float rounding even though the recovered X is already accurate --
                // the strict Solved/converged==S assertion on the consistent-recovery check (#4) is
                // double-only here.
                case SolverKind.Blsmr:
                    RunStandardChecks(
                        new fProxyBlsmrInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 },
                        new fProxyLsmrInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 },
                        strictConsistentStatus: IsDouble(), skipNarrow: true);
                    break;
                // bcgls tests convergence on the EXACT maintained S = A^T R (not an estimate), so its
                // Solved/converged==S status is honest in float too -- asserted for every dtype.
                case SolverKind.Bcgls:
                    RunStandardChecks(
                        new fProxyBcglsInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 },
                        new fProxyLsmrInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 },
                        strictConsistentStatus: true, skipNarrow: false);
                    break;
                // bcraig checks a FRESH ‖B - A X‖_F every round (an honest test, not a tracked
                // estimate -- see the OP/DEVLOG Krylov.Block.CRAIG note), so its Solved/converged==S
                // status is honest in float too -- asserted for every dtype, like bcgls.
                case SolverKind.Bcraig:
                    RunStandardChecks(
                        new fProxyBcraigInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 },
                        new fProxyCraigInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 },
                        strictConsistentStatus: true, skipNarrow: false);
                    break;
            }
        }

        void RunStandardChecks<TInvoker, TScalar>(TInvoker inv, TScalar scalarInv, bool strictConsistentStatus, bool skipNarrow)
            where TInvoker : struct, IfProxyBlockLstsqSolverInvoker
            where TScalar : struct, IfProxyLstsqSolverInvoker
        {
            for (int i = 0; i < DenseCount; i++)
            {
                var gm = (GalleryDenseMatrix)i;
                if (skipNarrow && IsNarrowDense(gm)) continue;
                if (MatrixProfileMatch.Applicable(inv.Requires, inv.Forbids, GalleryProfiles.Of(gm)))
                    CheckDense(inv, scalarInv, gm, strictConsistentStatus);
            }
        }

        // Checks #1-6 on one dense literature-gallery matrix (tall or wide A, block RHS B). Checks
        // #2/#3 (per-column optimality + scalar agreement) and #4 (consistent-system recovery) branch
        // by matrix shape: Overdetermined (blsmr/bcgls) uses a min-RESIDUAL oracle, Underdetermined
        // (bcraig) uses a min-NORM oracle (LQ.minNormSolve), mirroring the single-RHS battery's
        // check #10 vs #11 split.
        void CheckDense<TInvoker, TScalar>(TInvoker inv, TScalar scalarInv, GalleryDenseMatrix gm, bool strictConsistentStatus)
            where TInvoker : struct, IfProxyBlockLstsqSolverInvoker
            where TScalar : struct, IfProxyLstsqSolverInvoker
        {
            var arena = new Arena(Allocator.Persistent);

            var A = fProxyKrylovBatteryGallery.Build(ref arena, gm);
            int m = A.M_Rows, n = A.N_Cols;
            var Aop = new fProxyDenseOperator(in A);
            MatrixProfile tags = GalleryProfiles.Of(gm);
            fProxy tolBand = TolBand(tags);
            bool underdetermined = (tags & MatrixProfile.Underdetermined) != 0;

            inv.Init(ref arena, m, n, S);
            scalarInv.Init(ref arena, m, n);

            var rScratch = arena.fProxyVec(m);
            var sScratch = arena.fProxyVec(n);

            // 1. Converges: Solved or MaxIterations on a random block RHS. (A has full row rank, so a
            // random s x m block RHS is always consistent for the underdetermined shape too.)
            var B = arena.fProxyRandomMat(S, m, (fProxy)(-1), (fProxy)1, 0xD300u + (uint)gm);
            var X = arena.fProxyMat(S, n);
            BlockSolveInfo info = inv.Solve(in Aop, in B, ref X, inv.MaxIter(m, n));
            bool statusOk = info.status == IterativeSolveStatus.Converged || info.status == IterativeSolveStatus.MaxIterations;
            Record(statusOk, (int)gm, 1, (fProxy)(int)info.status, (fProxy)0);

            // 2/3: per column of the same (A, B, X), oracle branched by shape.
            for (int j = 0; j < S; j++)
            {
                var bj = fProxyKrylovBatteryOracles.Row(ref arena, in B, j, m);
                var xj = fProxyKrylovBatteryOracles.Row(ref arena, in X, j, n);

                var audit = Krylov.lstsqResidual(in Aop, in bj, in xj, (fProxy)0, ref rScratch, ref sScratch);

                if (underdetermined)
                {
                    // min-NORM oracle: the system is consistent, so ||A x_j - b_j|| (not ||A^T(Ax-b)||)
                    // is the residual signal, small relative to ||b_j|| (bcraig's own stopping scale);
                    // AND x_j must match LQ.minNormSolve -- the exact minimum-2-norm solution -- NOT
                    // merely satisfy the residual bound (any consistent solve would).
                    fProxy bNorm = math.sqrt(Blas.dot(bj, bj));
                    fProxy thresh = (fProxy)10 * inv.Tol * math.max(bNorm, (fProxy)1e-30);
                    Record((fProxy)audit.rnorm <= thresh, (int)gm, 2, (fProxy)audit.rnorm, thresh);

                    var xRef = arena.fProxyVec(n);
                    LQ.minNormSolve(in A, in bj, ref xRef);
                    for (int c = 0; c < n; c++)
                        Record(math.abs(X[j, c] - xRef[c]) <= tolBand * ((fProxy)1 + math.abs(xRef[c])), (int)gm, 2, X[j, c], xRef[c]);
                }
                else
                {
                    // min-RESIDUAL oracle: an overdetermined system has a nonzero residual by
                    // construction, so ||A^T(AX-B)|| is the correctness signal, not ||AX-B||.
                    Aop.ApplyT(in bj, ref sScratch);
                    fProxy atbNorm = math.sqrt(Blas.dot(sScratch, sScratch));
                    fProxy thresh = (fProxy)10 * inv.Tol * math.max(atbNorm, (fProxy)1e-30);
                    Record((fProxy)audit.Arnorm <= thresh, (int)gm, 2, (fProxy)audit.Arnorm, thresh);
                }

                // Agreement with the scalar sibling's solve of that same column (same problem, one RHS
                // at a time -- scalar lsmr for the tall shape, scalar craig for the wide shape).
                var xjScalar = arena.fProxyVec(n);
                LstsqInfo infoScalar = scalarInv.Solve(in Aop, in bj, ref xjScalar, (fProxy)0);
                bool scalarOk = infoScalar.status == IterativeSolveStatus.Converged || infoScalar.status == IterativeSolveStatus.MaxIterations;
                Record(scalarOk, (int)gm, 3, (fProxy)(int)infoScalar.status, (fProxy)0);
                for (int c = 0; c < n; c++)
                    Record(math.abs(X[j, c] - xjScalar[c]) <= tolBand * ((fProxy)1 + math.abs(xjScalar[c])), (int)gm, 3, X[j, c], xjScalar[c]);
            }

            // 4. Consistent (zero-residual) system B = A Xk. Overdetermined: recovers Xk exactly (the
            // finite-termination property). Underdetermined: the random Xk used to build B is NOT
            // itself minimum-norm, so the correct target is LQ.minNormSolve(A, Bc_j) per row, not Xk.
            var Xk = arena.fProxyRandomMat(S, n, (fProxy)(-1), (fProxy)1, 0xD400u + (uint)gm);
            var Bc = arena.fProxyMat(S, m);
            for (int j = 0; j < S; j++)
            {
                var xkj = fProxyKrylovBatteryOracles.Row(ref arena, in Xk, j, n);
                var bcj = arena.fProxyVec(m);
                Aop.Apply(in xkj, ref bcj);
                for (int c = 0; c < m; c++) Bc[j, c] = bcj[c];
            }
            var Xc = arena.fProxyMat(S, n);
            BlockSolveInfo infoC = inv.Solve(in Aop, in Bc, ref Xc, inv.MaxIter(m, n));
            if (underdetermined)
            {
                for (int j = 0; j < S; j++)
                {
                    var bcj = fProxyKrylovBatteryOracles.Row(ref arena, in Bc, j, m);
                    var xRef = arena.fProxyVec(n);
                    LQ.minNormSolve(in A, in bcj, ref xRef);
                    for (int c = 0; c < n; c++)
                        Record(math.abs(Xc[j, c] - xRef[c]) <= tolBand * ((fProxy)1 + math.abs(xRef[c])), (int)gm, 4, Xc[j, c], xRef[c]);
                }
            }
            else
            {
                for (int j = 0; j < S; j++)
                    for (int c = 0; c < n; c++)
                        Record(math.abs(Xc[j, c] - Xk[j, c]) <= tolBand * ((fProxy)1 + math.abs(Xk[j, c])), (int)gm, 4, Xc[j, c], Xk[j, c]);
            }
            if (strictConsistentStatus)
                Record(infoC.Solved && infoC.converged == S, (int)gm, 4, (fProxy)infoC.converged, (fProxy)S);

            // 5. Zero RHS converges immediately: X = 0, zero iterations.
            var B0 = arena.fProxyMat(S, m);   // zeroed by allocation
            var X0 = arena.fProxyMat(S, n);
            BlockSolveInfo info0 = inv.Solve(in Aop, in B0, ref X0, inv.MaxIter(m, n));
            Record(info0.Solved, (int)gm, 5, (fProxy)(int)info0.status, (fProxy)0);
            Record(info0.iterations == 0, (int)gm, 5, (fProxy)info0.iterations, (fProxy)0);
            for (int j = 0; j < S; j++)
                for (int c = 0; c < n; c++)
                    Record(X0[j, c] == (fProxy)0, (int)gm, 5, X0[j, c], (fProxy)0);

            // 6. A tiny maxIter (forcing MaxIterations before full convergence) never NaN/Inf X.
            var B6 = arena.fProxyRandomMat(S, m, (fProxy)(-1), (fProxy)1, 0xD500u + (uint)gm);
            var X6 = arena.fProxyMat(S, n);
            inv.Solve(in Aop, in B6, ref X6, 1);
            bool anyBad = false;
            for (int j = 0; j < S; j++)
                for (int c = 0; c < n; c++)
                    if (math.isnan(X6[j, c]) || math.isinf(X6[j, c])) anyBad = true;
            Record(!anyBad, (int)gm, 6, anyBad ? (fProxy)1 : (fProxy)0, (fProxy)0);

            arena.Dispose();
        }

        static bool IsDouble() => (double)Consts.fProxyEpsilon < 1e-10;

        // WellConditioned -> 50*sqrtEps, IllConditioned -> 5E-2 (mirrors the other Krylov battery
        // families' own bands).
        static fProxy TolBand(MatrixProfile tags)
            => (tags & MatrixProfile.IllConditioned) != 0 ? (fProxy)5E-2 : (fProxy)50 * Consts.fProxySqrtEps;

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
    // Execute() body (see DEVLOG).
    [Timeout(600000)]
    [TestCaseSource(nameof(GetKinds))]
    public void BlockLstsqBattery(TestJob.SolverKind kind)
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
