using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.Control;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of the determinism conformance harness's optimization/control groups
    // (lp-lad, qp, mip, control, nls-optimize). See DeterminismDirect.fProxy.cs's header for the
    // shared job/case-method convention and docs/dev/spec-determinism-conformance-harness.md for the
    // frozen op/group/root hash contract.
    //
    // The `control` group also covers the UKF (Kalman.ukfPredict/ukfUpdate) with a LINEAR model/
    // measurement (DetLinearKFModelFProxy/DetLinearKFMeasFProxy below): the sigma-point machinery
    // itself is +-*/sqrt-only, so a linear model keeps the whole path section-A-safe (see the spec's
    // group 31 note) -- folded into `control` rather than a separate native-sensitive group.

    public struct DetLinearKFModelFProxy : IfProxyKFModel
    {
        public fProxyMxN A;
        public fProxyMxN B;
        public fProxyN Scratch;

        public void F(in fProxyN x, in fProxyN u, ref fProxyN xNext)
        {
            Blas.dot(in A, in x, ref xNext);
            Blas.dot(in B, in u, ref Scratch);
            xNext.addInPlace(Scratch);
        }

        public void JacobianF(in fProxyN x, in fProxyN u, ref fProxyMxN J)
        {
            for (int r = 0; r < A.M_Rows; r++) for (int c = 0; c < A.N_Cols; c++) J[r, c] = A[r, c];
        }
    }

    public struct DetLinearKFMeasFProxy : IfProxyKFMeasurement
    {
        public fProxyMxN Hmat;

        public void H(in fProxyN x, ref fProxyN z) => Blas.dot(in Hmat, in x, ref z);

        public void JacobianH(in fProxyN x, ref fProxyMxN J)
        {
            for (int r = 0; r < Hmat.M_Rows; r++) for (int c = 0; c < Hmat.N_Cols; c++) J[r, c] = Hmat[r, c];
        }
    }

    public struct DetPolyResidualFProxy : IfProxyResidualFunction
    {
        public fProxyN X, Y;

        public void Residuals(in fProxyN p, ref fProxyN r)
        {
            for (int i = 0; i < r.N; i++)
            {
                fProxy x = X[i];
                fProxy model = p[0] + p[1] * x + p[2] * x * x;
                r[i] = model - Y[i];
            }
        }
    }

    public struct DetPolyCurveModelFProxy : IfProxyCurveModel
    {
        public fProxy Eval(fProxy x, in fProxyN p) => p[0] + p[1] * x + p[2] * x * x;
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetLpLadJobFProxy : IJob
    {
        public fProxyMxN A; public fProxyN b, c; public NativeArray<ConstraintSense> senses; public fProxyN x;
        public fProxyMxN ABr; public fProxyN bBr; public fProxyN xBr;
        public fProxyMxN AFn; public fProxyN bFn; public fProxyN xFn;

        public NativeArray<uint> HashOut; // 3 slots

        public void Execute()
        {
            var lpInfo = LP.solve(in A, in b, in c, in senses, ref x, out double lpObj, LPMethod.RevisedSimplex, 0);
            uint h = Hash.hash(in x);
            h = DetHash.Combine(h, lpObj);
            h = DetHash.Combine(h, lpInfo.iterations);
            h = DetHash.Combine(h, (int)lpInfo.status);
            HashOut[0] = h;

            var brInfo = LP.ladBR(in ABr, in bBr, ref xBr, out double brObj, 0);
            h = Hash.hash(in xBr);
            h = DetHash.Combine(h, brObj);
            h = DetHash.Combine(h, brInfo.iterations);
            h = DetHash.Combine(h, (int)brInfo.status);
            HashOut[1] = h;

            var fnInfo = LP.ladFN(in AFn, in bFn, ref xFn, out double fnObj, 0);
            h = Hash.hash(in xFn);
            h = DetHash.Combine(h, fnObj);
            h = DetHash.Combine(h, fnInfo.iterations);
            h = DetHash.Combine(h, (int)fnInfo.status);
            HashOut[2] = h;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetQpJobFProxy : IJob
    {
        public fProxyMxN Q; public fProxyN c; public fProxyMxN A; public fProxyN b;
        public NativeArray<ConstraintSense> senses; public fProxyN xl, xu; public fProxyN x;

        public NativeArray<uint> HashOut; // 1 slot

        public void Execute()
        {
            var info = QP.solve(in Q, in c, in A, in b, in senses, in xl, in xu, ref x, out double obj, 0);
            uint h = Hash.hash(in x);
            h = DetHash.Combine(h, obj);
            h = DetHash.Combine(h, info.iterations);
            h = DetHash.Combine(h, (int)info.status);
            h = DetHash.Combine(h, info.stationarityResidual);
            h = DetHash.Combine(h, info.feasibilityResidual);
            HashOut[0] = h;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetMipJobFProxy : IJob
    {
        public fProxyMxN A; public fProxyN b, c;
        public NativeArray<ConstraintSense> senses; public fProxyN xl, xu;
        public NativeArray<byte> integrality; public fProxyN x;

        public NativeArray<uint> HashOut; // 1 slot

        public void Execute()
        {
            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integrality, ref x, out double obj,
                                  maxNodes: 2000, maxIter: 0, absGap: 0.0, relGap: 0.0);
            uint h = Hash.hash(in x);
            h = DetHash.Combine(h, obj);
            h = DetHash.Combine(h, info.nodes);
            h = DetHash.Combine(h, info.lpIterations);
            h = DetHash.Combine(h, info.dualBound);
            h = DetHash.Combine(h, info.gap);
            h = DetHash.Combine(h, (int)info.status);
            HashOut[0] = h;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetControlJobFProxy : IJob
    {
        public fProxyMxN A, B, Qc, Rc;
        public fProxyMxN K;
        public fProxyMxN Sdare;

        public fProxyKFState kfState;
        public fProxyMxN Hmat, Qkf, Rkf;
        public fProxyN z1, z2;
        public fProxyMxN Kss;

        public fProxyMPCState mpcState;
        public fProxyN x0, reference, u0out;

        public fProxyKFState ukfState;
        public fProxyUKFCache ukfCache;
        public DetLinearKFModelFProxy model;
        public DetLinearKFMeasFProxy meas;
        public fProxyMxN Qukf;
        public fProxyN uZero, zUkf;

        public NativeArray<uint> HashOut; // 6 slots

        public void Execute()
        {
            var lqrInfo = LQR.lqr(in A, in B, in Qc, in Rc, ref K, 0);
            uint h = Hash.hash(in K);
            h = DetHash.Combine(h, lqrInfo.iterations);
            h = DetHash.Combine(h, lqrInfo.residual);
            h = DetHash.Combine(h, (int)lqrInfo.status);
            h = DetHash.Combine(h, lqrInfo.rankDeficient);
            HashOut[0] = h;

            var dareInfo = Riccati.dare(in A, in B, in Qc, in Rc, ref Sdare, 0);
            h = Hash.hash(in Sdare);
            h = DetHash.Combine(h, dareInfo.iterations);
            h = DetHash.Combine(h, dareInfo.residual);
            h = DetHash.Combine(h, (int)dareInfo.status);
            HashOut[1] = h;

            Kalman.predict(ref kfState, in A, in Qkf);
            var kfInfo1 = Kalman.update(ref kfState, in Hmat, in Rkf, in z1);
            Kalman.predict(ref kfState, in A, in Qkf);
            var kfInfo2 = Kalman.update(ref kfState, in Hmat, in Rkf, in z2);
            h = Hash.hash(in kfState.x);
            h = Hash.combine(h, Hash.hash(in kfState.P));
            h = DetHash.Combine(h, kfInfo1.innovationNorm);
            h = DetHash.Combine(h, (int)kfInfo1.status);
            h = DetHash.Combine(h, kfInfo2.innovationNorm);
            h = DetHash.Combine(h, (int)kfInfo2.status);
            HashOut[2] = h;

            var steadyInfo = Kalman.steadyStateGain(in A, in Hmat, in Qkf, in Rkf, ref Kss, 0);
            h = Hash.hash(in Kss);
            h = DetHash.Combine(h, steadyInfo.iterations);
            h = DetHash.Combine(h, steadyInfo.residual);
            h = DetHash.Combine(h, (int)steadyInfo.status);
            HashOut[3] = h;

            var mpcInfo = MPC.solve(ref mpcState, in x0, in reference, ref u0out, 0);
            h = Hash.hash(in u0out);
            h = DetHash.Combine(h, (int)mpcInfo.status);
            h = DetHash.Combine(h, mpcInfo.iterations);
            h = DetHash.Combine(h, mpcInfo.activeSetChanges);
            h = DetHash.Combine(h, mpcInfo.maxSlackViolation);
            h = DetHash.Combine(h, mpcInfo.objective);
            HashOut[4] = h;

            Kalman.ukfPredict(ref ukfState, ref ukfCache, in model, in uZero, in Qukf);
            var ukfInfo = Kalman.ukfUpdate(ref ukfState, ref ukfCache, in meas, in Rkf, in zUkf);
            h = Hash.hash(in ukfState.x);
            h = Hash.combine(h, Hash.hash(in ukfState.P));
            h = DetHash.Combine(h, ukfInfo.innovationNorm);
            h = DetHash.Combine(h, (int)ukfInfo.status);
            HashOut[5] = h;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetNlsOptimizeJobFProxy : IJob
    {
        public DetPolyResidualFProxy residual;
        public fProxyN pNls;
        public int MData;

        public fProxyN xdata, ydata;
        public DetPolyCurveModelFProxy curveModel;
        public fProxyN pCurve;

        public fProxyMxN AIrls; public fProxyN bIrls; public fProxyN xIrls;

        public NativeArray<uint> HashOut; // 3 slots

        public void Execute()
        {
            var nlsInfo = Optimize.nlsSolve(ref residual, ref pNls, MData);
            uint h = Hash.hash(in pNls);
            h = DetHash.Combine(h, nlsInfo.objective);
            h = DetHash.Combine(h, nlsInfo.residualNorm);
            h = DetHash.Combine(h, nlsInfo.gradientNorm);
            h = DetHash.Combine(h, nlsInfo.iterations);
            h = DetHash.Combine(h, (int)nlsInfo.status);
            HashOut[0] = h;

            var curveInfo = Optimize.curveFit(in xdata, in ydata, ref curveModel, ref pCurve);
            h = Hash.hash(in pCurve);
            h = DetHash.Combine(h, curveInfo.objective);
            h = DetHash.Combine(h, curveInfo.residualNorm);
            h = DetHash.Combine(h, curveInfo.gradientNorm);
            h = DetHash.Combine(h, curveInfo.iterations);
            h = DetHash.Combine(h, (int)curveInfo.status);
            HashOut[1] = h;

            var irlsInfo = Optimize.ladIRLS(in AIrls, in bIrls, ref xIrls);
            h = Hash.hash(in xIrls);
            h = DetHash.Combine(h, irlsInfo.iterations);
            h = DetHash.Combine(h, (int)irlsInfo.status);
            HashOut[2] = h;
        }
    }

    public static partial class DeterminismOptimize
    {
        public static (string id, uint hash)[] Case_LpLadFProxy()
        {
            var rng = new Random(2654435761u ^ 0x0011u);

            const int m = 20, n = 30;
            var A = new fProxyMxN(m, n, Allocator.Persistent);
            for (int r = 0; r < m; r++) for (int cc = 0; cc < n; cc++) A[r, cc] = rng.NextFProxy(0f, 1f);
            var x0 = new fProxyN(n, Allocator.Persistent); for (int i = 0; i < n; i++) x0[i] = rng.NextFProxy(0f, 1f);
            var Ax0 = new fProxyN(m, Allocator.Persistent); Blas.dot(in A, in x0, ref Ax0);
            var b = new fProxyN(m, Allocator.Persistent); for (int i = 0; i < m; i++) b[i] = Ax0[i] + rng.NextFProxy(0.1f, 1f);
            var c = new fProxyN(n, Allocator.Persistent); for (int i = 0; i < n; i++) c[i] = rng.NextFProxy(-1f, 1f);
            var senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
            for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;
            var x = new fProxyN(n, Allocator.Persistent, true);

            const int mBr = 120, nBr = 5;
            var ABr = new fProxyMxN(mBr, nBr, Allocator.Persistent);
            for (int r = 0; r < mBr; r++) for (int cc = 0; cc < nBr; cc++) ABr[r, cc] = (cc == 0) ? (fProxy)1 : rng.NextFProxy(-1f, 1f);
            var xtBr = new fProxyN(nBr, Allocator.Persistent); for (int i = 0; i < nBr; i++) xtBr[i] = rng.NextFProxy(-1f, 1f);
            var AxtBr = new fProxyN(mBr, Allocator.Persistent); Blas.dot(in ABr, in xtBr, ref AxtBr);
            var bBr = new fProxyN(mBr, Allocator.Persistent);
            for (int i = 0; i < mBr; i++)
            {
                fProxy v = AxtBr[i] + rng.NextFProxy(-(fProxy)0.05, (fProxy)0.05);
                if (i % 10 == 0) v += (fProxy)5;
                bBr[i] = v;
            }
            var xBr = new fProxyN(nBr, Allocator.Persistent, true);

            const int mFn = 600, nFn = 5;
            var AFn = new fProxyMxN(mFn, nFn, Allocator.Persistent);
            for (int r = 0; r < mFn; r++) for (int cc = 0; cc < nFn; cc++) AFn[r, cc] = (cc == 0) ? (fProxy)1 : rng.NextFProxy(-1f, 1f);
            var xtFn = new fProxyN(nFn, Allocator.Persistent); for (int i = 0; i < nFn; i++) xtFn[i] = rng.NextFProxy(-1f, 1f);
            var AxtFn = new fProxyN(mFn, Allocator.Persistent); Blas.dot(in AFn, in xtFn, ref AxtFn);
            var bFn = new fProxyN(mFn, Allocator.Persistent);
            for (int i = 0; i < mFn; i++)
            {
                fProxy v = AxtFn[i] + rng.NextFProxy(-(fProxy)0.05, (fProxy)0.05);
                if (i % 10 == 0) v += (fProxy)5;
                bFn[i] = v;
            }
            var xFn = new fProxyN(nFn, Allocator.Persistent, true);

            var hashOut = new NativeArray<uint>(3, Allocator.Persistent);
            var job = new DetLpLadJobFProxy
            {
                A = A, b = b, c = c, senses = senses, x = x,
                ABr = ABr, bBr = bBr, xBr = xBr,
                AFn = AFn, bFn = bFn, xFn = xFn, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("lp-lad/lp.solve.fProxy.m20n30", hashOut[0]),
                ("lp-lad/lad.ladBR.fProxy.120x5", hashOut[1]),
                ("lp-lad/lad.ladFN.fProxy.600x5", hashOut[2]),
            };
            hashOut.Dispose(); senses.Dispose();
            A.Dispose(); x0.Dispose(); Ax0.Dispose(); b.Dispose(); c.Dispose(); x.Dispose();
            ABr.Dispose(); xtBr.Dispose(); AxtBr.Dispose(); bBr.Dispose(); xBr.Dispose();
            AFn.Dispose(); xtFn.Dispose(); AxtFn.Dispose(); bFn.Dispose(); xFn.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_QpFProxy()
        {
            var rng = new Random(2654435761u ^ 0x0012u);

            const int n = 24, m = 6;
            var Mfac = new fProxyMxN(n, n, Allocator.Persistent);
            for (int r = 0; r < n; r++) for (int cc = 0; cc < n; cc++) Mfac[r, cc] = rng.NextFProxy(-1f, 1f);
            var Q = new fProxyMxN(n, n, Allocator.Persistent);
            Blas.dot(in Mfac, in Mfac, ref Q, transposeA: true);
            for (int d = 0; d < n; d++) Q[d, d] += (fProxy)n;

            var c = new fProxyN(n, Allocator.Persistent); for (int i = 0; i < n; i++) c[i] = rng.NextFProxy(-1f, 1f);
            var A = new fProxyMxN(m, n, Allocator.Persistent);
            for (int r = 0; r < m; r++) for (int cc = 0; cc < n; cc++) A[r, cc] = rng.NextFProxy(0f, 1f);
            var x0 = new fProxyN(n, Allocator.Persistent); for (int i = 0; i < n; i++) x0[i] = rng.NextFProxy(0.2f, 0.8f);
            var Ax0 = new fProxyN(m, Allocator.Persistent); Blas.dot(in A, in x0, ref Ax0);
            var b = new fProxyN(m, Allocator.Persistent); for (int i = 0; i < m; i++) b[i] = Ax0[i] + rng.NextFProxy(0.1f, 1f);
            var senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
            for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;
            var xl = new fProxyN(n, Allocator.Persistent);
            var xu = GenerateOP.fProxyVec(n, (fProxy)3, Allocator.Persistent);
            var x = new fProxyN(n, Allocator.Persistent, true);

            var hashOut = new NativeArray<uint>(1, Allocator.Persistent);
            var job = new DetQpJobFProxy { Q = Q, c = c, A = A, b = b, senses = senses, xl = xl, xu = xu, x = x, HashOut = hashOut };
            job.Run();

            var result = new[] { ("qp/qp.solve.fProxy.n24m6", hashOut[0]) };
            hashOut.Dispose(); senses.Dispose();
            Mfac.Dispose(); Q.Dispose(); c.Dispose(); A.Dispose(); x0.Dispose(); Ax0.Dispose(); b.Dispose();
            xl.Dispose(); xu.Dispose(); x.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_MipFProxy()
        {
            // stein9-class set-cover instance: 9 binaries, 13 rows (12 triples covering each pair,
            // + a "select >= 4 of 9" cut row), objective = minimize count. Known optimum 5.
            const int n = 9, m = 13;
            var A = new fProxyMxN(m, n, Allocator.Persistent);
            int[,] triples =
            {
                {0,1,2},{0,3,4},{0,5,6},{0,7,8},
                {1,3,5},{1,4,7},{1,6,8},
                {2,3,7},{2,4,6},
                {2,5,8},{3,6,8},{4,5,8},
            };
            for (int row = 0; row < 12; row++)
            {
                A[row, triples[row, 0]] = (fProxy)1;
                A[row, triples[row, 1]] = (fProxy)1;
                A[row, triples[row, 2]] = (fProxy)1;
            }
            for (int j = 0; j < n; j++) A[12, j] = (fProxy)1;

            var b = new fProxyN(m, Allocator.Persistent);
            for (int row = 0; row < 12; row++) b[row] = (fProxy)1;
            b[12] = (fProxy)4;

            var senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
            for (int i = 0; i < m; i++) senses[i] = ConstraintSense.GreaterEqual;

            var c = GenerateOP.fProxyVec(n, (fProxy)1, Allocator.Persistent);
            var xl = new fProxyN(n, Allocator.Persistent);
            var xu = GenerateOP.fProxyVec(n, (fProxy)1, Allocator.Persistent);
            var integrality = new NativeArray<byte>(n, Allocator.Persistent);
            for (int i = 0; i < n; i++) integrality[i] = 1;
            var x = new fProxyN(n, Allocator.Persistent, true);

            var hashOut = new NativeArray<uint>(1, Allocator.Persistent);
            var job = new DetMipJobFProxy { A = A, b = b, c = c, senses = senses, xl = xl, xu = xu, integrality = integrality, x = x, HashOut = hashOut };
            job.Run();

            var result = new[] { ("mip/mip.solve.fProxy.stein9", hashOut[0]) };
            hashOut.Dispose(); senses.Dispose(); integrality.Dispose();
            A.Dispose(); b.Dispose(); c.Dispose(); xl.Dispose(); xu.Dispose(); x.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_ControlFProxy()
        {
            const int n = 4, m = 2;

            var A = new fProxyMxN(n, n, Allocator.Persistent);
            A[0, 0] = (fProxy)0.9; A[0, 1] = (fProxy)0.1; A[0, 2] = (fProxy)0; A[0, 3] = (fProxy)0;
            A[1, 0] = (fProxy)0; A[1, 1] = (fProxy)0.9; A[1, 2] = (fProxy)0.1; A[1, 3] = (fProxy)0;
            A[2, 0] = (fProxy)0; A[2, 1] = (fProxy)0; A[2, 2] = (fProxy)0.9; A[2, 3] = (fProxy)0.1;
            A[3, 0] = (fProxy)0.05; A[3, 1] = (fProxy)0; A[3, 2] = (fProxy)0; A[3, 3] = (fProxy)0.85;

            var B = new fProxyMxN(n, m, Allocator.Persistent);
            B[0, 0] = (fProxy)0.5; B[0, 1] = (fProxy)0;
            B[1, 0] = (fProxy)0; B[1, 1] = (fProxy)0.5;
            B[2, 0] = (fProxy)0.2; B[2, 1] = (fProxy)0.1;
            B[3, 0] = (fProxy)0.1; B[3, 1] = (fProxy)0.2;

            var Qc = new fProxyMxN(n, n, Allocator.Persistent); for (int i = 0; i < n; i++) Qc[i, i] = (fProxy)1;
            var Rc = new fProxyMxN(m, m, Allocator.Persistent); for (int i = 0; i < m; i++) Rc[i, i] = (fProxy)1;
            var K = new fProxyMxN(m, n, Allocator.Persistent);
            var Sdare = new fProxyMxN(n, n, Allocator.Persistent);

            var kfState = new fProxyKFState(n, m, Allocator.Persistent);
            for (int i = 0; i < n; i++) kfState.x[i] = (fProxy)1;
            for (int i = 0; i < n; i++) kfState.P[i, i] = (fProxy)1;

            var Hmat = new fProxyMxN(m, n, Allocator.Persistent);
            Hmat[0, 0] = (fProxy)1; Hmat[1, 2] = (fProxy)1;
            var Qkf = new fProxyMxN(n, n, Allocator.Persistent); for (int i = 0; i < n; i++) Qkf[i, i] = (fProxy)0.01;
            var Rkf = new fProxyMxN(m, m, Allocator.Persistent); for (int i = 0; i < m; i++) Rkf[i, i] = (fProxy)0.1;
            var z1 = new fProxyN(m, Allocator.Persistent); z1[0] = (fProxy)0.9; z1[1] = (fProxy)0.05;
            var z2 = new fProxyN(m, Allocator.Persistent); z2[0] = (fProxy)0.8; z2[1] = (fProxy)0.1;
            var Kss = new fProxyMxN(n, m, Allocator.Persistent);

            var mpcUlo = GenerateOP.fProxyVec(m, (fProxy)(-1), Allocator.Persistent);
            var mpcUhi = GenerateOP.fProxyVec(m, (fProxy)1, Allocator.Persistent);
            var mpcState = new fProxyMPCState(n, m, 5, Allocator.Persistent, in A, in B, in Qc, in Rc,
                                              mpcUlo, mpcUhi);
            var x0 = new fProxyN(n, Allocator.Persistent); x0[0] = (fProxy)1;
            var reference = new fProxyN(n, Allocator.Persistent);
            var u0out = new fProxyN(m, Allocator.Persistent, true);

            var ukfState = new fProxyKFState(n, m, Allocator.Persistent);
            for (int i = 0; i < n; i++) ukfState.x[i] = (fProxy)1;
            for (int i = 0; i < n; i++) ukfState.P[i, i] = (fProxy)1;
            var ukfCache = new fProxyUKFCache(n, Allocator.Persistent);
            var modelScratch = new fProxyN(n, Allocator.Persistent);
            var model = new DetLinearKFModelFProxy { A = A, B = B, Scratch = modelScratch };
            var meas = new DetLinearKFMeasFProxy { Hmat = Hmat };
            var Qukf = Qkf;
            var uZero = new fProxyN(m, Allocator.Persistent);
            var zUkf = new fProxyN(m, Allocator.Persistent); zUkf[0] = (fProxy)0.9; zUkf[1] = (fProxy)0.05;

            var hashOut = new NativeArray<uint>(6, Allocator.Persistent);
            var job = new DetControlJobFProxy
            {
                A = A, B = B, Qc = Qc, Rc = Rc, K = K, Sdare = Sdare,
                kfState = kfState, Hmat = Hmat, Qkf = Qkf, Rkf = Rkf, z1 = z1, z2 = z2, Kss = Kss,
                mpcState = mpcState, x0 = x0, reference = reference, u0out = u0out,
                ukfState = ukfState, ukfCache = ukfCache, model = model, meas = meas, Qukf = Qukf, uZero = uZero, zUkf = zUkf,
                HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("control/lqr.lqr.fProxy.n4m2", hashOut[0]),
                ("control/riccati.dare.fProxy.n4m2", hashOut[1]),
                ("control/kalman.predictUpdate.fProxy.n4m2", hashOut[2]),
                ("control/kalman.steadyStateGain.fProxy.n4m2", hashOut[3]),
                ("control/mpc.solve.fProxy.n4m2.N5", hashOut[4]),
                ("control/kalman.ukfLinear.fProxy.n4m2", hashOut[5]),
            };
            hashOut.Dispose();
            A.Dispose(); B.Dispose(); Qc.Dispose(); Rc.Dispose(); K.Dispose(); Sdare.Dispose();
            kfState.Dispose(); Hmat.Dispose(); Qkf.Dispose(); Rkf.Dispose(); z1.Dispose(); z2.Dispose(); Kss.Dispose();
            mpcState.Dispose(); mpcUlo.Dispose(); mpcUhi.Dispose(); x0.Dispose(); reference.Dispose(); u0out.Dispose();
            ukfState.Dispose(); ukfCache.Dispose(); modelScratch.Dispose(); uZero.Dispose(); zUkf.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_NlsOptimizeFProxy()
        {
            const int mData = 12;

            var xdata = new fProxyN(mData, Allocator.Persistent);
            var ydata = new fProxyN(mData, Allocator.Persistent);
            for (int i = 0; i < mData; i++) xdata[i] = (fProxy)(i - mData / 2);
            for (int i = 0; i < mData; i++)
            {
                fProxy xv = xdata[i];
                ydata[i] = (fProxy)1 + (fProxy)0.5 * xv + (fProxy)0.25 * xv * xv;
            }

            var residual = new DetPolyResidualFProxy { X = xdata, Y = ydata };
            var pNls = new fProxyN(3, Allocator.Persistent);

            var curveModel = new DetPolyCurveModelFProxy();
            var pCurve = new fProxyN(3, Allocator.Persistent);

            const int nCoef = 3;
            var AIrls = new fProxyMxN(mData, nCoef, Allocator.Persistent);
            for (int i = 0; i < mData; i++) { AIrls[i, 0] = (fProxy)1; AIrls[i, 1] = xdata[i]; AIrls[i, 2] = xdata[i] * xdata[i]; }
            var bIrls = new fProxyN(mData, Allocator.Persistent);
            for (int i = 0; i < mData; i++) bIrls[i] = ydata[i] + ((i == 3) ? (fProxy)2 : (fProxy)0); // one outlier row
            var xIrls = new fProxyN(nCoef, Allocator.Persistent, true);

            var hashOut = new NativeArray<uint>(3, Allocator.Persistent);
            var job = new DetNlsOptimizeJobFProxy
            {
                residual = residual, pNls = pNls, MData = mData,
                xdata = xdata, ydata = ydata, curveModel = curveModel, pCurve = pCurve,
                AIrls = AIrls, bIrls = bIrls, xIrls = xIrls, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("nls-optimize/nlsSolve.fProxy.poly.n12", hashOut[0]),
                ("nls-optimize/curveFit.fProxy.poly.n12", hashOut[1]),
                ("nls-optimize/ladIRLS.fProxy.n12", hashOut[2]),
            };
            hashOut.Dispose();
            xdata.Dispose(); ydata.Dispose(); pNls.Dispose(); pCurve.Dispose();
            AIrls.Dispose(); bIrls.Dispose(); xIrls.Dispose();
            return result;
        }
    }
}
