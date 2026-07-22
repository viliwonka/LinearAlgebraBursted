using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of the determinism conformance harness's "direct" groups (dense
    // linear algebra: hash-selftest, blas-dense, elementwise-core, norms, qr-family, lu, cholesky).
    // The dtype-agnostic driver (group registration, root folding, report writing) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/DeterminismReport.cs. See
    // docs/dev/spec-determinism-conformance-harness.md for the frozen op/group/root hash contract.
    //
    // Every job below hashes its own outputs INSIDE Execute() (never after readback) and runs via
    // .Run() from its Case_*FProxy() builder, which is plain (non-Burst) code that seeds fixed
    // literal/RNG inputs into arena buffers, executes the job once, and returns (id, hash) pairs in
    // registration order.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetHashSelfTestJobFProxy : IJob
    {
        public fProxyN vec5;
        public fProxyMxN mat43;
        // rowHashes/colHashes' dest is always uint regardless of A's element type (see Hash/DEVLOG.md)
        // -- spelled via the choose marker, like Hash.fProxy.cs's own rowHashes/colHashes wrappers,
        // so the raw (un-substituted) template still compiles against the proxyStructs shim world.
        public /*+choose[uintN|uintN]*/iProxyN/*-choose*/ rowH;
        public /*+choose[uintN|uintN]*/iProxyN/*-choose*/ colH;
        public NativeArray<uint> HashOut; // [0]=vec, [1]=rowcolhash, [2]=combine-chain

        public void Execute()
        {
            uint vecHash = Hash.hash(in vec5);
            Hash.rowHashes(in mat43, ref rowH);
            Hash.colHashes(in mat43, ref colH);
            uint rc = Hash.hash(in rowH);
            rc = Hash.combine(rc, Hash.hash(in colH));
            uint chain = Hash.combine(0x9E3779B9u, vecHash);
            chain = Hash.combine(chain, rc);

            HashOut[0] = vecHash;
            HashOut[1] = rc;
            HashOut[2] = chain;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetBlasDenseJobFProxy : IJob
    {
        public fProxyN vecA, vecB;                 // dot(vec,vec), length 64
        public fProxyMxN P;                         // 53x37
        public fProxyN xVec37, yVec53;               // matvec/vecmat operands
        public fProxyN matvecOut, vecmatOut;
        public fProxyMxN gA, gB, gC;                 // 37x37 square, GEMM transpose combos
        public fProxyMxN dotSymC;                    // 37x37
        public fProxyN outerU, outerV;                // 53, 37
        public fProxyMxN outerC;                      // 53x37
        public fProxyMxN transT;                      // 37x53
        public NativeArray<uint> HashOut;             // 10 slots

        public void Execute()
        {
            fProxy dotVV = Blas.dot(vecA, vecB);
            HashOut[0] = DetHash.Combine(0u, dotVV);

            Blas.dot(in P, in xVec37, ref matvecOut);
            HashOut[1] = Hash.hash(in matvecOut);

            Blas.dot(in yVec53, in P, ref vecmatOut);
            HashOut[2] = Hash.hash(in vecmatOut);

            Blas.dot(in gA, in gB, ref gC, transposeA: false, transposeB: false);
            HashOut[3] = Hash.hash(in gC);
            Blas.dot(in gA, in gB, ref gC, transposeA: true, transposeB: false);
            HashOut[4] = Hash.hash(in gC);
            Blas.dot(in gA, in gB, ref gC, transposeA: false, transposeB: true);
            HashOut[5] = Hash.hash(in gC);
            Blas.dot(in gA, in gB, ref gC, transposeA: true, transposeB: true);
            HashOut[6] = Hash.hash(in gC);

            Blas.dotSym(in P, in P, ref dotSymC);
            HashOut[7] = Hash.hash(in dotSymC);

            Blas.outerDot(in outerU, in outerV, ref outerC);
            HashOut[8] = Hash.hash(in outerC);

            Blas.trans(in P, ref transT);
            HashOut[9] = Hash.hash(in transT);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetElementwiseCoreJobFProxy : IJob
    {
        public fProxyN baseVec, otherVec, otherVec2, nonNegVec;
        public fProxyN scratch;
        public NativeArray<uint> HashOut; // 12 slots: abs,sign,sqrt,clamp,lerp,min,max,mad,floor,ceil,round,saturate

        void CopyBaseIntoScratch()
        {
            for (int i = 0; i < baseVec.N; i++) scratch[i] = baseVec[i];
        }

        public void Execute()
        {
            CopyBaseIntoScratch(); scratch.absInPlace();
            HashOut[0] = Hash.hash(in scratch);

            CopyBaseIntoScratch(); scratch.signInPlace();
            HashOut[1] = Hash.hash(in scratch);

            for (int i = 0; i < nonNegVec.N; i++) scratch[i] = nonNegVec[i];
            scratch.sqrtInPlace();
            HashOut[2] = Hash.hash(in scratch);

            CopyBaseIntoScratch(); scratch.clampInPlace((fProxy)(-1), (fProxy)1);
            HashOut[3] = Hash.hash(in scratch);

            CopyBaseIntoScratch(); scratch.lerpInPlace(otherVec, (fProxy)0.5);
            HashOut[4] = Hash.hash(in scratch);

            CopyBaseIntoScratch(); scratch.minInPlace(otherVec);
            HashOut[5] = Hash.hash(in scratch);

            CopyBaseIntoScratch(); scratch.maxInPlace(otherVec);
            HashOut[6] = Hash.hash(in scratch);

            CopyBaseIntoScratch(); scratch.madInPlace(otherVec, otherVec2);
            HashOut[7] = Hash.hash(in scratch);

            CopyBaseIntoScratch(); scratch.floorInPlace();
            HashOut[8] = Hash.hash(in scratch);

            CopyBaseIntoScratch(); scratch.ceilInPlace();
            HashOut[9] = Hash.hash(in scratch);

            CopyBaseIntoScratch(); scratch.roundInPlace();
            HashOut[10] = Hash.hash(in scratch);

            CopyBaseIntoScratch(); scratch.saturateInPlace();
            HashOut[11] = Hash.hash(in scratch);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetNormsJobFProxy : IJob
    {
        public fProxyN vecX;          // length 37
        public fProxyMxN A;            // 53x37
        public fProxyN normalizeVec;   // copy of vecX, mutated in place
        public fProxyMxN normColsMat;  // copy of A, mutated in place
        public NativeArray<uint> HashOut; // 7 slots

        // Norms.LInf / Norms.normalizeRows are deliberately NOT exercised here: both reach
        // UnsafeOP.maxAbs, whose WideOP AVX max/abs intrinsics have no [IgnoreWarning(1305)] and
        // crash the Burst compiler under FloatMode.Strict (verified; see this folder's DEVLOG.md).
        // Norms.matrixLInf/normalizeColumns take a different (sumAbs/scalar math.max) code path and
        // are unaffected.
        public void Execute()
        {
            HashOut[0] = DetHash.Combine(0u, Norms.L1(in vecX));
            HashOut[1] = DetHash.Combine(0u, Norms.L2(in vecX));
            HashOut[2] = DetHash.Combine(0u, Norms.matrixL1(in A));
            HashOut[3] = DetHash.Combine(0u, Norms.matrixL2(in A));
            HashOut[4] = DetHash.Combine(0u, Norms.matrixLInf(in A));

            Norms.normalize(in normalizeVec, Norm.L2);
            HashOut[5] = Hash.hash(in normalizeVec);

            Norms.normalizeColumns(ref normColsMat, Norm.L2);
            HashOut[6] = Hash.hash(in normColsMat);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetQrFamilyJobFProxy : IJob
    {
        // Tall (53x37): QR, QRCP
        public fProxyMxN Atall;
        public fProxyMxN Q1, R1;
        public fProxyN bTall, xTall1;
        public fProxyMxN Q2, R2;
        public Pivot P1;
        public fProxyN bTall2, xTall2;
        public fProxyMxN AforMinNorm;
        public fProxyN bTall3, xTall3;

        // Wide (37x53): LQ, LQRP
        public fProxyMxN Awide;
        public fProxyMxN L1, Qw1;
        public fProxyN bWide1, xWide1;
        public fProxyMxN L2, Qw2;
        public Pivot P2;
        public fProxyMxN Bwide, Xwide;

        public NativeArray<uint> HashOut; // 9 slots

        public void Execute()
        {
            var qrInfo = QR.decomp(in Atall, ref Q1, ref R1);
            uint h = Hash.hash(in Q1);
            h = Hash.combine(h, Hash.hash(in R1));
            h = DetHash.Combine(h, (int)qrInfo.status);
            HashOut[0] = h;

            var qrSolveInfo = QR.decompSolve(ref Q1, ref R1, ref bTall, ref xTall1);
            h = Hash.hash(in xTall1);
            h = DetHash.Combine(h, (int)qrSolveInfo.status);
            HashOut[1] = h;

            var qrcpDecompInfo = QRCP.decomp(in Atall, ref Q2, ref R2, ref P1);
            h = Hash.hash(in Q2);
            h = Hash.combine(h, Hash.hash(in R2));
            h = DetHash.CombinePivot(h, in P1);
            h = DetHash.Combine(h, (int)qrcpDecompInfo.status);
            HashOut[2] = h;

            var qrcpSolveInfo = QRCP.decompSolve(ref Q2, ref R2, in P1, ref bTall2, ref xTall2);
            h = Hash.hash(in xTall2);
            h = DetHash.Combine(h, (int)qrcpSolveInfo.status);
            h = DetHash.Combine(h, qrcpSolveInfo.rank);
            HashOut[3] = h;

            var qrcpMinNormInfo = QRCP.minNormSolveInPlace(ref AforMinNorm, ref bTall3, ref xTall3);
            h = Hash.hash(in xTall3);
            h = DetHash.Combine(h, (int)qrcpMinNormInfo.status);
            h = DetHash.Combine(h, qrcpMinNormInfo.rank);
            HashOut[4] = h;

            var lqInfo = LQ.decomp(in Awide, ref L1, ref Qw1);
            h = Hash.hash(in L1);
            h = Hash.combine(h, Hash.hash(in Qw1));
            h = DetHash.Combine(h, (int)lqInfo.status);
            HashOut[5] = h;

            var lqSolveInfo = LQ.minNormSolve(in Awide, in bWide1, ref xWide1);
            h = Hash.hash(in xWide1);
            h = DetHash.Combine(h, (int)lqSolveInfo.status);
            HashOut[6] = h;

            var lqrpDecompInfo = LQRP.decomp(in Awide, ref L2, ref Qw2, ref P2);
            h = Hash.hash(in L2);
            h = Hash.combine(h, Hash.hash(in Qw2));
            h = DetHash.CombinePivot(h, in P2);
            h = DetHash.Combine(h, (int)lqrpDecompInfo.status);
            HashOut[7] = h;

            var lqrpSolveInfo = LQRP.minNormDecompSolve(ref L2, ref Qw2, in P2, ref Bwide, ref Xwide);
            h = Hash.hash(in Xwide);
            h = DetHash.Combine(h, (int)lqrpSolveInfo.status);
            h = DetHash.Combine(h, lqrpSolveInfo.rank);
            HashOut[8] = h;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetLuJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN L, U;
        public Pivot P1;
        public fProxyMxN LUPacked; // copy of A, destroyed by decompInPlace
        public Pivot P2;
        public fProxyN bSolve, bSolveTransA;

        public NativeArray<uint> HashOut; // 4 slots

        public void Execute()
        {
            var decompInfo = LU.decomp(in A, ref L, ref U, ref P1);
            uint h = Hash.hash(in L);
            h = Hash.combine(h, Hash.hash(in U));
            h = DetHash.CombinePivot(h, in P1);
            h = DetHash.Combine(h, (int)decompInfo.status);
            HashOut[0] = h;

            var inPlaceInfo = LU.decompInPlace(ref LUPacked, ref P2);
            h = Hash.hash(in LUPacked);
            h = DetHash.CombinePivot(h, in P2);
            h = DetHash.Combine(h, (int)inPlaceInfo.status);
            HashOut[1] = h;

            var solveInfo = LU.decompSolve(ref L, ref U, in P1, ref bSolve);
            h = Hash.hash(in bSolve);
            h = DetHash.Combine(h, (int)solveInfo.status);
            HashOut[2] = h;

            var solveTransAInfo = LU.decompSolveTransA(ref LUPacked, in P2, ref bSolveTransA);
            h = Hash.hash(in bSolveTransA);
            h = DetHash.Combine(h, (int)solveTransAInfo.status);
            HashOut[3] = h;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetCholeskyJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN L;
        public fProxyN bSolve;
        public fProxyMxN Lp;
        public Pivot P1;
        public fProxyMxN AtoL;
        public Pivot P2;
        public fProxyN bSolveInPlace;

        public NativeArray<uint> HashOut; // 4 slots

        public void Execute()
        {
            var decompInfo = CHO.decomp(in A, ref L);
            uint h = Hash.hash(in L);
            h = DetHash.Combine(h, (int)decompInfo.status);
            HashOut[0] = h;

            var solveInfo = CHO.decompSolve(ref L, ref bSolve);
            h = Hash.hash(in bSolve);
            h = DetHash.Combine(h, (int)solveInfo.status);
            HashOut[1] = h;

            var chopDecompInfo = CHOP.decomp(in A, ref Lp, ref P1);
            h = Hash.hash(in Lp);
            h = DetHash.CombinePivot(h, in P1);
            h = DetHash.Combine(h, (int)chopDecompInfo.status);
            h = DetHash.Combine(h, chopDecompInfo.rank);
            HashOut[2] = h;

            var chopSolveInfo = CHOP.solveInPlace(ref AtoL, ref P2, ref bSolveInPlace);
            h = Hash.hash(in AtoL);
            h = DetHash.CombinePivot(h, in P2);
            h = Hash.hash(in bSolveInPlace);
            h = DetHash.Combine(h, (int)chopSolveInfo.status);
            h = DetHash.Combine(h, chopSolveInfo.rank);
            HashOut[3] = h;
        }
    }

    public static partial class DeterminismDirect
    {
        public static (string id, uint hash)[] Case_HashSelfTestFProxy()
        {
            var arena = new Arena(Allocator.Persistent);

            var vec5 = arena.fProxyVec(5);
            vec5[0] = (fProxy)1; vec5[1] = (fProxy)(-2.5); vec5[2] = (fProxy)0;
            vec5[3] = -(fProxy)0; vec5[4] = fProxy.NaN;

            var mat43 = arena.fProxyMat(4, 3);
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 3; c++)
                    mat43[r, c] = (fProxy)(r * 3 + c) - (fProxy)6;

            var rowH = arena./*+choose[uintVec|uintVec]*/iProxyVec/*-choose*/(4);
            var colH = arena./*+choose[uintVec|uintVec]*/iProxyVec/*-choose*/(3);
            var hashOut = new NativeArray<uint>(3, Allocator.Persistent);

            var job = new DetHashSelfTestJobFProxy { vec5 = vec5, mat43 = mat43, rowH = rowH, colH = colH, HashOut = hashOut };
            job.Run();

            var result = new[]
            {
                ("hash-selftest/vec.fProxy.n5", hashOut[0]),
                ("hash-selftest/rowcolhash.fProxy.4x3", hashOut[1]),
                ("hash-selftest/combine-chain.fProxy", hashOut[2]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_BlasDenseFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2654435761u ^ 0x0002u);

            var vecA = arena.fProxyVec(64); var vecB = arena.fProxyVec(64);
            for (int i = 0; i < 64; i++) { vecA[i] = rng.NextFProxy(-1f, 1f); vecB[i] = rng.NextFProxy(-1f, 1f); }

            var P = arena.fProxyMat(53, 37);
            for (int r = 0; r < 53; r++) for (int c = 0; c < 37; c++) P[r, c] = rng.NextFProxy(-1f, 1f);

            var xVec37 = arena.fProxyVec(37); for (int i = 0; i < 37; i++) xVec37[i] = rng.NextFProxy(-1f, 1f);
            var yVec53 = arena.fProxyVec(53); for (int i = 0; i < 53; i++) yVec53[i] = rng.NextFProxy(-1f, 1f);
            var matvecOut = arena.fProxyVec(53);
            var vecmatOut = arena.fProxyVec(37);

            var gA = arena.fProxyMat(37, 37); var gB = arena.fProxyMat(37, 37); var gC = arena.fProxyMat(37, 37);
            for (int r = 0; r < 37; r++) for (int c = 0; c < 37; c++) { gA[r, c] = rng.NextFProxy(-1f, 1f); gB[r, c] = rng.NextFProxy(-1f, 1f); }

            var dotSymC = arena.fProxyMat(37, 37);
            var outerU = arena.fProxyVec(53); for (int i = 0; i < 53; i++) outerU[i] = rng.NextFProxy(-1f, 1f);
            var outerV = arena.fProxyVec(37); for (int i = 0; i < 37; i++) outerV[i] = rng.NextFProxy(-1f, 1f);
            var outerC = arena.fProxyMat(53, 37);
            var transT = arena.fProxyMat(37, 53);

            var hashOut = new NativeArray<uint>(10, Allocator.Persistent);
            var job = new DetBlasDenseJobFProxy
            {
                vecA = vecA, vecB = vecB, P = P, xVec37 = xVec37, yVec53 = yVec53,
                matvecOut = matvecOut, vecmatOut = vecmatOut,
                gA = gA, gB = gB, gC = gC, dotSymC = dotSymC,
                outerU = outerU, outerV = outerV, outerC = outerC, transT = transT,
                HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("blas-dense/dot.vv.fProxy.n64", hashOut[0]),
                ("blas-dense/dot.matvec.fProxy.53x37", hashOut[1]),
                ("blas-dense/dot.vecmat.fProxy.53x37", hashOut[2]),
                ("blas-dense/gemm.nn.fProxy.37x37", hashOut[3]),
                ("blas-dense/gemm.tn.fProxy.37x37", hashOut[4]),
                ("blas-dense/gemm.nt.fProxy.37x37", hashOut[5]),
                ("blas-dense/gemm.tt.fProxy.37x37", hashOut[6]),
                ("blas-dense/dotSym.fProxy.37x37", hashOut[7]),
                ("blas-dense/outerDot.fProxy.53x37", hashOut[8]),
                ("blas-dense/trans.fProxy.53x37", hashOut[9]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_ElementwiseCoreFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2654435761u ^ 0x0003u);

            const int n = 1000;
            var baseVec = arena.fProxyVec(n);
            var otherVec = arena.fProxyVec(n);
            var otherVec2 = arena.fProxyVec(n);
            var nonNegVec = arena.fProxyVec(n);
            var scratch = arena.fProxyVec(n);

            for (int i = 0; i < n; i++)
            {
                baseVec[i] = rng.NextFProxy(-5f, 5f);
                otherVec[i] = rng.NextFProxy(-5f, 5f);
                otherVec2[i] = rng.NextFProxy(-2f, 2f);
                nonNegVec[i] = math.abs(rng.NextFProxy(-5f, 5f));
            }
            // Fixed special-value overrides: denormal, +0, -0, a large negative, a large positive.
            baseVec[0] = (fProxy)1e-40; baseVec[1] = (fProxy)0; baseVec[2] = -(fProxy)0;
            baseVec[3] = (fProxy)(-1e6); baseVec[4] = (fProxy)1e6;
            nonNegVec[0] = (fProxy)1e-40; nonNegVec[1] = (fProxy)0;

            var hashOut = new NativeArray<uint>(12, Allocator.Persistent);
            var job = new DetElementwiseCoreJobFProxy
            {
                baseVec = baseVec, otherVec = otherVec, otherVec2 = otherVec2, nonNegVec = nonNegVec,
                scratch = scratch, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("elementwise-core/abs.fProxy.n1000", hashOut[0]),
                ("elementwise-core/sign.fProxy.n1000", hashOut[1]),
                ("elementwise-core/sqrt.fProxy.n1000", hashOut[2]),
                ("elementwise-core/clamp.fProxy.n1000", hashOut[3]),
                ("elementwise-core/lerp.fProxy.n1000", hashOut[4]),
                ("elementwise-core/min.fProxy.n1000", hashOut[5]),
                ("elementwise-core/max.fProxy.n1000", hashOut[6]),
                ("elementwise-core/mad.fProxy.n1000", hashOut[7]),
                ("elementwise-core/floor.fProxy.n1000", hashOut[8]),
                ("elementwise-core/ceil.fProxy.n1000", hashOut[9]),
                ("elementwise-core/round.fProxy.n1000", hashOut[10]),
                ("elementwise-core/saturate.fProxy.n1000", hashOut[11]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_NormsFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2654435761u ^ 0x0004u);

            var vecX = arena.fProxyVec(37); for (int i = 0; i < 37; i++) vecX[i] = rng.NextFProxy(-3f, 3f);
            var A = arena.fProxyMat(53, 37);
            for (int r = 0; r < 53; r++) for (int c = 0; c < 37; c++) A[r, c] = rng.NextFProxy(-3f, 3f);

            var normalizeVec = arena.fProxyVec(37); for (int i = 0; i < 37; i++) normalizeVec[i] = vecX[i];
            var normColsMat = arena.fProxyMat(53, 37);
            for (int r = 0; r < 53; r++) for (int c = 0; c < 37; c++) normColsMat[r, c] = A[r, c];

            var hashOut = new NativeArray<uint>(7, Allocator.Persistent);
            var job = new DetNormsJobFProxy
            {
                vecX = vecX, A = A, normalizeVec = normalizeVec, normColsMat = normColsMat, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("norms/L1.fProxy.n37", hashOut[0]),
                ("norms/L2.fProxy.n37", hashOut[1]),
                ("norms/matrixL1.fProxy.53x37", hashOut[2]),
                ("norms/matrixL2.fProxy.53x37", hashOut[3]),
                ("norms/matrixLInf.fProxy.53x37", hashOut[4]),
                ("norms/normalize.fProxy.n37", hashOut[5]),
                ("norms/normalizeColumns.fProxy.53x37", hashOut[6]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_QrFamilyFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2654435761u ^ 0x0006u);

            var Atall = arena.fProxyMat(53, 37);
            for (int r = 0; r < 53; r++) for (int c = 0; c < 37; c++) Atall[r, c] = rng.NextFProxy(-1f, 1f);
            var Q1 = arena.fProxyMat(53, 37); var R1 = arena.fProxyMat(37, 37);
            var bTall = arena.fProxyVec(53); for (int i = 0; i < 53; i++) bTall[i] = rng.NextFProxy(-1f, 1f);
            var xTall1 = arena.fProxyVec(37);

            var Q2 = arena.fProxyMat(53, 37); var R2 = arena.fProxyMat(37, 37);
            var P1 = new Pivot(37, Allocator.Persistent);
            var bTall2 = arena.fProxyVec(53); for (int i = 0; i < 53; i++) bTall2[i] = rng.NextFProxy(-1f, 1f);
            var xTall2 = arena.fProxyVec(37);

            var AforMinNorm = arena.fProxyMat(53, 37);
            for (int r = 0; r < 53; r++) for (int c = 0; c < 37; c++) AforMinNorm[r, c] = Atall[r, c];
            var bTall3 = arena.fProxyVec(53); for (int i = 0; i < 53; i++) bTall3[i] = rng.NextFProxy(-1f, 1f);
            var xTall3 = arena.fProxyVec(37);

            var Awide = arena.fProxyMat(37, 53);
            for (int r = 0; r < 37; r++) for (int c = 0; c < 53; c++) Awide[r, c] = rng.NextFProxy(-1f, 1f);
            var L1 = arena.fProxyMat(37, 37); var Qw1 = arena.fProxyMat(37, 53);
            var bWide1 = arena.fProxyVec(37); for (int i = 0; i < 37; i++) bWide1[i] = rng.NextFProxy(-1f, 1f);
            var xWide1 = arena.fProxyVec(53);

            var L2 = arena.fProxyMat(37, 37); var Qw2 = arena.fProxyMat(37, 53);
            var P2 = new Pivot(37, Allocator.Persistent);
            var Bwide = arena.fProxyMat(37, 1); for (int i = 0; i < 37; i++) Bwide[i, 0] = rng.NextFProxy(-1f, 1f);
            var Xwide = arena.fProxyMat(53, 1);

            var hashOut = new NativeArray<uint>(9, Allocator.Persistent);
            var job = new DetQrFamilyJobFProxy
            {
                Atall = Atall, Q1 = Q1, R1 = R1, bTall = bTall, xTall1 = xTall1,
                Q2 = Q2, R2 = R2, P1 = P1, bTall2 = bTall2, xTall2 = xTall2,
                AforMinNorm = AforMinNorm, bTall3 = bTall3, xTall3 = xTall3,
                Awide = Awide, L1 = L1, Qw1 = Qw1, bWide1 = bWide1, xWide1 = xWide1,
                L2 = L2, Qw2 = Qw2, P2 = P2, Bwide = Bwide, Xwide = Xwide,
                HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("qr-family/qr.decomp.fProxy.53x37", hashOut[0]),
                ("qr-family/qr.decompSolve.fProxy.53x37", hashOut[1]),
                ("qr-family/qrcp.decomp.fProxy.53x37", hashOut[2]),
                ("qr-family/qrcp.decompSolve.fProxy.53x37", hashOut[3]),
                ("qr-family/qrcp.minNormSolveInPlace.fProxy.53x37", hashOut[4]),
                ("qr-family/lq.decomp.fProxy.37x53", hashOut[5]),
                ("qr-family/lq.minNormSolve.fProxy.37x53", hashOut[6]),
                ("qr-family/lqrp.decomp.fProxy.37x53", hashOut[7]),
                ("qr-family/lqrp.minNormDecompSolve.fProxy.37x53", hashOut[8]),
            };
            hashOut.Dispose();
            P1.Dispose(); P2.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_LuFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2654435761u ^ 0x0007u);

            const int n = 48;
            var A = arena.fProxyMat(n, n);
            for (int r = 0; r < n; r++) for (int c = 0; c < n; c++) A[r, c] = rng.NextFProxy(-1f, 1f);
            for (int d = 0; d < n; d++) A[d, d] += (fProxy)n;

            var L = arena.fProxyMat(n, n); var U = arena.fProxyMat(n, n);
            var P1 = new Pivot(n, Allocator.Persistent);

            var LUPacked = arena.fProxyMat(n, n);
            for (int r = 0; r < n; r++) for (int c = 0; c < n; c++) LUPacked[r, c] = A[r, c];
            var P2 = new Pivot(n, Allocator.Persistent);

            var bSolve = arena.fProxyVec(n); for (int i = 0; i < n; i++) bSolve[i] = rng.NextFProxy(-1f, 1f);
            var bSolveTransA = arena.fProxyVec(n); for (int i = 0; i < n; i++) bSolveTransA[i] = rng.NextFProxy(-1f, 1f);

            var hashOut = new NativeArray<uint>(4, Allocator.Persistent);
            var job = new DetLuJobFProxy
            {
                A = A, L = L, U = U, P1 = P1, LUPacked = LUPacked, P2 = P2,
                bSolve = bSolve, bSolveTransA = bSolveTransA, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("lu/decomp.fProxy.n48", hashOut[0]),
                ("lu/decompInPlace.fProxy.n48", hashOut[1]),
                ("lu/decompSolve.fProxy.n48", hashOut[2]),
                ("lu/decompSolveTransA.fProxy.n48", hashOut[3]),
            };
            hashOut.Dispose();
            P1.Dispose(); P2.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_CholeskyFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2654435761u ^ 0x0008u);

            const int n = 48;
            var A = arena.fProxyMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    fProxy v = rng.NextFProxy(-1f, 1f);
                    A[i, j] = v; A[j, i] = v;
                }
            for (int d = 0; d < n; d++) A[d, d] += (fProxy)n;

            var L = arena.fProxyMat(n, n);
            var bSolve = arena.fProxyVec(n); for (int i = 0; i < n; i++) bSolve[i] = rng.NextFProxy(-1f, 1f);

            var Lp = arena.fProxyMat(n, n);
            var P1 = new Pivot(n, Allocator.Persistent);

            var AtoL = arena.fProxyMat(n, n);
            for (int r = 0; r < n; r++) for (int c = 0; c < n; c++) AtoL[r, c] = A[r, c];
            var P2 = new Pivot(n, Allocator.Persistent);
            var bSolveInPlace = arena.fProxyVec(n); for (int i = 0; i < n; i++) bSolveInPlace[i] = rng.NextFProxy(-1f, 1f);

            var hashOut = new NativeArray<uint>(4, Allocator.Persistent);
            var job = new DetCholeskyJobFProxy
            {
                A = A, L = L, bSolve = bSolve, Lp = Lp, P1 = P1,
                AtoL = AtoL, P2 = P2, bSolveInPlace = bSolveInPlace, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("cholesky/cho.decomp.fProxy.n48", hashOut[0]),
                ("cholesky/cho.decompSolve.fProxy.n48", hashOut[1]),
                ("cholesky/chop.decomp.fProxy.n48", hashOut[2]),
                ("cholesky/chop.solveInPlace.fProxy.n48", hashOut[3]),
            };
            hashOut.Dispose();
            P1.Dispose(); P2.Dispose();
            arena.Dispose();
            return result;
        }
    }
}
