using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of the determinism conformance harness's iterative/sparse groups
    // (krylov-dense, sparse-bsr, krylov-sparse-precond). See DeterminismDirect.fProxy.cs's header
    // for the shared job/case-method convention and
    // docs/dev/spec-determinism-conformance-harness.md for the frozen op/group/root hash contract.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetKrylovDenseJobFProxy : IJob
    {
        public fProxyMxN Aspd;
        public fProxyN bSpd;
        public fProxyN xCg, xMinres, xBicg;
        public fProxyMxN Als;
        public fProxyN bLs;
        public fProxyN xLsqr, xLsmr;

        public NativeArray<uint> HashOut; // 5 slots

        public void Execute()
        {
            var cgInfo = Krylov.cg(in Aspd, in bSpd, ref xCg);
            uint h = Hash.hash(in xCg);
            h = DetHash.Combine(h, cgInfo.iterations);
            h = DetHash.Combine(h, (int)cgInfo.status);
            HashOut[0] = h;

            var mrInfo = Krylov.minres(in Aspd, in bSpd, ref xMinres);
            h = Hash.hash(in xMinres);
            h = DetHash.Combine(h, mrInfo.iterations);
            h = DetHash.Combine(h, (int)mrInfo.status);
            HashOut[1] = h;

            var bicgInfo = Krylov.biCGStab(in Aspd, in bSpd, ref xBicg);
            h = Hash.hash(in xBicg);
            h = DetHash.Combine(h, bicgInfo.iterations);
            h = DetHash.Combine(h, (int)bicgInfo.status);
            HashOut[2] = h;

            var lsqrInfo = Krylov.lsqr(in Als, in bLs, ref xLsqr);
            h = Hash.hash(in xLsqr);
            h = DetHash.Combine(h, lsqrInfo.iterations);
            h = DetHash.Combine(h, (int)lsqrInfo.status);
            HashOut[3] = h;

            var lsmrInfo = Krylov.lsmr(in Als, in bLs, ref xLsmr);
            h = Hash.hash(in xLsmr);
            h = DetHash.Combine(h, lsmrInfo.iterations);
            h = DetHash.Combine(h, (int)lsmrInfo.status);
            HashOut[4] = h;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetSparseBsrJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyBlockJacobi Jac;
        public fProxyN x, y, yT, sweepLowerOut, sweepUpperOut;
        public fProxyMxN Vrows, AVrows;
        public fProxyBSR built; // small, manually assembled
        public fProxyBSR randomSpd;
        public fProxyN xRandom, yRandom;

        public NativeArray<uint> HashOut; // 7 slots

        public unsafe void Execute()
        {
            BSR.spMV(in A, in x, ref y);
            HashOut[0] = Hash.hash(in y);

            BSR.spMVT(in A, in x, ref yT);
            HashOut[1] = Hash.hash(in yT);

            BSR.spMM(in A, in Vrows, ref AVrows, Vrows.M_Rows);
            HashOut[2] = Hash.hash(in AVrows);

            BSR.sweepLower(in A, in Jac, in x, ref sweepLowerOut);
            HashOut[3] = Hash.hash(in sweepLowerOut);

            BSR.sweepUpper(in A, in Jac, in x, ref sweepUpperOut);
            HashOut[4] = Hash.hash(in sweepUpperOut);

            uint h = DetHash.Combine(0u, (byte*)built.RowPtr.Ptr, built.RowPtr.Length * sizeof(int));
            h = DetHash.Combine(h, (byte*)built.ColInd.Ptr, built.ColInd.Length * sizeof(int));
            h = DetHash.Combine(h, (byte*)built.Values.Ptr, built.Values.Length * sizeof(fProxy));
            HashOut[5] = h;

            BSR.spMV(in randomSpd, in xRandom, ref yRandom);
            HashOut[6] = Hash.hash(in yRandom);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetKrylovSparsePrecondJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b;
        public fProxyBlockJacobi MJacobi;
        public fProxyN xJacobi;
        public fProxySSOR MSsor;
        public fProxyN xSsor;
        public fProxyIC0 MIc0;
        public fProxyN xIc0;
        public fProxyILU0 MIlu0;
        public fProxyN xIlu0;

        public NativeArray<uint> HashOut; // 4 slots

        public unsafe void Execute()
        {
            var jInfo = Krylov.cg(in A, in MJacobi, in b, ref xJacobi);
            uint h = DetHash.Combine(0u, (byte*)MJacobi.DInv.Ptr, MJacobi.DInv.Length * sizeof(fProxy));
            h = Hash.combine(h, Hash.hash(in xJacobi));
            h = DetHash.Combine(h, jInfo.iterations);
            h = DetHash.Combine(h, (int)jInfo.status);
            HashOut[0] = h;

            var sInfo = Krylov.cg(in A, in MSsor, in b, ref xSsor);
            h = Hash.hash(in MSsor.ScaledD);
            h = DetHash.Combine(h, MSsor.Omega);
            h = Hash.combine(h, Hash.hash(in xSsor));
            h = DetHash.Combine(h, sInfo.iterations);
            h = DetHash.Combine(h, (int)sInfo.status);
            HashOut[1] = h;

            var iInfo = Krylov.cg(in A, in MIc0, in b, ref xIc0);
            h = DetHash.Combine(0u, (byte*)MIc0.L.RowPtr.Ptr, MIc0.L.RowPtr.Length * sizeof(int));
            h = DetHash.Combine(h, (byte*)MIc0.L.ColInd.Ptr, MIc0.L.ColInd.Length * sizeof(int));
            h = DetHash.Combine(h, (byte*)MIc0.L.Values.Ptr, MIc0.L.Values.Length * sizeof(fProxy));
            h = Hash.combine(h, Hash.hash(in xIc0));
            h = DetHash.Combine(h, iInfo.iterations);
            h = DetHash.Combine(h, (int)iInfo.status);
            HashOut[2] = h;

            // fProxyILU0.IsSpd == false: never a valid cg/minres preconditioner. biCGStab is its
            // shipped solver path (see docs/dev/spec-determinism-conformance-harness.md group 15).
            var lInfo = Krylov.biCGStab(in A, in MIlu0, in b, ref xIlu0);
            h = DetHash.Combine(0u, (byte*)MIlu0.F.RowPtr.Ptr, MIlu0.F.RowPtr.Length * sizeof(int));
            h = DetHash.Combine(h, (byte*)MIlu0.F.ColInd.Ptr, MIlu0.F.ColInd.Length * sizeof(int));
            h = DetHash.Combine(h, (byte*)MIlu0.F.Values.Ptr, MIlu0.F.Values.Length * sizeof(fProxy));
            h = Hash.combine(h, Hash.hash(in xIlu0));
            h = DetHash.Combine(h, lInfo.iterations);
            h = DetHash.Combine(h, (int)lInfo.status);
            HashOut[3] = h;
        }
    }

    public static partial class DeterminismIterativeSparse
    {
        public static (string id, uint hash)[] Case_KrylovDenseFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2654435761u ^ 0x000Du);

            const int n = 64;
            var M = arena.fProxyMat(n, n);
            for (int r = 0; r < n; r++) for (int c = 0; c < n; c++) M[r, c] = rng.NextFProxy(-1f, 1f);
            var Aspd = arena.fProxyMat(n, n);
            Blas.dot(in M, in M, ref Aspd, transposeA: true);
            for (int d = 0; d < n; d++) Aspd[d, d] += (fProxy)n;
            var bSpd = arena.fProxyVec(n); for (int i = 0; i < n; i++) bSpd[i] = rng.NextFProxy(-1f, 1f);
            var xCg = arena.fProxyVec(n, true); var xMinres = arena.fProxyVec(n, true); var xBicg = arena.fProxyVec(n, true);

            const int mLs = 96, nLs = 48;
            var Als = arena.fProxyMat(mLs, nLs);
            for (int r = 0; r < mLs; r++) for (int c = 0; c < nLs; c++) Als[r, c] = rng.NextFProxy(-1f, 1f);
            var bLs = arena.fProxyVec(mLs); for (int i = 0; i < mLs; i++) bLs[i] = rng.NextFProxy(-1f, 1f);
            var xLsqr = arena.fProxyVec(nLs, true); var xLsmr = arena.fProxyVec(nLs, true);

            var hashOut = new NativeArray<uint>(5, Allocator.Persistent);
            var job = new DetKrylovDenseJobFProxy
            {
                Aspd = Aspd, bSpd = bSpd, xCg = xCg, xMinres = xMinres, xBicg = xBicg,
                Als = Als, bLs = bLs, xLsqr = xLsqr, xLsmr = xLsmr, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("krylov-dense/cg.fProxy.spd.n64", hashOut[0]),
                ("krylov-dense/minres.fProxy.spd.n64", hashOut[1]),
                ("krylov-dense/biCGStab.fProxy.spd.n64", hashOut[2]),
                ("krylov-dense/lsqr.fProxy.96x48", hashOut[3]),
                ("krylov-dense/lsmr.fProxy.96x48", hashOut[4]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_SparseBsrFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyLaplacian2D(32, 32); // N=1024
            int n = A.M_Rows;
            var Jac = arena.fProxyBlockJacobi(in A);

            var rng = new Random(2654435761u ^ 0x000Eu);
            var x = arena.fProxyVec(n); for (int i = 0; i < n; i++) x[i] = rng.NextFProxy(-1f, 1f);
            var y = arena.fProxyVec(n); var yT = arena.fProxyVec(n);
            var sweepLowerOut = arena.fProxyVec(n); var sweepUpperOut = arena.fProxyVec(n);

            var Vrows = arena.fProxyMat(4, n);
            for (int r = 0; r < 4; r++) for (int c = 0; c < n; c++) Vrows[r, c] = rng.NextFProxy(-1f, 1f);
            var AVrows = arena.fProxyMat(4, n);

            // Small manually-assembled BSR: hashes the assembly buffers themselves (block CSR
            // RowPtr/ColInd/Values), independent of any gallery generator.
            var builder = arena.fProxyBSRBuilder(3, 3, 2, 2, 5);
            var diag = arena.fProxyMat(2, 2);
            diag[0, 0] = (fProxy)4; diag[0, 1] = (fProxy)1; diag[1, 0] = (fProxy)1; diag[1, 1] = (fProxy)4;
            builder.AddBlock(0, 0, in diag); builder.AddBlock(1, 1, in diag); builder.AddBlock(2, 2, in diag);
            var off = arena.fProxyMat(2, 2);
            off[0, 0] = (fProxy)(-1); off[0, 1] = (fProxy)0; off[1, 0] = (fProxy)0; off[1, 1] = (fProxy)(-1);
            builder.AddBlock(1, 0, in off); builder.AddBlock(2, 1, in off);
            var built = builder.ToBSR(ref arena);

            var randomSpd = arena.fProxyRandomSparseSPD(12, 4, (fProxy)0.3, 0xC0FFEEu); // 48x48-ish
            var xRandom = arena.fProxyVec(randomSpd.M_Rows);
            for (int i = 0; i < xRandom.N; i++) xRandom[i] = rng.NextFProxy(-1f, 1f);
            var yRandom = arena.fProxyVec(randomSpd.M_Rows);

            var hashOut = new NativeArray<uint>(7, Allocator.Persistent);
            var job = new DetSparseBsrJobFProxy
            {
                A = A, Jac = Jac, x = x, y = y, yT = yT, sweepLowerOut = sweepLowerOut, sweepUpperOut = sweepUpperOut,
                Vrows = Vrows, AVrows = AVrows, built = built, randomSpd = randomSpd, xRandom = xRandom, yRandom = yRandom,
                HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("sparse-bsr/spMV.fProxy.laplacian2d.n1024", hashOut[0]),
                ("sparse-bsr/spMVT.fProxy.laplacian2d.n1024", hashOut[1]),
                ("sparse-bsr/spMM.fProxy.laplacian2d.n1024", hashOut[2]),
                ("sparse-bsr/sweepLower.fProxy.laplacian2d.n1024", hashOut[3]),
                ("sparse-bsr/sweepUpper.fProxy.laplacian2d.n1024", hashOut[4]),
                ("sparse-bsr/assembly.fProxy.builder.3x3blocks", hashOut[5]),
                ("sparse-bsr/spMV.fProxy.randomSparseSPD", hashOut[6]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_KrylovSparsePrecondFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyLaplacian2D(8, 128); // N=1024, BR=8 (IC0/ILU0 cap block size at 16)
            int n = A.M_Rows;
            var rng = new Random(2654435761u ^ 0x000Fu);
            var b = arena.fProxyVec(n); for (int i = 0; i < n; i++) b[i] = rng.NextFProxy(-1f, 1f);

            var MJacobi = arena.fProxyBlockJacobi(in A);
            var xJacobi = arena.fProxyVec(n, true);
            var MSsor = arena.fProxySSOR(in A);
            var xSsor = arena.fProxyVec(n, true);
            var MIc0 = arena.fProxyIC0(in A);
            var xIc0 = arena.fProxyVec(n, true);
            var MIlu0 = arena.fProxyILU0(in A);
            var xIlu0 = arena.fProxyVec(n, true);

            var hashOut = new NativeArray<uint>(4, Allocator.Persistent);
            var job = new DetKrylovSparsePrecondJobFProxy
            {
                A = A, b = b,
                MJacobi = MJacobi, xJacobi = xJacobi,
                MSsor = MSsor, xSsor = xSsor,
                MIc0 = MIc0, xIc0 = xIc0,
                MIlu0 = MIlu0, xIlu0 = xIlu0,
                HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("krylov-sparse-precond/cg.blockJacobi.fProxy.laplacian2d.n1024", hashOut[0]),
                ("krylov-sparse-precond/cg.ssor.fProxy.laplacian2d.n1024", hashOut[1]),
                ("krylov-sparse-precond/cg.ic0.fProxy.laplacian2d.n1024", hashOut[2]),
                ("krylov-sparse-precond/biCGStab.ilu0.fProxy.laplacian2d.n1024", hashOut[3]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }
    }
}
