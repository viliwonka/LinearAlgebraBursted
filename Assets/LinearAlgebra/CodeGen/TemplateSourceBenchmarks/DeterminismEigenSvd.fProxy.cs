using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.Gallery;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of the determinism conformance harness's eigen/SVD/LOBPCG groups
    // (eigen-sym, eigen-nonsym, svd, lobpcg). See DeterminismDirect.fProxy.cs's header for the shared
    // job/case-method convention and docs/dev/spec-determinism-conformance-harness.md for the frozen
    // op/group/root hash contract.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetEigenSymJobFProxy : IJob
    {
        public fProxyMxN A1, A2;                 // copies of W, destroyed by the two InPlace ops
        public fProxyN eigenvalues1, eigenvalues2;
        public fProxyMxN V;
        public fProxyMxN Wlanczos;                // undestroyed original (lanczos takes `in`)
        public fProxyLanczosCache lws;
        public fProxyN lanczosEigenvalues;

        public NativeArray<uint> HashOut; // 3 slots

        public void Execute()
        {
            var info1 = Eigen.symmetricInPlace(ref A1, ref eigenvalues1, ref V);
            uint h = Hash.hash(in eigenvalues1);
            h = Hash.combine(h, Hash.hash(in V));
            h = DetHash.Combine(h, (int)info1.status);
            h = DetHash.Combine(h, info1.sweeps);
            h = DetHash.Combine(h, info1.converged);
            HashOut[0] = h;

            var info2 = Eigen.valuesSymmetricInPlace(ref A2, ref eigenvalues2);
            h = Hash.hash(in eigenvalues2);
            h = DetHash.Combine(h, (int)info2.status);
            HashOut[1] = h;

            var info3 = Eigen.lanczos(in Wlanczos, ref lws, ref lanczosEigenvalues, lanczosEigenvalues.N);
            h = Hash.hash(in lanczosEigenvalues);
            h = DetHash.Combine(h, info3.produced);
            h = DetHash.Combine(h, (int)info3.status);
            HashOut[2] = h;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetEigenNonsymJobFProxy : IJob
    {
        public fProxyMxN Afrank;                  // copy of Frank, destroyed by valuesQRInPlace
        public fProxyN eigRe, eigIm;
        public fProxyMxN FrankForPower;
        public fProxyN pv, pw;
        public fProxyMxN GrcarForPower;
        public fProxyN gv, gw;
        public fProxyMxN SpdForInvPower;           // Laplacian1D: SPD precondition for inversePowerIteration
        public fProxyN ipv;

        public NativeArray<uint> HashOut; // 4 slots

        public void Execute()
        {
            var qi = Eigen.valuesQRInPlace(ref Afrank, ref eigRe, ref eigIm);
            uint h = Hash.hash(in eigRe);
            h = Hash.combine(h, Hash.hash(in eigIm));
            h = DetHash.Combine(h, (int)qi.status);
            HashOut[0] = h;

            var pi1 = Eigen.powerIteration(in FrankForPower, ref pv, ref pw, out fProxy lambda1);
            h = Hash.hash(in pv);
            h = DetHash.Combine(h, lambda1);
            h = DetHash.Combine(h, pi1.iterations);
            h = DetHash.Combine(h, (int)pi1.status);
            HashOut[1] = h;

            var pi2 = Eigen.powerIteration(in GrcarForPower, ref gv, ref gw, out fProxy lambda2);
            h = Hash.hash(in gv);
            h = DetHash.Combine(h, lambda2);
            h = DetHash.Combine(h, pi2.iterations);
            h = DetHash.Combine(h, (int)pi2.status);
            HashOut[2] = h;

            var ipi = Eigen.inversePowerIteration(in SpdForInvPower, ref ipv, out fProxy lambda3);
            h = Hash.hash(in ipv);
            h = DetHash.Combine(h, lambda3);
            h = DetHash.Combine(h, ipi.iterations);
            h = DetHash.Combine(h, (int)ipi.status);
            HashOut[3] = h;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetSvdJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN U; public fProxyN S; public fProxyMxN V;
        public fProxyN Svalues;
        public fProxyMxN Uk; public fProxyN Sk; public fProxyMxN Vk; public fProxySVDTruncatedCache wsT;
        public fProxyMxN Uk2; public fProxyN Sk2; public fProxyMxN Vk2; public fProxySVDRandomizedCache wsR;
        public fProxyMxN Apinv; public fProxyN bPinv; public fProxyN xPinv; public fProxySVDCache wsS;
        public fProxyMxN Apinv2; public fProxyMxN Aplus; public fProxySVDCache wsS2;
        public fProxyMxN Lauchli; public fProxyMxN basis;

        public NativeArray<uint> HashOut; // 7 slots

        public void Execute()
        {
            var i1 = SVD.thin(in A, ref U, ref S, ref V);
            uint h = Hash.hash(in U);
            h = Hash.combine(h, Hash.hash(in S));
            h = Hash.combine(h, Hash.hash(in V));
            h = DetHash.Combine(h, (int)i1.status);
            HashOut[0] = h;

            var i2 = SVD.values(in A, ref Svalues);
            h = Hash.hash(in Svalues);
            h = DetHash.Combine(h, (int)i2.status);
            HashOut[1] = h;

            var i3 = SVD.truncated(in A, ref Uk, ref Sk, ref Vk, Sk.N, ref wsT);
            h = Hash.hash(in Uk);
            h = Hash.combine(h, Hash.hash(in Sk));
            h = Hash.combine(h, Hash.hash(in Vk));
            h = DetHash.Combine(h, (int)i3.status);
            HashOut[2] = h;

            var i4 = SVD.randomized(in A, ref Uk2, ref Sk2, ref Vk2, Sk2.N, ref wsR);
            h = Hash.hash(in Uk2);
            h = Hash.combine(h, Hash.hash(in Sk2));
            h = Hash.combine(h, Hash.hash(in Vk2));
            h = DetHash.Combine(h, (int)i4.status);
            HashOut[3] = h;

            var i5 = SVD.pinvSolve(ref Apinv, in bPinv, ref xPinv, ref wsS);
            h = Hash.hash(in xPinv);
            h = DetHash.Combine(h, (int)i5.status);
            h = DetHash.Combine(h, i5.rank);
            HashOut[4] = h;

            var i6 = SVD.pseudoInverse(ref Apinv2, ref Aplus, ref wsS2);
            h = Hash.hash(in Aplus);
            h = DetHash.Combine(h, (int)i6.status);
            h = DetHash.Combine(h, i6.rank);
            HashOut[5] = h;

            var i7 = SVD.nullspaceBasis(in Lauchli, ref basis);
            h = Hash.hash(in basis);
            h = DetHash.Combine(h, (int)i7.status);
            h = DetHash.Combine(h, i7.rank);
            HashOut[6] = h;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetLobpcgJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyLOBPCGCache ws;
        public int K;

        public NativeArray<uint> HashOut; // 1 slot

        public void Execute()
        {
            for (int r = 0; r < ws.X.M_Rows; r++)
                for (int c = 0; c < ws.X.N_Cols; c++)
                    ws.X[r, c] = (fProxy)0;

            var info = Eigen.lobpcg(in A, ref ws, K);
            uint h = Hash.hash(in ws.lambda);
            h = Hash.combine(h, Hash.hash(in ws.X));
            h = DetHash.Combine(h, info.iterations);
            h = DetHash.Combine(h, info.converged);
            h = DetHash.Combine(h, info.maxResidual);
            h = DetHash.Combine(h, (int)info.status);
            HashOut[0] = h;
        }
    }

    public static partial class DeterminismEigenSvd
    {
        public static (string id, uint hash)[] Case_EigenSymFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            const int n = 47; // fProxyWilkinsonPlus requires odd n >= 3
            var W = arena.fProxyWilkinsonPlus(n);

            var A1 = arena.fProxyMat(n, n);
            var A2 = arena.fProxyMat(n, n);
            for (int r = 0; r < n; r++) for (int c = 0; c < n; c++) { A1[r, c] = W[r, c]; A2[r, c] = W[r, c]; }

            var eigenvalues1 = arena.fProxyVec(n);
            var eigenvalues2 = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);
            var lws = arena.fProxyLanczosCache(n, n);
            var lanczosEigenvalues = arena.fProxyVec(n);

            var hashOut = new NativeArray<uint>(3, Allocator.Persistent);
            var job = new DetEigenSymJobFProxy
            {
                A1 = A1, A2 = A2, eigenvalues1 = eigenvalues1, eigenvalues2 = eigenvalues2, V = V,
                Wlanczos = W, lws = lws, lanczosEigenvalues = lanczosEigenvalues, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("eigen-sym/symmetricInPlace.fProxy.n47", hashOut[0]),
                ("eigen-sym/valuesSymmetricInPlace.fProxy.n47", hashOut[1]),
                ("eigen-sym/lanczos.fProxy.n47", hashOut[2]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_EigenNonsymFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            const int n = 32;
            var Frank = arena.fProxyFrank(n);
            var Grcar = arena.fProxyGrcar(n);
            var Spd = arena.fProxyLaplacian1D(n);

            var Afrank = arena.fProxyMat(n, n);
            for (int r = 0; r < n; r++) for (int c = 0; c < n; c++) Afrank[r, c] = Frank[r, c];
            var eigRe = arena.fProxyVec(n); var eigIm = arena.fProxyVec(n);

            var pv = arena.fProxyVec(n, (fProxy)1); var pw = arena.fProxyVec(n);
            var gv = arena.fProxyVec(n, (fProxy)1); var gw = arena.fProxyVec(n);
            var ipv = arena.fProxyVec(n, (fProxy)1);

            var hashOut = new NativeArray<uint>(4, Allocator.Persistent);
            var job = new DetEigenNonsymJobFProxy
            {
                Afrank = Afrank, eigRe = eigRe, eigIm = eigIm,
                FrankForPower = Frank, pv = pv, pw = pw,
                GrcarForPower = Grcar, gv = gv, gw = gw,
                SpdForInvPower = Spd, ipv = ipv, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("eigen-nonsym/valuesQRInPlace.fProxy.frank.n32", hashOut[0]),
                ("eigen-nonsym/powerIteration.fProxy.frank.n32", hashOut[1]),
                ("eigen-nonsym/powerIteration.fProxy.grcar.n32", hashOut[2]),
                ("eigen-nonsym/inversePowerIteration.fProxy.spd.n32", hashOut[3]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_SvdFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2654435761u ^ 0x000Bu);

            const int m = 53, n = 37, k = 16;
            var A = arena.fProxyMat(m, n);
            for (int r = 0; r < m; r++) for (int c = 0; c < n; c++) A[r, c] = rng.NextFProxy(-1f, 1f);

            var U = arena.fProxyMat(m, n); var S = arena.fProxyVec(n); var V = arena.fProxyMat(n, n);
            var Svalues = arena.fProxyVec(n);

            var Uk = arena.fProxyMat(m, k); var Sk = arena.fProxyVec(k); var Vk = arena.fProxyMat(n, k);
            var wsT = arena.fProxySVDTruncatedCache(m, n, k);

            var Uk2 = arena.fProxyMat(m, k); var Sk2 = arena.fProxyVec(k); var Vk2 = arena.fProxyMat(n, k);
            var wsR = arena.fProxySVDRandomizedCache(m, n, k);

            var Apinv = arena.fProxyMat(m, n);
            for (int r = 0; r < m; r++) for (int c = 0; c < n; c++) Apinv[r, c] = A[r, c];
            var bPinv = arena.fProxyVec(m); for (int i = 0; i < m; i++) bPinv[i] = rng.NextFProxy(-1f, 1f);
            var xPinv = arena.fProxyVec(n);
            var wsS = arena.fProxySVDCache(m, n);

            var Apinv2 = arena.fProxyMat(m, n);
            for (int r = 0; r < m; r++) for (int c = 0; c < n; c++) Apinv2[r, c] = A[r, c];
            var Aplus = arena.fProxyMat(n, m);
            var wsS2 = arena.fProxySVDCache(m, n);

            var Lauchli = arena.fProxyLauchli(n, (fProxy)1e-6);
            var basis = arena.fProxyMat(n, n);

            var hashOut = new NativeArray<uint>(7, Allocator.Persistent);
            var job = new DetSvdJobFProxy
            {
                A = A, U = U, S = S, V = V, Svalues = Svalues,
                Uk = Uk, Sk = Sk, Vk = Vk, wsT = wsT,
                Uk2 = Uk2, Sk2 = Sk2, Vk2 = Vk2, wsR = wsR,
                Apinv = Apinv, bPinv = bPinv, xPinv = xPinv, wsS = wsS,
                Apinv2 = Apinv2, Aplus = Aplus, wsS2 = wsS2,
                Lauchli = Lauchli, basis = basis, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("svd/thin.fProxy.53x37", hashOut[0]),
                ("svd/values.fProxy.53x37", hashOut[1]),
                ("svd/truncated.fProxy.53x37.k16", hashOut[2]),
                ("svd/randomized.fProxy.53x37.k16", hashOut[3]),
                ("svd/pinvSolve.fProxy.53x37", hashOut[4]),
                ("svd/pseudoInverse.fProxy.53x37", hashOut[5]),
                ("svd/nullspaceBasis.fProxy.lauchli.n37", hashOut[6]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_LobpcgFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            const int n = 48, k = 4;
            var A = arena.fProxyLaplacian1D(n);
            var ws = arena.fProxyLOBPCGCache(n, k);

            var hashOut = new NativeArray<uint>(1, Allocator.Persistent);
            var job = new DetLobpcgJobFProxy { A = A, ws = ws, K = k, HashOut = hashOut };
            job.Run();

            var result = new[] { ("lobpcg/lobpcg.fProxy.laplacian1d.n48.k4", hashOut[0]) };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }
    }
}
