using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.Control;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of LQRBenchmark (the timed IJobs + the instance builder + build+measure
    // methods). The dtype-agnostic harness (sizes/seeds, row formatter, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/LQRBenchmark.cs.
    //
    // Three variants, all through the PUBLIC Control API only (Benchmarks has no InternalsVisibleTo
    // grant -- same constraint QPBenchmark.fProxy.cs notes for qpActiveSetCore):
    //   - cold-SDA: the plain LQR.lqr(...) overload (structure-preserving doubling).
    //   - cold-recursion: the warm LQR.lqr(..., ref state) overload, but with a FRESH state whose
    //     S is force-seeded at zero and populated=true (both public fields) -- this makes the warm
    //     overload take its plain-fixed-point-recursion branch starting cold, the "naive" baseline the
    //     spec wants SDA/warm compared against, reached without touching Control's internal
    //     RiccatiIterate directly.
    //   - warm: an UNTIMED cold solve (managed thread, not inside the timed job) seeds Sprev with the
    //     converged S, A is perturbed ~1e-3 relative, then the TIMED job re-seeds a fresh state from
    //     Sprev each Execute and warm-solves the perturbed system -- same re-copy-before-timed-call
    //     idiom CholeskyBenchmark's face-off section uses for its destructive decompInPlace calls.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LqrColdSdaJobFProxy : IJob
    {
        public fProxyMxN A, B, Q, R, K;
        public int reps;    // identical solves per Execute; per-solve time = job time / reps
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;

        public void Execute()
        {
            for (int r = 0; r < reps; r++)
            {
                var info = LQR.lqr(in A, in B, in Q, in R, ref K);
                itersOut[0] = info.iterations;
                statusOut[0] = (int)info.status;
            }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LqrColdRecursionJobFProxy : IJob
    {
        public fProxyMxN A, B, Q, R, K;
        public int reps;
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;

        public void Execute()
        {
            for (int r = 0; r < reps; r++)
            {
                var state = new fProxyLQRState(A.M_Rows, Allocator.Temp);   // S starts zero-filled
                state.populated = true;                                    // forces the plain-recursion branch, cold-started
                var info = LQR.lqr(in A, in B, in Q, in R, ref K, ref state);
                itersOut[0] = info.iterations;
                statusOut[0] = (int)info.status;
                state.Dispose();
            }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LqrWarmJobFProxy : IJob
    {
        public fProxyMxN Aperturbed, B, Q, R, K;
        public fProxyMxN Sprev;   // pre-perturbation converged S; re-seeded into a fresh state each rep
        public int reps;
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;

        public void Execute()
        {
            for (int r = 0; r < reps; r++)
            {
                var state = new fProxyLQRState(Aperturbed.M_Rows, Allocator.Temp);
                state.S.Data.CopyFrom(Sprev.Data);
                state.populated = true;
                var info = LQR.lqr(in Aperturbed, in B, in Q, in R, ref K, ref state);
                itersOut[0] = info.iterations;
                statusOut[0] = (int)info.status;
                state.Dispose();
            }
        }
    }

    public static partial class LQRBenchmark
    {
        // Trivially stabilizable random instance (already stable, so any n/m/seed combination is a
        // valid LQR instance without needing a controllability check): diagonal in [0.2,0.4), off-
        // diagonal magnitude scaled 0.2/n so the Gershgorin bound stays under ~0.6 at EVERY n. Q=I, R=I.
        static void BuildInstanceFProxy(int n, int m, uint seed, bool nearMarginal,
                                        out fProxyMxN A, out fProxyMxN B, out fProxyMxN Q, out fProxyMxN R)
        {
            var rng = new Unity.Mathematics.Random(seed);
            // nearMarginal: diagonal in [0.90,0.98) pushes the spectrum toward the unit circle -- the
            // regime where the plain recursion's LINEAR convergence rate collapses and SDA's quadratic
            // convergence is supposed to earn its keep. Off-diagonal shrunk with it to stay stable.
            fProxy off = (fProxy)((nearMarginal ? 0.02 : 0.2) / n);

            A = new fProxyMxN(n, n, Allocator.Persistent);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (i == j)
                        ? (nearMarginal ? rng.NextFProxy(0.90f, 0.98f) : rng.NextFProxy(0.2f, 0.4f))
                        : rng.NextFProxy(-1f, 1f) * off;

            B = new fProxyMxN(n, m, Allocator.Persistent);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++)
                    B[i, j] = rng.NextFProxy(-1f, 1f);

            Q = new fProxyMxN(n, n, Allocator.Persistent);
            for (int i = 0; i < n; i++) Q[i, i] = (fProxy)1;

            R = new fProxyMxN(m, m, Allocator.Persistent);
            for (int i = 0; i < m; i++) R[i, i] = (fProxy)1;
        }

        static string ColdSdaFProxy(int n, int m, int reps, uint seed, bool nearMarginal = false)
        {
            BuildInstanceFProxy(n, m, seed, nearMarginal, out var A, out var B, out var Q, out var R);
            var K = new fProxyMxN(m, n, Allocator.Persistent);

            var itersOut = new NativeArray<int>(1, Allocator.Persistent);
            var statusOut = new NativeArray<int>(1, Allocator.Persistent);
            var job = new LqrColdSdaJobFProxy { A = A, B = B, Q = Q, R = R, K = K, reps = reps, itersOut = itersOut, statusOut = statusOut };
            var stat = Bench.Time(() => job.Run());
            string row = LQRBenchmarkFmt.Row("fProxy", nearMarginal ? "cold-SDA(marg)" : "cold-SDA", n, m, reps, stat, itersOut[0], statusOut[0]);

            itersOut.Dispose(); statusOut.Dispose();
            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); K.Dispose();
            return row;
        }

        static string ColdRecursionFProxy(int n, int m, int reps, uint seed, bool nearMarginal = false)
        {
            BuildInstanceFProxy(n, m, seed, nearMarginal, out var A, out var B, out var Q, out var R);
            var K = new fProxyMxN(m, n, Allocator.Persistent);

            var itersOut = new NativeArray<int>(1, Allocator.Persistent);
            var statusOut = new NativeArray<int>(1, Allocator.Persistent);
            var job = new LqrColdRecursionJobFProxy { A = A, B = B, Q = Q, R = R, K = K, reps = reps, itersOut = itersOut, statusOut = statusOut };
            var stat = Bench.Time(() => job.Run());
            string row = LQRBenchmarkFmt.Row("fProxy", nearMarginal ? "cold-rec(marg)" : "cold-recursion", n, m, reps, stat, itersOut[0], statusOut[0]);

            itersOut.Dispose(); statusOut.Dispose();
            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); K.Dispose();
            return row;
        }

        static string WarmFProxy(int n, int m, int reps, uint seed, bool nearMarginal = false)
        {
            BuildInstanceFProxy(n, m, seed, nearMarginal, out var A, out var B, out var Q, out var R);
            var K = new fProxyMxN(m, n, Allocator.Persistent);

            // untimed setup (managed thread -- LQR.lqr is plain Burst-compatible code, just not
            // Burst-JITted here; only the warm re-solve below is measured): cold-solve the unperturbed
            // system for its converged S, then perturb A by ~1e-3 relative per entry.
            var coldState = new fProxyLQRState(n, Allocator.Persistent);
            LQR.lqr(in A, in B, in Q, in R, ref K, ref coldState);
            var Sprev = new fProxyMxN(n, n, Allocator.Persistent);
            Sprev.Data.CopyFrom(coldState.S.Data);
            coldState.Dispose();

            var rng = new Unity.Mathematics.Random(seed ^ 0x9E3779B9u);
            var Aperturbed = new fProxyMxN(n, n, Allocator.Persistent);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    fProxy a = A[i, j];
                    fProxy scale = (a == (fProxy)0) ? (fProxy)1 : math.abs(a);
                    Aperturbed[i, j] = a + (fProxy)1e-3 * scale * rng.NextFProxy(-1f, 1f);
                }

            var itersOut = new NativeArray<int>(1, Allocator.Persistent);
            var statusOut = new NativeArray<int>(1, Allocator.Persistent);
            var job = new LqrWarmJobFProxy { Aperturbed = Aperturbed, B = B, Q = Q, R = R, K = K, Sprev = Sprev, reps = reps, itersOut = itersOut, statusOut = statusOut };
            var stat = Bench.Time(() => job.Run());
            string row = LQRBenchmarkFmt.Row("fProxy", nearMarginal ? "warm(marg)" : "warm(1e-3 pert)", n, m, reps, stat, itersOut[0], statusOut[0]);

            itersOut.Dispose(); statusOut.Dispose();
            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); K.Dispose();
            Sprev.Dispose(); Aperturbed.Dispose();
            return row;
        }
    }
}
