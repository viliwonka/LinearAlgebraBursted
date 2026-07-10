using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of LQRBenchmark (the timed IJobs + the instance builder + build+measure
    // methods). The dtype-agnostic harness (sizes/seeds, row formatter, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/LQRBenchmark.cs.
    //
    // Three variants, all through the PUBLIC Control API only (Benchmarks has no InternalsVisibleTo
    // grant -- same constraint QPBenchmark.float.cs notes for qpActiveSetCore):
    //   - cold-SDA: the plain Control.lqr(...) overload (structure-preserving doubling).
    //   - cold-recursion: the warm Control.lqr(..., ref state) overload, but with a FRESH state whose
    //     S is force-seeded at zero and populated=true (both public fields) -- this makes the warm
    //     overload take its plain-fixed-point-recursion branch starting cold, the "naive" baseline the
    //     spec wants SDA/warm compared against, reached without touching Control's internal
    //     RiccatiIterate directly.
    //   - warm: an UNTIMED cold solve (managed thread, not inside the timed job) seeds Sprev with the
    //     converged S, A is perturbed ~1e-3 relative, then the TIMED job re-seeds a fresh state from
    //     Sprev each Execute and warm-solves the perturbed system -- same re-copy-before-timed-call
    //     idiom CholeskyBenchmark's face-off section uses for its destructive decompInPlace calls.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LqrColdSdaJobFloat : IJob
    {
        public floatMxN A, B, Q, R, K;
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;

        public void Execute()
        {
            var info = Control.lqr(in A, in B, in Q, in R, ref K);
            itersOut[0] = info.iterations;
            statusOut[0] = (int)info.status;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LqrColdRecursionJobFloat : IJob
    {
        public floatMxN A, B, Q, R, K;
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;

        public void Execute()
        {
            var state = new floatLQRState(A.M_Rows, Allocator.Temp);   // S starts zero-filled
            state.populated = true;                                    // forces the plain-recursion branch, cold-started
            var info = Control.lqr(in A, in B, in Q, in R, ref K, ref state);
            itersOut[0] = info.iterations;
            statusOut[0] = (int)info.status;
            state.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LqrWarmJobFloat : IJob
    {
        public floatMxN Aperturbed, B, Q, R, K;
        public floatMxN Sprev;   // pre-perturbation converged S; re-seeded into a fresh state each Execute
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;

        public void Execute()
        {
            var state = new floatLQRState(Aperturbed.M_Rows, Allocator.Temp);
            state.S.Data.CopyFrom(Sprev.Data);
            state.populated = true;
            var info = Control.lqr(in Aperturbed, in B, in Q, in R, ref K, ref state);
            itersOut[0] = info.iterations;
            statusOut[0] = (int)info.status;
            state.Dispose();
        }
    }

    public static partial class LQRBenchmark
    {
        // Trivially stabilizable random instance (already stable, so any n/m/seed combination is a
        // valid LQR instance without needing a controllability check): diagonal in [0.2,0.4), off-
        // diagonal in [-0.05,0.05) keeps the Gershgorin bound comfortably under 1 up to n=12. Q=I, R=I.
        static void BuildInstanceFloat(int n, int m, uint seed, in Arena arena,
                                        out floatMxN A, out floatMxN B, out floatMxN Q, out floatMxN R)
        {
            var rng = new Unity.Mathematics.Random(seed);

            A = arena.floatMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (i == j) ? rng.NextFloat(0.2f, 0.4f) : rng.NextFloat(-0.05f, 0.05f);

            B = arena.floatMat(n, m);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++)
                    B[i, j] = rng.NextFloat(-1f, 1f);

            Q = arena.floatMat(n, n);
            for (int i = 0; i < n; i++) Q[i, i] = (float)1;

            R = arena.floatMat(m, m);
            for (int i = 0; i < m; i++) R[i, i] = (float)1;
        }

        static string ColdSdaFloat(int n, int m, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            BuildInstanceFloat(n, m, seed, in arena, out var A, out var B, out var Q, out var R);
            var K = arena.floatMat(m, n);

            var itersOut = new NativeArray<int>(1, Allocator.Persistent);
            var statusOut = new NativeArray<int>(1, Allocator.Persistent);
            var job = new LqrColdSdaJobFloat { A = A, B = B, Q = Q, R = R, K = K, itersOut = itersOut, statusOut = statusOut };
            var stat = Bench.Time(() => job.Run());
            string row = LQRBenchmarkFmt.Row("float", "cold-SDA", n, m, stat, itersOut[0], statusOut[0]);

            itersOut.Dispose(); statusOut.Dispose(); arena.Dispose();
            return row;
        }

        static string ColdRecursionFloat(int n, int m, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            BuildInstanceFloat(n, m, seed, in arena, out var A, out var B, out var Q, out var R);
            var K = arena.floatMat(m, n);

            var itersOut = new NativeArray<int>(1, Allocator.Persistent);
            var statusOut = new NativeArray<int>(1, Allocator.Persistent);
            var job = new LqrColdRecursionJobFloat { A = A, B = B, Q = Q, R = R, K = K, itersOut = itersOut, statusOut = statusOut };
            var stat = Bench.Time(() => job.Run());
            string row = LQRBenchmarkFmt.Row("float", "cold-recursion", n, m, stat, itersOut[0], statusOut[0]);

            itersOut.Dispose(); statusOut.Dispose(); arena.Dispose();
            return row;
        }

        static string WarmFloat(int n, int m, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            BuildInstanceFloat(n, m, seed, in arena, out var A, out var B, out var Q, out var R);
            var K = arena.floatMat(m, n);

            // untimed setup (managed thread -- Control.lqr is plain Burst-compatible code, just not
            // Burst-JITted here; only the warm re-solve below is measured): cold-solve the unperturbed
            // system for its converged S, then perturb A by ~1e-3 relative per entry.
            var coldState = new floatLQRState(n, Allocator.Persistent);
            Control.lqr(in A, in B, in Q, in R, ref K, ref coldState);
            var Sprev = arena.floatMat(n, n);
            Sprev.Data.CopyFrom(coldState.S.Data);
            coldState.Dispose();

            var rng = new Unity.Mathematics.Random(seed ^ 0x9E3779B9u);
            var Aperturbed = arena.floatMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    float a = A[i, j];
                    float scale = (a == (float)0) ? (float)1 : math.abs(a);
                    Aperturbed[i, j] = a + (float)1e-3 * scale * rng.NextFloat(-1f, 1f);
                }

            var itersOut = new NativeArray<int>(1, Allocator.Persistent);
            var statusOut = new NativeArray<int>(1, Allocator.Persistent);
            var job = new LqrWarmJobFloat { Aperturbed = Aperturbed, B = B, Q = Q, R = R, K = K, Sprev = Sprev, itersOut = itersOut, statusOut = statusOut };
            var stat = Bench.Time(() => job.Run());
            string row = LQRBenchmarkFmt.Row("float", "warm(1e-3 pert)", n, m, stat, itersOut[0], statusOut[0]);

            itersOut.Dispose(); statusOut.Dispose(); arena.Dispose();
            return row;
        }
    }
}
