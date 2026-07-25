using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using BULA;
using BULA.Control;

namespace BULA.Benchmarks
{
    // GENERATED per-dtype half of MPCBenchmark (the timed IJobs + the instance builder + build+measure
    // methods). The dtype-agnostic harness (sizes/seeds, row formatter, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/MPCBenchmark.cs.
    //
    // Three job shapes:
    //   - MpcWarmFrameJobFProxy: ONE receding-horizon frame per Execute() call (solve, then advance the
    //     REAL plant x <- A x + B u0). s/x/u0 are NativeArray-backed fields, so state survives across
    //     repeated job.Run() calls from managed code -- the harness runs this job untimed
    //     (MPCBenchmarkFmt.WarmPrewarmFrames times) to burn off the cold-start/churn transient, THEN
    //     times it via Bench.Time, so each of Bench.Time's own 1 warmup + 4 timed calls is one already-
    //     warm frame.
    //   - MpcColdJobFProxy: resets the warm-start carry (z/uPlan/wstatus/populated) to its fresh-
    //     construction values before every rep, so every solve pays the first-ever-solve cost at the
    //     SAME x0 -- fProxyMPCState's own fields are public for exactly this purpose (mirrors
    //     LqrColdRecursionJobFProxy's force-cold-started state in LQRBenchmark.fProxy.cs).
    //   - MpcConstructJobFProxy: times fProxyMPCState's constructor itself (terminal DARE + Phi/Gamma/H
    //     condensing), constructing and disposing a fresh instance every rep.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MpcWarmFrameJobFProxy : IJob
    {
        public fProxyMPCState s;
        public fProxyMxN A, B;      // REAL (not condensed) dynamics, to advance the plant after each solve
        public fProxyN x;            // current plant state -- carried across repeated job.Run() calls
        public fProxyN u0;
        public fProxyN reference;
        public fProxyN xNext, Bu;    // plant-advance scratch
        public NativeArray<int> itersOut;
        public NativeArray<int> changesOut;
        public NativeArray<int> statusOut;

        public void Execute()
        {
            var info = MPC.solve(ref s, in x, in reference, ref u0);
            itersOut[0] = info.iterations;
            changesOut[0] = info.activeSetChanges;
            statusOut[0] = (int)info.status;

            Blas.dot(in A, in x, ref xNext);
            Blas.dot(in B, in u0, ref Bu);
            xNext.addInPlace(Bu);
            x.CopyFrom(xNext);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MpcColdJobFProxy : IJob
    {
        public fProxyMPCState s;
        public fProxyN x0;
        public fProxyN reference;
        public fProxyN u0;
        public int reps;
        public NativeArray<int> itersOut;
        public NativeArray<int> changesOut;
        public NativeArray<int> statusOut;

        public void Execute()
        {
            for (int r = 0; r < reps; r++)
            {
                // Reset the warm-start carry to its fresh-construction state (zero z/uPlan, every
                // wstatus row Inactive -- WorkingSetStatus.Inactive == 0 -- populated=false) so this rep
                // pays the FIRST-EVER-solve cost at x0, not a warm-started one.
                for (int i = 0; i < s.z.N; i++) s.z[i] = (fProxy)0;
                for (int i = 0; i < s.uPlan.N; i++) s.uPlan[i] = (fProxy)0;
                for (int i = 0; i < s.wstatus.Length; i++) s.wstatus[i] = 0;
                s.populated = false;

                var info = MPC.solve(ref s, in x0, in reference, ref u0);
                itersOut[0] = info.iterations;
                changesOut[0] = info.activeSetChanges;
                statusOut[0] = (int)info.status;
            }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MpcConstructJobFProxy : IJob
    {
        public fProxyMxN A, B, Q, R;
        public fProxyN uLo, uHi;
        public int N;
        public int reps;
        public NativeArray<double> checksumOut;   // consumed so the construction can't be DCE'd away

        public void Execute()
        {
            double checksum = 0;
            for (int r = 0; r < reps; r++)
            {
                var s = new fProxyMPCState(A.M_Rows, B.N_Cols, N, Allocator.Temp, in A, in B, in Q, in R, in uLo, in uHi);
                checksum += (double)s.Kinf[0, 0];
                s.Dispose();
            }
            checksumOut[0] = checksum;
        }
    }

    public static partial class MPCBenchmark
    {
        // Trivially stabilizable random plant (diagonal in [0.2,0.4), off-diagonal scaled 0.2/n) -- the
        // SAME recipe LQRBenchmark.fProxy.cs's own BuildInstanceFProxy uses, so the terminal DARE
        // fProxyMPCState's constructor solves is guaranteed to converge at every size below. Q=I, R=I.
        static void BuildPlantFProxy(int n, int m, uint seed,
                                     out fProxyMxN A, out fProxyMxN B, out fProxyMxN Q, out fProxyMxN R)
        {
            var rng = new Unity.Mathematics.Random(seed);
            fProxy off = (fProxy)(0.2 / n);

            A = new fProxyMxN(n, n, Allocator.Persistent);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (i == j) ? rng.NextFProxy(0.2f, 0.4f) : rng.NextFProxy(-1f, 1f) * off;

            B = new fProxyMxN(n, m, Allocator.Persistent);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++)
                    B[i, j] = rng.NextFProxy(-1f, 1f);

            Q = new fProxyMxN(n, n, Allocator.Persistent);
            for (int i = 0; i < n; i++) Q[i, i] = (fProxy)1;

            R = new fProxyMxN(m, m, Allocator.Persistent);
            for (int i = 0; i < m; i++) R[i, i] = (fProxy)1;
        }

        // Builds the box-only or box+soft-wall fProxyMPCState for a given size. The soft wall (when
        // enabled) is a single row selecting state[0] (C=[1,0,...,0], d=2) -- with x0[0]=4 (see
        // WarmFrameFProxy/ColdSolveFProxy), the wall is violated along the initial transient, exercising
        // the soft-row general-constraint machinery before the plant's own contractive dynamics settle
        // it near the origin.
        static fProxyMPCState BuildStateFProxy(int n, int m, int N, in fProxyMxN A, in fProxyMxN B,
                                               in fProxyMxN Q, in fProxyMxN R, in fProxyN uLo, in fProxyN uHi,
                                               bool softWall)
        {
            if (!softWall)
                return new fProxyMPCState(n, m, N, Allocator.Persistent, in A, in B, in Q, in R, in uLo, in uHi);

            var C = new fProxyMxN(1, n, Allocator.Persistent);
            C[0, 0] = (fProxy)1;
            var d = GenerateOP.fProxyVec(1, (fProxy)2, Allocator.Persistent);
            var s = new fProxyMPCState(n, m, N, Allocator.Persistent, in A, in B, in Q, in R, in uLo, in uHi, in C, in d);
            C.Dispose(); d.Dispose();
            return s;
        }

        // ==== Section 1: warm steady-state per-frame cost (the headline) ====
        static string WarmFrameFProxy(int N, int n, int m, uint seed, bool softWall)
        {
            BuildPlantFProxy(n, m, seed, out var A, out var B, out var Q, out var R);
            var uLo = GenerateOP.fProxyVec(m, (fProxy)(-1), Allocator.Persistent);
            var uHi = GenerateOP.fProxyVec(m, (fProxy)1, Allocator.Persistent);
            var s = BuildStateFProxy(n, m, N, in A, in B, in Q, in R, in uLo, in uHi, softWall);

            var x = new fProxyN(n, Allocator.Persistent);
            x[0] = (fProxy)4;
            var reference = new fProxyN(n, Allocator.Persistent);   // zero -- track to the origin
            var u0 = new fProxyN(m, Allocator.Persistent, true);
            var xNext = new fProxyN(n, Allocator.Persistent, true);
            var Bu = new fProxyN(n, Allocator.Persistent, true);

            var itersOut = new NativeArray<int>(1, Allocator.Persistent);
            var changesOut = new NativeArray<int>(1, Allocator.Persistent);
            var statusOut = new NativeArray<int>(1, Allocator.Persistent);
            var job = new MpcWarmFrameJobFProxy
            {
                s = s, A = A, B = B, x = x, u0 = u0, reference = reference, xNext = xNext, Bu = Bu,
                itersOut = itersOut, changesOut = changesOut, statusOut = statusOut,
            };

            // Untimed: burn off the cold-start + active-set-churn transient before any timed call.
            for (int i = 0; i < MPCBenchmarkFmt.WarmPrewarmFrames; i++) job.Run();

            var stat = Bench.Time(() => job.Run());
            string row = MPCBenchmarkFmt.Row("fProxy", softWall ? "warm+wall" : "warm-box", N, n, m, 1, stat,
                                             itersOut[0], changesOut[0], statusOut[0]);

            itersOut.Dispose(); changesOut.Dispose(); statusOut.Dispose();
            s.Dispose();
            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); uLo.Dispose(); uHi.Dispose();
            x.Dispose(); reference.Dispose(); u0.Dispose(); xNext.Dispose(); Bu.Dispose();
            return row;
        }

        // ==== Section 2: cold solve cost (fresh warm-start carry every call) ====
        static string ColdSolveFProxy(int N, int n, int m, uint seed, int reps, bool softWall)
        {
            BuildPlantFProxy(n, m, seed, out var A, out var B, out var Q, out var R);
            var uLo = GenerateOP.fProxyVec(m, (fProxy)(-1), Allocator.Persistent);
            var uHi = GenerateOP.fProxyVec(m, (fProxy)1, Allocator.Persistent);
            var s = BuildStateFProxy(n, m, N, in A, in B, in Q, in R, in uLo, in uHi, softWall);

            var x0 = new fProxyN(n, Allocator.Persistent);
            x0[0] = (fProxy)4;
            var reference = new fProxyN(n, Allocator.Persistent);
            var u0 = new fProxyN(m, Allocator.Persistent, true);

            var itersOut = new NativeArray<int>(1, Allocator.Persistent);
            var changesOut = new NativeArray<int>(1, Allocator.Persistent);
            var statusOut = new NativeArray<int>(1, Allocator.Persistent);
            var job = new MpcColdJobFProxy
            {
                s = s, x0 = x0, reference = reference, u0 = u0, reps = reps,
                itersOut = itersOut, changesOut = changesOut, statusOut = statusOut,
            };
            var stat = Bench.Time(() => job.Run());
            string row = MPCBenchmarkFmt.Row("fProxy", softWall ? "cold+wall" : "cold-box", N, n, m, reps, stat,
                                             itersOut[0], changesOut[0], statusOut[0]);

            itersOut.Dispose(); changesOut.Dispose(); statusOut.Dispose();
            s.Dispose();
            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); uLo.Dispose(); uHi.Dispose();
            x0.Dispose(); reference.Dispose(); u0.Dispose();
            return row;
        }

        // ==== Section 3: fProxyMPCState construction cost (one-shot) ====
        static string ConstructFProxy(int N, int n, int m, uint seed, int reps)
        {
            BuildPlantFProxy(n, m, seed, out var A, out var B, out var Q, out var R);
            var uLo = GenerateOP.fProxyVec(m, (fProxy)(-1), Allocator.Persistent);
            var uHi = GenerateOP.fProxyVec(m, (fProxy)1, Allocator.Persistent);

            var checksumOut = new NativeArray<double>(1, Allocator.Persistent);
            var job = new MpcConstructJobFProxy { A = A, B = B, Q = Q, R = R, uLo = uLo, uHi = uHi, N = N, reps = reps, checksumOut = checksumOut };
            var stat = Bench.Time(() => job.Run());
            string row = MPCBenchmarkFmt.ConstructionRow("fProxy", N, n, m, reps, stat);

            checksumOut.Dispose();
            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); uLo.Dispose(); uHi.Dispose();
            return row;
        }
    }
}
