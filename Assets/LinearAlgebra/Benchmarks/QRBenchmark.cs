using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

using Debug = UnityEngine.Debug;

namespace LinearAlgebra.Benchmarks
{
    // QR Householder factorization benchmark.
    //
    // WHY a job: library calls made from a plain managed method run under Mono, NOT Burst,
    // so they would mis-measure the kernel by an order of magnitude. The work therefore runs
    // inside a [BurstCompile] IJob executed with .Run() (synchronous, single-thread, on the
    // calling thread but through the Burst-compiled code path). CompileSynchronously = true
    // guarantees the FIRST run is already native, so warm-up only has to settle the caches.
    //
    // Each Execute() copies a pristine source matrix into the working matrix and then factors
    // it, so every timed sample does identical work (qrDecomposition overwrites its input with
    // the orthogonal factor). The copy is O(N^2) against an O(N^3) factorization, i.e. < 1% for
    // N >= 128; it is included in the reported time and noted in the header.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRJobFloat : IJob
    {
        public floatMxN Q;
        public floatMxN R;
        public floatMxN Src;

        public void Execute()
        {
            int rows = Q.M_Rows, cols = Q.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    Q[r, c] = Src[r, c];

            OrthoOP.qrDecomposition(ref Q, ref R);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRJobDouble : IJob
    {
        public doubleMxN Q;
        public doubleMxN R;
        public doubleMxN Src;

        public void Execute()
        {
            int rows = Q.M_Rows, cols = Q.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    Q[r, c] = Src[r, c];

            OrthoOP.qrDecomposition(ref Q, ref R);
        }
    }

    public static class QRBenchmark
    {
        // sizes and repetition counts. Big sizes (512, 1024) keep the run honest about cache
        // behaviour; small ones expose fixed overheads. Median over `Runs` after `Warmup`.
        static readonly int[] Sizes = { 64, 128, 256, 512, 1024 };
        const int Warmup = 3;
        const int Runs = 9;

        struct Stat { public double Min, Median, Mean, Max; }

        // -executeMethod entry point (see Tools/benchmark.ps1).
        public static void Run()
        {
            var sb = new StringBuilder();
            sb.AppendLine("=== QR Householder factorization benchmark ===");
            sb.AppendLine("Burst (CompileSynchronously), single-thread IJob.Run(), N x N matrices.");
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "Warmup={0} runs, timed={1} runs; time = full factorization (copy-in + qrDecomposition).", Warmup, Runs));
            sb.AppendLine("Burst enabled: " + BurstCompiler.Options.EnableBurstCompilation);
            sb.AppendLine();
            sb.AppendLine(string.Format("{0,-7} {1,-6} {2,11} {3,11} {4,11} {5,11} {6,12}",
                "dtype", "N", "min(ms)", "med(ms)", "mean(ms)", "max(ms)", "GFLOP/s~"));

            foreach (var n in Sizes)
                sb.AppendLine(BenchFloat(n));
            foreach (var n in Sizes)
                sb.AppendLine(BenchDouble(n));

            sb.AppendLine();
            sb.AppendLine("GFLOP/s~ uses the standard leading term (4/3)*N^3 for square Householder QR (approximate).");

            Directory.CreateDirectory("TestResults");
            string path = Path.Combine("TestResults", "benchmark-qr.txt");
            File.WriteAllText(path, sb.ToString());

            Debug.Log(sb.ToString());
            Debug.Log("Benchmark results written to " + path);
        }

        static string BenchFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.floatMat(n, n);
            var R = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < n; d++)
                Src[d, d] += n;                 // diagonal dominance => full rank, no zero-column early-out

            var job = new QRJobFloat { Q = Q, R = R, Src = Src };
            var stat = TimeJob(() => job.Run());

            arena.Dispose();
            return Format("float", n, stat);
        }

        static string BenchDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.doubleMat(n, n);
            var R = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRJobDouble { Q = Q, R = R, Src = Src };
            var stat = TimeJob(() => job.Run());

            arena.Dispose();
            return Format("double", n, stat);
        }

        static Stat TimeJob(Action run)
        {
            for (int i = 0; i < Warmup; i++) run();

            var times = new double[Runs];
            var sw = new Stopwatch();
            for (int i = 0; i < Runs; i++)
            {
                sw.Restart();
                run();
                sw.Stop();
                times[i] = sw.Elapsed.TotalMilliseconds;
            }
            return Summarize(times);
        }

        static Stat Summarize(double[] t)
        {
            var s = (double[])t.Clone();
            Array.Sort(s);
            double sum = 0;
            for (int i = 0; i < s.Length; i++) sum += s[i];
            return new Stat
            {
                Min = s[0],
                Max = s[s.Length - 1],
                Median = s[s.Length / 2],
                Mean = sum / s.Length,
            };
        }

        static string Format(string dtype, int n, Stat st)
        {
            double flops = (4.0 / 3.0) * n * (double)n * n;
            double gflops = flops / (st.Median / 1000.0) / 1e9;
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-7} {1,-6} {2,11:F4} {3,11:F4} {4,11:F4} {5,11:F4} {6,12:F2}",
                dtype, n, st.Min, st.Median, st.Mean, st.Max, gflops);
        }
    }
}
