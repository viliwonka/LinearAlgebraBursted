using System;
using System.Globalization;
using System.IO;
using System.Text;

using Unity.Burst;

using Debug = UnityEngine.Debug;

namespace LinearAlgebra.Benchmarks
{
    // Shared timing + reporting infrastructure for the kernel benchmarks.
    //
    // WHY a job everywhere: a library call made from a plain managed method runs under Mono, NOT
    // Burst, and mis-measures the kernel by an order of magnitude. Every benchmark therefore runs
    // its work inside a [BurstCompile] IJob executed with .Run() (synchronous, single-thread, on
    // the calling thread but through the Burst-compiled code path) and times it through this helper.
    // CompileSynchronously = true on the jobs guarantees the first run is already native, so warm-up
    // only has to settle the caches.
    //
    // Each kernel benchmark contributes a titled Section(...) to one combined report; AllBenchmarks
    // composes them. Sizes span small (fixed-overhead-dominated) to large (cache-behaviour-dominated)
    // so a reorder/vectorisation win shows up as a flattening of the GFLOP/s column at large N.
    public static class Bench
    {
        public static readonly int[] Sizes = { 64, 128, 256, 512, 1024 };
        public const int Warmup = 3;
        public const int Runs = 9;

        public struct Stat { public double Min, Median, Mean, Max; }

        public static Stat Time(Action run)
        {
            for (int i = 0; i < Warmup; i++) run();

            var times = new double[Runs];
            var sw = new System.Diagnostics.Stopwatch();
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

        public static string Header()
        {
            return string.Format("{0,-7} {1,-6} {2,11} {3,11} {4,11} {5,11} {6,12}",
                "dtype", "N", "min(ms)", "med(ms)", "mean(ms)", "max(ms)", "GFLOP/s~");
        }

        // flops = the kernel's leading-term operation count for this N; GFLOP/s is computed from the
        // median time so the throughput column is comparable across kernels and sizes.
        public static string Row(string dtype, int n, Stat st, double flops)
        {
            double gflops = flops / (st.Median / 1000.0) / 1e9;
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-7} {1,-6} {2,11:F4} {3,11:F4} {4,11:F4} {5,11:F4} {6,12:F2}",
                dtype, n, st.Min, st.Median, st.Mean, st.Max, gflops);
        }

        // Builds the common preamble, runs `body` to fill in the kernel section(s), writes the report
        // to TestResults/<fileName>, echoes it to the Editor log, and logs the path on its own line
        // (benchmark.ps1 parses that line to know which file to print).
        public static void WriteReport(string fileName, Action<StringBuilder> body)
        {
            var sb = new StringBuilder();
            sb.AppendLine("=== LinearAlgebra kernel benchmarks ===");
            sb.AppendLine("Burst (CompileSynchronously), single-thread IJob.Run(), N x N matrices.");
            sb.AppendLine(string.Format("Warmup={0} runs, timed={1} runs (median reported).", Warmup, Runs));
            sb.AppendLine("Burst enabled: " + BurstCompiler.Options.EnableBurstCompilation);
            sb.AppendLine();

            body(sb);

            Directory.CreateDirectory("TestResults");
            string path = Path.Combine("TestResults", fileName);
            File.WriteAllText(path, sb.ToString());

            Debug.Log(sb.ToString());
            Debug.Log("Benchmark results written to " + path);
        }
    }
}
