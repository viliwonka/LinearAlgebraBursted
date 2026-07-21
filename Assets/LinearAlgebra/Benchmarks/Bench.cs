using System;
using System.Globalization;
using System.IO;
using System.Text;

using Unity.Burst;

using LinearAlgebra;

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
        public const int Warmup = 1;
        public const int Runs = 4;

        public struct Stat { public double Min, Median, Mean, Max; public bool RanUnderMono; }

        // A job that falls back to Mono cannot be caught with try/catch here: Unity never rethrows
        // a job Execute() exception synchronously to the .Run() caller, it only logs it later (see
        // BurstProbe's doc comment). BurstProbe.RanUnderMono is the synchronous signal instead --
        // reset before timing, polled after every .Run() call so a mid-run fallback is caught
        // without wasting the remaining warmup/timed runs.
        public static Stat Time(Action run)
        {
            BurstProbe.RanUnderMono = false;

            for (int i = 0; i < Warmup; i++)
            {
                run();
                if (BurstProbe.RanUnderMono) return new Stat { RanUnderMono = true };
            }

            var times = new double[Runs];
            var sw = new System.Diagnostics.Stopwatch();
            for (int i = 0; i < Runs; i++)
            {
                sw.Restart();
                run();
                sw.Stop();
                if (BurstProbe.RanUnderMono) return new Stat { RanUnderMono = true };
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
            int n = s.Length;
            // True median: average of the two central samples for an even count, middle one for odd.
            double median = (n & 1) == 0 ? 0.5 * (s[n / 2 - 1] + s[n / 2]) : s[n / 2];
            return new Stat
            {
                Min = s[0],
                Max = s[s.Length - 1],
                Median = median,
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
            if (st.RanUnderMono) return NotBurstedRow(dtype, n);
            double gflops = flops / (st.Median / 1000.0) / 1e9;
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-7} {1,-6} {2,11:F4} {3,11:F4} {4,11:F4} {5,11:F4} {6,12:F2}",
                dtype, n, st.Min, st.Median, st.Mean, st.Max, gflops);
        }

        // Time-only header/row for iterative algorithms (SVD, eigen) whose flop count is
        // data-dependent (varies with iteration/sweep count), so a GFLOP/s column would be misleading.
        public static string HeaderTime()
        {
            return string.Format("{0,-7} {1,-6} {2,11} {3,11} {4,11} {5,11}",
                "dtype", "N", "min(ms)", "med(ms)", "mean(ms)", "max(ms)");
        }

        public static string RowTime(string dtype, int n, Stat st)
        {
            if (st.RanUnderMono) return NotBurstedRow(dtype, n);
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-7} {1,-6} {2,11:F4} {3,11:F4} {4,11:F4} {5,11:F4}",
                dtype, n, st.Min, st.Median, st.Mean, st.Max);
        }

        // Distinct, unmissable line for a job that fell back to Mono instead of Burst -- printed in
        // place of a timing row so the sweep keeps going instead of reporting a bogus (interpreter)
        // timing next to genuine Burst-native numbers.
        static string NotBurstedRow(string dtype, int n)
        {
            return string.Format("{0,-7} {1,-6} NOT BURSTED -- job fell back to Mono, see BurstProbe / Editor log", dtype, n);
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
