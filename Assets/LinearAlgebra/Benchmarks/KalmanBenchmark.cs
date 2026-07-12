using System.Globalization;
using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Shared, dtype-agnostic config + table formatter for KalmanBenchmark. Public so the code-generated
    // per-dtype build methods (in a separate template assembly) can reach the sizes/seeds and row writer.
    public static class KalmanBenchmarkFmt
    {
        // n in {2,4,6,12} (gamedev-scale) plus n in {24,32} (quadrotor-state-estimator scale: pose 3 +
        // vel 3 + attitude 3 + rates 3 + 4x rotor speed/current + gyro/accel biases lands around n=24-32).
        // m_meas in {1, n/2} for the small rows (unchanged from before -- n=2 collapses to a single m=1
        // since n/2==1 there too); m_meas in {n/4, n/2, n} for the two large rows (n=24: m=6,12,24; n=32:
        // m=8,16,32 -- the m=n row is heavy sensor fusion, e.g. IMU+GPS+visual-pose+rangefinders all
        // fused at once). steps is per row: 1000 (unchanged) for every existing small row, 200 for the
        // new n=24/32 rows -- their per-step covariance math is O(n^3)/O(m^3), so a shorter loop keeps
        // total wall time sane while us/step (med(ms)/steps*1000) stays directly derivable from the
        // reported steps column either way. ssgReps is Section 3's own per-row rep count, similarly
        // unchanged (100) for the small rows and reduced (20) for the large ones.
        public static readonly (int n, int m, int steps, int ssgReps)[] Sizes =
        {
            (2, 1, 1000, 100),
            (4, 1, 1000, 100),
            (4, 2, 1000, 100),
            (6, 1, 1000, 100),
            (6, 3, 1000, 100),
            (12, 1, 1000, 100),
            (12, 6, 1000, 100),
            (24, 6, 200, 20),
            (24, 12, 200, 20),
            (24, 24, 200, 20),
            (32, 8, 200, 20),
            (32, 16, 200, 20),
            (32, 32, 200, 20),
        };
        public static readonly uint[] Seeds = { 11u, 22u, 33u, 44u, 55u, 66u, 77u, 88u, 99u, 110u, 121u, 132u, 143u };
        public const int NonlinearSteps = 100;

        // status crosses the hand-written/template assembly boundary as a raw int (Burst-legal
        // enum-to-int cast), same CS0012 reason LQRBenchmarkFmt.Row's own doc comment explains. -1 is a
        // sentinel for "no KFInfo available" (predictFixed/updateFixed return void).
        public static string KfStatusName(int status)
        {
            if (status == -1) return "n/a";
            return ((KFStatus)status) switch
            {
                KFStatus.Ok => "Ok",
                KFStatus.InnovationSolveFailed => "Failed",
                _ => "Unknown",
            };
        }

        public static string Header() => string.Format("{0,-7} {1,-12} {2,4} {3,4} {4,7} {5,11} {6,11} {7,10} {8,10}",
            "dtype", "variant", "n", "m", "steps", "med(ms)", "min(ms)", "us/step", "status");

        // med/min are for the WHOLE `steps`-iteration loop the timed job runs; us/step = med(ms)/steps*1000
        // is the per-frame budget number. steps is reported per row so us/step stays derivable even where
        // it differs from the default (the n>=24 rows).
        public static string Row(string dtype, string variant, int n, int m, int steps, Bench.Stat st, int status) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-12} {2,4} {3,4} {4,7} {5,11:F4} {6,11:F4} {7,10:F4} {8,10}",
                dtype, variant, n, m, steps, st.Median, st.Min, st.Median / steps * 1000.0, KfStatusName(status));
    }

    // ================================================================================================
    // Kalman filter family (Kalman.predict/update, predictFixed/updateFixed, steadyStateGain, ekfPredict/
    // ekfUpdate, ukfPredict/ukfUpdate). Four sections:
    //   1. Linear predict+update, full covariance path -- the per-frame budget number for a plain KF.
    //   2. predictFixed+updateFixed (steady-state gain, no covariance math) at the SAME sizes -- shows
    //      the speedup a fixed-gain filter buys once steadyStateGain has converged.
    //   3. steadyStateGain itself -- a one-shot cost (typically paid once at filter setup, not per frame).
    //   4. EKF vs UKF, pendulum (n=2,m=1) and a synthetic drone-scale ring model (n=12,m=6) -- the
    //      Jacobian-free UKF's own per-step cost against the analytic-Jacobian EKF.
    //
    // Sections 1-3 span n<=12 (gamedev-scale) through n in {24,32} (quadrotor-state-estimator scale --
    // see KalmanBenchmarkFmt.Sizes' own comment), with m up to n (heavy sensor fusion) at the two large
    // sizes.
    //
    // Hand-written harness half. The timed IJobs and build+measure methods are code-generated per dtype
    // from Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/KalmanBenchmark.fProxy.cs.
    // ================================================================================================
    public static partial class KalmanBenchmark
    {
        public static void Run() => Bench.WriteReport("benchmark-kalman.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Kalman filter family (predict/update, fixed-gain, steadyStateGain, EKF vs UKF) ===");
            sb.AppendLine("Sections 1-2: linear predict/update (full covariance) vs predictFixed/updateFixed");
            sb.AppendLine("(steady-state gain, no covariance math) at the SAME sizes -- the fixed-gain row shows the");
            sb.AppendLine("per-frame speedup once steadyStateGain has converged. n<=12 rows are gamedev-scale; n in");
            sb.AppendLine("{24,32} rows are quadrotor-state-estimator scale (m up to n = heavy sensor fusion) and use");
            sb.AppendLine("a shorter per-Execute step count (see the steps column) to keep total wall time sane --");
            sb.AppendLine("us/step is still directly derivable from med(ms) and steps either way. Section 3:");
            sb.AppendLine("steadyStateGain's own one-shot cost (typically paid once at setup). Section 4: EKF vs UKF");
            sb.AppendLine("per-step cost, pendulum (n=2) and a synthetic drone-scale ring model (n=12,m=6) -- the cost");
            sb.AppendLine("of skipping analytic Jacobians entirely, at both a tiny and a drone-like state size.");
            sb.AppendLine();

            sb.AppendLine("--- 1. Linear predict+update, full covariance path [fProxy] ---");
            sb.AppendLine(KalmanBenchmarkFmt.Header());
            for (int i = 0; i < KalmanBenchmarkFmt.Sizes.Length; i++)
            {
                var (n, m, steps, _) = KalmanBenchmarkFmt.Sizes[i];
                sb.AppendLine(KfCycleFloat(n, m, steps, KalmanBenchmarkFmt.Seeds[i]));
            }
            for (int i = 0; i < KalmanBenchmarkFmt.Sizes.Length; i++)
            {
                var (n, m, steps, _) = KalmanBenchmarkFmt.Sizes[i];
                sb.AppendLine(KfCycleDouble(n, m, steps, KalmanBenchmarkFmt.Seeds[i]));
            }

            sb.AppendLine();
            sb.AppendLine("--- 2. predictFixed+updateFixed (steady-state gain, no covariance math) [fProxy] ---");
            sb.AppendLine(KalmanBenchmarkFmt.Header());
            for (int i = 0; i < KalmanBenchmarkFmt.Sizes.Length; i++)
            {
                var (n, m, steps, _) = KalmanBenchmarkFmt.Sizes[i];
                sb.AppendLine(KfFixedFloat(n, m, steps, KalmanBenchmarkFmt.Seeds[i]));
            }
            for (int i = 0; i < KalmanBenchmarkFmt.Sizes.Length; i++)
            {
                var (n, m, steps, _) = KalmanBenchmarkFmt.Sizes[i];
                sb.AppendLine(KfFixedDouble(n, m, steps, KalmanBenchmarkFmt.Seeds[i]));
            }

            sb.AppendLine();
            sb.AppendLine("--- 3. steadyStateGain: one-shot cost [fProxy] ---");
            sb.AppendLine(LQRBenchmarkFmt.Header());
            for (int i = 0; i < KalmanBenchmarkFmt.Sizes.Length; i++)
            {
                var (n, m, _, ssgReps) = KalmanBenchmarkFmt.Sizes[i];
                sb.AppendLine(SteadyStateGainFloat(n, m, ssgReps, KalmanBenchmarkFmt.Seeds[i]));
            }
            for (int i = 0; i < KalmanBenchmarkFmt.Sizes.Length; i++)
            {
                var (n, m, _, ssgReps) = KalmanBenchmarkFmt.Sizes[i];
                sb.AppendLine(SteadyStateGainDouble(n, m, ssgReps, KalmanBenchmarkFmt.Seeds[i]));
            }

            sb.AppendLine();
            sb.AppendLine("--- 4. EKF vs UKF: pendulum (n=2,m=1) and a synthetic ring model (n=12,m=6, drone-scale) [fProxy] ---");
            sb.AppendLine(KalmanBenchmarkFmt.Header());
            sb.AppendLine(EkfCycleFloat(KalmanBenchmarkFmt.NonlinearSteps));
            sb.AppendLine(UkfCycleFloat(KalmanBenchmarkFmt.NonlinearSteps));
            sb.AppendLine(EkfRingFloat(12, 6, KalmanBenchmarkFmt.NonlinearSteps, 999u));
            sb.AppendLine(UkfRingFloat(12, 6, KalmanBenchmarkFmt.NonlinearSteps, 999u));
            sb.AppendLine(EkfCycleDouble(KalmanBenchmarkFmt.NonlinearSteps));
            sb.AppendLine(UkfCycleDouble(KalmanBenchmarkFmt.NonlinearSteps));
            sb.AppendLine(EkfRingDouble(12, 6, KalmanBenchmarkFmt.NonlinearSteps, 999u));
            sb.AppendLine(UkfRingDouble(12, 6, KalmanBenchmarkFmt.NonlinearSteps, 999u));
        }
    }
}
