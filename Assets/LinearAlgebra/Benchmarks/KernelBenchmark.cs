using System.Text;

namespace BULA.Benchmarks
{
    // Primitive Level-1 / Level-2 BLAS kernels in isolation -- the layer every solver bottoms out on,
    // measured directly so a reduction-vectorisation change (e.g. the 4-accumulator matVecDot rework)
    // is visible at the kernel level instead of only through a whole-solver time.
    //
    // Two families here, distinguished by their loop SHAPE (which decides whether Burst can vectorise
    // under FloatMode.Default, i.e. no reassociation):
    //   * REDUCTIONS  (vecDot, L1, L2, LInf, sum) -- a serial `acc += f(x[i])` dependency chain. A
    //     single accumulator is latency-bound and does NOT auto-vectorise; the fix is explicit multiple
    //     accumulators in the kernel source. These are the optimisation targets.
    //   * axpy (y += a*x) and vecMat (Aᵀx) -- independent per output element, so they DO vectorise
    //     already; included as the "what a vectorised kernel looks like" baseline.
    //   * GEMV (A*x) -- the matVecDot kernel; a per-row reduction (same hazard as the Level-1 reductions).
    //
    // READING IT: the tell is the float-vs-double GFLOP/s ratio at large N. float ~2x double => the
    // inner loop vectorises (4 floats vs 2 doubles per 128-bit SIMD op). float ~= double => it runs
    // serial (headroom for the accumulator rework). These benchmarks report time/throughput only,
    // never a numeric result.
    //
    // The ping-pong kernels (GEMV, vecMat, axpy) feed each output back as the next input, which would
    // let the values GROW without bound over the REPS chain and overflow to Inf/NaN -- and because
    // float overflows ~230 orders of magnitude sooner than double, that would make float spend more of
    // the chain on penalised Inf/NaN arithmetic and read as artificially slow, corrupting the very
    // float-vs-double comparison we want. So the operators are scaled to be (near-)norm-preserving:
    // the matrix by 1/sqrt(N) (largest singular value ~1) and axpy's alpha kept tiny, keeping every
    // buffer finite across all REPS so the timing reflects the kernel, not overflow handling.
    //
    // Hand-written harness half. The timed IJobs (Reduce/MatVec/Axpy Job {Float,Double}) and
    // build+measure methods (Reduce/Axpy/MatVec {Float,Double}) are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/KernelBenchmark.fProxy.cs.

    // ---- kernel selectors (constant per job invocation; the switch is hoisted out of the REPS loop) --
    // Public so the code-generated job structs (in a separate template assembly) can reference it.
    public static class Kern
    {
        public const int VecDot = 0, L1 = 1, L2 = 2, LInf = 3, Sum = 4; // Level-1 reductions
        public const int Gemv = 5, VecMat = 6;                          // Level-2 matrix-vector
        public const int Max = 7, Min = 8;                              // Level-1 max/min reductions
    }

    public static partial class KernelBenchmark
    {
        const int RepsL1 = 512; // Level-1 kernels are O(N); many back-to-back reps amortise the timer
        const int RepsL2 = 64;  // Level-2 GEMV is O(N^2); fewer reps keep the section fast

        // (selector, label, flops-per-element): the reduction's leading-term op count per vector entry.
        // sum is one add/elem; the rest are ~two (mul+add, or abs+add/max) -- approximate, used only for
        // the GFLOP/s column, whose float-vs-double RATIO (not absolute value) is the signal of interest.
        static readonly (int kind, string name, double flopPerElem)[] Reductions =
        {
            (Kern.VecDot, "vecDot", 2.0),
            (Kern.L1,     "L1",     2.0),
            (Kern.L2,     "L2",     2.0),
            (Kern.LInf,   "LInf",   2.0),
            (Kern.Sum,    "sum",    1.0),
            (Kern.Max,    "max",    1.0),
            (Kern.Min,    "min",    1.0),
        };

        public static void Run() => Bench.WriteReport("benchmark-kernels.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Primitive BLAS kernels (Level-1 & Level-2) ===");
            sb.AppendLine("Isolates the kernels every solver bottoms out on. The signal is the float-vs-double");
            sb.AppendLine("GFLOP/s ratio at large N: ~2x => the inner loop vectorises, ~1x => it runs serial");
            sb.AppendLine("(headroom for the multi-accumulator rework). Reductions (vecDot/L1/L2/LInf/sum) are");
            sb.AppendLine("serial acc-chains; GEMV is a per-row reduction; vecMat (Aᵀx) and axpy are the");
            sb.AppendLine("already-vectorising baselines.");
            sb.AppendLine(string.Format("Level-1 REPS={0}, Level-2 REPS={1} back-to-back calls per timed sample.",
                RepsL1, RepsL2));
            sb.AppendLine();

            // ---- Level-1 reductions ----
            foreach (var r in Reductions)
            {
                sb.AppendLine(string.Format("--- L1 reduction: {0} (vector length N, REPS={1}) ---", r.name, RepsL1));
                sb.AppendLine(Bench.Header());
                foreach (var n in Bench.Sizes) sb.AppendLine(ReduceFloat(n, r.kind, r.flopPerElem, RepsL1));
                foreach (var n in Bench.Sizes) sb.AppendLine(ReduceDouble(n, r.kind, r.flopPerElem, RepsL1));
                sb.AppendLine();
            }

            // ---- Level-1 axpy ----
            sb.AppendLine(string.Format("--- L1 axpy: y += a*x (vector length N, REPS={0}) ---", RepsL1));
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(AxpyFloat(n, RepsL1));
            foreach (var n in Bench.Sizes) sb.AppendLine(AxpyDouble(n, RepsL1));
            sb.AppendLine();

            // ---- Level-2 matrix-vector (N x N) ----
            sb.AppendLine(string.Format("--- L2 GEMV: y = A*x (matVecDot, NxN, REPS={0}) ---", RepsL2));
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(MatVecFloat(n, Kern.Gemv, RepsL2));
            foreach (var n in Bench.Sizes) sb.AppendLine(MatVecDouble(n, Kern.Gemv, RepsL2));
            sb.AppendLine();

            sb.AppendLine(string.Format("--- L2 vecMat: y = Aᵀx (vecMatDot, NxN, REPS={0}) ---", RepsL2));
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(MatVecFloat(n, Kern.VecMat, RepsL2));
            foreach (var n in Bench.Sizes) sb.AppendLine(MatVecDouble(n, Kern.VecMat, RepsL2));
            sb.AppendLine();
        }
    }
}
