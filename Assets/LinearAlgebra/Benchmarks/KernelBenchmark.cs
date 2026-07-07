using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;
using LinearAlgebra.Internal;

namespace LinearAlgebra.Benchmarks
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

    // ---- kernel selectors (constant per job invocation; the switch is hoisted out of the REPS loop) --
    static class Kern
    {
        public const int VecDot = 0, L1 = 1, L2 = 2, LInf = 3, Sum = 4; // Level-1 reductions
        public const int Gemv = 5, VecMat = 6;                          // Level-2 matrix-vector
    }

    // ---- Level-1 reduction microbench: REPS back-to-back reductions over a length-N vector ------------
    //
    // Each reduction returns a scalar; a lone scalar could be dead-code-eliminated, and a loop-invariant
    // reduction over unchanging inputs could be hoisted to a single evaluation. Both are defeated the
    // same way: fold every result into `acc`, feed a negligible (1e-30-scaled, so no overflow) slice of
    // acc back into the input each rep (loop-carried dependency -> cannot hoist/collapse), and finally
    // store acc into the external `sink` buffer (-> cannot eliminate).

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ReduceJobFloat : IJob
    {
        public floatN a, b, sink;
        public int kind, reps;
        public void Execute()
        {
            int n = a.N;
            float acc = 0f;
            for (int k = 0; k < reps; k++)
            {
                switch (kind)
                {
                    case Kern.VecDot: acc += Blas.dot(a, b);  break;
                    case Kern.L1:     acc += Norms.L1(a);     break;
                    case Kern.L2:     acc += Norms.L2(a);     break;
                    case Kern.LInf:   acc += Norms.LInf(a);   break;
                    case Kern.Sum:    acc += Stats.sum(a);    break;
                }
                a[k % n] += acc * 1e-30f;
            }
            sink[0] = acc;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ReduceJobDouble : IJob
    {
        public doubleN a, b, sink;
        public int kind, reps;
        public void Execute()
        {
            int n = a.N;
            double acc = 0.0;
            for (int k = 0; k < reps; k++)
            {
                switch (kind)
                {
                    case Kern.VecDot: acc += Blas.dot(a, b);  break;
                    case Kern.L1:     acc += Norms.L1(a);     break;
                    case Kern.L2:     acc += Norms.L2(a);     break;
                    case Kern.LInf:   acc += Norms.LInf(a);   break;
                    case Kern.Sum:    acc += Stats.sum(a);    break;
                }
                a[k % n] += acc * 1e-30;
            }
            sink[0] = acc;
        }
    }

    // ---- Level-2 matrix-vector microbench: REPS back-to-back matvecs, ping-ponging x<->y -------------
    //
    // Ping-pong (each matvec feeds the next, alternating destination) both defeats dead-store
    // elimination and keeps the buffers live-out through external NativeArray memory, so no sink needed.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MatVecJobFloat : IJob
    {
        public floatMxN A;
        public floatN x, y;
        public int kind, reps;
        public void Execute()
        {
            for (int k = 0; k < reps; k++)
            {
                if (kind == Kern.Gemv)
                {
                    if ((k & 1) == 0) Blas.dot(in A, in x, ref y);   // y = A x
                    else              Blas.dot(in A, in y, ref x);
                }
                else // VecMat: Aᵀx via vector*matrix
                {
                    if ((k & 1) == 0) Blas.dot(in x, in A, ref y);   // y = Aᵀ x
                    else              Blas.dot(in y, in A, ref x);
                }
            }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MatVecJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN x, y;
        public int kind, reps;
        public void Execute()
        {
            for (int k = 0; k < reps; k++)
            {
                if (kind == Kern.Gemv)
                {
                    if ((k & 1) == 0) Blas.dot(in A, in x, ref y);
                    else              Blas.dot(in A, in y, ref x);
                }
                else
                {
                    if ((k & 1) == 0) Blas.dot(in x, in A, ref y);
                    else              Blas.dot(in y, in A, ref x);
                }
            }
        }
    }

    // ---- Level-1 axpy microbench: REPS back-to-back y += a*x, ping-ponging which vector is updated ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct AxpyJobFloat : IJob
    {
        public floatN x, y;
        public float alpha;
        public int reps;
        public unsafe void Execute()
        {
            int n = x.N;
            for (int k = 0; k < reps; k++)
            {
                if ((k & 1) == 0) UnsafeOP.axpy(y.Data.Ptr, x.Data.Ptr, alpha, n); // y += a x
                else              UnsafeOP.axpy(x.Data.Ptr, y.Data.Ptr, alpha, n);
            }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct AxpyJobDouble : IJob
    {
        public doubleN x, y;
        public double alpha;
        public int reps;
        public unsafe void Execute()
        {
            int n = x.N;
            for (int k = 0; k < reps; k++)
            {
                if ((k & 1) == 0) UnsafeOP.axpy(y.Data.Ptr, x.Data.Ptr, alpha, n);
                else              UnsafeOP.axpy(x.Data.Ptr, y.Data.Ptr, alpha, n);
            }
        }
    }

    public static class KernelBenchmark
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
                foreach (var n in Bench.Sizes) sb.AppendLine(ReduceFloat(n, r.kind, r.flopPerElem));
                foreach (var n in Bench.Sizes) sb.AppendLine(ReduceDouble(n, r.kind, r.flopPerElem));
                sb.AppendLine();
            }

            // ---- Level-1 axpy ----
            sb.AppendLine(string.Format("--- L1 axpy: y += a*x (vector length N, REPS={0}) ---", RepsL1));
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(AxpyFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(AxpyDouble(n));
            sb.AppendLine();

            // ---- Level-2 matrix-vector (N x N) ----
            sb.AppendLine(string.Format("--- L2 GEMV: y = A*x (matVecDot, NxN, REPS={0}) ---", RepsL2));
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(MatVecFloat(n, Kern.Gemv));
            foreach (var n in Bench.Sizes) sb.AppendLine(MatVecDouble(n, Kern.Gemv));
            sb.AppendLine();

            sb.AppendLine(string.Format("--- L2 vecMat: y = Aᵀx (vecMatDot, NxN, REPS={0}) ---", RepsL2));
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(MatVecFloat(n, Kern.VecMat));
            foreach (var n in Bench.Sizes) sb.AppendLine(MatVecDouble(n, Kern.VecMat));
            sb.AppendLine();
        }

        // ---- Level-1 reduction runners ----
        static string ReduceFloat(int n, int kind, double flopPerElem)
        {
            var arena = new Arena(Allocator.Persistent);
            var a = arena.floatVec(n);
            var b = arena.floatVec(n);
            var sink = arena.floatVec(1);
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++) { a[i] = rng.NextFloat(-1f, 1f); b[i] = rng.NextFloat(-1f, 1f); }

            var job = new ReduceJobFloat { a = a, b = b, sink = sink, kind = kind, reps = RepsL1 };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, RepsL1 * flopPerElem * n);
        }

        static string ReduceDouble(int n, int kind, double flopPerElem)
        {
            var arena = new Arena(Allocator.Persistent);
            var a = arena.doubleVec(n);
            var b = arena.doubleVec(n);
            var sink = arena.doubleVec(1);
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++) { a[i] = rng.NextDouble(-1.0, 1.0); b[i] = rng.NextDouble(-1.0, 1.0); }

            var job = new ReduceJobDouble { a = a, b = b, sink = sink, kind = kind, reps = RepsL1 };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, RepsL1 * flopPerElem * n);
        }

        // ---- Level-1 axpy runners (2 flops/elem: one multiply + one add) ----
        static string AxpyFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var x = arena.floatVec(n);
            var y = arena.floatVec(n);
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++) { x[i] = rng.NextFloat(-1f, 1f); y[i] = rng.NextFloat(-1f, 1f); }

            var job = new AxpyJobFloat { x = x, y = y, alpha = 0.001f, reps = RepsL1 };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, RepsL1 * 2.0 * n);
        }

        static string AxpyDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var x = arena.doubleVec(n);
            var y = arena.doubleVec(n);
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++) { x[i] = rng.NextDouble(-1.0, 1.0); y[i] = rng.NextDouble(-1.0, 1.0); }

            var job = new AxpyJobDouble { x = x, y = y, alpha = 0.001, reps = RepsL1 };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, RepsL1 * 2.0 * n);
        }

        // ---- Level-2 matrix-vector runners (2 N^2 flops/call) ----
        static string MatVecFloat(int n, int kind)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var x = arena.floatVec(n);
            var y = arena.floatVec(n);
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            float s = 1f / Unity.Mathematics.math.sqrt(n); // norm-preserving scale: keeps the ping-pong finite
            for (int i = 0; i < n; i++)
            {
                x[i] = rng.NextFloat(-1f, 1f);
                for (int j = 0; j < n; j++) A[i, j] = rng.NextFloat(-1f, 1f) * s;
            }

            var job = new MatVecJobFloat { A = A, x = x, y = y, kind = kind, reps = RepsL2 };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, RepsL2 * 2.0 * n * n);
        }

        static string MatVecDouble(int n, int kind)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var x = arena.doubleVec(n);
            var y = arena.doubleVec(n);
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            double s = 1.0 / Unity.Mathematics.math.sqrt(n); // norm-preserving scale: keeps the ping-pong finite
            for (int i = 0; i < n; i++)
            {
                x[i] = rng.NextDouble(-1.0, 1.0);
                for (int j = 0; j < n; j++) A[i, j] = rng.NextDouble(-1.0, 1.0) * s;
            }

            var job = new MatVecJobDouble { A = A, x = x, y = y, kind = kind, reps = RepsL2 };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, RepsL2 * 2.0 * n * n);
        }
    }
}
