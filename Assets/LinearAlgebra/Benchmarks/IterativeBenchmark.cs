using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;

// NOTE: PCG (preconditioned Conjugate Gradient) is not yet implemented in the library;
// only plain CG is benchmarked here.

namespace LinearAlgebra.Benchmarks
{
    // Conjugate Gradient solver for dense SPD systems. The cost per iteration is one dense GEMV
    // (A·p) plus vector ops — all O(n²). With maxIterations = 100 and tolerance = 0 every timed
    // sample runs exactly 100 iterations for a deterministic, representative measurement.
    //
    // SPD construction: A = MᵀM + I (M random n×n in [-1,1]). This is guaranteed SPD (all
    // eigenvalues >= 1), and the Frobenius-norm-n random M makes the condition number grow with n,
    // so 100 iterations do not converge early even at small sizes — timing is iteration-count-bounded.
    //
    // A and b are built once (A is `in` — CG does not modify it). x is zeroed at the start of each
    // Execute so every timed sample begins from the same zero initial guess.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGJobFloat : IJob
    {
        public floatMxN A;     // n x n SPD input, NOT modified
        public floatN b;       // rhs, NOT modified
        public floatN x;       // initial guess (zeroed each Execute) / solution output
        public floatN r;
        public floatN p;
        public floatN Ap;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Solvers.conjugateGradient(in A, in b, ref x, ref r, ref p, ref Ap, 100, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b;
        public doubleN x;
        public doubleN r;
        public doubleN p;
        public doubleN Ap;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0.0;
            Solvers.conjugateGradient(in A, in b, ref x, ref r, ref p, ref Ap, 100, 0.0);
        }
    }

    public static class IterativeBenchmark
    {
        public static void Run() => Bench.WriteReport("benchmark-iterative.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Conjugate Gradient (dense SPD A = MᵀM + I; 100 iterations, tol=0; ms) ===");
            sb.AppendLine("    Each iteration: one dense GEMV (A·p) + vector ops. " +
                          "tol=0 forces all 100 iterations for deterministic timing.");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchDouble(n));
            sb.AppendLine();
        }

        static string BenchFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var M   = arena.floatMat(n, n);    // scratch to build MᵀM
            var A   = arena.floatMat(n, n);    // SPD A = MᵀM + I
            var b   = arena.floatVec(n);
            var x   = arena.floatVec(n);
            var r   = arena.floatVec(n);
            var p   = arena.floatVec(n);
            var Ap  = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int row = 0; row < n; row++)
                for (int col = 0; col < n; col++)
                    M[row, col] = rng.NextFloat(-1f, 1f);

            // A = MᵀM (guaranteed positive semi-definite)
            Linear_OP.dot(in M, in M, ref A, true);

            // Add I: A becomes MᵀM + I (guaranteed SPD with min eigenvalue >= 1)
            for (int d = 0; d < n; d++) A[d, d] += 1f;

            for (int i = 0; i < n; i++) b[i] = rng.NextFloat(-1f, 1f);

            var job = new CGJobFloat { A = A, b = b, x = x, r = r, p = p, Ap = Ap };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string BenchDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var M   = arena.doubleMat(n, n);
            var A   = arena.doubleMat(n, n);
            var b   = arena.doubleVec(n);
            var x   = arena.doubleVec(n);
            var r   = arena.doubleVec(n);
            var p   = arena.doubleVec(n);
            var Ap  = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int row = 0; row < n; row++)
                for (int col = 0; col < n; col++)
                    M[row, col] = rng.NextDouble(-1.0, 1.0);

            // A = MᵀM (guaranteed positive semi-definite)
            Linear_OP.dot(in M, in M, ref A, true);

            // Add I: A becomes MᵀM + I (guaranteed SPD with min eigenvalue >= 1)
            for (int d = 0; d < n; d++) A[d, d] += 1.0;

            for (int i = 0; i < n; i++) b[i] = rng.NextDouble(-1.0, 1.0);

            var job = new CGJobDouble { A = A, b = b, x = x, r = r, p = p, Ap = Ap };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }
    }
}
