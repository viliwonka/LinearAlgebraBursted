using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of IterativeBenchmark (timed IJob + build+measure method). The
    // dtype-agnostic harness (Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/IterativeBenchmark.cs.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGJobFProxy : IJob
    {
        public fProxyMxN A;     // n x n SPD input, NOT modified
        public fProxyN b;       // rhs, NOT modified
        public fProxyN x;       // initial guess (zeroed each Execute) / solution output
        public fProxyN r;
        public fProxyN p;
        public fProxyN Ap;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.cg(in A, in b, ref x, ref r, ref p, ref Ap, 100, 0f);
        }
    }

    public static partial class IterativeBenchmark
    {
        static string BenchFProxy(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var M   = arena.fProxyMat(n, n);    // scratch to build MᵀM
            var A   = arena.fProxyMat(n, n);    // SPD A = MᵀM + I
            var b   = arena.fProxyVec(n);
            var x   = arena.fProxyVec(n);
            var r   = arena.fProxyVec(n);
            var p   = arena.fProxyVec(n);
            var Ap  = arena.fProxyVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int row = 0; row < n; row++)
                for (int col = 0; col < n; col++)
                    M[row, col] = rng.NextFProxy(-1f, 1f);

            // A = MᵀM (guaranteed positive semi-definite)
            Blas.dot(in M, in M, ref A, true);

            // Add I: A becomes MᵀM + I (guaranteed SPD with min eigenvalue >= 1)
            for (int d = 0; d < n; d++) A[d, d] += 1f;

            for (int i = 0; i < n; i++) b[i] = rng.NextFProxy(-1f, 1f);

            var job = new CGJobFProxy { A = A, b = b, x = x, r = r, p = p, Ap = Ap };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("fProxy", n, stat);
        }
    }
}
