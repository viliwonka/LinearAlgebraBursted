using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // Iterative spectral algorithms: one-sided Jacobi SVD, cyclic-Jacobi symmetric eigen, and the
    // general-matrix QR-iteration eigenvalues (elmhes + Francis hqr). Their cost is data-dependent
    // (sweep / iteration count), so only ms is reported, and sizes are smaller than the direct
    // decompositions because Jacobi is O(sweeps * n^3) with strided column access. Each Execute copies
    // a pristine source so every timed sample does identical (and identically-converging) work.
    //
    // Both float and double variants are benched so the float/double timing ratio diagnoses
    // SIMD vectorisation: a vectorised float path should run ~1.5-2x faster than double;
    // non-vectorised paths run at roughly equal speed.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdJobFloat : IJob
    {
        public floatMxN U;
        public floatMxN Src;
        public floatN S;
        public floatMxN V;

        public void Execute()
        {
            int rows = U.M_Rows, cols = U.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    U[r, c] = Src[r, c];
            SVD.svdDecomposition(ref U, ref S, ref V);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdJobDouble : IJob
    {
        public doubleMxN U;
        public doubleMxN Src;
        public doubleN S;
        public doubleMxN V;

        public void Execute()
        {
            int rows = U.M_Rows, cols = U.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    U[r, c] = Src[r, c];
            SVD.svdDecomposition(ref U, ref S, ref V);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EigJacobiJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN Src;
        public floatN E;
        public floatMxN V;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            Eigen.eigenDecomposition(ref A, ref E, ref V);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EigJacobiJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Src;
        public doubleN E;
        public doubleMxN V;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            Eigen.eigenDecomposition(ref A, ref E, ref V);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdValuesJobFloat : IJob
    {
        public floatMxN A;     // not modified (svdValues copies into the augmented matrix)
        public floatN S;

        public void Execute() => SVD.svdValues(in A, ref S);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdValuesJobDouble : IJob
    {
        public doubleMxN A;    // not modified (svdValues copies into the augmented matrix)
        public doubleN S;

        public void Execute() => SVD.svdValues(in A, ref S);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EigSymJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN Src;
        public floatN E;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            Eigen.eigenvaluesSymmetric(ref A, ref E);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EigSymJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Src;
        public doubleN E;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            Eigen.eigenvaluesSymmetric(ref A, ref E);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EigSymVecJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN Src;
        public floatN E;
        public floatMxN V;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            Eigen.eigenSymmetric(ref A, ref E, ref V);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EigSymVecJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Src;
        public doubleN E;
        public doubleMxN V;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            Eigen.eigenSymmetric(ref A, ref E, ref V);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EigQRJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN Src;
        public floatN Re;
        public floatN Im;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            Eigen.eigenvaluesQR(ref A, ref Re, ref Im, 100);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EigQRJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Src;
        public doubleN Re;
        public doubleN Im;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            Eigen.eigenvaluesQR(ref A, ref Re, ref Im, 100);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdGKJobFloat : IJob
    {
        public floatMxN A;     // input, not modified (svdGolubKahan takes A `in`)
        public floatMxN U;
        public floatN S;
        public floatMxN V;

        public void Execute() => SVD.svdGolubKahan(in A, ref U, ref S, ref V);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdGKJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN U;
        public doubleN S;
        public doubleMxN V;

        public void Execute() => SVD.svdGolubKahan(in A, ref U, ref S, ref V);
    }

    public static class EigenSvdBenchmark
    {
        static readonly int[] Sizes = { 32, 64, 128, 256 };

        public static void Run() => Bench.WriteReport("benchmark-eigensvd.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== One-sided Jacobi SVD (svdDecomposition; iterative, ms only) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Sizes) sb.AppendLine(Svd(n));
            foreach (var n in Sizes) sb.AppendLine(SvdD(n));
            sb.AppendLine();

            sb.AppendLine("=== Golub-Kahan full SVD (svdGolubKahan; bidiag + implicit-shift QR, ms) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Sizes) sb.AppendLine(SvdGK(n));
            foreach (var n in Sizes) sb.AppendLine(SvdGKD(n));
            sb.AppendLine();

            sb.AppendLine("=== SVD singular values only (svdValues, augmented Householder; ms) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Sizes) sb.AppendLine(SvdVals(n));
            foreach (var n in Sizes) sb.AppendLine(SvdValsD(n));
            sb.AppendLine();

            sb.AppendLine("=== Cyclic-Jacobi symmetric eigen (eigenDecomposition; iterative, ms only) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Sizes) sb.AppendLine(EigJacobi(n));
            foreach (var n in Sizes) sb.AppendLine(EigJacobiD(n));
            sb.AppendLine();

            sb.AppendLine("=== Householder symmetric eigenvalues (eigenvaluesSymmetric; values only, ms) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Sizes) sb.AppendLine(EigSym(n));
            foreach (var n in Sizes) sb.AppendLine(EigSymD(n));
            sb.AppendLine();

            sb.AppendLine("=== Householder symmetric eigen + vectors (eigenSymmetric; ms) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Sizes) sb.AppendLine(EigSymVec(n));
            foreach (var n in Sizes) sb.AppendLine(EigSymVecD(n));
            sb.AppendLine();

            sb.AppendLine("=== General eigenvalues, QR iteration (eigenvaluesQR; iterative, ms only) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Sizes) sb.AppendLine(EigQR(n));
            foreach (var n in Sizes) sb.AppendLine(EigQRD(n));
            sb.AppendLine();
        }

        static string SvdGK(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var U = arena.floatMat(n, n);
            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFloat(-1f, 1f);

            var job = new SvdGKJobFloat { A = A, U = U, S = S, V = V };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string SvdGKD(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var U = arena.doubleMat(n, n);
            var S = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextDouble(-1.0, 1.0);

            var job = new SvdGKJobDouble { A = A, U = U, S = S, V = V };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        static string Svd(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var U = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);
            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);

            var job = new SvdJobFloat { U = U, Src = Src, S = S, V = V };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string SvdD(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var U = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);
            var S = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);

            var job = new SvdJobDouble { U = U, Src = Src, S = S, V = V };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        static string EigJacobi(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);
            var E = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    float v = rng.NextFloat(-1f, 1f);
                    Src[i, j] = v;
                    Src[j, i] = v;              // exactly symmetric
                }

            var job = new EigJacobiJobFloat { A = A, Src = Src, E = E, V = V };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string EigJacobiD(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);
            var E = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    double v = rng.NextDouble(-1.0, 1.0);
                    Src[i, j] = v;
                    Src[j, i] = v;              // exactly symmetric
                }

            var job = new EigJacobiJobDouble { A = A, Src = Src, E = E, V = V };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        static string SvdVals(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var S = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFloat(-1f, 1f);

            var job = new SvdValuesJobFloat { A = A, S = S };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string SvdValsD(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var S = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextDouble(-1.0, 1.0);

            var job = new SvdValuesJobDouble { A = A, S = S };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        static string EigSym(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);
            var E = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    float v = rng.NextFloat(-1f, 1f);
                    Src[i, j] = v;
                    Src[j, i] = v;              // exactly symmetric
                }

            var job = new EigSymJobFloat { A = A, Src = Src, E = E };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string EigSymD(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);
            var E = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    double v = rng.NextDouble(-1.0, 1.0);
                    Src[i, j] = v;
                    Src[j, i] = v;              // exactly symmetric
                }

            var job = new EigSymJobDouble { A = A, Src = Src, E = E };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        static string EigSymVec(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);
            var E = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    float v = rng.NextFloat(-1f, 1f);
                    Src[i, j] = v;
                    Src[j, i] = v;
                }

            var job = new EigSymVecJobFloat { A = A, Src = Src, E = E, V = V };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string EigSymVecD(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);
            var E = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    double v = rng.NextDouble(-1.0, 1.0);
                    Src[i, j] = v;
                    Src[j, i] = v;
                }

            var job = new EigSymVecJobDouble { A = A, Src = Src, E = E, V = V };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        static string EigQR(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);
            var Re = arena.floatVec(n);
            var Im = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);

            var job = new EigQRJobFloat { A = A, Src = Src, Re = Re, Im = Im };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string EigQRD(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);
            var Re = arena.doubleVec(n);
            var Im = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);

            var job = new EigQRJobDouble { A = A, Src = Src, Re = Re, Im = Im };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }
    }
}
