#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of LPBenchmark (timed IJobs + the per-section build+measure methods).
    // The dtype-agnostic harness (config sizes, row formatters, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/LPBenchmark.cs (LPBenchmarkFmt + the partial class).
    //
    // Dtype-sensitive numeric literals (slack magnitude 0.1, noise 0.05, outlier 5) are wrapped in
    // (float) casts so the double build gets the true double value; the exact-in-float random-range
    // bounds (-1f/0f/1f) stay literal.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LpSolveJobFloat : IJob
    {
        public floatMxN A;
        public floatN b, c, x;
        public NativeArray<ConstraintSense> senses;
        public LPMethod method;
        public int maxIter;
        public void Execute() { LP.solve(in A, in b, in c, in senses, ref x, out double obj, method, maxIter); }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LadJobFloat : IJob
    {
        public floatMxN A;
        public floatN b, x;
        public LPMethod method;
        public void Execute() { LP.lad(in A, in b, ref x, out double obj, method, 0); }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct IrlsJobFloat : IJob
    {
        public floatMxN A;
        public floatN b, x;
        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = (float)0;   // reset the warm-start guess each timed rep
            Optimize.ladIRLS(in A, in b, ref x);
        }
    }

    public static partial class LPBenchmark
    {
        // ==== Section 1: LP.solve, random dense feasible LP -- simplex vs interior point ====
        static void SectionSolveFloat(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 1. LP.solve: random dense feasible LP (m = n/2, A>=0, Ax<=b), simplex vs interior point [float] ---");
            sb.AppendLine(LPBenchmarkFmt.SolveHeader());

            foreach (var n in LPBenchmarkFmt.SolveVarsN)
            {
                int m = n / 2;
                var arena = new Arena(Allocator.Persistent);
                var A = arena.floatRandomMat(m, n, 0f, 1f, (uint)(n * 7919 + 11));      // nonneg -> bounded
                var x0 = arena.floatRandomVec(n, 0f, 1f, (uint)(n * 104729 + 7));
                var Ax0 = Blas.dot(A, x0);
                var b = arena.floatVec(m);
                var rng = new Random((uint)(n * 1299709 + 3));
                for (int i = 0; i < m; i++) b[i] = Ax0[i] + rng.NextFloat((float)0.1, (float)1);  // slack -> x0 feasible
                var c = arena.floatRandomVec(n, -1f, 1f, (uint)(n * 15485863 + 5));
                var senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
                for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;

                var xS = arena.floatVec(n);
                var infoS = LP.solve(in A, in b, in c, in senses, ref xS, out double objS, LPMethod.Simplex);
                var jobS = new LpSolveJobFloat { A = A, b = b, c = c, senses = senses, x = xS, method = LPMethod.Simplex, maxIter = 0 };
                var statS = Bench.Time(() => jobS.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("float", n, m, "simplex", statS, infoS.iterations, objS));

                var xI = arena.floatVec(n);
                var infoI = LP.solve(in A, in b, in c, in senses, ref xI, out double objI, LPMethod.InteriorPoint);
                var jobI = new LpSolveJobFloat { A = A, b = b, c = c, senses = senses, x = xI, method = LPMethod.InteriorPoint, maxIter = 0 };
                var statI = Bench.Time(() => jobI.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("float", n, m, "interior-point", statI, infoI.iterations, objI));

                senses.Dispose();
                arena.Dispose();
            }
        }

        // ==== Section 2: LAD (L1) regression with outliers -- exact LP vs fast IRLS ====
        static void SectionLadFloat(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 2. LAD (L1) regression, gross outliers: exact LP.lad (simplex/interior) vs fast ladIRLS [float] ---");
            sb.AppendLine(LPBenchmarkFmt.LadHeader());

            int n = LPBenchmarkFmt.NCoef;
            foreach (var m in LPBenchmarkFmt.LadRowsM)
            {
                var arena = new Arena(Allocator.Persistent);
                var A = arena.floatRandomMat(m, n, -1f, 1f, (uint)(m * 7919 + 13));
                var xt = arena.floatRandomVec(n, -1f, 1f, (uint)(m * 104729 + 17));
                var Axt = Blas.dot(A, xt);
                var b = arena.floatVec(m);
                var rng = new Random((uint)(m * 1299709 + 19));
                for (int i = 0; i < m; i++)
                {
                    float val = Axt[i] + rng.NextFloat(-(float)0.05, (float)0.05);
                    if (i % 10 == 0) val += (float)5;         // gross outlier every 10th observation
                    b[i] = val;
                }

                var xLs = arena.floatVec(n);
                var infoLs = LP.lad(in A, in b, ref xLs, out double objLs, LPMethod.Simplex);
                var jobLs = new LadJobFloat { A = A, b = b, x = xLs, method = LPMethod.Simplex };
                var statLs = Bench.Time(() => jobLs.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("float", m, n, "LP.lad-simplex", statLs, infoLs.iterations, objLs));

                var xLi = arena.floatVec(n);
                var infoLi = LP.lad(in A, in b, ref xLi, out double objLi, LPMethod.InteriorPoint);
                var jobLi = new LadJobFloat { A = A, b = b, x = xLi, method = LPMethod.InteriorPoint };
                var statLi = Bench.Time(() => jobLi.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("float", m, n, "LP.lad-interior", statLi, infoLi.iterations, objLi));

                var xIr = arena.floatVec(n);
                var infoIr = Optimize.ladIRLS(in A, in b, ref xIr);
                var jobIr = new IrlsJobFloat { A = A, b = b, x = xIr };
                var statIr = Bench.Time(() => jobIr.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("float", m, n, "ladIRLS", statIr, infoIr.iterations, infoIr.objective));

                arena.Dispose();
            }
        }
    }
}
