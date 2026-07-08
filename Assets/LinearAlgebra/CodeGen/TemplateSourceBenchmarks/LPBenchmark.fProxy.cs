#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.Sparse;
using LinearAlgebra.Gallery;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of LPBenchmark (timed IJobs + the per-section build+measure methods).
    // The dtype-agnostic harness (config sizes, row formatters, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/LPBenchmark.cs (LPBenchmarkFmt + the partial class).
    //
    // Dtype-sensitive numeric literals (slack magnitude 0.1, noise 0.05, outlier 5) are wrapped in
    // (fProxy) casts so the double build gets the true double value; the exact-in-float random-range
    // bounds (-1f/0f/1f) stay literal.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LpSolveJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN b, c, x;
        public NativeArray<ConstraintSense> senses;
        public LPMethod method;
        public int maxIter;
        public void Execute() { LP.solve(in A, in b, in c, in senses, ref x, out double obj, method, maxIter); }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LadJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN b, x;
        public LPMethod method;
        public void Execute() { LP.lad(in A, in b, ref x, out double obj, method, 0); }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct IrlsJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN b, x;
        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = (fProxy)0;   // reset the warm-start guess each timed rep
            Optimize.ladIRLS(in A, in b, ref x);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SparseLadJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
        public void Execute() { LP.lad(in A, in b, ref x, out double obj, 0); }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LpSolveSparseJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, c, x;
        public NativeArray<ConstraintSense> senses;
        public void Execute() { LP.solve(in A, in b, in c, in senses, ref x, out double obj); }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PdlpJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN lc, uc, lv, uv, c, x;
        public int maxIter;
        public double eps;
        public void Execute()
        {
            for (int j = 0; j < x.N; j++) x[j] = (fProxy)0;   // cold start each timed rep (x is PDLP's initial iterate)
            LP.pdlp(in A, in lc, in uc, in lv, in uv, in c, ref x, out double obj, maxIter, eps);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PdlpSparseJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN lc, uc, lv, uv, c, x;
        public int maxIter;
        public double eps;
        public void Execute()
        {
            for (int j = 0; j < x.N; j++) x[j] = (fProxy)0;
            LP.pdlp(in A, in lc, in uc, in lv, in uv, in c, ref x, out double obj, maxIter, eps);
        }
    }

    public static partial class LPBenchmark
    {
        // ==== Section 1: LP.solve, random dense feasible LP -- simplex vs interior point ====
        static void SectionSolveFProxy(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 1. LP.solve: random dense feasible LP (m = n/2, A>=0, Ax<=b), simplex vs interior point [fProxy] ---");
            sb.AppendLine(LPBenchmarkFmt.SolveHeader());

            foreach (var n in LPBenchmarkFmt.SolveVarsN)
            {
                int m = n / 2;
                var arena = new Arena(Allocator.Persistent);
                var A = arena.fProxyRandomMat(m, n, 0f, 1f, (uint)(n * 7919 + 11));      // nonneg -> bounded
                var x0 = arena.fProxyRandomVec(n, 0f, 1f, (uint)(n * 104729 + 7));
                var Ax0 = Blas.dot(A, x0);
                var b = arena.fProxyVec(m);
                var rng = new Random((uint)(n * 1299709 + 3));
                for (int i = 0; i < m; i++) b[i] = Ax0[i] + rng.NextFProxy((fProxy)0.1, (fProxy)1);  // slack -> x0 feasible
                var c = arena.fProxyRandomVec(n, -1f, 1f, (uint)(n * 15485863 + 5));
                var senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
                for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;

                var xS = arena.fProxyVec(n);
                var infoS = LP.solve(in A, in b, in c, in senses, ref xS, out double objS, LPMethod.Simplex);
                var jobS = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xS, method = LPMethod.Simplex, maxIter = 0 };
                var statS = Bench.Time(() => jobS.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("fProxy", n, m, "simplex", statS, infoS.iterations, objS));

                var xI = arena.fProxyVec(n);
                var infoI = LP.solve(in A, in b, in c, in senses, ref xI, out double objI, LPMethod.InteriorPoint);
                var jobI = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xI, method = LPMethod.InteriorPoint, maxIter = 0 };
                var statI = Bench.Time(() => jobI.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("fProxy", n, m, "interior-point", statI, infoI.iterations, objI));

                senses.Dispose();
                arena.Dispose();
            }
        }

        // ==== Section 2: LAD (L1) regression with outliers -- exact LP vs fast IRLS ====
        static void SectionLadFProxy(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 2. LAD (L1) regression, gross outliers: exact LP.lad (simplex/interior) vs fast ladIRLS [fProxy] ---");
            sb.AppendLine(LPBenchmarkFmt.LadHeader());

            int n = LPBenchmarkFmt.NCoef;
            foreach (var m in LPBenchmarkFmt.LadRowsM)
            {
                var arena = new Arena(Allocator.Persistent);
                var A = arena.fProxyRandomMat(m, n, -1f, 1f, (uint)(m * 7919 + 13));
                var xt = arena.fProxyRandomVec(n, -1f, 1f, (uint)(m * 104729 + 17));
                var Axt = Blas.dot(A, xt);
                var b = arena.fProxyVec(m);
                var rng = new Random((uint)(m * 1299709 + 19));
                for (int i = 0; i < m; i++)
                {
                    fProxy val = Axt[i] + rng.NextFProxy(-(fProxy)0.05, (fProxy)0.05);
                    if (i % 10 == 0) val += (fProxy)5;         // gross outlier every 10th observation
                    b[i] = val;
                }

                var xLs = arena.fProxyVec(n);
                var infoLs = LP.lad(in A, in b, ref xLs, out double objLs, LPMethod.Simplex);
                var jobLs = new LadJobFProxy { A = A, b = b, x = xLs, method = LPMethod.Simplex };
                var statLs = Bench.Time(() => jobLs.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "LP.lad-simplex", statLs, infoLs.iterations, objLs));

                var xLi = arena.fProxyVec(n);
                var infoLi = LP.lad(in A, in b, ref xLi, out double objLi, LPMethod.InteriorPoint);
                var jobLi = new LadJobFProxy { A = A, b = b, x = xLi, method = LPMethod.InteriorPoint };
                var statLi = Bench.Time(() => jobLi.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "LP.lad-interior", statLi, infoLi.iterations, objLi));

                var xIr = arena.fProxyVec(n);
                var infoIr = Optimize.ladIRLS(in A, in b, ref xIr);
                var jobIr = new IrlsJobFProxy { A = A, b = b, x = xIr };
                var statIr = Bench.Time(() => jobIr.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "ladIRLS", statIr, infoIr.iterations, infoIr.objective));

                arena.Dispose();
            }
        }

        // ==== Section 3: SPARSE LAD -- matrix-free interior point over a tall block-sparse design ====
        static void SectionSparseLadFProxy(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 3. LAD over a tall BSR design (m x n, ~8 nnz/row): dense LP.lad (simplex vs interior) " +
                          "vs sparse matrix-free interior point; dense runs only where the m x m normal matrix fits [fProxy] ---");
            sb.AppendLine(LPBenchmarkFmt.LadHeader());

            int n = LPBenchmarkFmt.SparseLadCoef;
            fProxy density = (fProxy)8 / (fProxy)n;       // ~8 nonzeros per row
            foreach (var m in LPBenchmarkFmt.SparseLadRowsM)
            {
                var arena = new Arena(Allocator.Persistent);
                var As = arena.fProxyRandomSparse(m, n, 1, density, (uint)(m * 7919 + 23));   // tall, full column rank

                // b = A x_true + small noise + a gross outlier every 10th row
                var xt = arena.fProxyRandomVec(n, -1f, 1f, (uint)(m * 104729 + 29));
                var bx = arena.fProxyVec(m);
                BSR.spMV(in As, in xt, ref bx);
                var b = arena.fProxyVec(m);
                var rng = new Random((uint)(m * 1299709 + 31));
                for (int i = 0; i < m; i++)
                {
                    fProxy val = bx[i] + rng.NextFProxy(-(fProxy)0.05, (fProxy)0.05);
                    if (i % 10 == 0) val += (fProxy)5;
                    b[i] = val;
                }

                // dense interior-point baseline only where the m x m normal matrix is still practical
                // (dense simplex omitted here -- Bland's-rule LAD simplex at m=512 is the slow tail;
                //  it is already benchmarked at appropriate sizes in Section 2)
                if (m <= LPBenchmarkFmt.SparseLadDenseCap)
                {
                    var Ad = As.ToDense(ref arena);
                    var xd = arena.fProxyVec(n);
                    var infoD = LP.lad(in Ad, in b, ref xd, out double objD, LPMethod.InteriorPoint);
                    var jobD = new LadJobFProxy { A = Ad, b = b, x = xd, method = LPMethod.InteriorPoint };
                    var statD = Bench.Time(() => jobD.Run());
                    sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "dense LP.lad-ip", statD, infoD.iterations, objD));
                }

                var xs = arena.fProxyVec(n);
                var infoS = LP.lad(in As, in b, ref xs, out double objS, 0);
                var jobS = new SparseLadJobFProxy { A = As, b = b, x = xs };
                var statS = Bench.Time(() => jobS.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "sparse LP.lad", statS, infoS.iterations, objS));

                arena.Dispose();
            }
        }

        // ==== Section 4: PDLP vs simplex vs interior point on the SAME dense feasible LP as Section 1 ====
        static void SectionPdlpDenseFProxy(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 4. PDLP (matrix-free first-order PDHG) vs simplex vs interior point on the SAME dense " +
                          "feasible LP (Section 1's construction; PDLP tol " + LPBenchmarkFmt.PdlpEps.ToString("E0") +
                          ", cap " + LPBenchmarkFmt.PdlpMaxIter + " iters) [fProxy] ---");
            sb.AppendLine(LPBenchmarkFmt.SolveHeader());

            fProxy INF = (fProxy)1e30;
            foreach (var n in LPBenchmarkFmt.SolveVarsN)
            {
                int m = n / 2;
                var arena = new Arena(Allocator.Persistent);
                var A = arena.fProxyRandomMat(m, n, 0f, 1f, (uint)(n * 7919 + 11));       // SAME problem as Section 1
                var x0 = arena.fProxyRandomVec(n, 0f, 1f, (uint)(n * 104729 + 7));
                var Ax0 = Blas.dot(A, x0);
                var b = arena.fProxyVec(m);
                var rng = new Random((uint)(n * 1299709 + 3));
                for (int i = 0; i < m; i++) b[i] = Ax0[i] + rng.NextFProxy((fProxy)0.1, (fProxy)1);
                var c = arena.fProxyRandomVec(n, -1f, 1f, (uint)(n * 15485863 + 5));
                var senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
                for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;

                var xS = arena.fProxyVec(n);
                var infoS = LP.solve(in A, in b, in c, in senses, ref xS, out double objS, LPMethod.Simplex);
                var jobS = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xS, method = LPMethod.Simplex, maxIter = 0 };
                var statS = Bench.Time(() => jobS.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("fProxy", n, m, "simplex", statS, infoS.iterations, objS));

                var xI = arena.fProxyVec(n);
                var infoI = LP.solve(in A, in b, in c, in senses, ref xI, out double objI, LPMethod.InteriorPoint);
                var jobI = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xI, method = LPMethod.InteriorPoint, maxIter = 0 };
                var statI = Bench.Time(() => jobI.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("fProxy", n, m, "interior-point", statI, infoI.iterations, objI));

                // PDLP on the same LP as two-sided bounds: -inf <= A x <= b, 0 <= x <= +inf
                var lc = arena.fProxyVec(m); var uc = arena.fProxyVec(m);
                for (int i = 0; i < m; i++) { lc[i] = -INF; uc[i] = b[i]; }
                var lv = arena.fProxyVec(n); var uv = arena.fProxyVec(n);
                for (int j = 0; j < n; j++) { lv[j] = (fProxy)0; uv[j] = INF; }
                var xP = arena.fProxyVec(n);
                var infoP = LP.pdlp(in A, in lc, in uc, in lv, in uv, in c, ref xP, out double objP, LPBenchmarkFmt.PdlpMaxIter, LPBenchmarkFmt.PdlpEps);
                var jobP = new PdlpJobFProxy { A = A, lc = lc, uc = uc, lv = lv, uv = uv, c = c, x = xP, maxIter = LPBenchmarkFmt.PdlpMaxIter, eps = LPBenchmarkFmt.PdlpEps };
                var statP = Bench.Time(() => jobP.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("fProxy", n, m, "pdlp", statP, infoP.iterations, objP));

                senses.Dispose();
                arena.Dispose();
            }
        }

        // ==== Section 5: sparse PDLP vs sparse interior point on a block-sparse covering LP ====
        static void SectionPdlpSparseFProxy(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 5. Block-sparse covering LP (min cx s.t. A x >= b, x >= 0; A,b,c >= 0, ~" +
                          LPBenchmarkFmt.PdlpSparseNnzPerRow + " nnz/row): sparse interior point vs matrix-free PDLP " +
                          "(tol " + LPBenchmarkFmt.PdlpEps.ToString("E0") + ", cap " + LPBenchmarkFmt.PdlpMaxIter + " iters) [fProxy] ---");
            sb.AppendLine(LPBenchmarkFmt.SolveHeader());

            fProxy INF = (fProxy)1e30;
            int nnzPer = LPBenchmarkFmt.PdlpSparseNnzPerRow;
            foreach (var m in LPBenchmarkFmt.PdlpSparseM)
            {
                int n = m;                                        // square covering LP
                var arena = new Arena(Allocator.Persistent);

                // build a nonneg BSR (1x1 blocks): every row gets exactly nnzPer positive entries at
                // distinct columns (stride coprime-ish to n), so A >= 0 and each row is satisfiable.
                var builder = arena.fProxyBSRBuilder(m, n, 1, 1, m * nnzPer);
                var rng = new Random((uint)(m * 2654435 + 41));
                for (int i = 0; i < m; i++)
                {
                    int baseCol = (int)(rng.NextUInt() % (uint)n);
                    for (int t = 0; t < nnzPer; t++)
                    {
                        int j = (baseCol + t * 97) % n;
                        var blk = arena.fProxyMat(1, 1);
                        blk[0, 0] = (fProxy)0.1 + rng.NextFProxy(0f, 1f) * (fProxy)0.9;   // in (0.1, 1]
                        builder.AddBlock(i, j, in blk);
                    }
                }
                var As = builder.ToBSR(ref arena);

                var b = arena.fProxyVec(m);
                for (int i = 0; i < m; i++) b[i] = (fProxy)1 + rng.NextFProxy(0f, 1f);      // demand in [1, 2]
                var c = arena.fProxyVec(n);
                for (int j = 0; j < n; j++) c[j] = (fProxy)0.5 + rng.NextFProxy(0f, 1f);    // cost in [0.5, 1.5]
                var senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
                for (int i = 0; i < m; i++) senses[i] = ConstraintSense.GreaterEqual;

                var xI = arena.fProxyVec(n);
                var infoI = LP.solve(in As, in b, in c, in senses, ref xI, out double objI);
                var jobI = new LpSolveSparseJobFProxy { A = As, b = b, c = c, senses = senses, x = xI };
                var statI = Bench.Time(() => jobI.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("fProxy", n, m, "sparse ip", statI, infoI.iterations, objI));

                // PDLP: b <= A x <= +inf, 0 <= x <= +inf
                var lc = arena.fProxyVec(m); var uc = arena.fProxyVec(m);
                for (int i = 0; i < m; i++) { lc[i] = b[i]; uc[i] = INF; }
                var lv = arena.fProxyVec(n); var uv = arena.fProxyVec(n);
                for (int j = 0; j < n; j++) { lv[j] = (fProxy)0; uv[j] = INF; }
                var xP = arena.fProxyVec(n);
                var infoP = LP.pdlp(in As, in lc, in uc, in lv, in uv, in c, ref xP, out double objP, LPBenchmarkFmt.PdlpMaxIter, LPBenchmarkFmt.PdlpEps);
                var jobP = new PdlpSparseJobFProxy { A = As, lc = lc, uc = uc, lv = lv, uv = uv, c = c, x = xP, maxIter = LPBenchmarkFmt.PdlpMaxIter, eps = LPBenchmarkFmt.PdlpEps };
                var statP = Bench.Time(() => jobP.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("fProxy", n, m, "sparse pdlp", statP, infoP.iterations, objP));

                senses.Dispose();
                arena.Dispose();
            }
        }
    }
}
