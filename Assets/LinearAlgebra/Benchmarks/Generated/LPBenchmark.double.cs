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
    // (double) casts so the double build gets the true double value; the exact-in-float random-range
    // bounds (-1f/0f/1f) stay literal.
    //
    // Every job below carries its OWN reporting outputs (objOut/itersOut/statusOut, length-1 arrays)
    // written from inside Execute(). This is deliberate: the report used to harvest objective/iters/
    // status via a SEPARATE plain managed call to LP.solve/LP.lad/LP.pdlp before ever timing the Burst
    // job -- i.e. every row solved the SAME problem TWICE, once fully Mono-interpreted. That is fine at
    // n=24 but catastrophic at n=384 (seconds per solve) and worse for PDLP at its 50000-iter cap
    // (minutes) -- an extended benchmark run measured 13+ minutes and was killed because of it. Bench.
    // Time already runs the job once as a warmup before the 4 timed reps, so the outputs are populated
    // as a natural side effect of the SAME Burst-native call the report already needed to time -- no
    // second solve, managed or otherwise. status is written as `(int)info.status` inside Execute (an
    // enum-to-int cast is Burst-legal); LPBenchmarkFmt.InfeasRow casts it back on the harness side (see
    // that method's doc comment for why the raw int, not the enum, crosses the template/harness
    // assembly boundary).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LpSolveJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, c, x;
        public NativeArray<ConstraintSense> senses;
        public LPMethod method;
        public int maxIter;
        public NativeArray<double> objOut;
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;
        public void Execute()
        {
            var info = LP.solve(in A, in b, in c, in senses, ref x, out double obj, method, maxIter);
            objOut[0] = obj;
            itersOut[0] = info.iterations;
            statusOut[0] = (int)info.status;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LadJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, x;
        public LPMethod method;
        public NativeArray<double> objOut;
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;
        public void Execute()
        {
            var info = LP.lad(in A, in b, ref x, out double obj, method, 0);
            // Honest L1 residual: RECOMPUTE sum_i |A x - b| from the RETURNED x rather than reporting
            // LPInfo.objective/obj, which equals the true residual ONLY at a converged optimum. A
            // not-quite-converged float solve (or a genuinely non-Optimal status) can report an internal
            // objective BELOW the true residual -- impossible for a real sum-of-absolute-values -- which
            // silently misled the benchmark table (observed: m=192 float revised printed 4.37 vs a true
            // residual of 104.08). This also means a non-Optimal row now shows a LARGE honest residual
            // instead of a falsely-small one -- the failure becomes visible instead of hidden.
            double residual = 0;
            for (int i = 0; i < A.M_Rows; i++)
            {
                double rowDot = 0;
                for (int j = 0; j < A.N_Cols; j++) rowDot += (double)A[i, j] * (double)x[j];
                residual += math.abs(rowDot - (double)b[i]);
            }
            objOut[0] = residual;
            itersOut[0] = info.iterations;
            statusOut[0] = (int)info.status;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct IrlsJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, x;
        public NativeArray<double> objOut;
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;
        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = (double)0;   // reset the warm-start guess each timed rep
            var info = Optimize.ladIRLS(in A, in b, ref x);
            // Same honest recompute as LadJobDouble, for consistency (and the same reason: IRLS is
            // approximate, so info.objective is an internal estimate, not guaranteed to equal the true
            // residual at whatever iterate it stopped on).
            double residual = 0;
            for (int i = 0; i < A.M_Rows; i++)
            {
                double rowDot = 0;
                for (int j = 0; j < A.N_Cols; j++) rowDot += (double)A[i, j] * (double)x[j];
                residual += math.abs(rowDot - (double)b[i]);
            }
            objOut[0] = residual;
            itersOut[0] = info.iterations;
            statusOut[0] = (int)info.status;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SparseLadJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN b, x;
        public NativeArray<double> objOut;
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;
        public void Execute()
        {
            var info = LP.lad(in A, in b, ref x, out double obj, 0);
            objOut[0] = obj;
            itersOut[0] = info.iterations;
            statusOut[0] = (int)info.status;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LpSolveSparseJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN b, c, x;
        public NativeArray<ConstraintSense> senses;
        public NativeArray<double> objOut;
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;
        public void Execute()
        {
            var info = LP.solve(in A, in b, in c, in senses, ref x, out double obj);
            objOut[0] = obj;
            itersOut[0] = info.iterations;
            statusOut[0] = (int)info.status;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PdlpJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN lc, uc, lv, uv, c, x;
        public int maxIter;
        public double eps;
        public NativeArray<double> objOut;
        public NativeArray<int> itersOut;
        public void Execute()
        {
            for (int j = 0; j < x.N; j++) x[j] = (double)0;   // cold start each timed rep (x is PDLP's initial iterate)
            var info = LP.pdlp(in A, in lc, in uc, in lv, in uv, in c, ref x, out double obj, maxIter, eps);
            objOut[0] = obj;
            itersOut[0] = info.iterations;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PdlpSparseJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN lc, uc, lv, uv, c, x;
        public int maxIter;
        public double eps;
        public NativeArray<double> objOut;
        public NativeArray<int> itersOut;
        public void Execute()
        {
            for (int j = 0; j < x.N; j++) x[j] = (double)0;
            var info = LP.pdlp(in A, in lc, in uc, in lv, in uv, in c, ref x, out double obj, maxIter, eps);
            objOut[0] = obj;
            itersOut[0] = info.iterations;
        }
    }

    // Small Burst-native matvec (Ax) for RHS construction (Sections 1, 4, 7 all build b = A x0 + slack
    // this way) -- called via .Run() as one-off setup, NOT inside Bench.Time. Moved out of a plain
    // managed Blas.dot(A, x0) call (interpreted Mono, O(mn)) into this job for the same reason the
    // solves themselves moved: at n=384 (m=192), that is 73728 multiply-adds Mono-interpreted on every
    // report generation, which the coordinator's sanity-scan explicitly called out ("managed matvecs for
    // residual columns"). Not part of the timed measurement, so it doesn't affect any row's numbers.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LpRhsMatVecJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN x;
        public doubleN result;
        public void Execute() { Blas.dot(in A, in x, ref result); }
    }

    public static partial class LPBenchmark
    {
        // ==== Section 1: LP.solve, random dense feasible LP -- all FOUR backends on the SAME problem ====
        // (tableau simplex, Mehrotra interior point, bounded-variable revised primal simplex, bounded-
        // variable dual revised simplex -- docs/spec-revised-simplex.md stages 1+2). Same A/b/c/senses
        // instance for every method each n, so the objective column is a direct four-way agreement check
        // and the iters column is directly comparable pivot-for-pivot (revised/dual) or iteration-for-
        // iteration (interior point) against the tableau baseline.
        static void SectionSolveDouble(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 1. LP.solve: random dense feasible LP (m = n/2, A>=0, Ax<=b), simplex vs interior point " +
                          "vs revised primal vs dual simplex [double] ---");
            sb.AppendLine(LPBenchmarkFmt.SolveHeader());

            foreach (var n in LPBenchmarkFmt.SolveVarsN)
            {
                int m = n / 2;
                var arena = new Arena(Allocator.Persistent);
                var A = arena.doubleRandomMat(m, n, 0f, 1f, (uint)(n * 7919 + 11));      // nonneg -> bounded
                var x0 = arena.doubleRandomVec(n, 0f, 1f, (uint)(n * 104729 + 7));
                var Ax0 = arena.doubleVec(m);
                new LpRhsMatVecJobDouble { A = A, x = x0, result = Ax0 }.Run();                // Burst-native, not Mono
                var b = arena.doubleVec(m);
                var rng = new Random((uint)(n * 1299709 + 3));
                for (int i = 0; i < m; i++) b[i] = Ax0[i] + rng.NextDouble((double)0.1, (double)1);  // slack -> x0 feasible
                var c = arena.doubleRandomVec(n, -1f, 1f, (uint)(n * 15485863 + 5));
                var senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
                for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);

                var xS = arena.doubleVec(n);
                var jobS = new LpSolveJobDouble { A = A, b = b, c = c, senses = senses, x = xS, method = LPMethod.Simplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statS = Bench.Time(() => jobS.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("double", n, m, "simplex", statS, itersOut[0], objOut[0]));

                var xI = arena.doubleVec(n);
                var jobI = new LpSolveJobDouble { A = A, b = b, c = c, senses = senses, x = xI, method = LPMethod.InteriorPoint, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statI = Bench.Time(() => jobI.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("double", n, m, "interior-point", statI, itersOut[0], objOut[0]));

                var xR = arena.doubleVec(n);
                var jobR = new LpSolveJobDouble { A = A, b = b, c = c, senses = senses, x = xR, method = LPMethod.RevisedSimplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statR = Bench.Time(() => jobR.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("double", n, m, "revised-primal", statR, itersOut[0], objOut[0]));

                var xD = arena.doubleVec(n);
                var jobD = new LpSolveJobDouble { A = A, b = b, c = c, senses = senses, x = xD, method = LPMethod.DualSimplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statD = Bench.Time(() => jobD.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("double", n, m, "dual-simplex", statD, itersOut[0], objOut[0]));

                objOut.Dispose(); itersOut.Dispose(); statusOut.Dispose();
                senses.Dispose();
                arena.Dispose();
            }
        }

        // ==== Section 2: LAD (L1) regression with outliers -- exact LP (all four backends) vs fast IRLS ====
        static void SectionLadDouble(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 2. LAD (L1) regression, gross outliers: exact LP.lad (simplex/interior/revised/dual) " +
                          "vs fast ladIRLS [double] ---");
            sb.AppendLine(LPBenchmarkFmt.LadHeader());

            int n = LPBenchmarkFmt.NCoef;
            foreach (var m in LPBenchmarkFmt.LadRowsM)
            {
                var arena = new Arena(Allocator.Persistent);
                var A = arena.doubleRandomMat(m, n, -1f, 1f, (uint)(m * 7919 + 13));
                var xt = arena.doubleRandomVec(n, -1f, 1f, (uint)(m * 104729 + 17));
                var Axt = Blas.dot(A, xt);
                var b = arena.doubleVec(m);
                var rng = new Random((uint)(m * 1299709 + 19));
                for (int i = 0; i < m; i++)
                {
                    double val = Axt[i] + rng.NextDouble(-(double)0.05, (double)0.05);
                    if (i % 10 == 0) val += (double)5;         // gross outlier every 10th observation
                    b[i] = val;
                }

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);

                // Tableau-simplex LAD row only up to LadSimplexCap -- its O(m*nCols) per-pivot tableau
                // update makes it the slow tail of this section past there (measured ~101ms already at
                // m=192, double), the way SparseLadDenseCap caps Section 3's dense interior baseline.
                if (m <= LPBenchmarkFmt.LadSimplexCap)
                {
                    var xLs = arena.doubleVec(n);
                    var jobLs = new LadJobDouble { A = A, b = b, x = xLs, method = LPMethod.Simplex, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                    var statLs = Bench.Time(() => jobLs.Run());
                    sb.AppendLine(LPBenchmarkFmt.LadRow("double", m, n, "LP.lad-simplex", statLs, itersOut[0], objOut[0]));
                }

                var xLi = arena.doubleVec(n);
                var jobLi = new LadJobDouble { A = A, b = b, x = xLi, method = LPMethod.InteriorPoint, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statLi = Bench.Time(() => jobLi.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("double", m, n, "LP.lad-interior", statLi, itersOut[0], objOut[0]));

                var xLr = arena.doubleVec(n);
                var jobLr = new LadJobDouble { A = A, b = b, x = xLr, method = LPMethod.RevisedSimplex, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statLr = Bench.Time(() => jobLr.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("double", m, n, "LP.lad-revised", statLr, itersOut[0], objOut[0]));

                var xLd = arena.doubleVec(n);
                var jobLd = new LadJobDouble { A = A, b = b, x = xLd, method = LPMethod.DualSimplex, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statLd = Bench.Time(() => jobLd.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("double", m, n, "LP.lad-dual", statLd, itersOut[0], objOut[0]));

                var xIr = arena.doubleVec(n);
                var jobIr = new IrlsJobDouble { A = A, b = b, x = xIr, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statIr = Bench.Time(() => jobIr.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("double", m, n, "ladIRLS", statIr, itersOut[0], objOut[0]));

                objOut.Dispose(); itersOut.Dispose(); statusOut.Dispose();
                arena.Dispose();
            }
        }

        // ==== Section 3: SPARSE LAD -- matrix-free interior point over a tall block-sparse design ====
        static void SectionSparseLadDouble(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 3. LAD over a tall BSR design (m x n, ~8 nnz/row): dense LP.lad (simplex vs interior) " +
                          "vs sparse matrix-free interior point; dense runs only where the m x m normal matrix fits [double] ---");
            sb.AppendLine(LPBenchmarkFmt.LadHeader());

            int n = LPBenchmarkFmt.SparseLadCoef;
            double density = (double)8 / (double)n;       // ~8 nonzeros per row
            foreach (var m in LPBenchmarkFmt.SparseLadRowsM)
            {
                var arena = new Arena(Allocator.Persistent);
                var As = arena.doubleRandomSparse(m, n, 1, density, (uint)(m * 7919 + 23));   // tall, full column rank

                // b = A x_true + small noise + a gross outlier every 10th row
                var xt = arena.doubleRandomVec(n, -1f, 1f, (uint)(m * 104729 + 29));
                var bx = arena.doubleVec(m);
                BSR.spMV(in As, in xt, ref bx);
                var b = arena.doubleVec(m);
                var rng = new Random((uint)(m * 1299709 + 31));
                for (int i = 0; i < m; i++)
                {
                    double val = bx[i] + rng.NextDouble(-(double)0.05, (double)0.05);
                    if (i % 10 == 0) val += (double)5;
                    b[i] = val;
                }

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);

                // dense interior-point baseline only where the m x m normal matrix is still practical
                // (dense simplex omitted here -- Bland's-rule LAD simplex at m=512 is the slow tail;
                //  it is already benchmarked at appropriate sizes in Section 2)
                if (m <= LPBenchmarkFmt.SparseLadDenseCap)
                {
                    var Ad = As.ToDense(ref arena);
                    var xd = arena.doubleVec(n);
                    var jobD = new LadJobDouble { A = Ad, b = b, x = xd, method = LPMethod.InteriorPoint, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                    var statD = Bench.Time(() => jobD.Run());
                    sb.AppendLine(LPBenchmarkFmt.LadRow("double", m, n, "dense LP.lad-ip", statD, itersOut[0], objOut[0]));
                }

                var xs = arena.doubleVec(n);
                var jobS = new SparseLadJobDouble { A = As, b = b, x = xs, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statS = Bench.Time(() => jobS.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("double", m, n, "sparse LP.lad", statS, itersOut[0], objOut[0]));

                objOut.Dispose(); itersOut.Dispose(); statusOut.Dispose();
                arena.Dispose();
            }
        }

        // ==== Section 4: PDLP vs simplex vs interior point on the SAME dense feasible LP as Section 1 ====
        static void SectionPdlpDenseDouble(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 4. PDLP (matrix-free first-order PDHG) vs simplex vs interior point on the SAME dense " +
                          "feasible LP (Section 1's construction; PDLP tol " + LPBenchmarkFmt.PdlpEps.ToString("E0") +
                          ", cap " + LPBenchmarkFmt.PdlpMaxIter + " iters) [double] ---");
            sb.AppendLine(LPBenchmarkFmt.SolveHeader());

            double INF = (double)1e30;
            foreach (var n in LPBenchmarkFmt.PdlpDenseVarsN)
            {
                int m = n / 2;
                var arena = new Arena(Allocator.Persistent);
                var A = arena.doubleRandomMat(m, n, 0f, 1f, (uint)(n * 7919 + 11));       // SAME problem as Section 1
                var x0 = arena.doubleRandomVec(n, 0f, 1f, (uint)(n * 104729 + 7));
                var Ax0 = arena.doubleVec(m);
                new LpRhsMatVecJobDouble { A = A, x = x0, result = Ax0 }.Run();                // Burst-native, not Mono
                var b = arena.doubleVec(m);
                var rng = new Random((uint)(n * 1299709 + 3));
                for (int i = 0; i < m; i++) b[i] = Ax0[i] + rng.NextDouble((double)0.1, (double)1);
                var c = arena.doubleRandomVec(n, -1f, 1f, (uint)(n * 15485863 + 5));
                var senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
                for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);

                var xS = arena.doubleVec(n);
                var jobS = new LpSolveJobDouble { A = A, b = b, c = c, senses = senses, x = xS, method = LPMethod.Simplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statS = Bench.Time(() => jobS.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("double", n, m, "simplex", statS, itersOut[0], objOut[0]));

                var xI = arena.doubleVec(n);
                var jobI = new LpSolveJobDouble { A = A, b = b, c = c, senses = senses, x = xI, method = LPMethod.InteriorPoint, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statI = Bench.Time(() => jobI.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("double", n, m, "interior-point", statI, itersOut[0], objOut[0]));

                // PDLP on the same LP as two-sided bounds: -inf <= A x <= b, 0 <= x <= +inf
                var lc = arena.doubleVec(m); var uc = arena.doubleVec(m);
                for (int i = 0; i < m; i++) { lc[i] = -INF; uc[i] = b[i]; }
                var lv = arena.doubleVec(n); var uv = arena.doubleVec(n);
                for (int j = 0; j < n; j++) { lv[j] = (double)0; uv[j] = INF; }
                var xP = arena.doubleVec(n);
                var jobP = new PdlpJobDouble { A = A, lc = lc, uc = uc, lv = lv, uv = uv, c = c, x = xP, maxIter = LPBenchmarkFmt.PdlpMaxIter, eps = LPBenchmarkFmt.PdlpEps, objOut = objOut, itersOut = itersOut };
                var statP = Bench.Time(() => jobP.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("double", n, m, "pdlp", statP, itersOut[0], objOut[0]));

                objOut.Dispose(); itersOut.Dispose(); statusOut.Dispose();
                senses.Dispose();
                arena.Dispose();
            }
        }

        // ==== Section 5: sparse PDLP vs sparse interior point on a block-sparse covering LP ====
        static void SectionPdlpSparseDouble(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 5. Block-sparse covering LP (min cx s.t. A x >= b, x >= 0; A,b,c >= 0, ~" +
                          LPBenchmarkFmt.PdlpSparseNnzPerRow + " nnz/row): sparse interior point vs matrix-free PDLP " +
                          "(tol " + LPBenchmarkFmt.PdlpEps.ToString("E0") + ", cap " + LPBenchmarkFmt.PdlpMaxIter + " iters) [double] ---");
            sb.AppendLine(LPBenchmarkFmt.SolveHeader());

            double INF = (double)1e30;
            int nnzPer = LPBenchmarkFmt.PdlpSparseNnzPerRow;
            foreach (var m in LPBenchmarkFmt.PdlpSparseM)
            {
                int n = m;                                        // square covering LP
                var arena = new Arena(Allocator.Persistent);

                // build a nonneg BSR (1x1 blocks): every row gets exactly nnzPer positive entries at
                // distinct columns (stride coprime-ish to n), so A >= 0 and each row is satisfiable.
                var builder = arena.doubleBSRBuilder(m, n, 1, 1, m * nnzPer);
                var rng = new Random((uint)(m * 2654435 + 41));
                for (int i = 0; i < m; i++)
                {
                    int baseCol = (int)(rng.NextUInt() % (uint)n);
                    for (int t = 0; t < nnzPer; t++)
                    {
                        int j = (baseCol + t * 97) % n;
                        var blk = arena.doubleMat(1, 1);
                        blk[0, 0] = (double)0.1 + rng.NextDouble(0f, 1f) * (double)0.9;   // in (0.1, 1]
                        builder.AddBlock(i, j, in blk);
                    }
                }
                var As = builder.ToBSR(ref arena);

                var b = arena.doubleVec(m);
                for (int i = 0; i < m; i++) b[i] = (double)1 + rng.NextDouble(0f, 1f);      // demand in [1, 2]
                var c = arena.doubleVec(n);
                for (int j = 0; j < n; j++) c[j] = (double)0.5 + rng.NextDouble(0f, 1f);    // cost in [0.5, 1.5]
                var senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
                for (int i = 0; i < m; i++) senses[i] = ConstraintSense.GreaterEqual;

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);

                var xI = arena.doubleVec(n);
                var jobI = new LpSolveSparseJobDouble { A = As, b = b, c = c, senses = senses, x = xI, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statI = Bench.Time(() => jobI.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("double", n, m, "sparse ip", statI, itersOut[0], objOut[0]));

                // PDLP: b <= A x <= +inf, 0 <= x <= +inf
                var lc = arena.doubleVec(m); var uc = arena.doubleVec(m);
                for (int i = 0; i < m; i++) { lc[i] = b[i]; uc[i] = INF; }
                var lv = arena.doubleVec(n); var uv = arena.doubleVec(n);
                for (int j = 0; j < n; j++) { lv[j] = (double)0; uv[j] = INF; }
                var xP = arena.doubleVec(n);
                var jobP = new PdlpSparseJobDouble { A = As, lc = lc, uc = uc, lv = lv, uv = uv, c = c, x = xP, maxIter = LPBenchmarkFmt.PdlpMaxIter, eps = LPBenchmarkFmt.PdlpEps, objOut = objOut, itersOut = itersOut };
                var statP = Bench.Time(() => jobP.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("double", n, m, "sparse pdlp", statP, itersOut[0], objOut[0]));

                objOut.Dispose(); itersOut.Dispose(); statusOut.Dispose();
                senses.Dispose();
                arena.Dispose();
            }
        }

        // ==== Section 6: DENSE covering LP (min cx s.t. Ax>=b, x>=0; A,b,c>=0) -- dual-favorable ====
        // Dense analogue of Section 5's sparse covering LP. Every entry of A, b, c is strictly positive,
        // so the LP is feasible (scale any x up enough) and bounded (cᵀx >= 0 always). The construction
        // is deliberately lopsided the OPPOSITE way from Section 1: at the all-logical start every
        // structural cost is already >= 0, so d_j = c_j >= 0 for every nonbasic -> dual-feasible
        // immediately, no artificial-bounds phase 1 needed at all -- while every row is a >= constraint
        // with rhs > 0, so x=0 (the all-logical basis's primal state) violates EVERY row at once,
        // forcing a real primal phase 1 on the tableau and revised-primal backends. This is the fairness
        // counterpoint to Section 1 (which is comparatively primal-friendly) for the primal-vs-dual
        // default question -- objectives must still agree across all four backends on the same instance.
        static void SectionDenseCoveringDouble(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 6. DENSE covering LP (min cx s.t. Ax>=b, x>=0; A,b,c>=0, m=n) -- dual-favorable " +
                          "(dual-feasible at start, primal needs a real phase 1): simplex vs interior point vs " +
                          "revised primal vs dual simplex [double] ---");
            sb.AppendLine(LPBenchmarkFmt.SolveHeader());

            foreach (var n in LPBenchmarkFmt.MidVarsN)
            {
                int m = n;                                        // square covering LP, same as sparse Section 5
                var arena = new Arena(Allocator.Persistent);
                var rng = new Random((uint)(n * 2654435761u + 43));

                var A = arena.doubleMat(m, n);
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        A[i, j] = (double)0.1 + rng.NextDouble(0f, 1f) * (double)0.9;   // in (0.1, 1]
                var b = arena.doubleVec(m);
                for (int i = 0; i < m; i++) b[i] = (double)1 + rng.NextDouble(0f, 1f);      // demand in [1, 2]
                var c = arena.doubleVec(n);
                for (int j = 0; j < n; j++) c[j] = (double)0.5 + rng.NextDouble(0f, 1f);    // cost in [0.5, 1.5]
                var senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
                for (int i = 0; i < m; i++) senses[i] = ConstraintSense.GreaterEqual;

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);

                var xS = arena.doubleVec(n);
                var jobS = new LpSolveJobDouble { A = A, b = b, c = c, senses = senses, x = xS, method = LPMethod.Simplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statS = Bench.Time(() => jobS.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("double", n, m, "simplex", statS, itersOut[0], objOut[0]));

                var xI = arena.doubleVec(n);
                var jobI = new LpSolveJobDouble { A = A, b = b, c = c, senses = senses, x = xI, method = LPMethod.InteriorPoint, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statI = Bench.Time(() => jobI.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("double", n, m, "interior-point", statI, itersOut[0], objOut[0]));

                var xR = arena.doubleVec(n);
                var jobR = new LpSolveJobDouble { A = A, b = b, c = c, senses = senses, x = xR, method = LPMethod.RevisedSimplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statR = Bench.Time(() => jobR.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("double", n, m, "revised-primal", statR, itersOut[0], objOut[0]));

                var xD = arena.doubleVec(n);
                var jobD = new LpSolveJobDouble { A = A, b = b, c = c, senses = senses, x = xD, method = LPMethod.DualSimplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statD = Bench.Time(() => jobD.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("double", n, m, "dual-simplex", statD, itersOut[0], objOut[0]));

                objOut.Dispose(); itersOut.Dispose(); statusOut.Dispose();
                senses.Dispose();
                arena.Dispose();
            }
        }

        // ==== Section 7: infeasibility detection -- Section 1's construction + one contradictory row ====
        // Reuses Section 1's exact feasible construction (A m x n >= 0, b = A x0 + slack, c random), then
        // appends ONE extra row: a duplicate of row 0 with sense >= and rhs b0+10. Row 0 demands
        // A0.x <= b0; the new row demands A0.x >= b0+10 -- those two can never hold simultaneously
        // regardless of every other row/variable, so the augmented LP is infeasible by construction with
        // no subtler failure mode to get wrong (the same robust recipe the review that requested this
        // section specified). All four backends attempt the SAME augmented instance; the STATUS column
        // (not objective -- see InfeasRow's doc comment) shows which ones actually certify Infeasible.
        static void SectionInfeasibleDouble(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 7. Infeasibility detection: Section-1-style dense LP + one contradictory duplicated " +
                          "row (row 0 as both <= b0 and >= b0+10) -- simplex vs interior point vs revised primal " +
                          "vs dual simplex [double] ---");
            sb.AppendLine(LPBenchmarkFmt.InfeasHeader());

            foreach (var n in LPBenchmarkFmt.MidVarsN)
            {
                int m = n / 2;
                var arena = new Arena(Allocator.Persistent);
                var Abase = arena.doubleRandomMat(m, n, 0f, 1f, (uint)(n * 7919 + 11));
                var x0 = arena.doubleRandomVec(n, 0f, 1f, (uint)(n * 104729 + 7));
                var Ax0 = arena.doubleVec(m);
                new LpRhsMatVecJobDouble { A = Abase, x = x0, result = Ax0 }.Run();            // Burst-native, not Mono
                var bbase = arena.doubleVec(m);
                var rng = new Random((uint)(n * 1299709 + 3));
                for (int i = 0; i < m; i++) bbase[i] = Ax0[i] + rng.NextDouble((double)0.1, (double)1);
                var c = arena.doubleRandomVec(n, -1f, 1f, (uint)(n * 15485863 + 5));

                int mAug = m + 1;
                var A = arena.doubleMat(mAug, n);
                for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) A[i, j] = Abase[i, j];
                for (int j = 0; j < n; j++) A[m, j] = Abase[0, j];              // duplicate row 0
                var b = arena.doubleVec(mAug);
                for (int i = 0; i < m; i++) b[i] = bbase[i];
                b[m] = bbase[0] + (double)10;                                    // contradicts row 0 -> infeasible
                var senses = new NativeArray<ConstraintSense>(mAug, Allocator.Persistent);
                for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;
                senses[m] = ConstraintSense.GreaterEqual;

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);

                // statusOut[0] is the raw int the job wrote via (int)info.status -- passed straight
                // through to LPBenchmarkFmt.InfeasRow, which casts it back to its OWN (real-assembly)
                // LPStatus and formats the name there. See InfeasRow's doc comment for why the int
                // (not the enum, not a template-built string) is what crosses the template/harness
                // assembly boundary.
                var xS = arena.doubleVec(n);
                var jobS = new LpSolveJobDouble { A = A, b = b, c = c, senses = senses, x = xS, method = LPMethod.Simplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statS = Bench.Time(() => jobS.Run());
                sb.AppendLine(LPBenchmarkFmt.InfeasRow("double", n, mAug, "simplex", statS, itersOut[0], statusOut[0]));

                var xI = arena.doubleVec(n);
                var jobI = new LpSolveJobDouble { A = A, b = b, c = c, senses = senses, x = xI, method = LPMethod.InteriorPoint, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statI = Bench.Time(() => jobI.Run());
                sb.AppendLine(LPBenchmarkFmt.InfeasRow("double", n, mAug, "interior-point", statI, itersOut[0], statusOut[0]));

                var xR = arena.doubleVec(n);
                var jobR = new LpSolveJobDouble { A = A, b = b, c = c, senses = senses, x = xR, method = LPMethod.RevisedSimplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statR = Bench.Time(() => jobR.Run());
                sb.AppendLine(LPBenchmarkFmt.InfeasRow("double", n, mAug, "revised-primal", statR, itersOut[0], statusOut[0]));

                var xD = arena.doubleVec(n);
                var jobD = new LpSolveJobDouble { A = A, b = b, c = c, senses = senses, x = xD, method = LPMethod.DualSimplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statD = Bench.Time(() => jobD.Run());
                sb.AppendLine(LPBenchmarkFmt.InfeasRow("double", n, mAug, "dual-simplex", statD, itersOut[0], statusOut[0]));

                objOut.Dispose(); itersOut.Dispose(); statusOut.Dispose();
                senses.Dispose();
                arena.Dispose();
            }
        }
    }
}
