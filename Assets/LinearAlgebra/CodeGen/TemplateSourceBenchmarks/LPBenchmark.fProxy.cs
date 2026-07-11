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
    //
    // Every job below carries its OWN reporting outputs (objOut/itersOut/statusOut, length-1 arrays)
    // written from inside Execute(), so objective/iters/status come out of the SAME Burst-native call
    // the report already times, with no second solve. status crosses as `(int)info.status` (Burst-legal
    // enum-to-int cast); LPBenchmarkFmt.InfeasRow casts it back on the harness side (see that method's
    // doc comment for why the raw int, not the enum, crosses the template/harness assembly boundary).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LpSolveJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN b, c, x;
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
    public struct LadJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN b, x;
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
    public struct LadFNJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN b, x;
        public NativeArray<double> objOut;
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;
        public void Execute()
        {
            var info = LP.ladFN(in A, in b, ref x, out double obj, 0);
            // Same honest recompute as LadJobFProxy/IrlsJobFProxy -- see that job's comment.
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
    public struct LadBRJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN b, x;
        public NativeArray<double> objOut;
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;
        public void Execute()
        {
            var info = LP.ladBR(in A, in b, ref x, out double obj, 0);
            // Same honest recompute as LadFNJobFProxy -- see that job's comment.
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
    public struct IrlsJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN b, x;
        public NativeArray<double> objOut;
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;
        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = (fProxy)0;   // reset the warm-start guess each timed rep
            var info = Optimize.ladIRLS(in A, in b, ref x);
            // Same honest recompute as LadJobFProxy, for consistency (and the same reason: IRLS is
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
    public struct SparseLadJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x;
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

    // Small Burst-native matvec (Ax) for RHS construction (Sections 1, 5 both build b = A x0 + slack
    // this way) -- called via .Run() as one-off setup, NOT inside Bench.Time. Moved out of a plain
    // managed Blas.dot(A, x0) call (interpreted Mono, O(mn)) into this job for the same reason the
    // solves themselves moved: at n=384 (m=192), that is 73728 multiply-adds Mono-interpreted on every
    // report generation, which the coordinator's sanity-scan explicitly called out ("managed matvecs for
    // residual columns"). Not part of the timed measurement, so it doesn't affect any row's numbers.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LpRhsMatVecJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN x;
        public fProxyN result;
        public void Execute() { Blas.dot(in A, in x, ref result); }
    }

    // Warm re-solve chain (Section 6): 1 cold seeding solve + `resolves` rhs-perturbed re-solves,
    // all inside ONE Execute -- LPBasis.populated / fProxyLPCache validity are plain struct fields,
    // so warm state cannot survive across IJob.Run copies; the chain must live in a single job.
    // Every mode regenerates the IDENTICAL deterministic perturbation sequence (seeded per k), so
    // the three rows solve the same problems. itersOut = TOTAL pivots across the K re-solves
    // (seed solve excluded); objOut = the LAST re-solve's objective (three-mode agreement check).
    // Perturbation stays within the slack floor (bBase slack >= 0.1, noise <= 0.04), so every
    // perturbed instance remains feasible.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LpWarmResolveJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN bBase, b, c, x;
        public NativeArray<ConstraintSense> senses;
        public int mode;        // 0 = cold every solve, 1 = ref LPBasis, 2 = ref LPBasis + fProxyLPCache
        public int resolves;
        public NativeArray<double> objOut;
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;
        public void Execute()
        {
            int n = A.N_Cols, m = A.M_Rows;
            var basis = new LPBasis(n, m, Allocator.Temp);
            var cache = new fProxyLPCache(n, m, Allocator.Temp);

            for (int i = 0; i < m; i++) b[i] = bBase[i];
            double obj;
            LPInfo info;
            if (mode == 0) info = LP.solve(in A, in b, in c, in senses, ref x, out obj, LPMethod.DualSimplex, 0);
            else if (mode == 1) info = LP.solve(in A, in b, in c, in senses, ref x, out obj, ref basis, 0);
            else info = LP.solve(in A, in b, in c, in senses, ref x, out obj, ref basis, ref cache, 0);

            int warmIters = 0;
            obj = 0;
            for (int k = 1; k <= resolves; k++)
            {
                var rng = new Random((uint)k * 2654435761u + 0x9E3779B9u);
                for (int i = 0; i < m; i++) b[i] = bBase[i] + rng.NextFProxy((fProxy)(-0.04), (fProxy)0.04);
                if (mode == 0) info = LP.solve(in A, in b, in c, in senses, ref x, out obj, LPMethod.DualSimplex, 0);
                else if (mode == 1) info = LP.solve(in A, in b, in c, in senses, ref x, out obj, ref basis, 0);
                else info = LP.solve(in A, in b, in c, in senses, ref x, out obj, ref basis, ref cache, 0);
                warmIters += info.iterations;
            }

            objOut[0] = obj;
            itersOut[0] = warmIters;
            statusOut[0] = (int)info.status;
            basis.Dispose(); cache.Dispose();
        }
    }

    public static partial class LPBenchmark
    {
        // ==== Section 1: LP.solve, random dense feasible LP -- all FOUR backends on the SAME problem ====
        // (tableau simplex, Mehrotra interior point, bounded-variable revised primal simplex, bounded-
        // variable dual revised simplex). Same A/b/c/senses instance for every method each n, so the
        // objective column is a direct four-way agreement check
        // and the iters column is directly comparable pivot-for-pivot (revised/dual) or iteration-for-
        // iteration (interior point) against the tableau baseline.
        static void SectionSolveFProxy(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 1. LP.solve: random dense feasible LP (m = n/2, A>=0, Ax<=b), simplex vs interior point " +
                          "vs revised primal vs dual simplex [fProxy] ---");
            sb.AppendLine(LPBenchmarkFmt.SolveHeader());

            foreach (var n in LPBenchmarkFmt.SolveVarsN)
            {
                int m = n / 2;
                var arena = new Arena(Allocator.Persistent);
                var A = arena.fProxyRandomMat(m, n, 0f, 1f, (uint)(n * 7919 + 11));      // nonneg -> bounded
                var x0 = arena.fProxyRandomVec(n, 0f, 1f, (uint)(n * 104729 + 7));
                var Ax0 = arena.fProxyVec(m);
                new LpRhsMatVecJobFProxy { A = A, x = x0, result = Ax0 }.Run();                // Burst-native, not Mono
                var b = arena.fProxyVec(m);
                var rng = new Random((uint)(n * 1299709 + 3));
                for (int i = 0; i < m; i++) b[i] = Ax0[i] + rng.NextFProxy((fProxy)0.1, (fProxy)1);  // slack -> x0 feasible
                var c = arena.fProxyRandomVec(n, -1f, 1f, (uint)(n * 15485863 + 5));
                var senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
                for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);

                var xS = arena.fProxyVec(n);
                var jobS = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xS, method = LPMethod.Simplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statS = Bench.Time(() => jobS.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("fProxy", n, m, "simplex", statS, itersOut[0], objOut[0]));

                var xI = arena.fProxyVec(n);
                var jobI = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xI, method = LPMethod.InteriorPoint, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statI = Bench.Time(() => jobI.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("fProxy", n, m, "interior-point", statI, itersOut[0], objOut[0]));

                var xR = arena.fProxyVec(n);
                var jobR = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xR, method = LPMethod.RevisedSimplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statR = Bench.Time(() => jobR.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("fProxy", n, m, "revised-primal", statR, itersOut[0], objOut[0]));

                var xD = arena.fProxyVec(n);
                var jobD = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xD, method = LPMethod.DualSimplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statD = Bench.Time(() => jobD.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("fProxy", n, m, "dual-simplex", statD, itersOut[0], objOut[0]));

                objOut.Dispose(); itersOut.Dispose(); statusOut.Dispose();
                senses.Dispose();
                arena.Dispose();
            }
        }

        // ==== Section 6: warm re-solve chain -- cold vs LPBasis vs LPBasis+fProxyLPCache ====
        // Section-1-style instance; see LpWarmResolveJobFProxy's comment for why the whole chain runs
        // inside one job.
        static void SectionWarmResolveFProxy(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 6. Warm re-solve chain: 1 cold seed + K=16 rhs-perturbed re-solves (Section-1-style " +
                          "instance; identical perturbation sequence per mode) [fProxy] ---");
            sb.AppendLine(LPBenchmarkFmt.WarmHeader());

            foreach (var n in LPBenchmarkFmt.WarmVarsN)
            {
                int m = n / 2;
                var arena = new Arena(Allocator.Persistent);
                var A = arena.fProxyRandomMat(m, n, 0f, 1f, (uint)(n * 7919 + 11));
                var x0 = arena.fProxyRandomVec(n, 0f, 1f, (uint)(n * 104729 + 7));
                var Ax0 = arena.fProxyVec(m);
                new LpRhsMatVecJobFProxy { A = A, x = x0, result = Ax0 }.Run();
                var bBase = arena.fProxyVec(m);
                var rng = new Random((uint)(n * 1299709 + 3));
                for (int i = 0; i < m; i++) bBase[i] = Ax0[i] + rng.NextFProxy((fProxy)0.1, (fProxy)1);
                var c = arena.fProxyRandomVec(n, -1f, 1f, (uint)(n * 15485863 + 5));
                var senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
                for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);

                var bScratch = arena.fProxyVec(m);
                for (int mode = 0; mode <= 2; mode++)
                {
                    var xW = arena.fProxyVec(n);
                    var job = new LpWarmResolveJobFProxy
                    {
                        A = A, bBase = bBase, b = bScratch, c = c, senses = senses, x = xW,
                        mode = mode, resolves = 16,
                        objOut = objOut, itersOut = itersOut, statusOut = statusOut
                    };
                    var stat = Bench.Time(() => job.Run());
                    string label = mode == 0 ? "cold" : (mode == 1 ? "warm-basis" : "warm+cache");
                    sb.AppendLine(LPBenchmarkFmt.WarmRow("fProxy", n, m, label, stat, itersOut[0], objOut[0]));
                }

                objOut.Dispose(); itersOut.Dispose(); statusOut.Dispose();
                senses.Dispose();
                arena.Dispose();
            }
        }

        // ==== Section 2: LAD (L1) regression with outliers -- exact LP (all four backends) vs fast IRLS ====
        static void SectionLadFProxy(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 2. LAD (L1) regression, gross outliers: exact LP.lad (simplex/interior/revised/dual) " +
                          "vs fast ladIRLS [fProxy] ---");
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

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);

                // Tableau-simplex LAD row only up to LadSimplexCap -- its O(m*nCols) per-pivot tableau
                // update makes it the slow tail of this section past there (measured ~101ms already at
                // m=192, double), the way SparseLadDenseCap caps Section 3's dense interior baseline.
                if (m <= LPBenchmarkFmt.LadSimplexCap)
                {
                    var xLs = arena.fProxyVec(n);
                    var jobLs = new LadJobFProxy { A = A, b = b, x = xLs, method = LPMethod.Simplex, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                    var statLs = Bench.Time(() => jobLs.Run());
                    sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "LP.lad-simplex", statLs, itersOut[0], objOut[0]));
                }

                var xLi = arena.fProxyVec(n);
                var jobLi = new LadJobFProxy { A = A, b = b, x = xLi, method = LPMethod.InteriorPoint, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statLi = Bench.Time(() => jobLi.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "LP.lad-interior", statLi, itersOut[0], objOut[0]));

                var xLr = arena.fProxyVec(n);
                var jobLr = new LadJobFProxy { A = A, b = b, x = xLr, method = LPMethod.RevisedSimplex, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statLr = Bench.Time(() => jobLr.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "LP.lad-revised", statLr, itersOut[0], objOut[0]));

                var xLd = arena.fProxyVec(n);
                var jobLd = new LadJobFProxy { A = A, b = b, x = xLd, method = LPMethod.DualSimplex, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statLd = Bench.Time(() => jobLd.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "LP.lad-dual", statLd, itersOut[0], objOut[0]));

                var xLf = arena.fProxyVec(n);
                var jobLf = new LadFNJobFProxy { A = A, b = b, x = xLf, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statLf = Bench.Time(() => jobLf.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "LP.ladFN", statLf, itersOut[0], objOut[0]));

                var xBr = arena.fProxyVec(n);
                var jobBr = new LadBRJobFProxy { A = A, b = b, x = xBr, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statBr = Bench.Time(() => jobBr.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "LP.ladBR", statBr, itersOut[0], objOut[0]));

                var xIr = arena.fProxyVec(n);
                var jobIr = new IrlsJobFProxy { A = A, b = b, x = xIr, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statIr = Bench.Time(() => jobIr.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "ladIRLS", statIr, itersOut[0], objOut[0]));

                objOut.Dispose(); itersOut.Dispose(); statusOut.Dispose();
                arena.Dispose();
            }

            // ==== Section 2b: LAD fast-route-only sweep -- extended m range, ladFN/ladBR/IRLS ONLY ====
            // The LP-reformulation backends above (simplex/interior/revised/dual, all via LadJobFProxy)
            // build an O(m) tableau or an O(m x m)-scaled normal/basis structure, so they are both far
            // over budget at m>=1024 and uninteresting at m=8 (too small to show any asymptotic trend).
            // This second sweep exists purely to bracket the Barrodale-Roberts vs Frisch-Newton
            // crossover the literature (Portnoy & Koenker 1997) predicts around m in [1e3,1e4] --
            // LadRowsM above tops out at 384, where ladBR was still winning every row; LadFastRowsM
            // adds one point below that range (m=8, near NCoef=4) and three points spanning past it
            // (1024, 4096, 16384).
            //
            // Budget estimate: ladFN and ladBR are each ~10-20 iterations (Newton steps / simplex
            // pivots) of ONE O(m*n) pass over the raw m x n design per iteration; IRLS is ~50
            // iterations of the same O(m*n) shape (an n x n normal solve built by one O(m*n) streaming
            // pass). At the top size m=16384, n=4: worst case ~50 * 16384 * 4 ~= 3.3M flops per solve --
            // sub-millisecond. Times 5 runs (1 warmup + 4 timed) times 3 routes times 5 sizes, the
            // added section is dominated by job-dispatch/array-alloc overhead rather than FLOPs at
            // these sizes; total added wall-clock is estimated at well under 10s (most rows sub-ms,
            // expected sum in the low hundreds of ms).
            sb.AppendLine();
            sb.AppendLine("--- 2b. LAD fast routes only (LP.ladFN / LP.ladBR / ladIRLS), extended m range " +
                          "for the Barrodale-Roberts vs Frisch-Newton crossover [fProxy] ---");
            sb.AppendLine(LPBenchmarkFmt.LadHeader());

            foreach (var m in LPBenchmarkFmt.LadFastRowsM)
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

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);

                var xLf = arena.fProxyVec(n);
                var jobLf = new LadFNJobFProxy { A = A, b = b, x = xLf, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statLf = Bench.Time(() => jobLf.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "LP.ladFN", statLf, itersOut[0], objOut[0]));

                var xBr = arena.fProxyVec(n);
                var jobBr = new LadBRJobFProxy { A = A, b = b, x = xBr, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statBr = Bench.Time(() => jobBr.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "LP.ladBR", statBr, itersOut[0], objOut[0]));

                var xIr = arena.fProxyVec(n);
                var jobIr = new IrlsJobFProxy { A = A, b = b, x = xIr, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statIr = Bench.Time(() => jobIr.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "ladIRLS", statIr, itersOut[0], objOut[0]));

                objOut.Dispose(); itersOut.Dispose(); statusOut.Dispose();
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

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);

                // dense interior-point baseline only where the m x m normal matrix is still practical
                // (dense simplex omitted here -- Bland's-rule LAD simplex at m=512 is the slow tail;
                //  it is already benchmarked at appropriate sizes in Section 2)
                if (m <= LPBenchmarkFmt.SparseLadDenseCap)
                {
                    var Ad = As.ToDense(ref arena);
                    var xd = arena.fProxyVec(n);
                    var jobD = new LadJobFProxy { A = Ad, b = b, x = xd, method = LPMethod.InteriorPoint, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                    var statD = Bench.Time(() => jobD.Run());
                    sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "dense LP.lad-ip", statD, itersOut[0], objOut[0]));
                }

                var xs = arena.fProxyVec(n);
                var jobS = new SparseLadJobFProxy { A = As, b = b, x = xs, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statS = Bench.Time(() => jobS.Run());
                sb.AppendLine(LPBenchmarkFmt.LadRow("fProxy", m, n, "sparse LP.lad", statS, itersOut[0], objOut[0]));

                objOut.Dispose(); itersOut.Dispose(); statusOut.Dispose();
                arena.Dispose();
            }
        }

        // ==== Section 4: DENSE covering LP (min cx s.t. Ax>=b, x>=0; A,b,c>=0) -- dual-favorable ====
        // Every entry of A, b, c is strictly positive, so the LP is feasible (scale any x up enough) and
        // bounded (cᵀx >= 0 always). The construction is deliberately lopsided the OPPOSITE way from
        // Section 1: at the all-logical start every structural cost is already >= 0, so d_j = c_j >= 0
        // for every nonbasic -> dual-feasible immediately, no artificial-bounds phase 1 needed at all --
        // while every row is a >= constraint with rhs > 0, so x=0 (the all-logical basis's primal state)
        // violates EVERY row at once, forcing a real primal phase 1 on the tableau and revised-primal
        // backends. This is the fairness counterpoint to Section 1 (which is comparatively primal-
        // friendly) for the primal-vs-dual default question -- objectives must still agree across all
        // four backends on the same instance.
        static void SectionDenseCoveringFProxy(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 4. DENSE covering LP (min cx s.t. Ax>=b, x>=0; A,b,c>=0, m=n) -- dual-favorable " +
                          "(dual-feasible at start, primal needs a real phase 1): simplex vs interior point vs " +
                          "revised primal vs dual simplex [fProxy] ---");
            sb.AppendLine(LPBenchmarkFmt.SolveHeader());

            foreach (var n in LPBenchmarkFmt.MidVarsN)
            {
                int m = n;                                        // square covering LP
                var arena = new Arena(Allocator.Persistent);
                var rng = new Random((uint)(n * 2654435761u + 43));

                var A = arena.fProxyMat(m, n);
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        A[i, j] = (fProxy)0.1 + rng.NextFProxy(0f, 1f) * (fProxy)0.9;   // in (0.1, 1]
                var b = arena.fProxyVec(m);
                for (int i = 0; i < m; i++) b[i] = (fProxy)1 + rng.NextFProxy(0f, 1f);      // demand in [1, 2]
                var c = arena.fProxyVec(n);
                for (int j = 0; j < n; j++) c[j] = (fProxy)0.5 + rng.NextFProxy(0f, 1f);    // cost in [0.5, 1.5]
                var senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
                for (int i = 0; i < m; i++) senses[i] = ConstraintSense.GreaterEqual;

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);

                var xS = arena.fProxyVec(n);
                var jobS = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xS, method = LPMethod.Simplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statS = Bench.Time(() => jobS.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("fProxy", n, m, "simplex", statS, itersOut[0], objOut[0]));

                var xI = arena.fProxyVec(n);
                var jobI = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xI, method = LPMethod.InteriorPoint, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statI = Bench.Time(() => jobI.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("fProxy", n, m, "interior-point", statI, itersOut[0], objOut[0]));

                var xR = arena.fProxyVec(n);
                var jobR = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xR, method = LPMethod.RevisedSimplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statR = Bench.Time(() => jobR.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("fProxy", n, m, "revised-primal", statR, itersOut[0], objOut[0]));

                var xD = arena.fProxyVec(n);
                var jobD = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xD, method = LPMethod.DualSimplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statD = Bench.Time(() => jobD.Run());
                sb.AppendLine(LPBenchmarkFmt.SolveRow("fProxy", n, m, "dual-simplex", statD, itersOut[0], objOut[0]));

                objOut.Dispose(); itersOut.Dispose(); statusOut.Dispose();
                senses.Dispose();
                arena.Dispose();
            }
        }

        // ==== Section 5: infeasibility detection -- Section 1's construction + one contradictory row ====
        // Reuses Section 1's exact feasible construction (A m x n >= 0, b = A x0 + slack, c random), then
        // appends ONE extra row: a duplicate of row 0 with sense >= and rhs b0+10. Row 0 demands
        // A0.x <= b0; the new row demands A0.x >= b0+10 -- those two can never hold simultaneously
        // regardless of every other row/variable, so the augmented LP is infeasible by construction with
        // no subtler failure mode to get wrong (the same robust recipe the review that requested this
        // section specified). All four backends attempt the SAME augmented instance; the STATUS column
        // (not objective -- see InfeasRow's doc comment) shows which ones actually certify Infeasible.
        static void SectionInfeasibleFProxy(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 5. Infeasibility detection: Section-1-style dense LP + one contradictory duplicated " +
                          "row (row 0 as both <= b0 and >= b0+10) -- simplex vs interior point vs revised primal " +
                          "vs dual simplex [fProxy] ---");
            sb.AppendLine(LPBenchmarkFmt.InfeasHeader());

            foreach (var n in LPBenchmarkFmt.MidVarsN)
            {
                int m = n / 2;
                var arena = new Arena(Allocator.Persistent);
                var Abase = arena.fProxyRandomMat(m, n, 0f, 1f, (uint)(n * 7919 + 11));
                var x0 = arena.fProxyRandomVec(n, 0f, 1f, (uint)(n * 104729 + 7));
                var Ax0 = arena.fProxyVec(m);
                new LpRhsMatVecJobFProxy { A = Abase, x = x0, result = Ax0 }.Run();            // Burst-native, not Mono
                var bbase = arena.fProxyVec(m);
                var rng = new Random((uint)(n * 1299709 + 3));
                for (int i = 0; i < m; i++) bbase[i] = Ax0[i] + rng.NextFProxy((fProxy)0.1, (fProxy)1);
                var c = arena.fProxyRandomVec(n, -1f, 1f, (uint)(n * 15485863 + 5));

                int mAug = m + 1;
                var A = arena.fProxyMat(mAug, n);
                for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) A[i, j] = Abase[i, j];
                for (int j = 0; j < n; j++) A[m, j] = Abase[0, j];              // duplicate row 0
                var b = arena.fProxyVec(mAug);
                for (int i = 0; i < m; i++) b[i] = bbase[i];
                b[m] = bbase[0] + (fProxy)10;                                    // contradicts row 0 -> infeasible
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
                var xS = arena.fProxyVec(n);
                var jobS = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xS, method = LPMethod.Simplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statS = Bench.Time(() => jobS.Run());
                sb.AppendLine(LPBenchmarkFmt.InfeasRow("fProxy", n, mAug, "simplex", statS, itersOut[0], statusOut[0]));

                var xI = arena.fProxyVec(n);
                var jobI = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xI, method = LPMethod.InteriorPoint, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statI = Bench.Time(() => jobI.Run());
                sb.AppendLine(LPBenchmarkFmt.InfeasRow("fProxy", n, mAug, "interior-point", statI, itersOut[0], statusOut[0]));

                var xR = arena.fProxyVec(n);
                var jobR = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xR, method = LPMethod.RevisedSimplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statR = Bench.Time(() => jobR.Run());
                sb.AppendLine(LPBenchmarkFmt.InfeasRow("fProxy", n, mAug, "revised-primal", statR, itersOut[0], statusOut[0]));

                var xD = arena.fProxyVec(n);
                var jobD = new LpSolveJobFProxy { A = A, b = b, c = c, senses = senses, x = xD, method = LPMethod.DualSimplex, maxIter = 0, objOut = objOut, itersOut = itersOut, statusOut = statusOut };
                var statD = Bench.Time(() => jobD.Run());
                sb.AppendLine(LPBenchmarkFmt.InfeasRow("fProxy", n, mAug, "dual-simplex", statD, itersOut[0], statusOut[0]));

                objOut.Dispose(); itersOut.Dispose(); statusOut.Dispose();
                senses.Dispose();
                arena.Dispose();
            }
        }
    }
}
