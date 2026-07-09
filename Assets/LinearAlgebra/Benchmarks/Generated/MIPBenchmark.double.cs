#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of MIPBenchmark (the timed IJob + the per-section instance-builders +
    // build+measure methods). The dtype-agnostic harness (sizes/seeds, row formatters, Run, Section) is
    // hand-written in Assets/LinearAlgebra/Benchmarks/MIPBenchmark.cs (MIPBenchmarkFmt + the partial
    // class).
    //
    // The job carries its OWN reporting outputs (objOut/nodesOut/itersOut/statusOut/dualBoundOut/gapOut,
    // length-1 arrays) written from inside Execute() -- the same "no second, Mono-interpreted solve just
    // to harvest diagnostics" discipline LPBenchmark.double.cs's own header comment explains (Bench.Time
    // already runs the job once as a warmup before the timed reps, so the outputs are a side effect of
    // the SAME Burst-native call being timed, not an extra solve).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MipSolveJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, c, xl, xu, x;
        public NativeArray<ConstraintSense> senses;
        public NativeArray<byte> integrality;
        public int maxNodes;
        public int maxIter;
        public double absGap;
        public double relGap;
        public NativeArray<double> objOut;
        public NativeArray<int> nodesOut;
        public NativeArray<int> itersOut;
        public NativeArray<int> statusOut;
        public NativeArray<double> dualBoundOut;
        public NativeArray<double> gapOut;
        public void Execute()
        {
            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integrality, ref x, out double obj,
                                 maxNodes, maxIter, absGap, relGap);
            objOut[0] = obj;
            nodesOut[0] = info.nodes;
            itersOut[0] = info.lpIterations;
            statusOut[0] = (int)info.status;
            dualBoundOut[0] = info.dualBound;
            gapOut[0] = info.gap;
        }
    }

    public static partial class MIPBenchmark
    {
        // Sets a coefficient-1 covering triple in row r of A -- same helper as MIPTests.double.cs's
        // SetTriple (the stein set-covering rows).
        static void SetTripleDouble(doubleMxN A, int r, int i, int j, int k)
        {
            A[r, i] = (double)1; A[r, j] = (double)1; A[r, k] = (double)1;
        }

        // MIPLIB "stein9" -- SAME literal instance data as MIPTests.double.cs's BuildStein9 (Fulkerson,
        // Nemhauser & Trotter 1974; MIPLIB 3), not re-derived. 9 binaries, minimize count, proven optimum 5.
        static void BuildStein9Double(in Arena arena, out doubleMxN A, out doubleN b, out doubleN c,
                                      out NativeArray<ConstraintSense> senses, out doubleN xl, out doubleN xu,
                                      out NativeArray<byte> integ)
        {
            const int n = 9, m = 13;
            A = arena.doubleMat(m, n);   // zero-initialized
            SetTripleDouble(A, 0, 1, 2, 3); SetTripleDouble(A, 1, 0, 2, 4); SetTripleDouble(A, 2, 0, 1, 5); SetTripleDouble(A, 3, 4, 5, 6);
            SetTripleDouble(A, 4, 3, 5, 7); SetTripleDouble(A, 5, 3, 4, 8); SetTripleDouble(A, 6, 0, 7, 8); SetTripleDouble(A, 7, 1, 6, 8);
            SetTripleDouble(A, 8, 2, 6, 7); SetTripleDouble(A, 9, 0, 3, 6); SetTripleDouble(A, 10, 1, 4, 7); SetTripleDouble(A, 11, 2, 5, 8);
            for (int j = 0; j < n; j++) A[12, j] = (double)1;   // the OB2 "at least 4 of 9" cut

            b = arena.doubleVec(m); for (int i = 0; i < 12; i++) b[i] = (double)1; b[12] = (double)4;
            senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
            for (int i = 0; i < m; i++) senses[i] = ConstraintSense.GreaterEqual;
            c = arena.doubleVec(n); for (int j = 0; j < n; j++) c[j] = (double)1;
            xl = arena.doubleVec(n);
            xu = arena.doubleVec(n); for (int j = 0; j < n; j++) xu[j] = (double)1;
            integ = new NativeArray<byte>(n, Allocator.Persistent); for (int j = 0; j < n; j++) integ[j] = 1;
        }

        // MIPLIB "stein15" -- SAME literal instance data as MIPTests.double.cs's Stein15, not re-derived.
        // 15 binaries, minimize count, proven optimum 9 (double: ~275 nodes / ~4307 LP iterations per the
        // test file's own measured baseline).
        static void BuildStein15Double(in Arena arena, out doubleMxN A, out doubleN b, out doubleN c,
                                       out NativeArray<ConstraintSense> senses, out doubleN xl, out doubleN xu,
                                       out NativeArray<byte> integ)
        {
            const int n = 15, m = 36;
            A = arena.doubleMat(m, n);   // zero-initialized
            SetTripleDouble(A, 0, 2, 3, 5);   SetTripleDouble(A, 1, 3, 4, 6);   SetTripleDouble(A, 2, 0, 4, 7);   SetTripleDouble(A, 3, 0, 1, 8);   SetTripleDouble(A, 4, 1, 2, 9);
            SetTripleDouble(A, 5, 1, 4, 5);   SetTripleDouble(A, 6, 0, 2, 6);   SetTripleDouble(A, 7, 1, 3, 7);   SetTripleDouble(A, 8, 2, 4, 8);   SetTripleDouble(A, 9, 0, 3, 9);
            SetTripleDouble(A, 10, 7, 8, 10); SetTripleDouble(A, 11, 8, 9, 11); SetTripleDouble(A, 12, 5, 9, 12); SetTripleDouble(A, 13, 5, 6, 13); SetTripleDouble(A, 14, 6, 7, 14);
            SetTripleDouble(A, 15, 6, 9, 10); SetTripleDouble(A, 16, 5, 7, 11); SetTripleDouble(A, 17, 6, 8, 12); SetTripleDouble(A, 18, 7, 9, 13); SetTripleDouble(A, 19, 5, 8, 14);
            SetTripleDouble(A, 20, 0, 12, 13); SetTripleDouble(A, 21, 1, 13, 14); SetTripleDouble(A, 22, 2, 10, 14); SetTripleDouble(A, 23, 3, 10, 11); SetTripleDouble(A, 24, 4, 11, 12);
            SetTripleDouble(A, 25, 0, 11, 14); SetTripleDouble(A, 26, 1, 10, 12); SetTripleDouble(A, 27, 2, 11, 13); SetTripleDouble(A, 28, 3, 12, 14); SetTripleDouble(A, 29, 4, 10, 13);
            SetTripleDouble(A, 30, 0, 5, 10); SetTripleDouble(A, 31, 1, 6, 11); SetTripleDouble(A, 32, 2, 7, 12); SetTripleDouble(A, 33, 3, 8, 13); SetTripleDouble(A, 34, 4, 9, 14);
            for (int j = 0; j < n; j++) A[35, j] = (double)1;   // the "at least 7 of 15" cut

            b = arena.doubleVec(m); for (int i = 0; i < 35; i++) b[i] = (double)1; b[35] = (double)7;
            senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
            for (int i = 0; i < m; i++) senses[i] = ConstraintSense.GreaterEqual;
            c = arena.doubleVec(n); for (int j = 0; j < n; j++) c[j] = (double)1;
            xl = arena.doubleVec(n);
            xu = arena.doubleVec(n); for (int j = 0; j < n; j++) xu[j] = (double)1;
            integ = new NativeArray<byte>(n, Allocator.Persistent); for (int j = 0; j < n; j++) integ[j] = 1;
        }

        // MIPLIB "p0033" -- SAME literal instance data as MIPTests.double.cs's P0033 (Crowder, Johnson &
        // Padberg 1983), not re-derived. 33 binaries, 15 LessEqual rows, proven optimum 3089.
        static void BuildP0033Double(in Arena arena, out doubleMxN A, out doubleN b, out doubleN c,
                                     out NativeArray<ConstraintSense> senses, out doubleN xl, out doubleN xu,
                                     out NativeArray<byte> integ)
        {
            const int n = 33, m = 15;
            A = arena.doubleMat(m, n);   // zero-initialized
            c = arena.doubleVec(n);
            c[0] = (double)171; c[1] = (double)171; c[2] = (double)171; c[3] = (double)171; c[4] = (double)163;
            c[5] = (double)162; c[6] = (double)163; c[7] = (double)69; c[8] = (double)69; c[9] = (double)183;
            c[10] = (double)183; c[11] = (double)183; c[12] = (double)183; c[13] = (double)49; c[14] = (double)183;
            c[15] = (double)258; c[16] = (double)517; c[17] = (double)250; c[18] = (double)500; c[19] = (double)250;
            c[20] = (double)500; c[21] = (double)159; c[22] = (double)318; c[23] = (double)159; c[24] = (double)318;
            c[25] = (double)159; c[26] = (double)318; c[27] = (double)159; c[28] = (double)318; c[29] = (double)114;
            c[30] = (double)228; c[31] = (double)159; c[32] = (double)318;

            A[0, 0] = (double)1; A[0, 1] = (double)1; A[0, 2] = (double)1; A[0, 3] = (double)1;
            A[1, 4] = (double)1; A[1, 5] = (double)1; A[1, 6] = (double)1;
            A[2, 7] = (double)1; A[2, 8] = (double)1;
            A[3, 9] = (double)1; A[3, 10] = (double)1; A[3, 11] = (double)1; A[3, 12] = (double)1; A[3, 14] = (double)1;
            A[4, 9] = (double)(-230); A[4, 15] = (double)(-200); A[4, 16] = (double)(-400);
            A[5, 2] = (double)300; A[5, 3] = (double)300; A[5, 4] = (double)285; A[5, 5] = (double)285;
            A[5, 7] = (double)265; A[5, 8] = (double)265; A[5, 11] = (double)230; A[5, 12] = (double)230;
            A[5, 13] = (double)190; A[5, 21] = (double)200; A[5, 22] = (double)400; A[5, 23] = (double)200;
            A[5, 24] = (double)400; A[5, 25] = (double)200; A[5, 26] = (double)400; A[5, 27] = (double)200;
            A[5, 28] = (double)400; A[5, 29] = (double)200; A[5, 30] = (double)400;
            for (int j = 0; j < n; j++) A[6, j] = -A[5, j];   // row 6 = exact negation of row 5
            A[7, 3] = (double)(-300); A[7, 29] = (double)(-200); A[7, 30] = (double)(-400);
            A[8, 0] = (double)(-300); A[8, 5] = (double)(-285); A[8, 8] = (double)(-265); A[8, 13] = (double)(-190);
            A[8, 25] = (double)(-200); A[8, 26] = (double)(-400);
            A[9, 0] = (double)(-300); A[9, 2] = (double)(-300); A[9, 5] = (double)(-285); A[9, 8] = (double)(-265);
            A[9, 13] = (double)(-190); A[9, 25] = (double)(-200); A[9, 26] = (double)(-400); A[9, 27] = (double)(-200);
            A[9, 28] = (double)(-400);
            A[10, 4] = (double)(-285); A[10, 7] = (double)(-265); A[10, 10] = (double)(-230); A[10, 21] = (double)(-200);
            A[10, 22] = (double)(-400);
            A[11, 4] = (double)(-285); A[11, 7] = (double)(-265); A[11, 10] = (double)(-230); A[11, 11] = (double)(-230);
            A[11, 21] = (double)(-200); A[11, 22] = (double)(-400); A[11, 23] = (double)(-200); A[11, 24] = (double)(-400);
            A[12, 1] = (double)(-300); A[12, 17] = (double)(-200); A[12, 18] = (double)(-400);
            A[13, 1] = (double)(-300); A[13, 17] = (double)(-200); A[13, 18] = (double)(-400); A[13, 19] = (double)(-200);
            A[13, 20] = (double)(-400);
            A[14, 6] = (double)(-285); A[14, 31] = (double)(-200); A[14, 32] = (double)(-400);

            b = arena.doubleVec(m);
            b[0] = (double)1; b[1] = (double)1; b[2] = (double)1; b[3] = (double)1; b[4] = (double)(-5);
            b[5] = (double)2700; b[6] = (double)(-2600); b[7] = (double)(-100); b[8] = (double)(-900);
            b[9] = (double)(-1656); b[10] = (double)(-335); b[11] = (double)(-1026); b[12] = (double)(-5);
            b[13] = (double)(-500); b[14] = (double)(-270);
            senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
            for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;
            xl = arena.doubleVec(n);
            xu = arena.doubleVec(n); for (int j = 0; j < n; j++) xu[j] = (double)1;
            integ = new NativeArray<byte>(n, Allocator.Persistent); for (int j = 0; j < n; j++) integ[j] = 1;
        }

        // Synthetic "branchy" integer-box MIP -- SAME recipe as MIPTests.double.cs's BuildBranchy12,
        // generalized to any (n, m, seed): integer A/b/c built around a random feasible integer point x*
        // (guarantees feasible + bounded), box [0,3]^n, every variable integer. n=12 with seed 424242
        // exactly reproduces the test suite's own Branchy12 instance.
        static void BuildBranchyDouble(int n, int m, uint seed, in Arena arena, out doubleMxN A, out doubleN b, out doubleN c,
                                       out NativeArray<ConstraintSense> senses, out doubleN xl, out doubleN xu,
                                       out NativeArray<byte> integ)
        {
            var rng = new Random(seed);
            A = arena.doubleMat(m, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (double)rng.NextInt(-2, 3);

            var xstar = new NativeArray<int>(n, Allocator.Temp);
            for (int j = 0; j < n; j++) xstar[j] = rng.NextInt(0, 4);

            b = arena.doubleVec(m);
            senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
            for (int i = 0; i < m; i++)
            {
                int act = 0;
                for (int j = 0; j < n; j++) act += (int)A[i, j] * xstar[j];
                int r = rng.NextInt(0, 3);
                int slack = rng.NextInt(0, 3);
                if (r == 0) { senses[i] = ConstraintSense.LessEqual; b[i] = (double)(act + slack); }
                else if (r == 1) { senses[i] = ConstraintSense.GreaterEqual; b[i] = (double)(act - slack); }
                else { senses[i] = ConstraintSense.Equal; b[i] = (double)act; }
            }

            c = arena.doubleVec(n);
            for (int j = 0; j < n; j++) c[j] = (double)rng.NextInt(-3, 4);
            xl = arena.doubleVec(n);
            xu = arena.doubleVec(n); for (int j = 0; j < n; j++) xu[j] = (double)3;
            integ = new NativeArray<byte>(n, Allocator.Persistent); for (int j = 0; j < n; j++) integ[j] = 1;

            xstar.Dispose();
        }

        // ==== Section 1: MIPLIB oracles -- stein9 (both dtypes), stein15 + p0033 (double only) ====
        static void SectionOraclesDouble(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 1. MIPLIB oracles: stein9 (both dtypes); stein15 + p0033 (double only -- float " +
                          "cannot prove optimality within a sane node budget, see MIPTests.double.cs's Stein15/P0033) [double] ---");
            sb.AppendLine(MIPBenchmarkFmt.Header());

            {
                var arena = new Arena(Allocator.Persistent);
                BuildStein9Double(in arena, out var A, out var b, out var c, out var senses, out var xl, out var xu, out var integ);
                var x = arena.doubleVec(9);

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var nodesOut = new NativeArray<int>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);
                var dualBoundOut = new NativeArray<double>(1, Allocator.Persistent);
                var gapOut = new NativeArray<double>(1, Allocator.Persistent);
                var job = new MipSolveJobDouble
                {
                    A = A, b = b, c = c, xl = xl, xu = xu, x = x, senses = senses, integrality = integ,
                    maxNodes = MIPBenchmarkFmt.OracleMaxNodes, maxIter = 0, absGap = 0.0, relGap = 0.0,
                    objOut = objOut, nodesOut = nodesOut, itersOut = itersOut, statusOut = statusOut,
                    dualBoundOut = dualBoundOut, gapOut = gapOut,
                };
                var stat = Bench.Time(() => job.Run());
                sb.AppendLine(MIPBenchmarkFmt.Row("double", "stein9", 9, 13, stat, nodesOut[0], itersOut[0], statusOut[0], objOut[0]));

                objOut.Dispose(); nodesOut.Dispose(); itersOut.Dispose(); statusOut.Dispose(); dualBoundOut.Dispose(); gapOut.Dispose();
                senses.Dispose(); integ.Dispose(); arena.Dispose();
            }

            // stein15 / p0033: double only (float known not to converge / not to prove optimality within a
            // sane node budget -- same rationale as MIPTests.double.cs's Stein15/P0033).
            int cases = 1;
            for (int s = 0; s < cases; s++)
            {
                var arena = new Arena(Allocator.Persistent);
                BuildStein15Double(in arena, out var A, out var b, out var c, out var senses, out var xl, out var xu, out var integ);
                var x = arena.doubleVec(15);

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var nodesOut = new NativeArray<int>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);
                var dualBoundOut = new NativeArray<double>(1, Allocator.Persistent);
                var gapOut = new NativeArray<double>(1, Allocator.Persistent);
                var job = new MipSolveJobDouble
                {
                    A = A, b = b, c = c, xl = xl, xu = xu, x = x, senses = senses, integrality = integ,
                    maxNodes = MIPBenchmarkFmt.OracleMaxNodes, maxIter = 0, absGap = 0.0, relGap = 0.0,
                    objOut = objOut, nodesOut = nodesOut, itersOut = itersOut, statusOut = statusOut,
                    dualBoundOut = dualBoundOut, gapOut = gapOut,
                };
                var stat = Bench.Time(() => job.Run());
                sb.AppendLine(MIPBenchmarkFmt.Row("double", "stein15", 15, 36, stat, nodesOut[0], itersOut[0], statusOut[0], objOut[0]));

                objOut.Dispose(); nodesOut.Dispose(); itersOut.Dispose(); statusOut.Dispose(); dualBoundOut.Dispose(); gapOut.Dispose();
                senses.Dispose(); integ.Dispose(); arena.Dispose();
            }
            for (int s = 0; s < cases; s++)
            {
                var arena = new Arena(Allocator.Persistent);
                BuildP0033Double(in arena, out var A, out var b, out var c, out var senses, out var xl, out var xu, out var integ);
                var x = arena.doubleVec(33);

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var nodesOut = new NativeArray<int>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);
                var dualBoundOut = new NativeArray<double>(1, Allocator.Persistent);
                var gapOut = new NativeArray<double>(1, Allocator.Persistent);
                var job = new MipSolveJobDouble
                {
                    A = A, b = b, c = c, xl = xl, xu = xu, x = x, senses = senses, integrality = integ,
                    maxNodes = MIPBenchmarkFmt.OracleMaxNodes, maxIter = 0, absGap = 0.0, relGap = 0.0,
                    objOut = objOut, nodesOut = nodesOut, itersOut = itersOut, statusOut = statusOut,
                    dualBoundOut = dualBoundOut, gapOut = gapOut,
                };
                var stat = Bench.Time(() => job.Run());
                sb.AppendLine(MIPBenchmarkFmt.Row("double", "p0033", 33, 15, stat, nodesOut[0], itersOut[0], statusOut[0], objOut[0]));

                objOut.Dispose(); nodesOut.Dispose(); itersOut.Dispose(); statusOut.Dispose(); dualBoundOut.Dispose(); gapOut.Dispose();
                senses.Dispose(); integ.Dispose(); arena.Dispose();
            }
        }

        // ==== Section 2: synthetic scaling -- random branchy integer-box MIPs, n = 8/12/16, both dtypes,
        //      a TIGHT maxNodes safety cap (these are untested instances with no known baseline, unlike
        //      the MIPLIB oracles above) -- a diverging cell reports its own status honestly (e.g.
        //      NodeLimit) instead of burning the wall-clock budget. ====
        static void SectionScalingDouble(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 2. Synthetic scaling: random branchy integer-box MIPs (n=8,12,16; fixed seeds; n=12's " +
                          "seed reproduces MIPTests.double.cs's Branchy12), both dtypes, maxNodes safety cap -- a " +
                          "diverging cell reports its status honestly (e.g. NodeLimit) rather than burning the " +
                          "wall-clock budget [double] ---");
            sb.AppendLine(MIPBenchmarkFmt.Header());

            for (int idx = 0; idx < MIPBenchmarkFmt.ScalingN.Length; idx++)
            {
                int n = MIPBenchmarkFmt.ScalingN[idx];
                int m = n / 2;
                uint seed = MIPBenchmarkFmt.ScalingSeed[idx];

                var arena = new Arena(Allocator.Persistent);
                BuildBranchyDouble(n, m, seed, in arena, out var A, out var b, out var c, out var senses, out var xl, out var xu, out var integ);
                var x = arena.doubleVec(n);

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var nodesOut = new NativeArray<int>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);
                var dualBoundOut = new NativeArray<double>(1, Allocator.Persistent);
                var gapOut = new NativeArray<double>(1, Allocator.Persistent);
                var job = new MipSolveJobDouble
                {
                    A = A, b = b, c = c, xl = xl, xu = xu, x = x, senses = senses, integrality = integ,
                    maxNodes = MIPBenchmarkFmt.ScalingMaxNodes, maxIter = 0, absGap = 0.0, relGap = 0.0,
                    objOut = objOut, nodesOut = nodesOut, itersOut = itersOut, statusOut = statusOut,
                    dualBoundOut = dualBoundOut, gapOut = gapOut,
                };
                var stat = Bench.Time(() => job.Run());
                sb.AppendLine(MIPBenchmarkFmt.Row("double", "branchy" + n, n, m, stat, nodesOut[0], itersOut[0], statusOut[0], objOut[0]));

                objOut.Dispose(); nodesOut.Dispose(); itersOut.Dispose(); statusOut.Dispose(); dualBoundOut.Dispose(); gapOut.Dispose();
                senses.Dispose(); integ.Dispose(); arena.Dispose();
            }
        }

        // ==== Section 3: gap-limit economics -- p0033 (double only) at relGap = 0 / 0.01 / 0.05, showing
        //      time vs proof-quality tradeoff (the rounding heuristic finds the optimum 3089 early; the
        //      gap PROOF -- closing dualBound -- is what costs). DOUBLE-ONLY: same rationale as Section 1's
        //      p0033 row. ====
        static void SectionGapDouble(StringBuilder sb)
        {
            int cases = 1;
            for (int s = 0; s < cases; s++)
            {
                sb.AppendLine();
                sb.AppendLine("--- 3. Gap-limit economics: p0033 (double only) at relGap = 0 / 0.01 / 0.05 -- the " +
                              "rounding heuristic finds the optimum (3089) early, the gap PROOF is what costs [double] ---");
                sb.AppendLine(MIPBenchmarkFmt.GapHeader());

                foreach (var relGap in MIPBenchmarkFmt.GapRelGaps)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildP0033Double(in arena, out var A, out var b, out var c, out var senses, out var xl, out var xu, out var integ);
                    var x = arena.doubleVec(33);

                    var objOut = new NativeArray<double>(1, Allocator.Persistent);
                    var nodesOut = new NativeArray<int>(1, Allocator.Persistent);
                    var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                    var statusOut = new NativeArray<int>(1, Allocator.Persistent);
                    var dualBoundOut = new NativeArray<double>(1, Allocator.Persistent);
                    var gapOut = new NativeArray<double>(1, Allocator.Persistent);
                    var job = new MipSolveJobDouble
                    {
                        A = A, b = b, c = c, xl = xl, xu = xu, x = x, senses = senses, integrality = integ,
                        maxNodes = MIPBenchmarkFmt.GapMaxNodes, maxIter = 0, absGap = 0.0, relGap = relGap,
                        objOut = objOut, nodesOut = nodesOut, itersOut = itersOut, statusOut = statusOut,
                        dualBoundOut = dualBoundOut, gapOut = gapOut,
                    };
                    var stat = Bench.Time(() => job.Run());
                    sb.AppendLine(MIPBenchmarkFmt.GapRow("double", "p0033", relGap, stat, nodesOut[0], itersOut[0], statusOut[0], objOut[0], gapOut[0]));

                    objOut.Dispose(); nodesOut.Dispose(); itersOut.Dispose(); statusOut.Dispose(); dualBoundOut.Dispose(); gapOut.Dispose();
                    senses.Dispose(); integ.Dispose(); arena.Dispose();
                }
            }
        }

        // ==== Section 4: warm-start accounting -- one mid-size instance (stein15, double only), reporting
        //      lpIterations/nodes: the average simplex pivots the warm-started dual simplex needs to
        //      restore optimality PER NODE (already in MIPInfo -- no instrumentation needed). A low ratio
        //      is the warm-start payoff -- each node reuses the persistent LPBasis rather than cold-
        //      starting a fresh LP. DOUBLE-ONLY: same rationale as Section 1's stein15 row. ====
        static void SectionWarmStartDouble(StringBuilder sb)
        {
            int cases = 1;
            for (int s = 0; s < cases; s++)
            {
                sb.AppendLine();
                sb.AppendLine("--- 4. Warm-start accounting: stein15 (double only, mid-size), lpIterations/nodes -- " +
                              "average simplex pivots per warm-started node re-solve [double] ---");
                sb.AppendLine(MIPBenchmarkFmt.Header());

                var arena = new Arena(Allocator.Persistent);
                BuildStein15Double(in arena, out var A, out var b, out var c, out var senses, out var xl, out var xu, out var integ);
                var x = arena.doubleVec(15);

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var nodesOut = new NativeArray<int>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);
                var dualBoundOut = new NativeArray<double>(1, Allocator.Persistent);
                var gapOut = new NativeArray<double>(1, Allocator.Persistent);
                var job = new MipSolveJobDouble
                {
                    A = A, b = b, c = c, xl = xl, xu = xu, x = x, senses = senses, integrality = integ,
                    maxNodes = MIPBenchmarkFmt.WarmStartMaxNodes, maxIter = 0, absGap = 0.0, relGap = 0.0,
                    objOut = objOut, nodesOut = nodesOut, itersOut = itersOut, statusOut = statusOut,
                    dualBoundOut = dualBoundOut, gapOut = gapOut,
                };
                var stat = Bench.Time(() => job.Run());
                sb.AppendLine(MIPBenchmarkFmt.Row("double", "stein15", 15, 36, stat, nodesOut[0], itersOut[0], statusOut[0], objOut[0]));
                sb.AppendLine(MIPBenchmarkFmt.RatioLine("double", itersOut[0], nodesOut[0]));

                objOut.Dispose(); nodesOut.Dispose(); itersOut.Dispose(); statusOut.Dispose(); dualBoundOut.Dispose(); gapOut.Dispose();
                senses.Dispose(); integ.Dispose(); arena.Dispose();
            }
        }
    }
}
