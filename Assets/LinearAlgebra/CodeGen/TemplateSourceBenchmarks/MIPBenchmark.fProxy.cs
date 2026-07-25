using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using BULA;

namespace BULA.Benchmarks
{
    // GENERATED per-dtype half of MIPBenchmark (the timed IJob + the per-section instance-builders +
    // build+measure methods). The dtype-agnostic harness (sizes/seeds, row formatters, Run, Section) is
    // hand-written in Assets/LinearAlgebra/Benchmarks/MIPBenchmark.cs (MIPBenchmarkFmt + the partial
    // class).
    //
    // The job carries its OWN reporting outputs (objOut/nodesOut/itersOut/statusOut/dualBoundOut/gapOut,
    // length-1 arrays) written from inside Execute() -- the same "no second, Mono-interpreted solve just
    // to harvest diagnostics" discipline LPBenchmark.fProxy.cs's own header comment explains (Bench.Time
    // already runs the job once as a warmup before the timed reps, so the outputs are a side effect of
    // the SAME Burst-native call being timed, not an extra solve).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MipSolveJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN b, c, xl, xu, x;
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
        // Sets a coefficient-1 covering triple in row r of A -- same helper as MIPTests.fProxy.cs's
        // SetTriple (the stein set-covering rows).
        static void SetTripleFProxy(fProxyMxN A, int r, int i, int j, int k)
        {
            A[r, i] = (fProxy)1; A[r, j] = (fProxy)1; A[r, k] = (fProxy)1;
        }

        // MIPLIB "stein9" -- SAME literal instance data as MIPTests.fProxy.cs's BuildStein9 (Fulkerson,
        // Nemhauser & Trotter 1974; MIPLIB 3), not re-derived. 9 binaries, minimize count, proven optimum 5.
        static void BuildStein9FProxy(Allocator allocator, out fProxyMxN A, out fProxyN b, out fProxyN c,
                                      out NativeArray<ConstraintSense> senses, out fProxyN xl, out fProxyN xu,
                                      out NativeArray<byte> integ)
        {
            const int n = 9, m = 13;
            A = new fProxyMxN(m, n, allocator);   // zero-initialized
            SetTripleFProxy(A, 0, 1, 2, 3); SetTripleFProxy(A, 1, 0, 2, 4); SetTripleFProxy(A, 2, 0, 1, 5); SetTripleFProxy(A, 3, 4, 5, 6);
            SetTripleFProxy(A, 4, 3, 5, 7); SetTripleFProxy(A, 5, 3, 4, 8); SetTripleFProxy(A, 6, 0, 7, 8); SetTripleFProxy(A, 7, 1, 6, 8);
            SetTripleFProxy(A, 8, 2, 6, 7); SetTripleFProxy(A, 9, 0, 3, 6); SetTripleFProxy(A, 10, 1, 4, 7); SetTripleFProxy(A, 11, 2, 5, 8);
            for (int j = 0; j < n; j++) A[12, j] = (fProxy)1;   // the OB2 "at least 4 of 9" cut

            b = new fProxyN(m, allocator); for (int i = 0; i < 12; i++) b[i] = (fProxy)1; b[12] = (fProxy)4;
            senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
            for (int i = 0; i < m; i++) senses[i] = ConstraintSense.GreaterEqual;
            c = new fProxyN(n, allocator); for (int j = 0; j < n; j++) c[j] = (fProxy)1;
            xl = new fProxyN(n, allocator);
            xu = new fProxyN(n, allocator); for (int j = 0; j < n; j++) xu[j] = (fProxy)1;
            integ = new NativeArray<byte>(n, Allocator.Persistent); for (int j = 0; j < n; j++) integ[j] = 1;
        }

        // MIPLIB "stein15" -- SAME literal instance data as MIPTests.fProxy.cs's Stein15, not re-derived.
        // 15 binaries, minimize count, proven optimum 9 (double: ~275 nodes / ~4307 LP iterations per the
        // test file's own measured baseline).
        static void BuildStein15FProxy(Allocator allocator, out fProxyMxN A, out fProxyN b, out fProxyN c,
                                       out NativeArray<ConstraintSense> senses, out fProxyN xl, out fProxyN xu,
                                       out NativeArray<byte> integ)
        {
            const int n = 15, m = 36;
            A = new fProxyMxN(m, n, allocator);   // zero-initialized
            SetTripleFProxy(A, 0, 2, 3, 5);   SetTripleFProxy(A, 1, 3, 4, 6);   SetTripleFProxy(A, 2, 0, 4, 7);   SetTripleFProxy(A, 3, 0, 1, 8);   SetTripleFProxy(A, 4, 1, 2, 9);
            SetTripleFProxy(A, 5, 1, 4, 5);   SetTripleFProxy(A, 6, 0, 2, 6);   SetTripleFProxy(A, 7, 1, 3, 7);   SetTripleFProxy(A, 8, 2, 4, 8);   SetTripleFProxy(A, 9, 0, 3, 9);
            SetTripleFProxy(A, 10, 7, 8, 10); SetTripleFProxy(A, 11, 8, 9, 11); SetTripleFProxy(A, 12, 5, 9, 12); SetTripleFProxy(A, 13, 5, 6, 13); SetTripleFProxy(A, 14, 6, 7, 14);
            SetTripleFProxy(A, 15, 6, 9, 10); SetTripleFProxy(A, 16, 5, 7, 11); SetTripleFProxy(A, 17, 6, 8, 12); SetTripleFProxy(A, 18, 7, 9, 13); SetTripleFProxy(A, 19, 5, 8, 14);
            SetTripleFProxy(A, 20, 0, 12, 13); SetTripleFProxy(A, 21, 1, 13, 14); SetTripleFProxy(A, 22, 2, 10, 14); SetTripleFProxy(A, 23, 3, 10, 11); SetTripleFProxy(A, 24, 4, 11, 12);
            SetTripleFProxy(A, 25, 0, 11, 14); SetTripleFProxy(A, 26, 1, 10, 12); SetTripleFProxy(A, 27, 2, 11, 13); SetTripleFProxy(A, 28, 3, 12, 14); SetTripleFProxy(A, 29, 4, 10, 13);
            SetTripleFProxy(A, 30, 0, 5, 10); SetTripleFProxy(A, 31, 1, 6, 11); SetTripleFProxy(A, 32, 2, 7, 12); SetTripleFProxy(A, 33, 3, 8, 13); SetTripleFProxy(A, 34, 4, 9, 14);
            for (int j = 0; j < n; j++) A[35, j] = (fProxy)1;   // the "at least 7 of 15" cut

            b = new fProxyN(m, allocator); for (int i = 0; i < 35; i++) b[i] = (fProxy)1; b[35] = (fProxy)7;
            senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
            for (int i = 0; i < m; i++) senses[i] = ConstraintSense.GreaterEqual;
            c = new fProxyN(n, allocator); for (int j = 0; j < n; j++) c[j] = (fProxy)1;
            xl = new fProxyN(n, allocator);
            xu = new fProxyN(n, allocator); for (int j = 0; j < n; j++) xu[j] = (fProxy)1;
            integ = new NativeArray<byte>(n, Allocator.Persistent); for (int j = 0; j < n; j++) integ[j] = 1;
        }

        // MIPLIB "p0033" -- SAME literal instance data as MIPTests.fProxy.cs's P0033 (Crowder, Johnson &
        // Padberg 1983), not re-derived. 33 binaries, 15 LessEqual rows, proven optimum 3089.
        static void BuildP0033FProxy(Allocator allocator, out fProxyMxN A, out fProxyN b, out fProxyN c,
                                     out NativeArray<ConstraintSense> senses, out fProxyN xl, out fProxyN xu,
                                     out NativeArray<byte> integ)
        {
            const int n = 33, m = 15;
            A = new fProxyMxN(m, n, allocator);   // zero-initialized
            c = new fProxyN(n, allocator);
            c[0] = (fProxy)171; c[1] = (fProxy)171; c[2] = (fProxy)171; c[3] = (fProxy)171; c[4] = (fProxy)163;
            c[5] = (fProxy)162; c[6] = (fProxy)163; c[7] = (fProxy)69; c[8] = (fProxy)69; c[9] = (fProxy)183;
            c[10] = (fProxy)183; c[11] = (fProxy)183; c[12] = (fProxy)183; c[13] = (fProxy)49; c[14] = (fProxy)183;
            c[15] = (fProxy)258; c[16] = (fProxy)517; c[17] = (fProxy)250; c[18] = (fProxy)500; c[19] = (fProxy)250;
            c[20] = (fProxy)500; c[21] = (fProxy)159; c[22] = (fProxy)318; c[23] = (fProxy)159; c[24] = (fProxy)318;
            c[25] = (fProxy)159; c[26] = (fProxy)318; c[27] = (fProxy)159; c[28] = (fProxy)318; c[29] = (fProxy)114;
            c[30] = (fProxy)228; c[31] = (fProxy)159; c[32] = (fProxy)318;

            A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)1; A[0, 2] = (fProxy)1; A[0, 3] = (fProxy)1;
            A[1, 4] = (fProxy)1; A[1, 5] = (fProxy)1; A[1, 6] = (fProxy)1;
            A[2, 7] = (fProxy)1; A[2, 8] = (fProxy)1;
            A[3, 9] = (fProxy)1; A[3, 10] = (fProxy)1; A[3, 11] = (fProxy)1; A[3, 12] = (fProxy)1; A[3, 14] = (fProxy)1;
            A[4, 9] = (fProxy)(-230); A[4, 15] = (fProxy)(-200); A[4, 16] = (fProxy)(-400);
            A[5, 2] = (fProxy)300; A[5, 3] = (fProxy)300; A[5, 4] = (fProxy)285; A[5, 5] = (fProxy)285;
            A[5, 7] = (fProxy)265; A[5, 8] = (fProxy)265; A[5, 11] = (fProxy)230; A[5, 12] = (fProxy)230;
            A[5, 13] = (fProxy)190; A[5, 21] = (fProxy)200; A[5, 22] = (fProxy)400; A[5, 23] = (fProxy)200;
            A[5, 24] = (fProxy)400; A[5, 25] = (fProxy)200; A[5, 26] = (fProxy)400; A[5, 27] = (fProxy)200;
            A[5, 28] = (fProxy)400; A[5, 29] = (fProxy)200; A[5, 30] = (fProxy)400;
            for (int j = 0; j < n; j++) A[6, j] = -A[5, j];   // row 6 = exact negation of row 5
            A[7, 3] = (fProxy)(-300); A[7, 29] = (fProxy)(-200); A[7, 30] = (fProxy)(-400);
            A[8, 0] = (fProxy)(-300); A[8, 5] = (fProxy)(-285); A[8, 8] = (fProxy)(-265); A[8, 13] = (fProxy)(-190);
            A[8, 25] = (fProxy)(-200); A[8, 26] = (fProxy)(-400);
            A[9, 0] = (fProxy)(-300); A[9, 2] = (fProxy)(-300); A[9, 5] = (fProxy)(-285); A[9, 8] = (fProxy)(-265);
            A[9, 13] = (fProxy)(-190); A[9, 25] = (fProxy)(-200); A[9, 26] = (fProxy)(-400); A[9, 27] = (fProxy)(-200);
            A[9, 28] = (fProxy)(-400);
            A[10, 4] = (fProxy)(-285); A[10, 7] = (fProxy)(-265); A[10, 10] = (fProxy)(-230); A[10, 21] = (fProxy)(-200);
            A[10, 22] = (fProxy)(-400);
            A[11, 4] = (fProxy)(-285); A[11, 7] = (fProxy)(-265); A[11, 10] = (fProxy)(-230); A[11, 11] = (fProxy)(-230);
            A[11, 21] = (fProxy)(-200); A[11, 22] = (fProxy)(-400); A[11, 23] = (fProxy)(-200); A[11, 24] = (fProxy)(-400);
            A[12, 1] = (fProxy)(-300); A[12, 17] = (fProxy)(-200); A[12, 18] = (fProxy)(-400);
            A[13, 1] = (fProxy)(-300); A[13, 17] = (fProxy)(-200); A[13, 18] = (fProxy)(-400); A[13, 19] = (fProxy)(-200);
            A[13, 20] = (fProxy)(-400);
            A[14, 6] = (fProxy)(-285); A[14, 31] = (fProxy)(-200); A[14, 32] = (fProxy)(-400);

            b = new fProxyN(m, allocator);
            b[0] = (fProxy)1; b[1] = (fProxy)1; b[2] = (fProxy)1; b[3] = (fProxy)1; b[4] = (fProxy)(-5);
            b[5] = (fProxy)2700; b[6] = (fProxy)(-2600); b[7] = (fProxy)(-100); b[8] = (fProxy)(-900);
            b[9] = (fProxy)(-1656); b[10] = (fProxy)(-335); b[11] = (fProxy)(-1026); b[12] = (fProxy)(-5);
            b[13] = (fProxy)(-500); b[14] = (fProxy)(-270);
            senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
            for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;
            xl = new fProxyN(n, allocator);
            xu = new fProxyN(n, allocator); for (int j = 0; j < n; j++) xu[j] = (fProxy)1;
            integ = new NativeArray<byte>(n, Allocator.Persistent); for (int j = 0; j < n; j++) integ[j] = 1;
        }

        // Synthetic "branchy" integer-box MIP -- SAME recipe as MIPTests.fProxy.cs's BuildBranchy12,
        // generalized to any (n, m, seed): integer A/b/c built around a random feasible integer point x*
        // (guarantees feasible + bounded), box [0,3]^n, every variable integer. n=12 with seed 424242
        // exactly reproduces the test suite's own Branchy12 instance.
        static void BuildBranchyFProxy(int n, int m, uint seed, Allocator allocator, out fProxyMxN A, out fProxyN b, out fProxyN c,
                                       out NativeArray<ConstraintSense> senses, out fProxyN xl, out fProxyN xu,
                                       out NativeArray<byte> integ)
        {
            var rng = new Random(seed);
            A = new fProxyMxN(m, n, allocator);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (fProxy)rng.NextInt(-2, 3);

            var xstar = new NativeArray<int>(n, Allocator.Temp);
            for (int j = 0; j < n; j++) xstar[j] = rng.NextInt(0, 4);

            b = new fProxyN(m, allocator);
            senses = new NativeArray<ConstraintSense>(m, Allocator.Persistent);
            for (int i = 0; i < m; i++)
            {
                int act = 0;
                for (int j = 0; j < n; j++) act += (int)A[i, j] * xstar[j];
                int r = rng.NextInt(0, 3);
                int slack = rng.NextInt(0, 3);
                if (r == 0) { senses[i] = ConstraintSense.LessEqual; b[i] = (fProxy)(act + slack); }
                else if (r == 1) { senses[i] = ConstraintSense.GreaterEqual; b[i] = (fProxy)(act - slack); }
                else { senses[i] = ConstraintSense.Equal; b[i] = (fProxy)act; }
            }

            c = new fProxyN(n, allocator);
            for (int j = 0; j < n; j++) c[j] = (fProxy)rng.NextInt(-3, 4);
            xl = new fProxyN(n, allocator);
            xu = new fProxyN(n, allocator); for (int j = 0; j < n; j++) xu[j] = (fProxy)3;
            integ = new NativeArray<byte>(n, Allocator.Persistent); for (int j = 0; j < n; j++) integ[j] = 1;

            xstar.Dispose();
        }

        // ==== Section 1: MIPLIB oracles -- stein9 (both dtypes), stein15 + p0033 (double only) ====
        static void SectionOraclesFProxy(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 1. MIPLIB oracles: stein9 (both dtypes); stein15 + p0033 (double only -- float " +
                          "cannot prove optimality within a sane node budget, see MIPTests.fProxy.cs's Stein15/P0033) [fProxy] ---");
            sb.AppendLine(MIPBenchmarkFmt.Header());

            {
                BuildStein9FProxy(Allocator.Persistent, out var A, out var b, out var c, out var senses, out var xl, out var xu, out var integ);
                var x = new fProxyN(9, Allocator.Persistent);

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var nodesOut = new NativeArray<int>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);
                var dualBoundOut = new NativeArray<double>(1, Allocator.Persistent);
                var gapOut = new NativeArray<double>(1, Allocator.Persistent);
                var job = new MipSolveJobFProxy
                {
                    A = A, b = b, c = c, xl = xl, xu = xu, x = x, senses = senses, integrality = integ,
                    maxNodes = MIPBenchmarkFmt.OracleMaxNodes, maxIter = 0, absGap = 0.0, relGap = 0.0,
                    objOut = objOut, nodesOut = nodesOut, itersOut = itersOut, statusOut = statusOut,
                    dualBoundOut = dualBoundOut, gapOut = gapOut,
                };
                var stat = Bench.Time(() => job.Run());
                sb.AppendLine(MIPBenchmarkFmt.Row("fProxy", "stein9", 9, 13, stat, nodesOut[0], itersOut[0], statusOut[0], objOut[0]));

                objOut.Dispose(); nodesOut.Dispose(); itersOut.Dispose(); statusOut.Dispose(); dualBoundOut.Dispose(); gapOut.Dispose();
                senses.Dispose(); integ.Dispose();
                A.Dispose(); b.Dispose(); c.Dispose(); xl.Dispose(); xu.Dispose(); x.Dispose();
            }

            // stein15 / p0033: double only (float known not to converge / not to prove optimality within a
            // sane node budget -- same rationale as MIPTests.fProxy.cs's Stein15/P0033).
            int cases = /*+choose[0|1]*/0/*-choose*/;
            for (int s = 0; s < cases; s++)
            {
                BuildStein15FProxy(Allocator.Persistent, out var A, out var b, out var c, out var senses, out var xl, out var xu, out var integ);
                var x = new fProxyN(15, Allocator.Persistent);

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var nodesOut = new NativeArray<int>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);
                var dualBoundOut = new NativeArray<double>(1, Allocator.Persistent);
                var gapOut = new NativeArray<double>(1, Allocator.Persistent);
                var job = new MipSolveJobFProxy
                {
                    A = A, b = b, c = c, xl = xl, xu = xu, x = x, senses = senses, integrality = integ,
                    maxNodes = MIPBenchmarkFmt.OracleMaxNodes, maxIter = 0, absGap = 0.0, relGap = 0.0,
                    objOut = objOut, nodesOut = nodesOut, itersOut = itersOut, statusOut = statusOut,
                    dualBoundOut = dualBoundOut, gapOut = gapOut,
                };
                var stat = Bench.Time(() => job.Run());
                sb.AppendLine(MIPBenchmarkFmt.Row("fProxy", "stein15", 15, 36, stat, nodesOut[0], itersOut[0], statusOut[0], objOut[0]));

                objOut.Dispose(); nodesOut.Dispose(); itersOut.Dispose(); statusOut.Dispose(); dualBoundOut.Dispose(); gapOut.Dispose();
                senses.Dispose(); integ.Dispose();
                A.Dispose(); b.Dispose(); c.Dispose(); xl.Dispose(); xu.Dispose(); x.Dispose();
            }
            for (int s = 0; s < cases; s++)
            {
                BuildP0033FProxy(Allocator.Persistent, out var A, out var b, out var c, out var senses, out var xl, out var xu, out var integ);
                var x = new fProxyN(33, Allocator.Persistent);

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var nodesOut = new NativeArray<int>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);
                var dualBoundOut = new NativeArray<double>(1, Allocator.Persistent);
                var gapOut = new NativeArray<double>(1, Allocator.Persistent);
                var job = new MipSolveJobFProxy
                {
                    A = A, b = b, c = c, xl = xl, xu = xu, x = x, senses = senses, integrality = integ,
                    maxNodes = MIPBenchmarkFmt.OracleMaxNodes, maxIter = 0, absGap = 0.0, relGap = 0.0,
                    objOut = objOut, nodesOut = nodesOut, itersOut = itersOut, statusOut = statusOut,
                    dualBoundOut = dualBoundOut, gapOut = gapOut,
                };
                var stat = Bench.Time(() => job.Run());
                sb.AppendLine(MIPBenchmarkFmt.Row("fProxy", "p0033", 33, 15, stat, nodesOut[0], itersOut[0], statusOut[0], objOut[0]));

                objOut.Dispose(); nodesOut.Dispose(); itersOut.Dispose(); statusOut.Dispose(); dualBoundOut.Dispose(); gapOut.Dispose();
                senses.Dispose(); integ.Dispose();
                A.Dispose(); b.Dispose(); c.Dispose(); xl.Dispose(); xu.Dispose(); x.Dispose();
            }
        }

        // ==== Section 2: synthetic scaling -- random branchy integer-box MIPs, n = 8/12/16, both dtypes,
        //      a TIGHT maxNodes safety cap (these are untested instances with no known baseline, unlike
        //      the MIPLIB oracles above) -- a diverging cell reports its own status honestly (e.g.
        //      NodeLimit) instead of burning the wall-clock budget. ====
        static void SectionScalingFProxy(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("--- 2. Synthetic scaling: random branchy integer-box MIPs (n=8,12,16; fixed seeds; n=12's " +
                          "seed reproduces MIPTests.fProxy.cs's Branchy12), both dtypes, maxNodes safety cap -- a " +
                          "diverging cell reports its status honestly (e.g. NodeLimit) rather than burning the " +
                          "wall-clock budget [fProxy] ---");
            sb.AppendLine(MIPBenchmarkFmt.Header());

            for (int idx = 0; idx < MIPBenchmarkFmt.ScalingN.Length; idx++)
            {
                int n = MIPBenchmarkFmt.ScalingN[idx];
                int m = n / 2;
                uint seed = MIPBenchmarkFmt.ScalingSeed[idx];

                BuildBranchyFProxy(n, m, seed, Allocator.Persistent, out var A, out var b, out var c, out var senses, out var xl, out var xu, out var integ);
                var x = new fProxyN(n, Allocator.Persistent);

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var nodesOut = new NativeArray<int>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);
                var dualBoundOut = new NativeArray<double>(1, Allocator.Persistent);
                var gapOut = new NativeArray<double>(1, Allocator.Persistent);
                var job = new MipSolveJobFProxy
                {
                    A = A, b = b, c = c, xl = xl, xu = xu, x = x, senses = senses, integrality = integ,
                    maxNodes = MIPBenchmarkFmt.ScalingMaxNodes, maxIter = 0, absGap = 0.0, relGap = 0.0,
                    objOut = objOut, nodesOut = nodesOut, itersOut = itersOut, statusOut = statusOut,
                    dualBoundOut = dualBoundOut, gapOut = gapOut,
                };
                var stat = Bench.Time(() => job.Run());
                sb.AppendLine(MIPBenchmarkFmt.Row("fProxy", "branchy" + n, n, m, stat, nodesOut[0], itersOut[0], statusOut[0], objOut[0]));

                objOut.Dispose(); nodesOut.Dispose(); itersOut.Dispose(); statusOut.Dispose(); dualBoundOut.Dispose(); gapOut.Dispose();
                senses.Dispose(); integ.Dispose();
                A.Dispose(); b.Dispose(); c.Dispose(); xl.Dispose(); xu.Dispose(); x.Dispose();
            }
        }

        // ==== Section 3: gap-limit economics -- p0033 (double only) at relGap = 0 / 0.01 / 0.05, showing
        //      time vs proof-quality tradeoff (the rounding heuristic finds the optimum 3089 early; the
        //      gap PROOF -- closing dualBound -- is what costs). DOUBLE-ONLY: same rationale as Section 1's
        //      p0033 row. ====
        static void SectionGapFProxy(StringBuilder sb)
        {
            int cases = /*+choose[0|1]*/0/*-choose*/;
            for (int s = 0; s < cases; s++)
            {
                sb.AppendLine();
                sb.AppendLine("--- 3. Gap-limit economics: p0033 (double only) at relGap = 0 / 0.01 / 0.05 -- the " +
                              "rounding heuristic finds the optimum (3089) early, the gap PROOF is what costs [fProxy] ---");
                sb.AppendLine(MIPBenchmarkFmt.GapHeader());

                foreach (var relGap in MIPBenchmarkFmt.GapRelGaps)
                {
                    BuildP0033FProxy(Allocator.Persistent, out var A, out var b, out var c, out var senses, out var xl, out var xu, out var integ);
                    var x = new fProxyN(33, Allocator.Persistent);

                    var objOut = new NativeArray<double>(1, Allocator.Persistent);
                    var nodesOut = new NativeArray<int>(1, Allocator.Persistent);
                    var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                    var statusOut = new NativeArray<int>(1, Allocator.Persistent);
                    var dualBoundOut = new NativeArray<double>(1, Allocator.Persistent);
                    var gapOut = new NativeArray<double>(1, Allocator.Persistent);
                    var job = new MipSolveJobFProxy
                    {
                        A = A, b = b, c = c, xl = xl, xu = xu, x = x, senses = senses, integrality = integ,
                        maxNodes = MIPBenchmarkFmt.GapMaxNodes, maxIter = 0, absGap = 0.0, relGap = relGap,
                        objOut = objOut, nodesOut = nodesOut, itersOut = itersOut, statusOut = statusOut,
                        dualBoundOut = dualBoundOut, gapOut = gapOut,
                    };
                    var stat = Bench.Time(() => job.Run());
                    sb.AppendLine(MIPBenchmarkFmt.GapRow("fProxy", "p0033", relGap, stat, nodesOut[0], itersOut[0], statusOut[0], objOut[0], gapOut[0]));

                    objOut.Dispose(); nodesOut.Dispose(); itersOut.Dispose(); statusOut.Dispose(); dualBoundOut.Dispose(); gapOut.Dispose();
                    senses.Dispose(); integ.Dispose();
                    A.Dispose(); b.Dispose(); c.Dispose(); xl.Dispose(); xu.Dispose(); x.Dispose();
                }
            }
        }

        // ==== Section 4: warm-start accounting -- one mid-size instance (stein15, double only), reporting
        //      lpIterations/nodes: the average simplex pivots the warm-started dual simplex needs to
        //      restore optimality PER NODE (already in MIPInfo -- no instrumentation needed). A low ratio
        //      is the warm-start payoff -- each node reuses the persistent LPBasis rather than cold-
        //      starting a fresh LP. DOUBLE-ONLY: same rationale as Section 1's stein15 row. ====
        static void SectionWarmStartFProxy(StringBuilder sb)
        {
            int cases = /*+choose[0|1]*/0/*-choose*/;
            for (int s = 0; s < cases; s++)
            {
                sb.AppendLine();
                sb.AppendLine("--- 4. Warm-start accounting: stein15 (double only, mid-size), lpIterations/nodes -- " +
                              "average simplex pivots per warm-started node re-solve [fProxy] ---");
                sb.AppendLine(MIPBenchmarkFmt.Header());

                BuildStein15FProxy(Allocator.Persistent, out var A, out var b, out var c, out var senses, out var xl, out var xu, out var integ);
                var x = new fProxyN(15, Allocator.Persistent);

                var objOut = new NativeArray<double>(1, Allocator.Persistent);
                var nodesOut = new NativeArray<int>(1, Allocator.Persistent);
                var itersOut = new NativeArray<int>(1, Allocator.Persistent);
                var statusOut = new NativeArray<int>(1, Allocator.Persistent);
                var dualBoundOut = new NativeArray<double>(1, Allocator.Persistent);
                var gapOut = new NativeArray<double>(1, Allocator.Persistent);
                var job = new MipSolveJobFProxy
                {
                    A = A, b = b, c = c, xl = xl, xu = xu, x = x, senses = senses, integrality = integ,
                    maxNodes = MIPBenchmarkFmt.WarmStartMaxNodes, maxIter = 0, absGap = 0.0, relGap = 0.0,
                    objOut = objOut, nodesOut = nodesOut, itersOut = itersOut, statusOut = statusOut,
                    dualBoundOut = dualBoundOut, gapOut = gapOut,
                };
                var stat = Bench.Time(() => job.Run());
                sb.AppendLine(MIPBenchmarkFmt.Row("fProxy", "stein15", 15, 36, stat, nodesOut[0], itersOut[0], statusOut[0], objOut[0]));
                sb.AppendLine(MIPBenchmarkFmt.RatioLine("fProxy", itersOut[0], nodesOut[0]));

                objOut.Dispose(); nodesOut.Dispose(); itersOut.Dispose(); statusOut.Dispose(); dualBoundOut.Dispose(); gapOut.Dispose();
                senses.Dispose(); integ.Dispose();
                A.Dispose(); b.Dispose(); c.Dispose(); xl.Dispose(); xu.Dispose(); x.Dispose();
            }
        }
    }
}
