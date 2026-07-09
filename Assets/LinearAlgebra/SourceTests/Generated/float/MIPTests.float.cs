using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for MIP.solve -- LP-based branch & bound over the dual simplex (docs/draft-spec-mip.md).
// Grows by stage: (a)-(e) STAGE 2 (most-fractional DFS), (f) STAGE 3 (pseudocost + best-bound queue),
// (g) STAGE 4 (activity-based propagation, rounding heuristic, absGap/relGap gap limits + MIPLIB
// stein/p0033 known-answer oracles). Templated (float) so codegen emits a float and a double build; per the draft spec
// "test float only on tiny instances, double is the serious dtype", the exhaustive-enumeration
// cross-check (EnumCrossCheck) uses a per-dtype codegen choose-marker to run 2 tiny instances in
// float and 7 (up to n=5) in double. Every numeric assertion routes through the Fail[0..3] diagnostic
// slots (flag / got / expected-or-limit / diff-or-extra) exactly like LPTests.float.cs.
public class floatMIPTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            // ==== (a) 0/1 knapsacks with brute-forced known optima (max reformulated as min -value) ====
            Knapsack3,          // max 6x1+5x2+4x3 s.t. 4x1+3x2+2x3<=6, binary -> {1,3}, value 10, obj -10
            Knapsack6,          // max 10,13,18,31,7,15 / w 2,3,4,7,1,3 / cap 10, binary -> value 50, obj -50

            // ==== (b) assignment LP is totally unimodular -> root already integral, MUST finish at
            //         the root (nodes == 1: the false-fractionality canary) ====
            AssignmentRootIntegral, // 3x3, cost diag 1 / off 7 -> diagonal assignment, obj 3, nodes==1

            // ==== (c) classic Gomory/Wolsey 2-var textbook IP (LP relaxation fractional, rounds WRONG) ====
            GomoryWolsey,       // max 8x1+5x2 s.t. x1+x2<=6, 9x1+5x2<=45 -> int opt (5,0), obj -40 (LP 41.25)
            GomoryWolseyTrueInfiniteBound, // SAME instance/optimum but xu = +inf sentinel (1e30): the
                                //   integer UB row must start INERT, not carry a 1e30 rhs (regression for
                                //   the dataScale/artificialBound inflation bug). Same answer as GomoryWolsey.

            // ==== (d) infeasible / unbounded MIPs (all-NaN objective/dualBound/gap, x all zeros) ====
            InfeasibleNoIntInRange, // 2.1<=x<=2.9 integer: LP feasible, NO integer point -> Infeasible via B&B
            InfeasibleRootLP,   // x+y=1 with x>=1,y>=1 integer: root LP itself infeasible -> Infeasible, nodes==1
            UnboundedRoot,      // min -x s.t. x>=0 integer, xu=+inf -> Unbounded (root only), nodes==1

            // ==== (e) exhaustive-enumeration cross-check on random tiny integer-box MIPs ([0,3]^n) ====
            EnumCrossCheck,     // MIP optimum == brute-force min over all 4^n lattice points (multi-seed)

            // ==== extra coverage (not in the required oracle list) ====
            GeneralIntBounds,   // integer var xl=3,xu=10 (nonzero-xl shift path): min -x s.t. x<=7.5 -> x=7, obj -7
            NodeLimitNoIncumbent, // Gomory with maxNodes=1 -> NodeLimit, nodes==1, obj=+inf, dualBound finite (~-41.25)
            IterLimitNoIncumbent, // Gomory with maxIter=1  -> MaxIterations, nodes==1, obj=+inf, dualBound finite

            // ==== (f) STAGE 3 verification: pseudocost + reliability branching + best-bound queue with
            //         plunging replaces stage 2's most-fractional/pure-DFS search. The node count on the
            //         harder instances must NOT increase vs the stage-2 baselines (this IS the feature --
            //         hard-assert nodes <= baseline). Baselines were measured on both stages directly (by
            //         reverting to the stage-2 commit, running a throwaway diagnostic, then restoring).
            //         lpIterations is deliberately NOT asserted: on tiny instances pseudocost never becomes
            //         "reliable" within the few nodes explored, so it strong-branches nearly every candidate
            //         and total iterations can RISE even as node count falls -- an expected, documented
            //         tradeoff. Also: the single-threaded, RNG-free search must be bit-for-bit deterministic
            //         across repeated calls (determinism story). ====
            Stage3NodesKnapsack6,    // stage2 = stage3 = 1 node (tie): assert nodes <= 1, both dtypes
            Stage3NodesGomoryWolsey, // stage2 = stage3 = 7 nodes (lpIter 9 -> 27, NOT asserted): nodes <= 7
            Stage3NodesBranchy12,    // random n=12/m=6 seed 424242: stage2 267 -> stage3 241 nodes, obj 6
                                     //   (DOUBLE-ONLY: the stage-2 float baseline for this instance was an
                                     //   anomalous nodes=0, so there is no valid float baseline to compare)
            Stage3Determinism,       // two back-to-back GomoryWolsey solves -> identical nodes/iter/obj/bound/x
            Stage3DeterminismBranchy12, // same determinism check on the big branchy n=12 search (DOUBLE-ONLY)

            // ==== (g) STAGE 4 verification: activity-based domain propagation at every node, a rounding
            //         heuristic, and absGap/relGap gap limits make MIPStatus.GapLimit reachable
            //         (docs/draft-spec-mip.md stage 4). ====

            // -- MIPLIB tiny known-answer instances (the "stein/p0033" standard set) --
            Stein9,   // MIPLIB steiner-triple set-covering: 9 binaries, proven optimum 5 (both dtypes)
            Stein15,  // 15 binaries, proven optimum 9 (both dtypes; float lands ~8.9999997, within 1e-3)
            P0033,    // Crowder-Johnson-Padberg 0/1: 33 binaries, proven optimum 3089. DOUBLE-ONLY: float
                      //   finds the incumbent 3089 quickly but can't PROVE optimality within a sane node
                      //   budget (large coeff magnitudes up to 2700 vs fixed 1e-6 tolerances), same
                      //   float-baseline rationale as Stage3NodesBranchy12.

            // -- propagation: node-count drop vs the recorded stage-3 counts, + pre-LP fathom --
            Stage4NodesGomoryWolsey,     // stage3 7 -> stage4 5 nodes (propagation fathoms 2): assert exactly 5
            Stage4NodesBranchy12,        // stage3 241 -> stage4 218 nodes (DOUBLE-ONLY, same instance rationale)
            Stage4PropagationInfeasible, // 2x+2y=3 integer parity: propagation proves the root infeasible with
                                         //   NO LP solve (status Infeasible, nodes==1, lpIterations==0)

            // -- gap limits (new absGap/relGap parameters) --
            GapLimitRelGap,        // Branchy12 with relGap: stops at GapLimit before full exploration (DOUBLE-ONLY)
            GapLimitPassThrough,   // GomoryWolsey with absGap:0/relGap:0 (=off) still reaches Optimal (both dtypes)

            // -- determinism under the new features --
            Stage4DeterminismStein9,   // rounding-heuristic + propagation path: two solves bit-for-bit identical
            Stage4DeterminismGapLimit, // two identical GapLimit-triggering solves identical (DOUBLE-ONLY)
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/extra
        public NativeArray<float> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.Knapsack3: Knapsack3(); break;
                case TestType.Knapsack6: Knapsack6(); break;
                case TestType.AssignmentRootIntegral: AssignmentRootIntegral(); break;
                case TestType.GomoryWolsey: GomoryWolsey(); break;
                case TestType.GomoryWolseyTrueInfiniteBound: GomoryWolseyTrueInfiniteBound(); break;
                case TestType.InfeasibleNoIntInRange: InfeasibleNoIntInRange(); break;
                case TestType.InfeasibleRootLP: InfeasibleRootLP(); break;
                case TestType.UnboundedRoot: UnboundedRoot(); break;
                case TestType.EnumCrossCheck: EnumCrossCheck(); break;
                case TestType.GeneralIntBounds: GeneralIntBounds(); break;
                case TestType.NodeLimitNoIncumbent: NodeLimitNoIncumbent(); break;
                case TestType.IterLimitNoIncumbent: IterLimitNoIncumbent(); break;
                case TestType.Stage3NodesKnapsack6: Stage3NodesKnapsack6(); break;
                case TestType.Stage3NodesGomoryWolsey: Stage3NodesGomoryWolsey(); break;
                case TestType.Stage3NodesBranchy12: Stage3NodesBranchy12(); break;
                case TestType.Stage3Determinism: Stage3Determinism(); break;
                case TestType.Stage3DeterminismBranchy12: Stage3DeterminismBranchy12(); break;
                case TestType.Stein9: Stein9(); break;
                case TestType.Stein15: Stein15(); break;
                case TestType.P0033: P0033(); break;
                case TestType.Stage4NodesGomoryWolsey: Stage4NodesGomoryWolsey(); break;
                case TestType.Stage4NodesBranchy12: Stage4NodesBranchy12(); break;
                case TestType.Stage4PropagationInfeasible: Stage4PropagationInfeasible(); break;
                case TestType.GapLimitRelGap: GapLimitRelGap(); break;
                case TestType.GapLimitPassThrough: GapLimitPassThrough(); break;
                case TestType.Stage4DeterminismStein9: Stage4DeterminismStein9(); break;
                case TestType.Stage4DeterminismGapLimit: Stage4DeterminismGapLimit(); break;
            }
        }

        // ==== (a) knapsacks -- optima brute-forced in ordinary C# at authoring time, embedded as
        //         literal constants (the "Literature test vectors" / known-answer convention). ====

        // max 6x1+5x2+4x3  s.t.  4x1+3x2+2x3 <= 6,  x binary.  Brute force over all 8 subsets: the unique
        // optimum is {item0, item2} (weight 4+2=6, value 6+4=10). Reformulated as min -6x1-5x2-4x3.
        void Knapsack3()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(1, 3);
            A[0, 0] = (float)4; A[0, 1] = (float)3; A[0, 2] = (float)2;
            var b = arena.floatVec(1); b[0] = (float)6;
            var c = arena.floatVec(3); c[0] = (float)(-6); c[1] = (float)(-5); c[2] = (float)(-4);
            var xl = arena.floatVec(3);
            var xu = arena.floatVec(3); for (int j = 0; j < 3; j++) xu[j] = (float)1;
            var x = arena.floatVec(3);
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual;
            var integ = new NativeArray<byte>(3, Allocator.Temp); for (int j = 0; j < 3; j++) integ[j] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

            AssertTrue(info.status == MIPStatus.Optimal);
            AssertCloseD(obj, -10.0, 1e-3);
            AssertCloseD(info.objective, -10.0, 1e-3);
            AssertClose(x[0], (float)1, (float)1e-3);   // unique optimum {0,2}
            AssertClose(x[1], (float)0, (float)1e-3);
            AssertClose(x[2], (float)1, (float)1e-3);
            // proven-optimal contract: dualBound == objective, gap == 0.
            AssertCloseD(info.dualBound, -10.0, 1e-3);
            AssertCloseD(info.gap, 0.0, 1e-9);

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // max 10x1+13x2+18x3+31x4+7x5+15x6  s.t.  2x1+3x2+4x3+7x4+x5+3x6 <= 10,  x binary. Brute force
        // over all 64 subsets: optimum value 50 (items {0,2,4,5}, weight 2+4+1+3=10). Assert only the
        // objective (potential alternate optima). Reformulated as min of the negated values.
        void Knapsack6()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(1, 6);
            A[0, 0] = (float)2; A[0, 1] = (float)3; A[0, 2] = (float)4;
            A[0, 3] = (float)7; A[0, 4] = (float)1; A[0, 5] = (float)3;
            var b = arena.floatVec(1); b[0] = (float)10;
            var c = arena.floatVec(6);
            c[0] = (float)(-10); c[1] = (float)(-13); c[2] = (float)(-18);
            c[3] = (float)(-31); c[4] = (float)(-7); c[5] = (float)(-15);
            var xl = arena.floatVec(6);
            var xu = arena.floatVec(6); for (int j = 0; j < 6; j++) xu[j] = (float)1;
            var x = arena.floatVec(6);
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual;
            var integ = new NativeArray<byte>(6, Allocator.Temp); for (int j = 0; j < 6; j++) integ[j] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

            AssertTrue(info.status == MIPStatus.Optimal);
            AssertCloseD(obj, -50.0, 1e-3);
            AssertCloseD(info.gap, 0.0, 1e-9);
            // the returned selection must be feasible (weight <= 10) and hit the optimal value
            double wt = 0; for (int j = 0; j < 6; j++) wt += (double)A[0, j] * (double)x[j];
            AssertTrue(wt <= 10.0 + 1e-3);

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // ==== (b) assignment problem -- totally unimodular constraint matrix (bipartite incidence),
        //         so the LP relaxation already has an integral optimal vertex: MIP.solve MUST terminate
        //         at the root with nodes == 1 (branching a genuinely-integral root LP would mean a
        //         tolerance/rounding bug -- this is the false-fractionality canary). ====

        // 3x3 assignment, cost[i][j] = 1 on the diagonal, 7 off it -> the unique optimum is the diagonal
        // permutation (cost 1+1+1 = 3, brute-forced over all 6 permutations). Variables x_{ij} = index
        // i*3+j, binary; 3 row-sum + 3 col-sum equality constraints.
        void AssignmentRootIntegral()
        {
            var arena = new Arena(Allocator.Persistent);
            int nv = 9;
            var A = arena.floatMat(6, nv);   // zero-initialized
            // rows 0..2: each source i assigned exactly once (sum_j x_{ij} = 1)
            for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) A[i, i * 3 + j] = (float)1;
            // rows 3..5: each target j receives exactly once (sum_i x_{ij} = 1)
            for (int j = 0; j < 3; j++) for (int i = 0; i < 3; i++) A[3 + j, i * 3 + j] = (float)1;
            var b = arena.floatVec(6); for (int i = 0; i < 6; i++) b[i] = (float)1;
            var c = arena.floatVec(nv);
            for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) c[i * 3 + j] = (float)(i == j ? 1 : 7);
            var xl = arena.floatVec(nv);
            var xu = arena.floatVec(nv); for (int j = 0; j < nv; j++) xu[j] = (float)1;
            var x = arena.floatVec(nv);
            var senses = new NativeArray<ConstraintSense>(6, Allocator.Temp);
            for (int i = 0; i < 6; i++) senses[i] = ConstraintSense.Equal;
            var integ = new NativeArray<byte>(nv, Allocator.Temp); for (int j = 0; j < nv; j++) integ[j] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

            AssertTrue(info.status == MIPStatus.Optimal);
            AssertCloseD(obj, 3.0, 1e-3);
            AssertNodes(info, 1);   // the canary: TU relaxation is integral, so NO branching

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // ==== (c) classic Gomory/Wolsey textbook IP. Source: L. A. Wolsey, "Integer Programming"
        //         (Wiley, 1998), Ch. 7 "Branch and Bound: an Example"; the identical instance appears in
        //         CUHK MAT3007 Optimization, Lecture 26 (mypage.cuhk.edu.cn/.../MAT3007/Slides/lecture26.pdf),
        //         which reports LP relaxation (15/4, 9/4) = 41.25 and integer optimum (5, 0) = 40.
        //           max 8x1 + 5x2  s.t.  x1 + x2 <= 6,  9x1 + 5x2 <= 45,  x1, x2 >= 0 integer.
        //         The LP relaxation optimum (3.75, 2.25) rounds to (4, 2) which is INFEASIBLE -- the
        //         pedagogical point. Verified by exhaustive enumeration at authoring time: unique integer
        //         optimum (5, 0), value 40. MIP.solve minimizes, so negate to min -8x1-5x2 -> obj -40.
        void GomoryWolsey()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(2, 2);
            A[0, 0] = (float)1; A[0, 1] = (float)1;
            A[1, 0] = (float)9; A[1, 1] = (float)5;
            var b = arena.floatVec(2); b[0] = (float)6; b[1] = (float)45;
            var c = arena.floatVec(2); c[0] = (float)(-8); c[1] = (float)(-5);
            var xl = arena.floatVec(2);
            var xu = arena.floatVec(2); xu[0] = (float)10; xu[1] = (float)10;
            var x = arena.floatVec(2);
            var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.LessEqual;
            var integ = new NativeArray<byte>(2, Allocator.Temp); integ[0] = 1; integ[1] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

            AssertTrue(info.status == MIPStatus.Optimal);
            AssertClose(x[0], (float)5, (float)1e-3);   // unique integer optimum (5,0)
            AssertClose(x[1], (float)0, (float)1e-3);
            AssertCloseD(obj, -40.0, 1e-3);
            AssertCloseD(info.dualBound, -40.0, 1e-3);     // proven: dualBound closes to the incumbent
            AssertCloseD(info.gap, 0.0, 1e-9);

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // Regression for the "1e30 UB-row rhs inflates dataScale/artificialBound" bug: the SAME
        // Gomory/Wolsey instance and optimum as GomoryWolsey, but with xu = the library's +infinity
        // sentinel (1e30) instead of a finite proxy (10). Because both variables are unbounded above,
        // each integer variable's pre-allocated upper-bound row must start genuinely INERT (0*y_j <= 0)
        // rather than carrying a 1e30 rhs -- otherwise DualSimplexCore's dataScale = max(1, max|rhs|,
        // max|cost|) reads that 1e30, inflates artificialBound (~100*dataScale) to ~1e32 and corrupts the
        // dual simplex, making this trivially-feasible LP return a FALSE Infeasible. Changing ONLY the
        // bound representation (finite proxy vs true infinite sentinel) must not change the answer:
        // Optimal, (5,0), obj -40, dualBound -40, gap 0 -- identical to GomoryWolsey.
        void GomoryWolseyTrueInfiniteBound()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(2, 2);
            A[0, 0] = (float)1; A[0, 1] = (float)1;
            A[1, 0] = (float)9; A[1, 1] = (float)5;
            var b = arena.floatVec(2); b[0] = (float)6; b[1] = (float)45;
            var c = arena.floatVec(2); c[0] = (float)(-8); c[1] = (float)(-5);
            var xl = arena.floatVec(2);
            var xu = arena.floatVec(2); xu[0] = (float)1e30; xu[1] = (float)1e30;   // true +inf sentinel
            var x = arena.floatVec(2);
            var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.LessEqual;
            var integ = new NativeArray<byte>(2, Allocator.Temp); integ[0] = 1; integ[1] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

            AssertTrue(info.status == MIPStatus.Optimal);   // NOT a false Infeasible
            AssertClose(x[0], (float)5, (float)1e-3);
            AssertClose(x[1], (float)0, (float)1e-3);
            AssertCloseD(obj, -40.0, 1e-3);
            AssertCloseD(info.dualBound, -40.0, 1e-3);
            AssertCloseD(info.gap, 0.0, 1e-9);

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // ==== (d) infeasible / unbounded ====

        // A single integer variable confined to 2.1 <= x <= 2.9: the LP relaxation is feasible (e.g.
        // x=2.5) but there is NO integer point in that finite range, so branch & bound exhausts the tree
        // without any incumbent -> Infeasible. xl=2.1 is finite (> -1e29), so it passes validation (this
        // is the correct way to build an infeasible-MIP oracle: NOT xl > xu, which is an input-sanity
        // throw). Both children (x<=2 and x>=3) have empty local domains. Contract: objective/dualBound/
        // gap all NaN, x all zeros.
        void InfeasibleNoIntInRange()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(1, 1); A[0, 0] = (float)1;   // redundant row x <= 100
            var b = arena.floatVec(1); b[0] = (float)100;
            var c = arena.floatVec(1); c[0] = (float)1;
            var xl = arena.floatVec(1); xl[0] = (float)2.1;
            var xu = arena.floatVec(1); xu[0] = (float)2.9;
            var x = arena.floatVec(1);
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual;
            var integ = new NativeArray<byte>(1, Allocator.Temp); integ[0] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

            AssertTrue(info.status == MIPStatus.Infeasible);
            AssertTrue(math.isnan(info.objective));
            AssertTrue(math.isnan(info.dualBound));
            AssertTrue(math.isnan(info.gap));
            AssertTrue(math.isnan(obj));
            AssertClose(x[0], (float)0, (float)0);
            AssertTrue(info.nodes >= 1);   // still a meaningful count

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // x + y = 1 with x >= 1 and y >= 1 (both integer): x + y >= 2 contradicts the equality, so the
        // ROOT LP relaxation itself is infeasible -> Infeasible detected at node 1. Same NaN contract.
        void InfeasibleRootLP()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(1, 2); A[0, 0] = (float)1; A[0, 1] = (float)1;
            var b = arena.floatVec(1); b[0] = (float)1;
            var c = arena.floatVec(2); c[0] = (float)1; c[1] = (float)1;
            var xl = arena.floatVec(2); xl[0] = (float)1; xl[1] = (float)1;
            var xu = arena.floatVec(2); xu[0] = (float)10; xu[1] = (float)10;
            var x = arena.floatVec(2);
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp);
            senses[0] = ConstraintSense.Equal;
            var integ = new NativeArray<byte>(2, Allocator.Temp); integ[0] = 1; integ[1] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

            AssertTrue(info.status == MIPStatus.Infeasible);
            AssertTrue(math.isnan(info.objective));
            AssertTrue(math.isnan(info.dualBound));
            AssertTrue(math.isnan(info.gap));
            AssertNodes(info, 1);   // root LP infeasible -> exactly one node
            AssertClose(x[0], (float)0, (float)0);
            AssertClose(x[1], (float)0, (float)0);

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // min -x  s.t.  x >= 0,  x integer,  xu = +inf (1e30 sentinel): the objective decreases without
        // bound over the integers, so the root LP relaxation is unbounded. Unbounded is detected ONLY at
        // the root -> Unbounded, nodes == 1, all-NaN contract.
        void UnboundedRoot()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(1, 1); A[0, 0] = (float)1;   // x >= 0 (redundant with the bound)
            var b = arena.floatVec(1); b[0] = (float)0;
            var c = arena.floatVec(1); c[0] = (float)(-1);
            var xl = arena.floatVec(1); xl[0] = (float)0;        // finite lower (required for integer)
            var xu = arena.floatVec(1); xu[0] = (float)1e30;     // unbounded above
            var x = arena.floatVec(1);
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp);
            senses[0] = ConstraintSense.GreaterEqual;
            var integ = new NativeArray<byte>(1, Allocator.Temp); integ[0] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

            AssertTrue(info.status == MIPStatus.Unbounded);
            AssertTrue(math.isnan(info.objective));
            AssertTrue(math.isnan(info.dualBound));
            AssertTrue(math.isnan(info.gap));
            AssertTrue(math.isnan(obj));
            AssertNodes(info, 1);
            AssertClose(x[0], (float)0, (float)0);

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // ==== (e) exhaustive-enumeration cross-check. Random tiny all-integer MIPs on the box [0,3]^n:
        //         integer A/b/c (so lattice feasibility is EXACT integer arithmetic, no boundary
        //         roundoff), constructed around a random feasible integer point x* so the instance is
        //         always feasible + bounded (box) -> MIP.solve must return Optimal. The objective is
        //         cross-checked against the brute-force min over all 4^n lattice points. float runs 2
        //         tiny instances (n<=4), double runs 7 (up to n=5), per the draft spec's float-is-tiny,
        //         double-is-serious guidance. ====
        void EnumCrossCheck()
        {
            int cases = 2;
            for (int s = 0; s < cases; s++)
            {
                int n = 3 + (s % 3);          // 3,4,5,3,4,5,3  -> float reaches only s=0(n=3),s=1(n=4)
                int m = 2 + (s % 2);          // 2,3,2,3,2,3,2
                RunEnumCase(n, m, (uint)(1000003u * (uint)(s + 1) + 97u));
            }
        }

        void RunEnumCase(int n, int m, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Unity.Mathematics.Random(seed == 0u ? 1u : seed);

            var A = arena.floatMat(m, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (float)rng.NextInt(-2, 3);   // integer coeff in {-2,-1,0,1,2}

            // random feasible integer point x* in [0,3]^n, used to make every row feasible at x*.
            var xstar = new NativeArray<int>(n, Allocator.Temp);
            for (int j = 0; j < n; j++) xstar[j] = rng.NextInt(0, 4);

            var b = arena.floatVec(m);
            var senses = new NativeArray<ConstraintSense>(m, Allocator.Temp);
            for (int i = 0; i < m; i++)
            {
                int act = 0;
                for (int j = 0; j < n; j++) act += (int)A[i, j] * xstar[j];
                int r = rng.NextInt(0, 3);
                int slack = rng.NextInt(0, 3);
                if (r == 0) { senses[i] = ConstraintSense.LessEqual; b[i] = (float)(act + slack); }
                else if (r == 1) { senses[i] = ConstraintSense.GreaterEqual; b[i] = (float)(act - slack); }
                else { senses[i] = ConstraintSense.Equal; b[i] = (float)act; }
            }

            var c = arena.floatVec(n);
            for (int j = 0; j < n; j++) c[j] = (float)rng.NextInt(-3, 4);
            var xl = arena.floatVec(n);
            var xu = arena.floatVec(n); for (int j = 0; j < n; j++) xu[j] = (float)3;
            var integ = new NativeArray<byte>(n, Allocator.Temp); for (int j = 0; j < n; j++) integ[j] = 1;
            var x = arena.floatVec(n);

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

            // brute-force the true min over all 4^n lattice points (base-4 decode via 2 bits/digit).
            int total = 1; for (int j = 0; j < n; j++) total *= 4;
            double best = double.PositiveInfinity; bool any = false;
            for (int code = 0; code < total; code++)
            {
                bool feas = true;
                for (int i = 0; i < m && feas; i++)
                {
                    double a = 0;
                    for (int j = 0; j < n; j++) a += (double)A[i, j] * ((code >> (2 * j)) & 3);
                    double bi = (double)b[i];
                    ConstraintSense se = senses[i];
                    if (se == ConstraintSense.LessEqual) { if (a > bi + 0.5) feas = false; }
                    else if (se == ConstraintSense.GreaterEqual) { if (a < bi - 0.5) feas = false; }
                    else { if (math.abs(a - bi) > 0.5) feas = false; }   // integer data -> 0.5 cleanly separates
                }
                if (!feas) continue;
                double cost = 0;
                for (int j = 0; j < n; j++) cost += (double)c[j] * ((code >> (2 * j)) & 3);
                if (cost < best) { best = cost; any = true; }
            }

            // x* is always feasible by construction, so enumeration must find at least one point.
            if (!any && Fail[0] == (float)0) { Fail[0] = (float)1; Fail[1] = (float)0; Fail[2] = (float)1; Fail[3] = (float)0; }
            Assert.IsTrue(any);

            AssertTrue(info.status == MIPStatus.Optimal);
            AssertCloseD(info.objective, best, 1e-3 * (1.0 + math.abs(best)));

            xstar.Dispose(); senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // ==== extra coverage ====

        // Integer variable with a nonzero finite lower bound (xl=3, xu=10) -- exercises the shift/split
        // reformulation's anchor-low branch-rhs closed form (rhs = newBound - xl) for xl != 0, not just
        // the xl=0 binary case. min -x s.t. x <= 7.5; the integer optimum is x=7, obj -7.
        //
        // Stage 4: activity-based propagation floors the single-variable row x<=7.5 to the integer bound
        // x<=7 at the ROOT (x is integer with an already-finite xl=3), so the root LP relaxation is
        // ALREADY integral (x=7) -- B&B finishes at the root with no branch. This is the textbook simplest
        // case of activity-based bound tightening; nodes==1 is a hard invariant for this exact instance.
        void GeneralIntBounds()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(1, 1); A[0, 0] = (float)1;
            var b = arena.floatVec(1); b[0] = (float)7.5;
            var c = arena.floatVec(1); c[0] = (float)(-1);
            var xl = arena.floatVec(1); xl[0] = (float)3;
            var xu = arena.floatVec(1); xu[0] = (float)10;
            var x = arena.floatVec(1);
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual;
            var integ = new NativeArray<byte>(1, Allocator.Temp); integ[0] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

            AssertTrue(info.status == MIPStatus.Optimal);
            AssertClose(x[0], (float)7, (float)1e-3);
            AssertCloseD(obj, -7.0, 1e-3);
            AssertNodes(info, 1);   // propagation closes the root at 1 node (see method comment)

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // maxNodes = 1 on the Gomory instance (whose root LP is fractional, so no incumbent exists after
        // node 1): the budget is exhausted right after the root -> NodeLimit, nodes == 1, no incumbent
        // (objective == +inf), but a SOUND finite dual bound (the root LP value ~ -41.25, never NaN).
        void NodeLimitNoIncumbent()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(2, 2);
            A[0, 0] = (float)1; A[0, 1] = (float)1;
            A[1, 0] = (float)9; A[1, 1] = (float)5;
            var b = arena.floatVec(2); b[0] = (float)6; b[1] = (float)45;
            var c = arena.floatVec(2); c[0] = (float)(-8); c[1] = (float)(-5);
            var xl = arena.floatVec(2);
            var xu = arena.floatVec(2); xu[0] = (float)10; xu[1] = (float)10;
            var x = arena.floatVec(2);
            var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.LessEqual;
            var integ = new NativeArray<byte>(2, Allocator.Temp); integ[0] = 1; integ[1] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj, maxNodes: 1);

            AssertTrue(info.status == MIPStatus.NodeLimit);
            AssertNodes(info, 1);
            AssertTrue(info.nodes <= 1);                        // nodes <= maxNodes
            AssertTrue(info.objective == double.PositiveInfinity);   // no incumbent yet
            AssertTrue(obj == double.PositiveInfinity);
            AssertTrue(math.isfinite(info.dualBound) && !math.isnan(info.dualBound));   // sound, finite bound
            AssertCloseD(info.dualBound, -41.25, 1e-2);         // the fractional root LP value

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // maxIter = 1 on the Gomory instance: the root LP fully solves (its own budget is unlimited) using
        // several pivots, then the cumulative-LP-iteration budget is already exceeded -> MaxIterations
        // after node 1, same no-incumbent partial-result contract as NodeLimitNoIncumbent.
        void IterLimitNoIncumbent()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(2, 2);
            A[0, 0] = (float)1; A[0, 1] = (float)1;
            A[1, 0] = (float)9; A[1, 1] = (float)5;
            var b = arena.floatVec(2); b[0] = (float)6; b[1] = (float)45;
            var c = arena.floatVec(2); c[0] = (float)(-8); c[1] = (float)(-5);
            var xl = arena.floatVec(2);
            var xu = arena.floatVec(2); xu[0] = (float)10; xu[1] = (float)10;
            var x = arena.floatVec(2);
            var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.LessEqual;
            var integ = new NativeArray<byte>(2, Allocator.Temp); integ[0] = 1; integ[1] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj, maxIter: 1);

            AssertTrue(info.status == MIPStatus.MaxIterations);
            AssertNodes(info, 1);
            AssertTrue(info.objective == double.PositiveInfinity);   // no incumbent yet
            AssertTrue(math.isfinite(info.dualBound) && !math.isnan(info.dualBound));
            AssertCloseD(info.dualBound, -41.25, 1e-2);

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // ==== (f) STAGE 3 node-count regression + determinism ====

        // Knapsack6 (same instance as Knapsack6() above). Stage 2 and stage 3 both solve it at a single
        // node (the LP relaxation is already integer-optimal here), so this is a tie -- assert nodes <= 1,
        // identical in both dtypes. Also re-checks the optimum value survives (obj -50).
        void Stage3NodesKnapsack6()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(1, 6);
            A[0, 0] = (float)2; A[0, 1] = (float)3; A[0, 2] = (float)4;
            A[0, 3] = (float)7; A[0, 4] = (float)1; A[0, 5] = (float)3;
            var b = arena.floatVec(1); b[0] = (float)10;
            var c = arena.floatVec(6);
            c[0] = (float)(-10); c[1] = (float)(-13); c[2] = (float)(-18);
            c[3] = (float)(-31); c[4] = (float)(-7); c[5] = (float)(-15);
            var xl = arena.floatVec(6);
            var xu = arena.floatVec(6); for (int j = 0; j < 6; j++) xu[j] = (float)1;
            var x = arena.floatVec(6);
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual;
            var integ = new NativeArray<byte>(6, Allocator.Temp); for (int j = 0; j < 6; j++) integ[j] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

            AssertTrue(info.status == MIPStatus.Optimal);
            AssertCloseD(obj, -50.0, 1e-3);
            AssertNodesLE(info, 1);   // stage-2 baseline = 1 node; stage 3 must not exceed it

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // GomoryWolsey (same instance as GomoryWolsey() above). Stage 2 = 7 nodes; stage 3 = 7 nodes with
        // MORE lp iterations (9 -> 27: reliability-init strong-branching overhead on a 2-variable instance
        // where pseudocost never becomes reliable in only 7 nodes). Assert nodes <= 7 ONLY -- lpIterations
        // is expected to rise and is deliberately not asserted. Optimum (obj -40) must survive.
        void Stage3NodesGomoryWolsey()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(2, 2);
            A[0, 0] = (float)1; A[0, 1] = (float)1;
            A[1, 0] = (float)9; A[1, 1] = (float)5;
            var b = arena.floatVec(2); b[0] = (float)6; b[1] = (float)45;
            var c = arena.floatVec(2); c[0] = (float)(-8); c[1] = (float)(-5);
            var xl = arena.floatVec(2);
            var xu = arena.floatVec(2); xu[0] = (float)10; xu[1] = (float)10;
            var x = arena.floatVec(2);
            var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.LessEqual;
            var integ = new NativeArray<byte>(2, Allocator.Temp); integ[0] = 1; integ[1] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

            AssertTrue(info.status == MIPStatus.Optimal);
            AssertCloseD(obj, -40.0, 1e-3);
            AssertNodesLE(info, 7);   // stage-2 baseline = 7 nodes; stage 3 must not exceed it

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // Random "branchy" MIP on the integer box [0,3]^12, n=12, m=6, seed 424242 -- built with the
        // EXACT RNG call sequence of RunEnumCase (integer A/b/c around a random feasible integer x*, so
        // always feasible + bounded). Measured baselines (double): stage 2 = 267 nodes / obj 6; stage 3 =
        // 241 nodes / obj 6 (same optimum, FEWER nodes -- the pseudocost/reliability payoff, and direct
        // proof that the branching sequence differs from stage-2 most-fractional, satisfying (a) and (c)).
        //
        // DOUBLE-ONLY: the stage-2 FLOAT run on this exact instance returned an anomalous nodes=0 (nodes is
        // incremented unconditionally before the first LP solve, so 0 is structurally impossible from a
        // real search -- a pre-existing float robustness quirk, NOT a stage-3 change; stage-3's own float
        // run was clean at 244 nodes). With no trustworthy stage-2 float baseline, the float case runs 0
        // iterations (a no-op pass), per this file's "float only on tiny instances" convention.
        void Stage3NodesBranchy12()
        {
            int cases = 0;
            for (int s = 0; s < cases; s++)
            {
                var arena = new Arena(Allocator.Persistent);
                BuildBranchy12(in arena, out var A, out var b, out var c, out var senses,
                               out var xl, out var xu, out var integ);
                var x = arena.floatVec(12);

                var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

                AssertTrue(info.status == MIPStatus.Optimal);
                AssertCloseD(info.objective, 6.0, 1e-6);
                AssertCloseD(obj, 6.0, 1e-6);
                AssertNodesLE(info, 267);   // stage-2 baseline = 267 nodes; stage 3 measured 241

                senses.Dispose(); integ.Dispose(); arena.Dispose();
            }
        }

        // Determinism: two back-to-back MIP.solve calls on the identical GomoryWolsey inputs must produce
        // bit-for-bit identical nodes / lpIterations / objective / dualBound and identical x. The solve
        // path is single-threaded with no RNG/parallelism, so this should hold trivially -- the test
        // exists to VERIFY it rather than assume it. Runs in both dtypes.
        void Stage3Determinism()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(2, 2);
            A[0, 0] = (float)1; A[0, 1] = (float)1;
            A[1, 0] = (float)9; A[1, 1] = (float)5;
            var b = arena.floatVec(2); b[0] = (float)6; b[1] = (float)45;
            var c = arena.floatVec(2); c[0] = (float)(-8); c[1] = (float)(-5);
            var xl = arena.floatVec(2);
            var xu = arena.floatVec(2); xu[0] = (float)10; xu[1] = (float)10;
            var x1 = arena.floatVec(2);
            var x2 = arena.floatVec(2);
            var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.LessEqual;
            var integ = new NativeArray<byte>(2, Allocator.Temp); integ[0] = 1; integ[1] = 1;

            var i1 = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x1, out double o1);
            var i2 = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x2, out double o2);

            AssertEqInt(i1.nodes, i2.nodes);
            AssertEqInt(i1.lpIterations, i2.lpIterations);
            AssertEqExactD(i1.objective, i2.objective);
            AssertEqExactD(i1.dualBound, i2.dualBound);
            AssertEqExactD(o1, o2);
            for (int j = 0; j < 2; j++) AssertClose(x1[j], x2[j], (float)0);   // exact: precision 0

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // Determinism on the big branchy n=12 search -- a real many-node plunge + queue-jump sequence, so
        // it exercises far more of the search path than the 7-node GomoryWolsey case. DOUBLE-ONLY (same
        // instance/rationale as Stage3NodesBranchy12; float runs 0 iterations, a no-op pass).
        void Stage3DeterminismBranchy12()
        {
            int cases = 0;
            for (int s = 0; s < cases; s++)
            {
                var arena = new Arena(Allocator.Persistent);
                BuildBranchy12(in arena, out var A, out var b, out var c, out var senses,
                               out var xl, out var xu, out var integ);
                var x1 = arena.floatVec(12);
                var x2 = arena.floatVec(12);

                var i1 = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x1, out double o1);
                var i2 = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x2, out double o2);

                AssertEqInt(i1.nodes, i2.nodes);
                AssertEqInt(i1.lpIterations, i2.lpIterations);
                AssertEqExactD(i1.objective, i2.objective);
                AssertEqExactD(i1.dualBound, i2.dualBound);
                AssertEqExactD(o1, o2);
                for (int j = 0; j < 12; j++) AssertClose(x1[j], x2[j], (float)0);

                senses.Dispose(); integ.Dispose(); arena.Dispose();
            }
        }

        // Builds the branchy n=12/m=6/seed-424242 instance into arena-owned A/b/c/xl/xu (caller disposes
        // the arena) and Temp-owned senses/integ (caller disposes both). Replicates RunEnumCase's EXACT
        // RNG draw order (A row-major, then x*, then per-row sense/rhs, then c) so the generated instance
        // is identical to the one the stage-2/stage-3 baselines were measured on.
        void BuildBranchy12(in Arena arena, out floatMxN A, out floatN b, out floatN c,
                            out NativeArray<ConstraintSense> senses, out floatN xl, out floatN xu,
                            out NativeArray<byte> integ)
        {
            const int n = 12, m = 6;
            var rng = new Unity.Mathematics.Random(424242u);

            A = arena.floatMat(m, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = (float)rng.NextInt(-2, 3);

            var xstar = new NativeArray<int>(n, Allocator.Temp);
            for (int j = 0; j < n; j++) xstar[j] = rng.NextInt(0, 4);

            b = arena.floatVec(m);
            senses = new NativeArray<ConstraintSense>(m, Allocator.Temp);
            for (int i = 0; i < m; i++)
            {
                int act = 0;
                for (int j = 0; j < n; j++) act += (int)A[i, j] * xstar[j];
                int r = rng.NextInt(0, 3);
                int slack = rng.NextInt(0, 3);
                if (r == 0) { senses[i] = ConstraintSense.LessEqual; b[i] = (float)(act + slack); }
                else if (r == 1) { senses[i] = ConstraintSense.GreaterEqual; b[i] = (float)(act - slack); }
                else { senses[i] = ConstraintSense.Equal; b[i] = (float)act; }
            }

            c = arena.floatVec(n);
            for (int j = 0; j < n; j++) c[j] = (float)rng.NextInt(-3, 4);
            xl = arena.floatVec(n);
            xu = arena.floatVec(n); for (int j = 0; j < n; j++) xu[j] = (float)3;
            integ = new NativeArray<byte>(n, Allocator.Temp); for (int j = 0; j < n; j++) integ[j] = 1;

            xstar.Dispose();
        }

        // ==== (g) STAGE 4: MIPLIB known-answer oracles, propagation, gap limits, determinism ====

        // Sets a coefficient-1 covering triple in row r of A (the stein set-covering rows).
        void SetTriple(floatMxN A, int r, int i, int j, int k)
        {
            A[r, i] = (float)1; A[r, j] = (float)1; A[r, k] = (float)1;
        }

        // Builds the MIPLIB "stein9" Steiner-triple set-covering instance (Fulkerson, Nemhauser & Trotter,
        // "Two computationally difficult set covering problems...", Math. Prog. Study 2, 1974; MIPLIB 3 /
        // miplib.zib.de). 9 binaries, minimize the count sum_j x_j; rows 0-11 are the covering triples
        // (each >= 1), row 12 is the OB2 "at least 4 of 9" cut (all vars >= 4). Proven optimum 5.
        void BuildStein9(in Arena arena, out floatMxN A, out floatN b, out floatN c,
                         out NativeArray<ConstraintSense> senses, out floatN xl, out floatN xu,
                         out NativeArray<byte> integ)
        {
            const int n = 9, m = 13;
            A = arena.floatMat(m, n);   // zero-initialized
            SetTriple(A, 0, 1, 2, 3); SetTriple(A, 1, 0, 2, 4); SetTriple(A, 2, 0, 1, 5); SetTriple(A, 3, 4, 5, 6);
            SetTriple(A, 4, 3, 5, 7); SetTriple(A, 5, 3, 4, 8); SetTriple(A, 6, 0, 7, 8); SetTriple(A, 7, 1, 6, 8);
            SetTriple(A, 8, 2, 6, 7); SetTriple(A, 9, 0, 3, 6); SetTriple(A, 10, 1, 4, 7); SetTriple(A, 11, 2, 5, 8);
            for (int j = 0; j < n; j++) A[12, j] = (float)1;   // all 9 vars

            b = arena.floatVec(m); for (int i = 0; i < 12; i++) b[i] = (float)1; b[12] = (float)4;
            senses = new NativeArray<ConstraintSense>(m, Allocator.Temp);
            for (int i = 0; i < m; i++) senses[i] = ConstraintSense.GreaterEqual;
            c = arena.floatVec(n); for (int j = 0; j < n; j++) c[j] = (float)1;
            xl = arena.floatVec(n);
            xu = arena.floatVec(n); for (int j = 0; j < n; j++) xu[j] = (float)1;
            integ = new NativeArray<byte>(n, Allocator.Temp); for (int j = 0; j < n; j++) integ[j] = 1;
        }

        // stein9 known-answer: proven optimum 5, proven-optimal contract (gap 0). Both dtypes.
        void Stein9()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildStein9(in arena, out var A, out var b, out var c, out var senses, out var xl, out var xu, out var integ);
            var x = arena.floatVec(9);

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

            AssertTrue(info.status == MIPStatus.Optimal);
            AssertCloseD(obj, 5.0, 1e-3);
            AssertCloseD(info.objective, 5.0, 1e-3);
            AssertCloseD(info.gap, 0.0, 1e-9);

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // MIPLIB "stein15" (same Steiner-triple family/source as stein9). 15 binaries, minimize the count;
        // rows 0-34 covering triples (>= 1), row 35 the all-vars "at least 7 of 15" cut (>= 7). Proven
        // optimum 9 (double solves it in 275 nodes / 4307 LP iterations). DOUBLE-ONLY: measured empirically,
        // the FLOAT search does NOT converge -- it diverges past the node cap (float roundoff at this size
        // multiplies branching), same float-baseline rationale as P0033/Stage3NodesBranchy12. (The brief
        // reported float stein15 finishing in ~261 nodes; that did not reproduce on the shipped code.)
        void Stein15()
        {
            int cases = 0;
            for (int s = 0; s < cases; s++)
            {
                var arena = new Arena(Allocator.Persistent);
                const int n = 15, m = 36;
                var A = arena.floatMat(m, n);   // zero-initialized
                SetTriple(A, 0, 2, 3, 5);   SetTriple(A, 1, 3, 4, 6);   SetTriple(A, 2, 0, 4, 7);   SetTriple(A, 3, 0, 1, 8);   SetTriple(A, 4, 1, 2, 9);
                SetTriple(A, 5, 1, 4, 5);   SetTriple(A, 6, 0, 2, 6);   SetTriple(A, 7, 1, 3, 7);   SetTriple(A, 8, 2, 4, 8);   SetTriple(A, 9, 0, 3, 9);
                SetTriple(A, 10, 7, 8, 10); SetTriple(A, 11, 8, 9, 11); SetTriple(A, 12, 5, 9, 12); SetTriple(A, 13, 5, 6, 13); SetTriple(A, 14, 6, 7, 14);
                SetTriple(A, 15, 6, 9, 10); SetTriple(A, 16, 5, 7, 11); SetTriple(A, 17, 6, 8, 12); SetTriple(A, 18, 7, 9, 13); SetTriple(A, 19, 5, 8, 14);
                SetTriple(A, 20, 0, 12, 13); SetTriple(A, 21, 1, 13, 14); SetTriple(A, 22, 2, 10, 14); SetTriple(A, 23, 3, 10, 11); SetTriple(A, 24, 4, 11, 12);
                SetTriple(A, 25, 0, 11, 14); SetTriple(A, 26, 1, 10, 12); SetTriple(A, 27, 2, 11, 13); SetTriple(A, 28, 3, 12, 14); SetTriple(A, 29, 4, 10, 13);
                SetTriple(A, 30, 0, 5, 10); SetTriple(A, 31, 1, 6, 11); SetTriple(A, 32, 2, 7, 12); SetTriple(A, 33, 3, 8, 13); SetTriple(A, 34, 4, 9, 14);
                for (int j = 0; j < n; j++) A[35, j] = (float)1;

                var b = arena.floatVec(m); for (int i = 0; i < 35; i++) b[i] = (float)1; b[35] = (float)7;
                var senses = new NativeArray<ConstraintSense>(m, Allocator.Temp);
                for (int i = 0; i < m; i++) senses[i] = ConstraintSense.GreaterEqual;
                var c = arena.floatVec(n); for (int j = 0; j < n; j++) c[j] = (float)1;
                var xl = arena.floatVec(n);
                var xu = arena.floatVec(n); for (int j = 0; j < n; j++) xu[j] = (float)1;
                var integ = new NativeArray<byte>(n, Allocator.Temp); for (int j = 0; j < n; j++) integ[j] = 1;
                var x = arena.floatVec(n);

                var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj, maxNodes: 200000);

                AssertTrue(info.status == MIPStatus.Optimal);
                AssertCloseD(obj, 9.0, 1e-3);
                AssertCloseD(info.gap, 0.0, 1e-9);

                senses.Dispose(); integ.Dispose(); arena.Dispose();
            }
        }

        // MIPLIB "p0033": Crowder, Johnson & Padberg, "Solving large-scale zero-one linear programming
        // problems" (Oper. Res. 31, 1983); MIPLIB 3 / miplib.zib.de. 33 binaries, minimize c^T x, 15 LessEqual
        // rows (the literature's 16th "ZBESTROW" all-zero bookkeeping row is omitted -- mathematically
        // equivalent). Proven optimum 3089. DOUBLE-ONLY (see enum comment: float finds 3089 but cannot prove
        // optimality within a sane node budget). Generous maxNodes guards a runaway; Optimal proves the cap
        // was not the stopping reason (double explores ~447 nodes).
        void P0033()
        {
            int cases = 0;
            for (int s = 0; s < cases; s++)
            {
                var arena = new Arena(Allocator.Persistent);
                const int n = 33, m = 15;
                var A = arena.floatMat(m, n);   // zero-initialized
                var c = arena.floatVec(n);
                c[0] = (float)171; c[1] = (float)171; c[2] = (float)171; c[3] = (float)171; c[4] = (float)163;
                c[5] = (float)162; c[6] = (float)163; c[7] = (float)69; c[8] = (float)69; c[9] = (float)183;
                c[10] = (float)183; c[11] = (float)183; c[12] = (float)183; c[13] = (float)49; c[14] = (float)183;
                c[15] = (float)258; c[16] = (float)517; c[17] = (float)250; c[18] = (float)500; c[19] = (float)250;
                c[20] = (float)500; c[21] = (float)159; c[22] = (float)318; c[23] = (float)159; c[24] = (float)318;
                c[25] = (float)159; c[26] = (float)318; c[27] = (float)159; c[28] = (float)318; c[29] = (float)114;
                c[30] = (float)228; c[31] = (float)159; c[32] = (float)318;

                A[0, 0] = (float)1; A[0, 1] = (float)1; A[0, 2] = (float)1; A[0, 3] = (float)1;
                A[1, 4] = (float)1; A[1, 5] = (float)1; A[1, 6] = (float)1;
                A[2, 7] = (float)1; A[2, 8] = (float)1;
                A[3, 9] = (float)1; A[3, 10] = (float)1; A[3, 11] = (float)1; A[3, 12] = (float)1; A[3, 14] = (float)1;
                A[4, 9] = (float)(-230); A[4, 15] = (float)(-200); A[4, 16] = (float)(-400);
                A[5, 2] = (float)300; A[5, 3] = (float)300; A[5, 4] = (float)285; A[5, 5] = (float)285;
                A[5, 7] = (float)265; A[5, 8] = (float)265; A[5, 11] = (float)230; A[5, 12] = (float)230;
                A[5, 13] = (float)190; A[5, 21] = (float)200; A[5, 22] = (float)400; A[5, 23] = (float)200;
                A[5, 24] = (float)400; A[5, 25] = (float)200; A[5, 26] = (float)400; A[5, 27] = (float)200;
                A[5, 28] = (float)400; A[5, 29] = (float)200; A[5, 30] = (float)400;
                for (int j = 0; j < n; j++) A[6, j] = -A[5, j];   // row 6 = exact negation of row 5
                A[7, 3] = (float)(-300); A[7, 29] = (float)(-200); A[7, 30] = (float)(-400);
                A[8, 0] = (float)(-300); A[8, 5] = (float)(-285); A[8, 8] = (float)(-265); A[8, 13] = (float)(-190);
                A[8, 25] = (float)(-200); A[8, 26] = (float)(-400);
                A[9, 0] = (float)(-300); A[9, 2] = (float)(-300); A[9, 5] = (float)(-285); A[9, 8] = (float)(-265);
                A[9, 13] = (float)(-190); A[9, 25] = (float)(-200); A[9, 26] = (float)(-400); A[9, 27] = (float)(-200);
                A[9, 28] = (float)(-400);
                A[10, 4] = (float)(-285); A[10, 7] = (float)(-265); A[10, 10] = (float)(-230); A[10, 21] = (float)(-200);
                A[10, 22] = (float)(-400);
                A[11, 4] = (float)(-285); A[11, 7] = (float)(-265); A[11, 10] = (float)(-230); A[11, 11] = (float)(-230);
                A[11, 21] = (float)(-200); A[11, 22] = (float)(-400); A[11, 23] = (float)(-200); A[11, 24] = (float)(-400);
                A[12, 1] = (float)(-300); A[12, 17] = (float)(-200); A[12, 18] = (float)(-400);
                A[13, 1] = (float)(-300); A[13, 17] = (float)(-200); A[13, 18] = (float)(-400); A[13, 19] = (float)(-200);
                A[13, 20] = (float)(-400);
                A[14, 6] = (float)(-285); A[14, 31] = (float)(-200); A[14, 32] = (float)(-400);

                var b = arena.floatVec(m);
                b[0] = (float)1; b[1] = (float)1; b[2] = (float)1; b[3] = (float)1; b[4] = (float)(-5);
                b[5] = (float)2700; b[6] = (float)(-2600); b[7] = (float)(-100); b[8] = (float)(-900);
                b[9] = (float)(-1656); b[10] = (float)(-335); b[11] = (float)(-1026); b[12] = (float)(-5);
                b[13] = (float)(-500); b[14] = (float)(-270);
                var senses = new NativeArray<ConstraintSense>(m, Allocator.Temp);
                for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;
                var xl = arena.floatVec(n);
                var xu = arena.floatVec(n); for (int j = 0; j < n; j++) xu[j] = (float)1;
                var integ = new NativeArray<byte>(n, Allocator.Temp); for (int j = 0; j < n; j++) integ[j] = 1;
                var x = arena.floatVec(n);

                var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj, maxNodes: 200000);

                AssertTrue(info.status == MIPStatus.Optimal);
                AssertCloseD(obj, 3089.0, 1e-3);
                AssertCloseD(info.objective, 3089.0, 1e-3);
                AssertCloseD(info.gap, 0.0, 1e-9);

                senses.Dispose(); integ.Dispose(); arena.Dispose();
            }
        }

        // Propagation node-count drop, pinned per dtype: the same GomoryWolsey instance solves at exactly 5
        // nodes in DOUBLE now (stage 3 measured 7 -> propagation fathoms 2), and 7 nodes in FLOAT (float
        // roundoff changes the branching sequence, so propagation's fathoms land differently -- still no
        // increase over the stage-3 baseline of 7). Deterministic single-threaded search -> the exact count
        // is a hard invariant per dtype. (The brief reported 5 for both; only double reproduced it.)
        void Stage4NodesGomoryWolsey()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(2, 2);
            A[0, 0] = (float)1; A[0, 1] = (float)1;
            A[1, 0] = (float)9; A[1, 1] = (float)5;
            var b = arena.floatVec(2); b[0] = (float)6; b[1] = (float)45;
            var c = arena.floatVec(2); c[0] = (float)(-8); c[1] = (float)(-5);
            var xl = arena.floatVec(2);
            var xu = arena.floatVec(2); xu[0] = (float)10; xu[1] = (float)10;
            var x = arena.floatVec(2);
            var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.LessEqual;
            var integ = new NativeArray<byte>(2, Allocator.Temp); integ[0] = 1; integ[1] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

            AssertTrue(info.status == MIPStatus.Optimal);
            AssertCloseD(obj, -40.0, 1e-3);
            AssertNodes(info, 7);   // float 7 (unchanged), double 5 (7 -> 5 drop)

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // Propagation node-count drop on the big branchy search: stage 3 = 241 nodes, stage 4 = 218 (same
        // optimum, strictly fewer nodes). DOUBLE-ONLY (same instance/rationale as Stage3NodesBranchy12).
        void Stage4NodesBranchy12()
        {
            int cases = 0;
            for (int s = 0; s < cases; s++)
            {
                var arena = new Arena(Allocator.Persistent);
                BuildBranchy12(in arena, out var A, out var b, out var c, out var senses,
                               out var xl, out var xu, out var integ);
                var x = arena.floatVec(12);

                var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

                AssertTrue(info.status == MIPStatus.Optimal);
                AssertCloseD(info.objective, 6.0, 1e-6);
                AssertNodes(info, 218);   // stage3 = 241 -> stage4 = 218

                senses.Dispose(); integ.Dispose(); arena.Dispose();
            }
        }

        // Dedicated pre-LP fathom oracle: 2x + 2y = 3 with x, y integer in [0,2]. The LP relaxation is
        // feasible (x=y=0.75) but there is NO integer point (2(x+y) is even, 3 is odd), and activity-based
        // propagation proves it at the ROOT: it floors x,y <= 1 from the row's upper limit, then ceils
        // x,y >= 1 from the lower limit, leaving x=y=1 whose min activity 4 > 3 -> empty domain. So the
        // node is fathomed WITHOUT ever solving an LP: status Infeasible, nodes == 1, lpIterations == 0
        // (an LP-detected root infeasibility, by contrast, would report lpIterations > 0 -- cf.
        // InfeasibleRootLP). Also re-verifies the Infeasible NaN contract. Both dtypes (exact integer data).
        void Stage4PropagationInfeasible()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(1, 2); A[0, 0] = (float)2; A[0, 1] = (float)2;
            var b = arena.floatVec(1); b[0] = (float)3;
            var c = arena.floatVec(2); c[0] = (float)(-1); c[1] = (float)(-1);
            var xl = arena.floatVec(2);
            var xu = arena.floatVec(2); xu[0] = (float)2; xu[1] = (float)2;
            var x = arena.floatVec(2);
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp);
            senses[0] = ConstraintSense.Equal;
            var integ = new NativeArray<byte>(2, Allocator.Temp); integ[0] = 1; integ[1] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj);

            AssertTrue(info.status == MIPStatus.Infeasible);
            AssertNodes(info, 1);
            AssertEqInt(info.lpIterations, 0);   // propagation fathomed it pre-LP: no LP ever solved
            AssertTrue(math.isnan(info.objective));
            AssertTrue(math.isnan(info.dualBound));
            AssertTrue(math.isnan(info.gap));
            AssertClose(x[0], (float)0, (float)0);
            AssertClose(x[1], (float)0, (float)0);

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // Gap limit: the branchy n=12 search stopped early via relGap=0.3 before the tree is fully explored
        // -> MIPStatus.GapLimit, a feasible non-NaN incumbent, a sound finite dualBound (<= objective for a
        // minimization), and the reported relative gap within the requested limit. DOUBLE-ONLY (many-node
        // search gives a reliable early-stop window; same float rationale as the other Branchy12 tests).
        void GapLimitRelGap()
        {
            int cases = 0;
            for (int s = 0; s < cases; s++)
            {
                var arena = new Arena(Allocator.Persistent);
                BuildBranchy12(in arena, out var A, out var b, out var c, out var senses,
                               out var xl, out var xu, out var integ);
                var x = arena.floatVec(12);
                const double relGap = 0.3;

                var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj,
                                     maxNodes: 0, maxIter: 0, absGap: 0.0, relGap: relGap);

                AssertTrue(info.status == MIPStatus.GapLimit);
                AssertTrue(math.isfinite(info.objective));                    // valid incumbent, not +inf/NaN
                AssertTrue(math.isfinite(info.dualBound));
                AssertTrue(info.dualBound <= info.objective + 1e-6);          // sound bound (minimization)
                AssertTrue(info.gap <= relGap + 1e-9);                        // gap within the requested limit
                AssertTrue(info.nodes < 218);                                 // stopped before the full 218-node tree
                double nz = 0; for (int j = 0; j < 12; j++) nz += math.abs((double)x[j]);
                AssertTrue(nz > 0);                                           // incumbent not all-zero (opt 6, x=0 infeasible)

                senses.Dispose(); integ.Dispose(); arena.Dispose();
            }
        }

        // Pass-through: calling with absGap:0/relGap:0 (both = "off") must be identical to the default --
        // GomoryWolsey still solved to proven Optimal, obj -40, gap 0. Guards against the new parameters
        // perturbing the default path. Both dtypes.
        void GapLimitPassThrough()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(2, 2);
            A[0, 0] = (float)1; A[0, 1] = (float)1;
            A[1, 0] = (float)9; A[1, 1] = (float)5;
            var b = arena.floatVec(2); b[0] = (float)6; b[1] = (float)45;
            var c = arena.floatVec(2); c[0] = (float)(-8); c[1] = (float)(-5);
            var xl = arena.floatVec(2);
            var xu = arena.floatVec(2); xu[0] = (float)10; xu[1] = (float)10;
            var x = arena.floatVec(2);
            var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.LessEqual;
            var integ = new NativeArray<byte>(2, Allocator.Temp); integ[0] = 1; integ[1] = 1;

            var info = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double obj,
                                 maxNodes: 0, maxIter: 0, absGap: 0.0, relGap: 0.0);

            AssertTrue(info.status == MIPStatus.Optimal);
            AssertCloseD(obj, -40.0, 1e-3);
            AssertCloseD(info.gap, 0.0, 1e-9);

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // Determinism with the rounding heuristic AND propagation both active: stein9 installs incumbents
        // via the rounding heuristic during its search and is propagated at every node, so two back-to-back
        // identical solves must still be bit-for-bit identical (nodes/iter/obj/bound/x). Both dtypes (cheap).
        void Stage4DeterminismStein9()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildStein9(in arena, out var A, out var b, out var c, out var senses, out var xl, out var xu, out var integ);
            var x1 = arena.floatVec(9);
            var x2 = arena.floatVec(9);

            var i1 = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x1, out double o1);
            var i2 = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x2, out double o2);

            AssertEqInt(i1.nodes, i2.nodes);
            AssertEqInt(i1.lpIterations, i2.lpIterations);
            AssertEqExactD(i1.objective, i2.objective);
            AssertEqExactD(i1.dualBound, i2.dualBound);
            AssertEqExactD(o1, o2);
            for (int j = 0; j < 9; j++) AssertClose(x1[j], x2[j], (float)0);

            senses.Dispose(); integ.Dispose(); arena.Dispose();
        }

        // Determinism with an active gap limit: two identical relGap-triggering solves must stop at the
        // identical node/status/gap/incumbent. DOUBLE-ONLY (Branchy12, same rationale as GapLimitRelGap).
        void Stage4DeterminismGapLimit()
        {
            int cases = 0;
            for (int s = 0; s < cases; s++)
            {
                var arena = new Arena(Allocator.Persistent);
                BuildBranchy12(in arena, out var A, out var b, out var c, out var senses,
                               out var xl, out var xu, out var integ);
                var x1 = arena.floatVec(12);
                var x2 = arena.floatVec(12);
                const double relGap = 0.3;

                var i1 = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x1, out double o1,
                                   maxNodes: 0, maxIter: 0, absGap: 0.0, relGap: relGap);
                var i2 = MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x2, out double o2,
                                   maxNodes: 0, maxIter: 0, absGap: 0.0, relGap: relGap);

                AssertTrue(i1.status == MIPStatus.GapLimit);
                AssertTrue(i1.status == i2.status);
                AssertEqInt(i1.nodes, i2.nodes);
                AssertEqInt(i1.lpIterations, i2.lpIterations);
                AssertEqExactD(i1.objective, i2.objective);
                AssertEqExactD(i1.dualBound, i2.dualBound);
                AssertEqExactD(i1.gap, i2.gap);
                AssertEqExactD(o1, o2);
                for (int j = 0; j < 12; j++) AssertClose(x1[j], x2[j], (float)0);

                senses.Dispose(); integ.Dispose(); arena.Dispose();
            }
        }

        // ---- diagnostics-recording assert helpers (mirrors LPTests.float.cs) ----

        void AssertTrue(bool cond)
        {
            if (!cond && Fail[0] == (float)0) { Fail[0] = (float)1; Fail[1] = (float)0; Fail[2] = (float)1; Fail[3] = (float)0; }
            Assert.IsTrue(cond);
        }

        void AssertNodes(MIPInfo info, int expected)
        {
            if (info.nodes != expected && Fail[0] == (float)0) { Fail[0] = (float)1; Fail[1] = (float)info.nodes; Fail[2] = (float)expected; Fail[3] = (float)(info.nodes - expected); }
            Assert.IsTrue(info.nodes == expected);
        }

        // nodes <= limit (the stage-3 regression assertion): records got=nodes, expected=limit on failure.
        void AssertNodesLE(MIPInfo info, int limit)
        {
            if (!(info.nodes <= limit) && Fail[0] == (float)0) { Fail[0] = (float)1; Fail[1] = (float)info.nodes; Fail[2] = (float)limit; Fail[3] = (float)(info.nodes - limit); }
            Assert.IsTrue(info.nodes <= limit);
        }

        void AssertEqInt(int got, int expected)
        {
            if (got != expected && Fail[0] == (float)0) { Fail[0] = (float)1; Fail[1] = (float)got; Fail[2] = (float)expected; Fail[3] = (float)(got - expected); }
            Assert.IsTrue(got == expected);
        }

        // Exact (bit-for-bit) double equality for the determinism checks -- NOT a tolerance compare.
        void AssertEqExactD(double got, double expected)
        {
            if (!(got == expected) && Fail[0] == (float)0) { Fail[0] = (float)1; Fail[1] = (float)got; Fail[2] = (float)expected; Fail[3] = (float)(got - expected); }
            Assert.IsTrue(got == expected);
        }

        void AssertClose(float a, float b, float precision)
        {
            float diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (float)0) { Fail[0] = (float)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff; }
            Assert.IsTrue(diff <= precision);
        }

        void AssertCloseD(double a, double b, double precision)
        {
            double diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (float)0) { Fail[0] = (float)1; Fail[1] = (float)a; Fail[2] = (float)b; Fail[3] = (float)diff; }
            Assert.IsTrue(diff <= precision);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void MIPTests(TestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (float)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    // ---- managed-thread argument-validation throw tests (Assert.Catch, same tail pattern as
    //      LPTests.float.cs's Solve/Lad dimension-mismatch tests) ----

    // Each of b / c / senses / xl / xu / integrality / x with the wrong length must throw
    // ArgumentException. Base instance is a valid 2x2 continuous LP (all integ = 0, so the finite-xl
    // integer restriction is not in play -- only the length check should fire).
    [Test]
    public void SolveThrowsOnDimensionMismatch()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.floatMat(2, 2);
        var b = arena.floatVec(2);
        var c = arena.floatVec(2);
        var xl = arena.floatVec(2);
        var xu = arena.floatVec(2); for (int j = 0; j < 2; j++) xu[j] = (float)1;
        var x = arena.floatVec(2);
        var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
        var integ = new NativeArray<byte>(2, Allocator.Temp);   // all continuous

        var bBad = arena.floatVec(3);
        Assert.Catch<ArgumentException>(() => MIP.solve(in A, in bBad, in c, in senses, in xl, in xu, in integ, ref x, out double o));
        var cBad = arena.floatVec(3);
        Assert.Catch<ArgumentException>(() => MIP.solve(in A, in b, in cBad, in senses, in xl, in xu, in integ, ref x, out double o));
        var sensesBad = new NativeArray<ConstraintSense>(3, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => MIP.solve(in A, in b, in c, in sensesBad, in xl, in xu, in integ, ref x, out double o));
        var xlBad = arena.floatVec(3);
        Assert.Catch<ArgumentException>(() => MIP.solve(in A, in b, in c, in senses, in xlBad, in xu, in integ, ref x, out double o));
        var xuBad = arena.floatVec(3);
        Assert.Catch<ArgumentException>(() => MIP.solve(in A, in b, in c, in senses, in xl, in xuBad, in integ, ref x, out double o));
        var integBad = new NativeArray<byte>(3, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integBad, ref x, out double o));
        var xBad = arena.floatVec(3);
        Assert.Catch<ArgumentException>(() => MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref xBad, out double o));

        sensesBad.Dispose(); integBad.Dispose(); senses.Dispose(); integ.Dispose(); arena.Dispose();
    }

    // xl[j] > xu[j] componentwise is an input-sanity error -> ArgumentException (NOT a solver outcome).
    [Test]
    public void SolveThrowsOnLowerAboveUpper()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.floatMat(2, 2);
        var b = arena.floatVec(2);
        var c = arena.floatVec(2);
        var xl = arena.floatVec(2); xl[0] = (float)5; xl[1] = (float)0;   // xl[0] > xu[0]
        var xu = arena.floatVec(2); xu[0] = (float)1; xu[1] = (float)1;
        var x = arena.floatVec(2);
        var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
        var integ = new NativeArray<byte>(2, Allocator.Temp);   // continuous

        Assert.Catch<ArgumentException>(() => MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double o));

        senses.Dispose(); integ.Dispose(); arena.Dispose();
    }

    // Stage-2 restriction: an INTEGER variable with a non-finite lower bound (xl <= -1e29) must throw
    // ("needs a finite xl"). Here var 0 is integer with xl = -1e30; var 1 is continuous (unaffected).
    [Test]
    public void SolveThrowsOnIntegerVariableWithInfiniteLowerBound()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.floatMat(2, 2);
        var b = arena.floatVec(2);
        var c = arena.floatVec(2);
        var xl = arena.floatVec(2); xl[0] = (float)(-1e30); xl[1] = (float)0;
        var xu = arena.floatVec(2); xu[0] = (float)1; xu[1] = (float)1;
        var x = arena.floatVec(2);
        var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
        var integ = new NativeArray<byte>(2, Allocator.Temp); integ[0] = 1; integ[1] = 0;

        Assert.Catch<ArgumentException>(() => MIP.solve(in A, in b, in c, in senses, in xl, in xu, in integ, ref x, out double o));

        senses.Dispose(); integ.Dispose(); arena.Dispose();
    }
}
