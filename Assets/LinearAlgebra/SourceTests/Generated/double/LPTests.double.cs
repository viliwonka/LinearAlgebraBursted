using System;

using LinearAlgebra;
using LinearAlgebra.Sparse;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class doubleLPTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            LpMax2Var,          // min -2x-3y  s.t. x+y<=4, x+3y<=6         -> (3,1), obj -9
            LpEquality,         // min 2x+y    s.t. x+y=2                    -> (0,2), obj 2
            LpGreaterEqual,     // min x+2y    s.t. x+y>=3                   -> (3,0), obj 3
            LpNegativeRhs,      // min x+y     s.t. -x-y<=-2  (i.e. x+y>=2)  -> obj 2
            LpInfeasible,       // x+y<=1 AND x+y>=3                         -> Infeasible
            LpUnbounded,        // min -x      s.t. x-y<=1                   -> Unbounded
            LadExactFit,        // L1 fit of exact line b=1+2t              -> (1,2), obj 0
            LadOutlier,         // L1 fit, 4 collinear pts + 1 outlier       -> (0,1), obj 8
            IRLSExactFit,       // IRLS on exact line                        -> (1,2), obj ~0
            IRLSOutlier,        // IRLS on outlier set                       -> (0,1), obj ~8
            IpMax2Var,          // interior point: same LP as LpMax2Var      -> (3,1), obj -9
            IpEquality,         // interior point: equality LP               -> (0,2), obj 2
            IpGreaterEqual,     // interior point: >= LP                     -> (3,0), obj 3
            IpLadExactFit,      // interior-point LAD, exact line            -> (1,2), obj ~0
            IpLadOutlier,       // interior-point LAD, outlier set           -> (0,1), obj ~8
            WyndorGlass,        // Hillier-Lieberman classic LP              -> (2,6), Z 36
            LadStackloss,       // Brownlee stack-loss LAD (published coeffs)
            SparseLadExactFit,  // sparse (BSR) matrix-free LAD, exact line  -> (1,2), obj ~0
            SparseVsDenseLad,   // sparse LAD objective == dense LAD (outlier set)
            SparseLadStackloss, // sparse LAD objective == dense LAD (stack-loss)
            SparseWyndorGlass,  // sparse (BSR) LP.solve, Wyndor Glass        -> (2,6), Z 36
            SparseVsDenseLp,    // sparse LP.solve == dense LP.solve (mixed <=/>= senses)
            PdlpWyndor,         // PDLP (first-order PDHG) on Wyndor Glass    -> (2,6), Z 36
            PdlpSparseWyndor,   // PDLP over a BSR Wyndor Glass               -> (2,6), Z 36
            PdlpVsDense,        // PDLP(BSR) == PDLP(dense) on a mixed two-sided LP
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/extra
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.LpMax2Var: LpMax2Var(); break;
                case TestType.LpEquality: LpEquality(); break;
                case TestType.LpGreaterEqual: LpGreaterEqual(); break;
                case TestType.LpNegativeRhs: LpNegativeRhs(); break;
                case TestType.LpInfeasible: LpInfeasible(); break;
                case TestType.LpUnbounded: LpUnbounded(); break;
                case TestType.LadExactFit: LadExactFit(); break;
                case TestType.LadOutlier: LadOutlier(); break;
                case TestType.IRLSExactFit: IRLSExactFit(); break;
                case TestType.IRLSOutlier: IRLSOutlier(); break;
                case TestType.IpMax2Var: IpMax2Var(); break;
                case TestType.IpEquality: IpEquality(); break;
                case TestType.IpGreaterEqual: IpGreaterEqual(); break;
                case TestType.IpLadExactFit: IpLadExactFit(); break;
                case TestType.IpLadOutlier: IpLadOutlier(); break;
                case TestType.WyndorGlass: WyndorGlass(); break;
                case TestType.LadStackloss: LadStackloss(); break;
                case TestType.SparseLadExactFit: SparseLadExactFit(); break;
                case TestType.SparseVsDenseLad: SparseVsDenseLad(); break;
                case TestType.SparseLadStackloss: SparseLadStackloss(); break;
                case TestType.SparseWyndorGlass: SparseWyndorGlass(); break;
                case TestType.SparseVsDenseLp: SparseVsDenseLp(); break;
                case TestType.PdlpWyndor: PdlpWyndor(); break;
                case TestType.PdlpSparseWyndor: PdlpSparseWyndor(); break;
                case TestType.PdlpVsDense: PdlpVsDense(); break;
            }
        }

        // --- min -2x-3y  s.t.  x+y<=4, x+3y<=6, x,y>=0  ->  optimal vertex (3,1), obj -9 ---
        void LpMax2Var()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(2, 2);
            A[0, 0] = (double)1; A[0, 1] = (double)1;
            A[1, 0] = (double)1; A[1, 1] = (double)3;
            var b = arena.doubleVec(2); b[0] = (double)4; b[1] = (double)6;
            var c = arena.doubleVec(2); c[0] = (double)(-2); c[1] = (double)(-3);
            var x = arena.doubleVec(2);
            var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.LessEqual;

            var info = LP.solve(in A, in b, in c, in senses, ref x, out double obj);

            AssertTrue(info.status == LPStatus.Optimal);
            AssertClose(x[0], (double)3, (double)1e-3);
            AssertClose(x[1], (double)1, (double)1e-3);
            AssertCloseD(obj, -9.0, 1e-3);

            senses.Dispose(); arena.Dispose();
        }

        // --- min 2x+y  s.t.  x+y=2, x,y>=0  ->  (0,2), obj 2 (exercises equality -> phase-1 artificial) ---
        void LpEquality()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(1, 2); A[0, 0] = (double)1; A[0, 1] = (double)1;
            var b = arena.doubleVec(1); b[0] = (double)2;
            var c = arena.doubleVec(2); c[0] = (double)2; c[1] = (double)1;
            var x = arena.doubleVec(2);
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp);
            senses[0] = ConstraintSense.Equal;

            var info = LP.solve(in A, in b, in c, in senses, ref x, out double obj);

            AssertTrue(info.status == LPStatus.Optimal);
            AssertClose(x[0], (double)0, (double)1e-3);
            AssertClose(x[1], (double)2, (double)1e-3);
            AssertCloseD(obj, 2.0, 1e-3);

            senses.Dispose(); arena.Dispose();
        }

        // --- min x+2y  s.t.  x+y>=3, x,y>=0  ->  (3,0), obj 3 (exercises surplus + artificial) ---
        void LpGreaterEqual()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(1, 2); A[0, 0] = (double)1; A[0, 1] = (double)1;
            var b = arena.doubleVec(1); b[0] = (double)3;
            var c = arena.doubleVec(2); c[0] = (double)1; c[1] = (double)2;
            var x = arena.doubleVec(2);
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp);
            senses[0] = ConstraintSense.GreaterEqual;

            var info = LP.solve(in A, in b, in c, in senses, ref x, out double obj);

            AssertTrue(info.status == LPStatus.Optimal);
            AssertClose(x[0], (double)3, (double)1e-3);
            AssertClose(x[1], (double)0, (double)1e-3);
            AssertCloseD(obj, 3.0, 1e-3);

            senses.Dispose(); arena.Dispose();
        }

        // --- min x+y  s.t.  -x-y <= -2 (negative rhs -> internal row negation to x+y>=2) -> obj 2 ---
        void LpNegativeRhs()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(1, 2); A[0, 0] = (double)(-1); A[0, 1] = (double)(-1);
            var b = arena.doubleVec(1); b[0] = (double)(-2);
            var c = arena.doubleVec(2); c[0] = (double)1; c[1] = (double)1;
            var x = arena.doubleVec(2);
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual;

            var info = LP.solve(in A, in b, in c, in senses, ref x, out double obj);

            AssertTrue(info.status == LPStatus.Optimal);
            AssertTrue(x[0] >= (double)(-1e-4) && x[1] >= (double)(-1e-4));   // x >= 0
            AssertCloseD((double)x[0] + (double)x[1], 2.0, 1e-3);            // x+y = 2 (any vertex on the edge)
            AssertCloseD(obj, 2.0, 1e-3);

            senses.Dispose(); arena.Dispose();
        }

        // --- x+y<=1 AND x+y>=3: empty feasible region -> Infeasible ---
        void LpInfeasible()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(2, 2);
            A[0, 0] = (double)1; A[0, 1] = (double)1;
            A[1, 0] = (double)1; A[1, 1] = (double)1;
            var b = arena.doubleVec(2); b[0] = (double)1; b[1] = (double)3;
            var c = arena.doubleVec(2); c[0] = (double)1; c[1] = (double)1;
            var x = arena.doubleVec(2);
            var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.GreaterEqual;

            var info = LP.solve(in A, in b, in c, in senses, ref x, out double obj);

            AssertTrue(info.status == LPStatus.Infeasible);

            senses.Dispose(); arena.Dispose();
        }

        // --- min -x  s.t.  x-y<=1, x,y>=0: x grows without bound (y=x-1) -> Unbounded ---
        void LpUnbounded()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(1, 2); A[0, 0] = (double)1; A[0, 1] = (double)(-1);
            var b = arena.doubleVec(1); b[0] = (double)1;
            var c = arena.doubleVec(2); c[0] = (double)(-1); c[1] = (double)0;
            var x = arena.doubleVec(2);
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual;

            var info = LP.solve(in A, in b, in c, in senses, ref x, out double obj);

            AssertTrue(info.status == LPStatus.Unbounded);

            senses.Dispose(); arena.Dispose();
        }

        // Build the 2-column design A=[1, t] and observations b for the LAD/IRLS fit tests.
        static void BuildLine(ref Arena arena, out doubleMxN A, out doubleN b, bool outlier)
        {
            int m = outlier ? 5 : 4;
            A = arena.doubleMat(m, 2);
            b = arena.doubleVec(m);
            if (!outlier)
            {
                // exact line b = 1 + 2t at t = 0,1,2,3
                for (int i = 0; i < 4; i++) { A[i, 0] = (double)1; A[i, 1] = (double)i; b[i] = (double)(1 + 2 * i); }
            }
            else
            {
                // line b = t at t = 0,1,2,3,4 with a gross outlier at t=2 (b=10 instead of 2)
                for (int i = 0; i < 5; i++) { A[i, 0] = (double)1; A[i, 1] = (double)i; b[i] = (double)i; }
                b[2] = (double)10;
            }
        }

        // --- LAD on an exactly-collinear set: residual 0, coefficients (1,2) ---
        void LadExactFit()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildLine(ref arena, out var A, out var b, false);
            var x = arena.doubleVec(2);

            var info = LP.lad(in A, in b, ref x, out double obj);

            AssertTrue(info.status == LPStatus.Optimal);
            AssertClose(x[0], (double)1, (double)1e-2);
            AssertClose(x[1], (double)2, (double)1e-2);
            AssertCloseD(obj, 0.0, 1e-2);

            arena.Dispose();
        }

        // --- LAD with 4 collinear points + 1 gross outlier: robustly ignores it ->
        //     line b=t (coeffs 0,1), L1 residual = |10-2| = 8 ---
        void LadOutlier()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildLine(ref arena, out var A, out var b, true);
            var x = arena.doubleVec(2);

            var info = LP.lad(in A, in b, ref x, out double obj);

            AssertTrue(info.status == LPStatus.Optimal);
            AssertClose(x[0], (double)0, (double)1e-2);
            AssertClose(x[1], (double)1, (double)1e-2);
            AssertCloseD(obj, 8.0, 1e-2);

            arena.Dispose();
        }

        // --- IRLS on the exact line: approximate but should nail a zero-residual fit ---
        void IRLSExactFit()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildLine(ref arena, out var A, out var b, false);
            var x = arena.doubleVec(2);   // zero start

            var info = Optimize.ladIRLS(in A, in b, ref x);

            AssertTrue(info.status == LPStatus.Optimal);
            AssertClose(x[0], (double)1, (double)1e-2);
            AssertClose(x[1], (double)2, (double)1e-2);
            AssertCloseD(info.objective, 0.0, 1e-2);

            arena.Dispose();
        }

        // --- IRLS on the outlier set: down-weights the outlier, approaches the LAD line (0,1) ---
        void IRLSOutlier()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildLine(ref arena, out var A, out var b, true);
            var x = arena.doubleVec(2);

            var info = Optimize.ladIRLS(in A, in b, ref x);

            // IRLS is approximate: looser tolerances than the exact LP.lad path.
            AssertClose(x[0], (double)0, (double)5e-2);
            AssertClose(x[1], (double)1, (double)5e-2);
            AssertCloseD(info.objective, 8.0, 2e-1);

            arena.Dispose();
        }

        // ==== interior-point backend: same optima as simplex, looser (interior) tolerances ====

        void IpMax2Var()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(2, 2);
            A[0, 0] = (double)1; A[0, 1] = (double)1;
            A[1, 0] = (double)1; A[1, 1] = (double)3;
            var b = arena.doubleVec(2); b[0] = (double)4; b[1] = (double)6;
            var c = arena.doubleVec(2); c[0] = (double)(-2); c[1] = (double)(-3);
            var x = arena.doubleVec(2);
            var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.LessEqual;

            var info = LP.solve(in A, in b, in c, in senses, ref x, out double obj, LPMethod.InteriorPoint);

            AssertTrue(info.status == LPStatus.Optimal);
            AssertClose(x[0], (double)3, (double)3e-2);
            AssertClose(x[1], (double)1, (double)3e-2);
            AssertCloseD(obj, -9.0, 3e-2);

            senses.Dispose(); arena.Dispose();
        }

        void IpEquality()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(1, 2); A[0, 0] = (double)1; A[0, 1] = (double)1;
            var b = arena.doubleVec(1); b[0] = (double)2;
            var c = arena.doubleVec(2); c[0] = (double)2; c[1] = (double)1;
            var x = arena.doubleVec(2);
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp);
            senses[0] = ConstraintSense.Equal;

            var info = LP.solve(in A, in b, in c, in senses, ref x, out double obj, LPMethod.InteriorPoint);

            AssertTrue(info.status == LPStatus.Optimal);
            AssertClose(x[0], (double)0, (double)3e-2);
            AssertClose(x[1], (double)2, (double)3e-2);
            AssertCloseD(obj, 2.0, 3e-2);

            senses.Dispose(); arena.Dispose();
        }

        void IpGreaterEqual()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(1, 2); A[0, 0] = (double)1; A[0, 1] = (double)1;
            var b = arena.doubleVec(1); b[0] = (double)3;
            var c = arena.doubleVec(2); c[0] = (double)1; c[1] = (double)2;
            var x = arena.doubleVec(2);
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp);
            senses[0] = ConstraintSense.GreaterEqual;

            var info = LP.solve(in A, in b, in c, in senses, ref x, out double obj, LPMethod.InteriorPoint);

            AssertTrue(info.status == LPStatus.Optimal);
            AssertClose(x[0], (double)3, (double)3e-2);
            AssertClose(x[1], (double)0, (double)3e-2);
            AssertCloseD(obj, 3.0, 3e-2);

            senses.Dispose(); arena.Dispose();
        }

        void IpLadExactFit()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildLine(ref arena, out var A, out var b, false);
            var x = arena.doubleVec(2);

            // LAD LPs are highly degenerate; interior point may stop just shy of the tight tolerance
            // while still landing on an accurate solution -- assert on the solution, not the status.
            var info = LP.lad(in A, in b, ref x, out double obj, LPMethod.InteriorPoint);

            AssertClose(x[0], (double)1, (double)5e-2);
            AssertClose(x[1], (double)2, (double)5e-2);
            AssertCloseD(obj, 0.0, 5e-2);

            arena.Dispose();
        }

        void IpLadOutlier()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildLine(ref arena, out var A, out var b, true);
            var x = arena.doubleVec(2);

            var info = LP.lad(in A, in b, ref x, out double obj, LPMethod.InteriorPoint);

            AssertClose(x[0], (double)0, (double)5e-2);
            AssertClose(x[1], (double)1, (double)5e-2);
            AssertCloseD(obj, 8.0, 1e-1);

            arena.Dispose();
        }

        // ==== literature known-answer vectors ====

        // Wyndor Glass Co. (Hillier & Lieberman, "Introduction to Operations Research"):
        //   max 3x1 + 5x2  s.t.  x1 <= 4,  2x2 <= 12,  3x1 + 2x2 <= 18,  x >= 0
        // Optimal vertex (2, 6), Z = 36. Solved as a minimization of -3x1 - 5x2 (obj -36).
        void WyndorGlass()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(3, 2);
            A[0, 0] = (double)1; A[0, 1] = (double)0;
            A[1, 0] = (double)0; A[1, 1] = (double)2;
            A[2, 0] = (double)3; A[2, 1] = (double)2;
            var b = arena.doubleVec(3); b[0] = (double)4; b[1] = (double)12; b[2] = (double)18;
            var c = arena.doubleVec(2); c[0] = (double)(-3); c[1] = (double)(-5);
            var x = arena.doubleVec(2);
            var senses = new NativeArray<ConstraintSense>(3, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.LessEqual; senses[2] = ConstraintSense.LessEqual;

            var info = LP.solve(in A, in b, in c, in senses, ref x, out double obj);

            AssertTrue(info.status == LPStatus.Optimal);
            AssertClose(x[0], (double)2, (double)1e-3);
            AssertClose(x[1], (double)6, (double)1e-3);
            AssertCloseD(obj, -36.0, 1e-3);

            senses.Dispose(); arena.Dispose();
        }

        // Brownlee's stack-loss plant data (R `stackloss`, 21 obs). LAD (L1) regression coefficients,
        // cross-verified across R's quantreg and ROI: intercept -39.68986, AirFlow 0.83188,
        // WaterTemp 0.57391, AcidConc -0.06087. The L1 fit interpolates 4 of the 21 points, so the
        // vertex (hence these coefficients) is exact -- float reaches it comfortably within 5e-2.
        void LadStackloss()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildStackloss(ref arena, out var A, out var b);
            var x = arena.doubleVec(4);

            var info = LP.lad(in A, in b, ref x, out double obj);

            AssertTrue(info.status == LPStatus.Optimal);
            AssertClose(x[0], (double)(-39.68985507), (double)5e-2);   // intercept
            AssertClose(x[1], (double)0.83188406, (double)5e-2);       // Air.Flow
            AssertClose(x[2], (double)0.57391304, (double)5e-2);       // Water.Temp
            AssertClose(x[3], (double)(-0.06086957), (double)5e-2);    // Acid.Conc.

            arena.Dispose();
        }

        // A = [1, AirFlow, WaterTemp, AcidConc], b = stack.loss. All 21 rows are integer-valued.
        static void BuildStackloss(ref Arena arena, out doubleMxN A, out doubleN b)
        {
            A = arena.doubleMat(21, 4);
            b = arena.doubleVec(21);
            SetObs(A, b, 0, 80, 27, 89, 42); SetObs(A, b, 1, 80, 27, 88, 37); SetObs(A, b, 2, 75, 25, 90, 37);
            SetObs(A, b, 3, 62, 24, 87, 28); SetObs(A, b, 4, 62, 22, 87, 18); SetObs(A, b, 5, 62, 23, 87, 18);
            SetObs(A, b, 6, 62, 24, 93, 19); SetObs(A, b, 7, 62, 24, 93, 20); SetObs(A, b, 8, 58, 23, 87, 15);
            SetObs(A, b, 9, 58, 18, 80, 14); SetObs(A, b, 10, 58, 18, 89, 14); SetObs(A, b, 11, 58, 17, 88, 13);
            SetObs(A, b, 12, 58, 18, 82, 11); SetObs(A, b, 13, 58, 19, 93, 12); SetObs(A, b, 14, 50, 18, 89, 8);
            SetObs(A, b, 15, 50, 18, 86, 7); SetObs(A, b, 16, 50, 19, 72, 8); SetObs(A, b, 17, 50, 19, 79, 8);
            SetObs(A, b, 18, 50, 20, 80, 9); SetObs(A, b, 19, 56, 20, 82, 15); SetObs(A, b, 20, 70, 20, 91, 15);
        }

        static void SetObs(doubleMxN A, doubleN b, int i, int airflow, int watertemp, int acidconc, int stackloss)
        {
            A[i, 0] = (double)1; A[i, 1] = (double)airflow; A[i, 2] = (double)watertemp; A[i, 3] = (double)acidconc;
            b[i] = (double)stackloss;
        }

        // ==== sparse (BSR) matrix-free interior-point LAD ====

        // Convert a dense matrix to BSR with 1×1 blocks (nonzeros only) -- exercises the sparse LAD
        // path (doubleLadOperator / matrix-free interior point) on data whose dense answer is known.
        static doubleBSR BuildBSR1x1(ref Arena arena, in doubleMxN dense)
        {
            int m = dense.M_Rows, n = dense.N_Cols;
            int nnz = 0;
            for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) if (dense[i, j] != (double)0) nnz++;
            var builder = arena.doubleBSRBuilder(m, n, 1, 1, nnz);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    if (dense[i, j] != (double)0)
                    {
                        var blk = arena.doubleMat(1, 1);
                        blk[0, 0] = dense[i, j];
                        builder.AddBlock(i, j, in blk);
                    }
            return builder.ToBSR(ref arena);
        }

        // Sparse LAD on an exactly-collinear set: matrix-free interior point recovers (1,2), residual ~0.
        void SparseLadExactFit()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildLine(ref arena, out var A, out var b, false);
            var As = BuildBSR1x1(ref arena, in A);
            var x = arena.doubleVec(2);

            var info = LP.lad(in As, in b, ref x, out double obj);

            AssertClose(x[0], (double)1, (double)1e-1);
            AssertClose(x[1], (double)2, (double)1e-1);
            AssertCloseD(obj, 0.0, 1e-1);

            arena.Dispose();
        }

        // The sparse (matrix-free interior-point) LAD must reach the SAME L1 optimum as the exact dense
        // LP.lad on the identical outlier-laden data -- objective and coefficients agree.
        void SparseVsDenseLad()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildLine(ref arena, out var A, out var b, true);
            var As = BuildBSR1x1(ref arena, in A);
            var xd = arena.doubleVec(2);
            var xs = arena.doubleVec(2);

            var infoD = LP.lad(in A, in b, ref xd, out double objD);     // dense, exact
            var infoS = LP.lad(in As, in b, ref xs, out double objS);    // sparse, matrix-free IP

            AssertTrue(infoD.status == LPStatus.Optimal);
            AssertCloseD(objS, objD, 0.08 * (1.0 + objD));
            AssertClose(xs[0], xd[0], (double)2e-1);
            AssertClose(xs[1], xd[1], (double)2e-1);

            arena.Dispose();
        }

        // Sparse LAD on the real stack-loss data must match the dense LAD L1 residual.
        void SparseLadStackloss()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildStackloss(ref arena, out var A, out var b);
            var As = BuildBSR1x1(ref arena, in A);
            var xd = arena.doubleVec(4);
            var xs = arena.doubleVec(4);

            var infoD = LP.lad(in A, in b, ref xd, out double objD);
            var infoS = LP.lad(in As, in b, ref xs, out double objS);

            AssertTrue(infoD.status == LPStatus.Optimal);
            AssertCloseD(objS, objD, 0.08 * (1.0 + objD));

            arena.Dispose();
        }

        // ==== sparse (BSR) matrix-free interior-point general LP.solve (slack-augmented operator) ====

        // Sparse (BSR) LP.solve on the Wyndor Glass problem must reach the same vertex (2,6), Z 36, as the
        // dense simplex -- exercises the slack-augmented operator (all-≤ inequalities).
        void SparseWyndorGlass()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(3, 2);
            A[0, 0] = (double)1; A[0, 1] = (double)0;
            A[1, 0] = (double)0; A[1, 1] = (double)2;
            A[2, 0] = (double)3; A[2, 1] = (double)2;
            var b = arena.doubleVec(3); b[0] = (double)4; b[1] = (double)12; b[2] = (double)18;
            var c = arena.doubleVec(2); c[0] = (double)(-3); c[1] = (double)(-5);
            var As = BuildBSR1x1(ref arena, in A);
            var x = arena.doubleVec(2);
            var senses = new NativeArray<ConstraintSense>(3, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.LessEqual; senses[2] = ConstraintSense.LessEqual;

            var info = LP.solve(in As, in b, in c, in senses, ref x, out double obj);

            AssertClose(x[0], (double)2, (double)2e-1);
            AssertClose(x[1], (double)6, (double)2e-1);
            AssertCloseD(obj, -36.0, 0.05 * (1.0 + 36.0));

            senses.Dispose(); arena.Dispose();
        }

        // General sparse LP.solve (matrix-free interior point) must reach the SAME optimum as the dense
        // LP.solve on an identical LP with MIXED senses (<= and >=): min -x-2y s.t. x+y<=4, x+y>=1, y<=3.
        void SparseVsDenseLp()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(3, 2);
            A[0, 0] = (double)1; A[0, 1] = (double)1;   // x + y <= 4
            A[1, 0] = (double)1; A[1, 1] = (double)1;   // x + y >= 1
            A[2, 0] = (double)0; A[2, 1] = (double)1;   // y <= 3
            var b = arena.doubleVec(3); b[0] = (double)4; b[1] = (double)1; b[2] = (double)3;
            var c = arena.doubleVec(2); c[0] = (double)(-1); c[1] = (double)(-2);
            var senses = new NativeArray<ConstraintSense>(3, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.GreaterEqual; senses[2] = ConstraintSense.LessEqual;
            var As = BuildBSR1x1(ref arena, in A);
            var xd = arena.doubleVec(2);
            var xs = arena.doubleVec(2);

            var infoD = LP.solve(in A, in b, in c, in senses, ref xd, out double objD);   // dense simplex
            var infoS = LP.solve(in As, in b, in c, in senses, ref xs, out double objS);  // sparse IP

            AssertTrue(infoD.status == LPStatus.Optimal);
            AssertCloseD(objS, objD, 0.05 * (1.0 + math.abs(objD)));
            AssertClose(xs[0], xd[0], (double)2e-1);
            AssertClose(xs[1], xd[1], (double)2e-1);

            senses.Dispose(); arena.Dispose();
        }

        // ==== PDLP (first-order matrix-free PDHG) ====

        // PDLP on Wyndor Glass in two-sided form: max 3x1+5x2 (min -3x1-5x2) s.t. x1<=4, 2x2<=12,
        // 3x1+2x2<=18, x>=0. Optimum (2,6), Z=36 (obj -36). First-order PDHG (restart + primal weight +
        // adaptive step + preconditioning), so it converges to a loose tolerance -- give it a generous
        // budget and check the optimum, not speed.
        void PdlpWyndor()
        {
            var arena = new Arena(Allocator.Persistent);
            double INF = (double)1e30;
            var A = arena.doubleMat(3, 2);
            A[0, 0] = (double)1; A[0, 1] = (double)0;
            A[1, 0] = (double)0; A[1, 1] = (double)2;
            A[2, 0] = (double)3; A[2, 1] = (double)2;
            var lc = arena.doubleVec(3); lc[0] = -INF; lc[1] = -INF; lc[2] = -INF;
            var uc = arena.doubleVec(3); uc[0] = (double)4; uc[1] = (double)12; uc[2] = (double)18;
            var lv = arena.doubleVec(2); lv[0] = (double)0; lv[1] = (double)0;
            var uv = arena.doubleVec(2); uv[0] = INF; uv[1] = INF;
            var c = arena.doubleVec(2); c[0] = (double)(-3); c[1] = (double)(-5);
            var x = arena.doubleVec(2);

            var info = LP.pdlp(in A, in lc, in uc, in lv, in uv, in c, ref x, out double obj, 200000, 1e-6);

            AssertClose(x[0], (double)2, (double)1.5e-1);
            AssertClose(x[1], (double)6, (double)1.5e-1);
            AssertCloseD(obj, -36.0, 0.5);

            arena.Dispose();
        }

        // Same Wyndor Glass optimum, but the constraint matrix is a BSR (1x1 blocks): exercises the sparse
        // PDLP path end-to-end -- doubleBSROperator's spMV/spMVT, the BSR block-traversal equilibration,
        // and the shared scaled-solve glue. Must land on the same (2,6) / -36.
        void PdlpSparseWyndor()
        {
            var arena = new Arena(Allocator.Persistent);
            double INF = (double)1e30;
            var A = arena.doubleMat(3, 2);
            A[0, 0] = (double)1; A[0, 1] = (double)0;
            A[1, 0] = (double)0; A[1, 1] = (double)2;
            A[2, 0] = (double)3; A[2, 1] = (double)2;
            var As = BuildBSR1x1(ref arena, in A);
            var lc = arena.doubleVec(3); lc[0] = -INF; lc[1] = -INF; lc[2] = -INF;
            var uc = arena.doubleVec(3); uc[0] = (double)4; uc[1] = (double)12; uc[2] = (double)18;
            var lv = arena.doubleVec(2); lv[0] = (double)0; lv[1] = (double)0;
            var uv = arena.doubleVec(2); uv[0] = INF; uv[1] = INF;
            var c = arena.doubleVec(2); c[0] = (double)(-3); c[1] = (double)(-5);
            var x = arena.doubleVec(2);

            var info = LP.pdlp(in As, in lc, in uc, in lv, in uv, in c, ref x, out double obj, 200000, 1e-6);

            AssertClose(x[0], (double)2, (double)1.5e-1);
            AssertClose(x[1], (double)6, (double)1.5e-1);
            AssertCloseD(obj, -36.0, 0.5);

            arena.Dispose();
        }

        // Sparse PDLP must reach the SAME optimum as dense PDLP on an identical two-sided LP with an
        // equality row and an inequality row: min -x-2y s.t. x+y=4 (ℓ_c=u_c), 0<=y<=3, x,y>=0. Optimum
        // (1,3), Z -7. Confirms the BSR operator + BSR equilibration agree with the dense path.
        void PdlpVsDense()
        {
            var arena = new Arena(Allocator.Persistent);
            double INF = (double)1e30;
            var A = arena.doubleMat(2, 2);
            A[0, 0] = (double)1; A[0, 1] = (double)1;   // x + y = 4
            A[1, 0] = (double)0; A[1, 1] = (double)1;   // y <= 3
            var lc = arena.doubleVec(2); lc[0] = (double)4; lc[1] = -INF;
            var uc = arena.doubleVec(2); uc[0] = (double)4; uc[1] = (double)3;
            var lv = arena.doubleVec(2); lv[0] = (double)0; lv[1] = (double)0;
            var uv = arena.doubleVec(2); uv[0] = INF; uv[1] = INF;
            var c = arena.doubleVec(2); c[0] = (double)(-1); c[1] = (double)(-2);
            var As = BuildBSR1x1(ref arena, in A);
            var xd = arena.doubleVec(2);
            var xs = arena.doubleVec(2);

            LP.pdlp(in A,  in lc, in uc, in lv, in uv, in c, ref xd, out double objD, 200000, 1e-6);
            LP.pdlp(in As, in lc, in uc, in lv, in uv, in c, ref xs, out double objS, 200000, 1e-6);

            AssertCloseD(objD, -7.0, 0.5);
            AssertCloseD(objS, objD, 0.05 * (1.0 + math.abs(objD)));
            AssertClose(xs[0], xd[0], (double)2e-1);
            AssertClose(xs[1], xd[1], (double)2e-1);

            arena.Dispose();
        }

        // ---- diagnostics-recording assert helpers (Burst-legal: Assert.Fail(string) is not) ----

        void AssertTrue(bool cond)
        {
            if (!cond && Fail[0] == (double)0) { Fail[0] = (double)1; Fail[1] = (double)0; Fail[2] = (double)1; Fail[3] = (double)0; }
            Assert.IsTrue(cond);
        }

        void AssertClose(double a, double b, double precision)
        {
            double diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (double)0) { Fail[0] = (double)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff; }
            Assert.IsTrue(diff <= precision);
        }

        void AssertCloseD(double a, double b, double precision)
        {
            double diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (double)0) { Fail[0] = (double)1; Fail[1] = (double)a; Fail[2] = (double)b; Fail[3] = (double)diff; }
            Assert.IsTrue(diff <= precision);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void LPTests(TestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (double)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    // ---- managed-thread argument-validation throw tests ----

    [Test]
    public void SolveThrowsOnDimensionMismatch()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.doubleMat(2, 2);
        var b = arena.doubleVec(2);
        var c = arena.doubleVec(2);
        var x = arena.doubleVec(3);   // wrong length
        var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);

        Assert.Catch<ArgumentException>(() => LP.solve(in A, in b, in c, in senses, ref x, out double obj));

        senses.Dispose(); arena.Dispose();
    }

    [Test]
    public void LadThrowsOnDimensionMismatch()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.doubleMat(4, 2);
        var b = arena.doubleVec(3);   // wrong length (should be 4)
        var x = arena.doubleVec(2);

        Assert.Catch<ArgumentException>(() => LP.lad(in A, in b, ref x, out double obj));

        arena.Dispose();
    }

    // Diagnostic: the matrix-free normal operator M = Aₛ diag(D) Aₛᵀ (with D = 1) must reproduce the
    // materialized 2·A·Aᵀ + 2·I column by column (Aₛ = [A|−A|−I|I] ⇒ Aₛ Aₛᵀ = 2 A Aᵀ + 2 I).
    [Test]
    public void SparseNormalOperatorMatchesDense()
    {
        var arena = new Arena(Allocator.Persistent);
        int m = 3, n = 2, nv = 2 * n + 2 * m;
        var Ad = arena.doubleMat(m, n);
        Ad[0, 0] = (double)1; Ad[0, 1] = (double)2;
        Ad[1, 0] = (double)3; Ad[1, 1] = (double)(-1);
        Ad[2, 0] = (double)0; Ad[2, 1] = (double)4;

        // dense BSR (1×1 blocks, nonzeros only)
        int nnz = 0;
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) if (Ad[i, j] != (double)0) nnz++;
        var builder = arena.doubleBSRBuilder(m, n, 1, 1, nnz);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                if (Ad[i, j] != (double)0) { var blk = arena.doubleMat(1, 1); blk[0, 0] = Ad[i, j]; builder.AddBlock(i, j, in blk); }
        var As = builder.ToBSR(ref arena);

        var ladSp = arena.doubleVec(n); var ladTm = arena.doubleVec(m); var ladAtr = arena.doubleVec(n);
        var d = arena.doubleVec(nv); for (int j = 0; j < nv; j++) d[j] = (double)1;
        var normNV = arena.doubleVec(nv);
        var lad = new doubleLadOperator(in As, in ladSp, in ladTm, in ladAtr);
        var Mop = new doubleNormalOperator<doubleLadOperator>(in lad, in d, in normNV, (double)0);

        var v = arena.doubleVec(m); var y = arena.doubleVec(m);
        for (int i = 0; i < m; i++)
        {
            for (int k = 0; k < m; k++) v[k] = (double)0;
            v[i] = (double)1;
            Mop.Apply(in v, ref y);
            for (int k = 0; k < m; k++)
            {
                double aat = 0;
                for (int j = 0; j < n; j++) aat += (double)Ad[k, j] * (double)Ad[i, j];
                double expected = 2.0 * aat + (k == i ? 2.0 : 0.0);
                Assert.That((double)y[k], Is.EqualTo(expected).Within(1e-3), $"M[{k},{i}]");
            }
        }
        arena.Dispose();
    }

    // Simplex and interior point must agree on the objective of a feasible bounded LP (they reach the
    // same optimal face). Run on the managed thread so a divergence surfaces with a clear message.
    [Test]
    public void SimplexAndInteriorPointAgree()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.doubleMat(2, 2);
        A[0, 0] = (double)1; A[0, 1] = (double)1;
        A[1, 0] = (double)1; A[1, 1] = (double)3;
        var b = arena.doubleVec(2); b[0] = (double)4; b[1] = (double)6;
        var c = arena.doubleVec(2); c[0] = (double)(-2); c[1] = (double)(-3);
        var xs = arena.doubleVec(2);
        var xi = arena.doubleVec(2);
        var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
        senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.LessEqual;

        var si = LP.solve(in A, in b, in c, in senses, ref xs, out double objS, LPMethod.Simplex);
        var ii = LP.solve(in A, in b, in c, in senses, ref xi, out double objI, LPMethod.InteriorPoint);

        Assert.IsTrue(si.status == LPStatus.Optimal);
        Assert.IsTrue(ii.status == LPStatus.Optimal);
        Assert.That(objI, Is.EqualTo(objS).Within(3e-2), $"simplex obj {objS} vs interior-point obj {objI}");

        senses.Dispose(); arena.Dispose();
    }
}
