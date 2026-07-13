using System;
using System.Collections.Generic;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

// Acceptance battery for the inequality-constrained active-set QP core (QP.qpActiveSetCore).
// INTERNAL entry point (the public surface is QP.solve) -- reached here via the InternalsVisibleTo
// grants on both BurstLinearAlgebra.Tests and BurstLinearAlgebra.TemplateSource.Tests-firstpass
// (TemplateSource/AssemblyInfo.cs), the same route QPEqpTests.fProxy.cs uses for QP.eqpSolve.
//
// Burst execution: compute runs inside [BurstCompile(CompileSynchronously = true)] IJob structs; NUnit
// Assert.IsTrue with == only inside the job, first failure recorded into a Fail[] diagnostic array read
// back on the managed side -- the exact pattern QPEqpTests.fProxy.cs / LPTests.fProxy.cs use.
//
// ---- Acceptance oracles ----
//
// (a) Hock-Schittkowski knowns: HS21/HS35/HS52/HS76, literals transcribed from the CUTEst SIF encoding
//     (github.com/lpoo/cutest-sif, mirroring the Hock & Schittkowski 1981 "Test examples for nonlinear
//     programming codes" collection; SIF input by A.R. Conn, 1990/1991). Each problem's SIF ELEMENT/
//     GROUP USES + CONSTANTS were expanded BY HAND into the Q/c/A/b/xl/xu literals below (see each
//     job's own comment for the expansion) and cross-checked two independent ways: scipy.optimize.
//     minimize(method='SLSQP') on the transcribed problem, AND (HS35/HS52/HS76) an exact closed-form
//     KKT/Lagrange solve (sympy) at the determined active set. HS21/HS35 additionally match their SIF
//     file's own "*LO SOLTN" comment (a published reference value) to 6-7 significant figures once the
//     CONSTANTS-section objective offset is added back; HS52's SOLTN comment (5.326643) is a truncated
//     literature value -- the exact rational optimum is 1859/349 = 5.3266475644699140..., used here
//     instead (see HS52Job's comment). HS76's SIF mirror carries no SOLTN comment at all; its optimum
//     -103/22 = -4.68181818... matches the value widely cited for this problem in the SQP literature.
//     All four use a HAND-CHOSEN feasible x0 (none of these SIF files' own START POINT satisfies this
//     library's phase-1-free active-set contract for every problem -- HS21/HS52's do not) -- see each job.
//
// (b) Brute-force oracle: random box-constrained (bounds only, no general A rows) STRICTLY convex QP
//     (SPD Q, so the box optimum is unique) at n <= 8. Enumerates every one of the 3^n possible
//     active-set assignments (each variable: free / at-lower / at-upper) via BASE-3 digit decoding;
//     for each, solves the resulting EQP EXACTLY (k == 0: direct CHO solve of Qx = -c; k >= 1: reuses
//     QP.eqpSolve on the selected unit rows -- exactly the kernel qpActiveSetCore itself is built
//     from), checks FULL-box feasibility (not just the active bounds), and keeps the minimum feasible
//     objective. The true optimum's active set is necessarily one of the 3^n combinations enumerated,
//     so this minimum IS the global box-constrained optimum -- an oracle independent of
//     qpActiveSetCore's own machinery.
//
// (c) LP limit: Q = 0 reproduces LP.solve's objective on a random dense feasible LP (A m x n
//     nonnegative in [0,1] with m = n/2, x0 random in [0,1], b = A x0 + slack (0.1..1, so x0 is
//     STRICTLY feasible), c random in [-1,1], all rows LessEqual, x >= 0). x0 doubles as the feasible
//     start.
//
// (d) Degenerate: a constraint row DUPLICATED verbatim (same A row, same b, same sense, tight at the
//     optimum) must not change the answer -- qpActiveSetCore on the duplicated (m+1-row) problem must
//     reach the SAME objective as the same problem with the duplicate removed (m rows). Exercises
//     SeedWorkingSet's rank-guarded seeding on a redundant row.
//
// (e) Unbounded: a genuine unbounded ray -- Q = diag(1, 0) (singular, null space spans x2), c = (0,-1),
//     no general constraints, x2 >= 0 with no upper bound, x1 free. f = 1/2 x1^2 - x2 decreases without
//     bound as x2 -> +infinity with x1 = 0 (no constraint ever blocks it). Hand-traced against the
//     kernel's own logic: x0 = (0,0) seeds with x2's lower bound tight; the first EQP step there is
//     already optimal in x1 (its multiplier sign is wrong, so it is dropped); the SECOND step (k = 0,
//     full space) hits Q's singularity directly, regularizes, finds zero curvature, an unblocked ray,
//     and gp &lt; 0 -- the real Unbounded condition.
public class fProxyQPActiveSetTests
{
    // ================================================================================================
    // (a) Hock-Schittkowski knowns
    // ================================================================================================

    // HS21 (Hock & Schittkowski 1981, problem 21; SIF: A.R. Conn, April 1990;
    // github.com/lpoo/cutest-sif/blob/master/HS21.SIF). min 0.01 x1^2 + x2^2 - 100 s.t.
    // 10 x1 - x2 >= 10, 2 <= x1 <= 50, -50 <= x2 <= 50. SOLTN comment (published reference): -99.96.
    // Our solver's objective omits the SIF CONSTANTS-section "-100" offset, so the acceptance value
    // here is 0.04 = -99.96 + 100. Optimum x* = (2, 0) (verified independently via scipy SLSQP). SIF's
    // own START POINT (-1,-1) is infeasible for our phase-1-free entry (x1 < its lower bound 2 AND the
    // >= row is violated); x0 = (5, 10) is a hand-chosen feasible substitute (2<=5<=50, -50<=10<=50,
    // 10*5-10=40>=10).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct HS21Job : IJob
    {
        public NativeArray<double> Fail;
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.fProxyMat(2, 2); Q[0, 0] = 0.02f; Q[1, 1] = 2f;
            var c = arena.fProxyVec(2);
            var A = arena.fProxyMat(1, 2); A[0, 0] = 10f; A[0, 1] = -1f;
            var b = arena.fProxyVec(1); b[0] = 10f;
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp); senses[0] = ConstraintSense.GreaterEqual;
            var xl = arena.fProxyVec(2); xl[0] = 2f; xl[1] = -50f;
            var xu = arena.fProxyVec(2); xu[0] = 50f; xu[1] = 50f;
            var x = arena.fProxyVec(2); x[0] = 5f; x[1] = 10f;

            var info = QP.qpActiveSetCore(in Q, in c, in A, in b, senses, in xl, in xu, ref x, out double obj, 0);

            H.AssertTrue(Fail, 1, info.status == QPStatus.Optimal);
            H.AssertLE(Fail, 2, math.abs(obj - 0.04), /*+choose[5e-4|1e-9]*/5e-4/*-choose*/);
            H.AssertLE(Fail, 3, math.abs((double)x[0] - 2.0), /*+choose[5e-3|1e-7]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 4, math.abs((double)x[1] - 0.0), /*+choose[5e-3|1e-7]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 5, info.iterations, 20);

            senses.Dispose(); arena.Dispose();
        }
    }

    // HS35 (Hock & Schittkowski 1981, problem 35; SIF: A.R. Conn, April 1990;
    // github.com/lpoo/cutest-sif/blob/master/HS35.SIF). Elements expand to
    // f(x) = 2x1^2+2x2^2+x3^2+2x1x2+2x1x3 - 8x1-6x2-4x3 + 9, i.e. Q = [[4,2,2],[2,4,0],[2,0,2]],
    // c = (-8,-6,-4), offset +9 (SIF CONSTANTS). Constraint: x1+x2+2x3 <= 3 (CONSTANTS -3.0 on a
    // G-row of -x1-x2-2x3 -> -x1-x2-2x3+3>=0 -> x1+x2+2x3<=3). No BOUNDS section -> SIF default x>=0.
    // SOLTN comment 0.1111111111 = 1/9; our objective omits the +9 offset, so the acceptance value is
    // 1/9 - 9 = -80/9. x* = (4/3, 7/9, 4/9) (verified via scipy SLSQP, matches this problem's widely
    // published "Betts' function" solution). x0 = (0.5,0.5,0.5) is the SIF file's own START POINT
    // (XV 'DEFAULT' 0.5) -- feasible here (0.5+0.5+1=2<=3, x>=0), so used verbatim.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct HS35Job : IJob
    {
        public NativeArray<double> Fail;
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.fProxyMat(3, 3);
            Q[0, 0] = 4f; Q[0, 1] = 2f; Q[0, 2] = 2f;
            Q[1, 0] = 2f; Q[1, 1] = 4f; Q[1, 2] = 0f;
            Q[2, 0] = 2f; Q[2, 1] = 0f; Q[2, 2] = 2f;
            var c = arena.fProxyVec(3); c[0] = -8f; c[1] = -6f; c[2] = -4f;
            var A = arena.fProxyMat(1, 3); A[0, 0] = 1f; A[0, 1] = 1f; A[0, 2] = 2f;
            var b = arena.fProxyVec(1); b[0] = 3f;
            var senses = new NativeArray<ConstraintSense>(1, Allocator.Temp); senses[0] = ConstraintSense.LessEqual;
            var xl = arena.fProxyVec(3);
            var xu = arena.fProxyVec(3); xu[0] = 1e30f; xu[1] = 1e30f; xu[2] = 1e30f;
            var x = arena.fProxyVec(3); x[0] = 0.5f; x[1] = 0.5f; x[2] = 0.5f;

            var info = QP.qpActiveSetCore(in Q, in c, in A, in b, senses, in xl, in xu, ref x, out double obj, 0);
            double expected = 1.0 / 9.0 - 9.0;

            H.AssertTrue(Fail, 1, info.status == QPStatus.Optimal);
            H.AssertLE(Fail, 2, math.abs(obj - expected), /*+choose[5e-4|1e-9]*/5e-4/*-choose*/);
            H.AssertLE(Fail, 3, math.abs((double)x[0] - 4.0 / 3.0), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 4, math.abs((double)x[1] - 7.0 / 9.0), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 5, math.abs((double)x[2] - 4.0 / 9.0), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 6, info.iterations, 20);

            senses.Dispose(); arena.Dispose();
        }
    }

    // HS52 (Hock & Schittkowski 1981, problem 52; SIF: A.R. Conn, April 1990;
    // github.com/lpoo/cutest-sif/blob/master/HS52.SIF). f(x) = (4x1-x2)^2+(x2+x3-2)^2+(x4-1)^2+(x5-1)^2
    // expands to Q = 2*[[16,-4,0,0,0],[-4,2,1,0,0],[0,1,1,0,0],[0,0,0,1,0],[0,0,0,0,1]] i.e.
    // Q11=32,Q22=4,Q33=2,Q44=2,Q55=2,Q12=Q21=-8,Q23=Q32=2 (rest 0), c=(0,-4,-4,-2,-2), offset +6.
    // Constraints (all equality, all RHS 0): x1+3x2=0; x3+x4-2x5=0; x2-x5=0. All variables FREE
    // (BOUNDS: FR 'DEFAULT'). SOLTN comment 5.326643 is a TRUNCATED literature value; the EXACT
    // rational optimum (sympy KKT solve of this linear-equality QP) is x* =
    // (-33/349, 11/349, 180/349, -158/349, 11/349), f* = 1859/349 = 5.32664756446991404... (matches
    // scipy SLSQP to 1e-10 independently) -- used here instead of the rounded SOLTN comment. Our
    // objective omits the +6 offset: expected = 1859/349 - 6 = -235/349. x0 = 0 (all equality RHS are
    // 0, so the origin is trivially exactly feasible -- no SIF START POINT reuse needed/possible, since
    // that file's own default (2,2,2,2,2) violates all three equalities).
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct HS52Job : IJob
    {
        public NativeArray<double> Fail;
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.fProxyMat(5, 5);
            Q[0, 0] = 32f; Q[0, 1] = -8f;
            Q[1, 0] = -8f; Q[1, 1] = 4f; Q[1, 2] = 2f;
            Q[2, 1] = 2f; Q[2, 2] = 2f;
            Q[3, 3] = 2f;
            Q[4, 4] = 2f;
            var c = arena.fProxyVec(5); c[1] = -4f; c[2] = -4f; c[3] = -2f; c[4] = -2f;
            var A = arena.fProxyMat(3, 5);
            A[0, 0] = 1f; A[0, 1] = 3f;
            A[1, 2] = 1f; A[1, 3] = 1f; A[1, 4] = -2f;
            A[2, 1] = 1f; A[2, 4] = -1f;
            var b = arena.fProxyVec(3);
            var senses = new NativeArray<ConstraintSense>(3, Allocator.Temp);
            senses[0] = ConstraintSense.Equal; senses[1] = ConstraintSense.Equal; senses[2] = ConstraintSense.Equal;
            var xl = arena.fProxyVec(5); for (int i = 0; i < 5; i++) xl[i] = -1e30f;
            var xu = arena.fProxyVec(5); for (int i = 0; i < 5; i++) xu[i] = 1e30f;
            var x = arena.fProxyVec(5);   // origin: all three equalities are homogeneous (RHS 0)

            var info = QP.qpActiveSetCore(in Q, in c, in A, in b, senses, in xl, in xu, ref x, out double obj, 0);
            double expected = 1859.0 / 349.0 - 6.0;

            H.AssertTrue(Fail, 1, info.status == QPStatus.Optimal);
            H.AssertLE(Fail, 2, math.abs(obj - expected), /*+choose[5e-4|1e-8]*/5e-4/*-choose*/);
            H.AssertLE(Fail, 3, info.iterations, 20);

            senses.Dispose(); arena.Dispose();
        }
    }

    // HS76 (Hock & Schittkowski 1981, problem 76; SIF: A.R. Conn, March 1991;
    // github.com/lpoo/cutest-sif/blob/master/HS76.SIF). f(x)=x1^2+0.5x2^2+x3^2+0.5x4^2-x1x3+x3x4
    // -x1-3x2+x3-x4 expands to Q11=2,Q22=1,Q33=2,Q44=1,Q13=Q31=-1,Q34=Q43=1 (rest 0),
    // c=(-1,-3,1,-1), no offset. Constraints: x1+2x2+x3+x4<=5 (C1, CONSTANTS 5.0); 3x1+x2+2x3-x4<=4
    // (C2, CONSTANTS 4.0); x2+4x3>=1.5 (C3, CONSTANTS 1.5). No BOUNDS section -> SIF default x>=0.
    // This SIF mirror carries no SOLTN comment; optimum x* = (3/11, 23/11, 0, 6/11), f* = -103/22
    // (exact sympy KKT solve at the determined active set {C1 tight, x3's lower bound tight}; matches
    // scipy SLSQP AND the -4.6818... value widely cited for this problem in the SQP literature). x0 =
    // (0.5,0.5,0.5,0.5) is the SIF file's own START POINT -- feasible here (C1: 2.5<=5, C2: 2.5<=4,
    // C3: 2.5>=1.5, x>=0), used verbatim.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct HS76Job : IJob
    {
        public NativeArray<double> Fail;
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.fProxyMat(4, 4);
            Q[0, 0] = 2f; Q[0, 2] = -1f;
            Q[1, 1] = 1f;
            Q[2, 0] = -1f; Q[2, 2] = 2f; Q[2, 3] = 1f;
            Q[3, 2] = 1f; Q[3, 3] = 1f;
            var c = arena.fProxyVec(4); c[0] = -1f; c[1] = -3f; c[2] = 1f; c[3] = -1f;
            var A = arena.fProxyMat(3, 4);
            A[0, 0] = 1f; A[0, 1] = 2f; A[0, 2] = 1f; A[0, 3] = 1f;
            A[1, 0] = 3f; A[1, 1] = 1f; A[1, 2] = 2f; A[1, 3] = -1f;
            A[2, 1] = 1f; A[2, 2] = 4f;
            var b = arena.fProxyVec(3); b[0] = 5f; b[1] = 4f; b[2] = 1.5f;
            var senses = new NativeArray<ConstraintSense>(3, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.LessEqual; senses[2] = ConstraintSense.GreaterEqual;
            var xl = arena.fProxyVec(4);
            var xu = arena.fProxyVec(4); for (int i = 0; i < 4; i++) xu[i] = 1e30f;
            var x = arena.fProxyVec(4); for (int i = 0; i < 4; i++) x[i] = 0.5f;

            var info = QP.qpActiveSetCore(in Q, in c, in A, in b, senses, in xl, in xu, ref x, out double obj, 0);
            double expected = -103.0 / 22.0;

            H.AssertTrue(Fail, 1, info.status == QPStatus.Optimal);
            H.AssertLE(Fail, 2, math.abs(obj - expected), /*+choose[5e-4|1e-8]*/5e-4/*-choose*/);
            H.AssertLE(Fail, 3, math.abs((double)x[0] - 3.0 / 11.0), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 4, math.abs((double)x[1] - 23.0 / 11.0), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 5, math.abs((double)x[2] - 0.0), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 6, math.abs((double)x[3] - 6.0 / 11.0), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 7, info.iterations, 20);

            senses.Dispose(); arena.Dispose();
        }
    }

    [Test] public void HS21() => H.Run(fail => new HS21Job { Fail = fail }.Run());
    [Test] public void HS35() => H.Run(fail => new HS35Job { Fail = fail }.Run());
    [Test] public void HS52() => H.Run(fail => new HS52Job { Fail = fail }.Run());
    [Test] public void HS76() => H.Run(fail => new HS76Job { Fail = fail }.Run());

    // ================================================================================================
    // (b) Brute-force oracle: random SPD box-constrained QP, n <= 8, enumerate all 3^n active-set
    // combinations (each variable free/lower/upper), solve each EQP exactly, take the best feasible.
    // ================================================================================================
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BruteForceJob : IJob
    {
        public int N, Seed;
        public NativeArray<double> Fail;
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random((uint)Seed | 1u);
            int n = N;

            var Q = arena.fProxyMat(n, n);
            Rand.spdInPlace(ref rng, ref Q, 1f, 10f);
            var c = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) c[i] = rng.NextFProxy(-1f, 1f);
            var xl = arena.fProxyVec(n);
            var xu = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) { xl[i] = rng.NextFProxy(-3f, -1f); xu[i] = rng.NextFProxy(1f, 3f); }
            var x0 = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x0[i] = 0.5f * (xl[i] + xu[i]);   // box midpoint, trivially feasible

            var A = arena.fProxyMat(0, n);
            var b = arena.fProxyVec(0);
            var senses = new NativeArray<ConstraintSense>(0, Allocator.Temp);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = x0[i];
            var info = QP.qpActiveSetCore(in Q, in c, in A, in b, senses, in xl, in xu, ref x, out double obj, 0);
            H.AssertTrue(Fail, 1, info.status == QPStatus.Optimal);

            // ---- brute force over 3^n active-set assignments ----
            int combos = 1; for (int i = 0; i < n; i++) combos *= 3;
            double bestObj = double.PositiveInfinity;
            var digit = new NativeArray<int>(n, Allocator.Temp);
            var xEq = arena.fProxyVec(n, true);
            var Qcopy = arena.fProxyMat(n, n, true);
            var rhs = arena.fProxyVec(n, true);

            for (int combo = 0; combo < combos; combo++)
            {
                int rem = combo, k = 0;
                for (int i = 0; i < n; i++) { digit[i] = rem % 3; rem /= 3; if (digit[i] != 0) k++; }

                bool feasibleCandidate;
                if (k == 0)
                {
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            Qcopy[i, j] = Q[i, j];
                    for (int i = 0; i < n; i++) rhs[i] = -c[i];
                    var choInfo = CHO.solveInPlace(ref Qcopy, ref rhs);
                    feasibleCandidate = choInfo.Solved;
                    if (feasibleCandidate)
                        for (int i = 0; i < n; i++) xEq[i] = rhs[i];
                }
                else
                {
                    var AWv = new fProxyMxN(k, n, Allocator.Temp);   // k x n, zero-init
                    var bWv = new fProxyN(k, Allocator.Temp);
                    var lamWv = new fProxyN(k, Allocator.Temp);      // eqpSolve requires lambda.N == A_W.M_Rows == k
                    int kk = 0;
                    for (int i = 0; i < n; i++)
                    {
                        if (digit[i] == 0) continue;
                        AWv[kk, i] = 1f;
                        bWv[kk] = digit[i] == 1 ? xl[i] : xu[i];
                        kk++;
                    }
                    var eqInfo = QP.eqpSolve(in Q, in c, in AWv, in bWv, ref xEq, ref lamWv);
                    feasibleCandidate = eqInfo.status == QPStatus.Optimal;
                    AWv.Dispose(); bWv.Dispose(); lamWv.Dispose();
                }

                if (feasibleCandidate)
                {
                    for (int i = 0; i < n; i++)
                        if ((double)xEq[i] < (double)xl[i] - /*+choose[1e-4|1e-9]*/1e-4/*-choose*/
                            || (double)xEq[i] > (double)xu[i] + /*+choose[1e-4|1e-9]*/1e-4/*-choose*/)
                        { feasibleCandidate = false; break; }
                }

                if (feasibleCandidate)
                {
                    double o = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double qxi = 0;
                        for (int j = 0; j < n; j++) qxi += (double)Q[i, j] * (double)xEq[j];
                        o += 0.5 * (double)xEq[i] * qxi + (double)c[i] * (double)xEq[i];
                    }
                    if (o < bestObj) bestObj = o;
                }
            }

            H.AssertTrue(Fail, 2, !double.IsPositiveInfinity(bestObj));
            H.AssertLE(Fail, 3, math.abs(obj - bestObj), /*+choose[2e-3|1e-8]*/2e-3/*-choose*/ * (1.0 + math.abs(bestObj)));

            digit.Dispose(); senses.Dispose(); arena.Dispose();
        }
    }

    static IEnumerable<TestCaseData> BruteForceCases()
    {
        int[] ns = { 2, 3, 5, 8 };
        int[] seeds = { 12345, 67890 };
        foreach (int n in ns)
            foreach (int s in seeds)
                yield return new TestCaseData(n, s).SetName($"BruteForce_n{n}_s{s}");
    }

    [TestCaseSource(nameof(BruteForceCases))]
    public void BruteForce(int n, int seed) => H.Run(fail => new BruteForceJob { N = n, Seed = seed, Fail = fail }.Run());

    // ================================================================================================
    // (c) LP limit: Q = 0 reproduces LP.solve's objective on a random dense feasible LP.
    // ================================================================================================
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LpLimitJob : IJob
    {
        public int N, Seed;
        public NativeArray<double> Fail;
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = N, m = n / 2;
            var rng = new Random((uint)Seed | 1u);

            var A = arena.fProxyMat(m, n);
            for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) A[i, j] = rng.NextFProxy(0f, 1f);
            var x0 = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x0[i] = rng.NextFProxy(0f, 1f);
            var Ax0 = arena.fProxyVec(m);
            Blas.dot(in A, in x0, ref Ax0);
            var b = arena.fProxyVec(m);
            for (int i = 0; i < m; i++) b[i] = Ax0[i] + rng.NextFProxy(0.1f, 1f);
            var c = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) c[i] = rng.NextFProxy(-1f, 1f);
            var senses = new NativeArray<ConstraintSense>(m, Allocator.Temp);
            for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;

            var Q = arena.fProxyMat(n, n);   // zero
            var xl = arena.fProxyVec(n);
            var xu = arena.fProxyVec(n); for (int i = 0; i < n; i++) xu[i] = 1e30f;

            var xQp = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) xQp[i] = x0[i];
            var qpInfo = QP.qpActiveSetCore(in Q, in c, in A, in b, senses, in xl, in xu, ref xQp, out double qpObj, 0);

            var xLp = arena.fProxyVec(n);
            var lpInfo = LP.solve(in A, in b, in c, senses, ref xLp, out double lpObj);

            H.AssertTrue(Fail, 1, qpInfo.status == QPStatus.Optimal);
            H.AssertTrue(Fail, 2, lpInfo.Solved);
            H.AssertLE(Fail, 3, math.abs(qpObj - lpObj), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/ * (1.0 + math.abs(lpObj)));

            senses.Dispose(); arena.Dispose();
        }
    }

    static IEnumerable<TestCaseData> LpLimitCases()
    {
        int[] ns = { 8, 16, 24 };
        int[] seeds = { 111, 222 };
        foreach (int n in ns) foreach (int s in seeds) yield return new TestCaseData(n, s).SetName($"LpLimit_n{n}_s{s}");
    }

    [TestCaseSource(nameof(LpLimitCases))]
    public void LpLimit(int n, int seed) => H.Run(fail => new LpLimitJob { N = n, Seed = seed, Fail = fail }.Run());

    // ================================================================================================
    // (d) Degenerate: a verbatim-duplicated tight constraint row must not change the answer.
    // ================================================================================================
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct DegenerateJob : IJob
    {
        public int Seed;
        public NativeArray<double> Fail;
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random((uint)Seed | 1u);
            int n = 3;
            var Q = arena.fProxyMat(n, n);
            Rand.spdInPlace(ref rng, ref Q, 1f, 5f);
            var c = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) c[i] = rng.NextFProxy(-1f, 1f);
            var xl = arena.fProxyVec(n);
            var xu = arena.fProxyVec(n); for (int i = 0; i < n; i++) xu[i] = 5f;
            var x0 = arena.fProxyVec(n); x0[0] = 2f; x0[1] = 2f; x0[2] = 2f;   // sum = 6, tight below

            // single copy
            var A1 = arena.fProxyMat(1, n); A1[0, 0] = 1f; A1[0, 1] = 1f; A1[0, 2] = 1f;
            var b1 = arena.fProxyVec(1); b1[0] = 6f;
            var senses1 = new NativeArray<ConstraintSense>(1, Allocator.Temp); senses1[0] = ConstraintSense.LessEqual;
            var x1 = arena.fProxyVec(n); x1[0] = x0[0]; x1[1] = x0[1]; x1[2] = x0[2];
            var info1 = QP.qpActiveSetCore(in Q, in c, in A1, in b1, senses1, in xl, in xu, ref x1, out double obj1, 0);

            // duplicated (verbatim second copy of the same row)
            var A2 = arena.fProxyMat(2, n);
            A2[0, 0] = 1f; A2[0, 1] = 1f; A2[0, 2] = 1f;
            A2[1, 0] = 1f; A2[1, 1] = 1f; A2[1, 2] = 1f;
            var b2 = arena.fProxyVec(2); b2[0] = 6f; b2[1] = 6f;
            var senses2 = new NativeArray<ConstraintSense>(2, Allocator.Temp);
            senses2[0] = ConstraintSense.LessEqual; senses2[1] = ConstraintSense.LessEqual;
            var x2 = arena.fProxyVec(n); x2[0] = x0[0]; x2[1] = x0[1]; x2[2] = x0[2];
            var info2 = QP.qpActiveSetCore(in Q, in c, in A2, in b2, senses2, in xl, in xu, ref x2, out double obj2, 0);

            H.AssertTrue(Fail, 1, info1.status == QPStatus.Optimal);
            H.AssertTrue(Fail, 2, info2.status == QPStatus.Optimal);
            H.AssertLE(Fail, 3, math.abs(obj1 - obj2), /*+choose[5e-4|1e-9]*/5e-4/*-choose*/ * (1.0 + math.abs(obj1)));
            for (int i = 0; i < n; i++)
                H.AssertLE(Fail, 4 + i, math.abs((double)x1[i] - (double)x2[i]), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);

            senses1.Dispose(); senses2.Dispose(); arena.Dispose();
        }
    }

    static IEnumerable<TestCaseData> DegenerateCases()
    {
        int[] seeds = { 111, 222, 333 };
        foreach (int s in seeds) yield return new TestCaseData(s).SetName($"Degenerate_s{s}");
    }

    [TestCaseSource(nameof(DegenerateCases))]
    public void Degenerate(int seed) => H.Run(fail => new DegenerateJob { Seed = seed, Fail = fail }.Run());

    // ================================================================================================
    // (e) Unbounded: Q = diag(1,0), c = (0,-1), no general constraints, x2 >= 0 unbounded above,
    // x1 free. f = 1/2 x1^2 - x2 -> -infinity as x2 -> +infinity with x1 = 0. See this file's header
    // comment for the hand-traced iteration path.
    // ================================================================================================
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct UnboundedJob : IJob
    {
        public NativeArray<double> Fail;
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.fProxyMat(2, 2); Q[0, 0] = 1f; Q[1, 1] = 0f;
            var c = arena.fProxyVec(2); c[1] = -1f;
            var A = arena.fProxyMat(0, 2);
            var b = arena.fProxyVec(0);
            var senses = new NativeArray<ConstraintSense>(0, Allocator.Temp);
            var xl = arena.fProxyVec(2); xl[0] = -1e30f; xl[1] = 0f;
            var xu = arena.fProxyVec(2); xu[0] = 1e30f; xu[1] = 1e30f;
            var x = arena.fProxyVec(2);   // (0,0), feasible

            var info = QP.qpActiveSetCore(in Q, in c, in A, in b, senses, in xl, in xu, ref x, out double obj, 0);

            H.AssertTrue(Fail, 1, info.status == QPStatus.Unbounded);

            senses.Dispose(); arena.Dispose();
        }
    }

    [Test] public void Unbounded() => H.Run(fail => new UnboundedJob { Fail = fail }.Run());

    // ---- shared test-side helpers (Fail[]-array Burst diagnostic pattern, see QPEqpTests.fProxy.cs) ----
    static class H
    {
        public static void AssertTrue(NativeArray<double> fail, int id, bool cond)
        {
            if (!cond && fail[0] == 0) { fail[0] = 1; fail[1] = id; fail[2] = 0; fail[3] = 1; fail[4] = 0; }
            Assert.IsTrue(cond);
        }
        public static void AssertLE(NativeArray<double> fail, int id, double val, double limit)
        {
            bool ok = val <= limit;
            if (!ok && fail[0] == 0) { fail[0] = 1; fail[1] = id; fail[2] = val; fail[3] = limit; fail[4] = val - limit; }
            Assert.IsTrue(ok);
        }

        // Managed-side orchestration only (job construction + .Run() + result check) -- NOT executed
        // inside Burst, so an Action closure is fine here even though the job's OWN Execute() must stay
        // Burst-legal (see AssertTrue/AssertLE above, called FROM Execute()).
        public static void Run(Action<NativeArray<double>> runJob)
        {
            var fail = new NativeArray<double>(5, Allocator.TempJob);
            try
            {
                runJob(fail);
                if (fail[0] != 0)
                    Assert.Fail($"check {fail[1]}: got {fail[2]:G6}, limit/expected {fail[3]:G6}, diff {fail[4]:G6}");
            }
            finally { fail.Dispose(); }
        }
    }
}
