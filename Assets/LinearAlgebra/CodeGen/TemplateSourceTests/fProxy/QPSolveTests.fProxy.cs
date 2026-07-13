using System;
using System.Collections.Generic;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

// Acceptance battery for the PUBLIC convex-QP facade QP.solve (phase-1 feasible start + HiGHS-style
// bound-perturbation hardening). QP.eqpSolve / QP.qpActiveSetCore are covered by
// QPEqpTests.fProxy.cs / QPActiveSetTests.fProxy.cs; this file adds ONLY the facade surface. Reached
// via the InternalsVisibleTo grants on both BurstLinearAlgebra.Tests and
// BurstLinearAlgebra.TemplateSource.Tests-firstpass (TemplateSource/AssemblyInfo.cs) -- needed here
// for the ill-conditioning / degeneracy cases that call the internal QP.qpActiveSetCore directly, to
// isolate the hardening path from phase-1's own LP as a confound.
//
// Burst execution / Fail[] diagnostic pattern is identical to QPActiveSetTests.fProxy.cs (see its
// header): compute in [BurstCompile(CompileSynchronously = true)] IJob structs, Assert.IsTrue with ==
// only inside the job, first failure recorded into a Fail[] array read back managed-side. The ONE
// exception is the validation-throw tests (Q asymmetry / dimension mismatch / xl>xu), which are plain
// [Test] methods calling QP.solve directly from managed code -- Assert.Throws must catch the
// ArgumentException on the managed side, and Arena / fProxyMxN are usable from the main thread outside
// a job (the arena authoring tier). Those three are double-only in this suite (the validation logic
// they exercise does not depend on numeric precision) -- preserved as double-only here via a
// skipFor(float) gate rather than adding new float coverage.
//
// ---- Acceptance oracles ----
//
// (1) FACADE end-to-end, phase 1 proven: the four Hock-Schittkowski knowns HS21/HS35/HS52/HS76 (SAME
//     Q/c/A/b/xl/xu literals and expected optima/tolerances as QPActiveSetTests.fProxy.cs -- see that
//     file's header for the full CUTEst-SIF / scipy-SLSQP / exact-KKT provenance of each) are
//     re-solved through the PUBLIC QP.solve, with x deliberately filled with GARBAGE on entry.
//     QPActiveSetTests hands qpActiveSetCore a hand-chosen feasible x0; here the facade must IGNORE
//     the incoming x and manufacture its own feasible start via PhaseOneFeasibleStart -- so a correct
//     answer is direct proof phase 1 works for GreaterEqual+box (HS21), LessEqual+x>=0 (HS35),
//     all-equality-free (HS52), and mixed <=/>=/x>=0 (HS76). Same status/objective/x tolerances as
//     the qpActiveSetCore versions.
//
// (2) INFEASIBLE -> QPStatus.Infeasible, objective == 0: contradictory rows (sum(x) <= 1 AND
//     sum(x) >= 11 over a box) make an empty feasible region; phase 1's LP reports infeasible and the
//     facade maps that straight to QPStatus.Infeasible with x zeroed and objective 0 (the documented
//     Infeasible contract). The same contradictory-inequality recipe LPBenchmark Section 5 uses.
//
// (3) NO-BOUNDS overload, closed-form unconstrained optimum: Q = diag(2,2), c = (-2,-4), no rows, no
//     bounds -> f = x1^2 + 2x2^2 - 2x1 - 4x2, unique unconstrained minimum at x* = (1,2), f* = -5
//     (grad Qx + c = 0 => (2,4)-scaled solve). Exercises the box-free convenience overload AND phase
//     1's mprime==0 early-return path (all variables free, no rows -> trivially feasible at 0).
//
// (4) VALIDATION throws (ArgumentException, managed [Test] + Assert.Throws): a Q whose asymmetry
//     exceeds the scaled symmetry tolerance (the v1 convexity-contract check -- the ONE place symmetry
//     is verified), a c/x dimension mismatch, and xl > xu componentwise. Cheap coverage of the facade's
//     up-front validation surface.
//
// (5) ILL-CONDITIONED stress (hardening path, via the PUBLIC facade QP.solve): Q = Rand.spdInPlace with
//     condition = maxEig/minEig up to ~1e6 (float) / ~1e12 (double) at n in {16, 32}, m = n/2 random
//     LessEqual rows + a wide box, feasible start x0 (LpLimit-style construction). Asserts Optimal with
//     residuals within tolerances SCALED the way the solver's own internal zeroThreshold is:
//     feasibilityResidual is essentially machine-precision-scale regardless of conditioning (tight,
//     lightly A-norm-scaled tol), but stationarityResidual is reported in ABSOLUTE terms and scales
//     with ||Q||_inf == maxEig, so its tolerance is relTol * maxEig (a fixed absolute tol would
//     spuriously fail at high conditioning).
//
// (6) HEAVY DEGENERACY / no-stall (hardening path, via qpActiveSetCore directly): a strictly convex
//     instance (Q = I) whose optimum x0 = (1/2,...,1/2) is a vertex where TWO independent LessEqual
//     rows are simultaneously tight, EACH DUPLICATED 40x (80 rows total) so the ratio test faces a
//     massive exact tie every time it approaches that vertex -- precisely the zero-length-step
//     degeneracy the HiGHS-style bound perturbation exists to break. Started from a feasible NON-optimal
//     point (the origin) so the solver genuinely traverses the loop into the degenerate vertex rather
//     than starting already-optimal. The optimum is known in CLOSED FORM by construction (c chosen as
//     c = -Q x0 - r1 - r2 makes x0 a KKT point with both multipliers = 1 >= 0, hence the unique convex
//     optimum), so the objective oracle f* = 1/2 x0^T Q x0 + c^T x0 = -5 is exact and
//     minimizer-independent. Asserts Optimal (NOT MaxIterations), obj == f*, x ~ x0, tiny feasibility,
//     and an iteration count far under the budget. NOTE: this provably stresses the redundant-row rank
//     guard and the loop's robustness under massive simultaneous binding; whether degenCount actually
//     crosses degenCap (= 3n) to ENGAGE perturbation on this particular instance is not directly
//     observable (there is no public "did perturbation fire" flag -- by design the final x/objective are
//     identical whether it fired or not), so this is the strongest available oracle attached to this
//     behavioral/no-stall proxy.
public class fProxyQPSolveTests
{
    // ================================================================================================
    // (1) Facade + phase 1: the four HS knowns through the PUBLIC QP.solve, x = garbage on entry.
    // ================================================================================================

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct HS21FacadeJob : IJob
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
            var x = arena.fProxyVec(2); x[0] = -999f; x[1] = 777f;   // garbage: facade must ignore it

            var info = QP.solve(in Q, in c, in A, in b, in senses, in xl, in xu, ref x, out double obj, 0);

            H.AssertTrue(Fail, 1, info.status == QPStatus.Optimal);
            H.AssertLE(Fail, 2, math.abs(obj - 0.04), /*+choose[5e-4|1e-9]*/5e-4/*-choose*/);
            H.AssertLE(Fail, 3, math.abs((double)x[0] - 2.0), /*+choose[5e-3|1e-7]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 4, math.abs((double)x[1] - 0.0), /*+choose[5e-3|1e-7]*/5e-3/*-choose*/);

            senses.Dispose(); arena.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct HS35FacadeJob : IJob
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
            var x = arena.fProxyVec(3); x[0] = -5f; x[1] = 42f; x[2] = -1f;   // garbage

            var info = QP.solve(in Q, in c, in A, in b, in senses, in xl, in xu, ref x, out double obj, 0);
            double expected = 1.0 / 9.0 - 9.0;

            H.AssertTrue(Fail, 1, info.status == QPStatus.Optimal);
            H.AssertLE(Fail, 2, math.abs(obj - expected), /*+choose[5e-4|1e-9]*/5e-4/*-choose*/);
            H.AssertLE(Fail, 3, math.abs((double)x[0] - 4.0 / 3.0), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 4, math.abs((double)x[1] - 7.0 / 9.0), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 5, math.abs((double)x[2] - 4.0 / 9.0), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);

            senses.Dispose(); arena.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct HS52FacadeJob : IJob
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
            var x = arena.fProxyVec(5); for (int i = 0; i < 5; i++) x[i] = 13f;   // garbage (violates all 3 equalities)

            var info = QP.solve(in Q, in c, in A, in b, in senses, in xl, in xu, ref x, out double obj, 0);
            double expected = 1859.0 / 349.0 - 6.0;

            H.AssertTrue(Fail, 1, info.status == QPStatus.Optimal);
            H.AssertLE(Fail, 2, math.abs(obj - expected), /*+choose[5e-4|1e-8]*/5e-4/*-choose*/);

            senses.Dispose(); arena.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct HS76FacadeJob : IJob
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
            var x = arena.fProxyVec(4); for (int i = 0; i < 4; i++) x[i] = -100f;   // garbage (violates x>=0)

            var info = QP.solve(in Q, in c, in A, in b, in senses, in xl, in xu, ref x, out double obj, 0);
            double expected = -103.0 / 22.0;

            H.AssertTrue(Fail, 1, info.status == QPStatus.Optimal);
            H.AssertLE(Fail, 2, math.abs(obj - expected), /*+choose[5e-4|1e-8]*/5e-4/*-choose*/);
            H.AssertLE(Fail, 3, math.abs((double)x[0] - 3.0 / 11.0), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 4, math.abs((double)x[1] - 23.0 / 11.0), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 5, math.abs((double)x[2] - 0.0), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 6, math.abs((double)x[3] - 6.0 / 11.0), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);

            senses.Dispose(); arena.Dispose();
        }
    }

    [Test] public void HS21_Facade() => H.Run(fail => new HS21FacadeJob { Fail = fail }.Run());
    [Test] public void HS35_Facade() => H.Run(fail => new HS35FacadeJob { Fail = fail }.Run());
    [Test] public void HS52_Facade() => H.Run(fail => new HS52FacadeJob { Fail = fail }.Run());
    [Test] public void HS76_Facade() => H.Run(fail => new HS76FacadeJob { Fail = fail }.Run());

    // ================================================================================================
    // (2) Infeasible: contradictory rows (sum(x) <= 1 AND sum(x) >= 11) -> QPStatus.Infeasible, obj 0.
    // ================================================================================================
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct InfeasibleJob : IJob
    {
        public NativeArray<double> Fail;
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 3;
            var Q = arena.fProxyMat(n, n); for (int i = 0; i < n; i++) Q[i, i] = 1f;
            var c = arena.fProxyVec(n);
            var A = arena.fProxyMat(2, n);
            for (int j = 0; j < n; j++) { A[0, j] = 1f; A[1, j] = 1f; }
            var b = arena.fProxyVec(2); b[0] = 1f; b[1] = 11f;
            var senses = new NativeArray<ConstraintSense>(2, Allocator.Temp);
            senses[0] = ConstraintSense.LessEqual; senses[1] = ConstraintSense.GreaterEqual;
            var xl = arena.fProxyVec(n);                          // 0
            var xu = arena.fProxyVec(n); for (int j = 0; j < n; j++) xu[j] = 10f;
            var x = arena.fProxyVec(n); for (int j = 0; j < n; j++) x[j] = 3f;

            var info = QP.solve(in Q, in c, in A, in b, in senses, in xl, in xu, ref x, out double obj, 0);

            H.AssertTrue(Fail, 1, info.status == QPStatus.Infeasible);
            H.AssertLE(Fail, 2, math.abs(obj), 0.0);
            for (int j = 0; j < n; j++) H.AssertLE(Fail, 3, math.abs((double)x[j]), 0.0);   // x zeroed

            senses.Dispose(); arena.Dispose();
        }
    }

    [Test] public void Infeasible() => H.Run(fail => new InfeasibleJob { Fail = fail }.Run());

    // ================================================================================================
    // (3) No-bounds overload: Q = diag(2,2), c = (-2,-4), no rows/bounds -> min at (1,2), f* = -5.
    // ================================================================================================
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct NoBoundsJob : IJob
    {
        public NativeArray<double> Fail;
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.fProxyMat(2, 2); Q[0, 0] = 2f; Q[1, 1] = 2f;
            var c = arena.fProxyVec(2); c[0] = -2f; c[1] = -4f;
            var A = arena.fProxyMat(0, 2);
            var b = arena.fProxyVec(0);
            var senses = new NativeArray<ConstraintSense>(0, Allocator.Temp);
            var x = arena.fProxyVec(2); x[0] = 55f; x[1] = -9f;   // garbage

            var info = QP.solve(in Q, in c, in A, in b, in senses, ref x, out double obj, 0);

            H.AssertTrue(Fail, 1, info.status == QPStatus.Optimal);
            H.AssertLE(Fail, 2, math.abs(obj - (-5.0)), /*+choose[5e-5|1e-10]*/5e-5/*-choose*/);
            H.AssertLE(Fail, 3, math.abs((double)x[0] - 1.0), /*+choose[5e-4|1e-9]*/5e-4/*-choose*/);
            H.AssertLE(Fail, 4, math.abs((double)x[1] - 2.0), /*+choose[5e-4|1e-9]*/5e-4/*-choose*/);

            senses.Dispose(); arena.Dispose();
        }
    }

    [Test] public void NoBounds() => H.Run(fail => new NoBoundsJob { Fail = fail }.Run());

    // ================================================================================================
    // (4) Validation throws -- managed [Test] + Assert.Throws (the exception must be caught managed-side).
    // Double-only (see file header): the validation logic under test does not depend on numeric
    // precision, so the original hand-written suite covered it once; preserved via the skipFor(float)
    // gate below.
    // ================================================================================================
    //+skipFor[float]
    [Test]
    public void Solve_AsymmetricQ_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var Q = arena.fProxyMat(2, 2);
            Q[0, 0] = 1f; Q[1, 1] = 1f; Q[0, 1] = 1f; Q[1, 0] = -1f;   // |Q01-Q10| = 2 >> scaled symTol
            var c = arena.fProxyVec(2);
            var A = arena.fProxyMat(0, 2);
            var b = arena.fProxyVec(0);
            var senses = new NativeArray<ConstraintSense>(0, Allocator.Temp);
            var x = arena.fProxyVec(2);
            Assert.Throws<ArgumentException>(() =>
                QP.solve(in Q, in c, in A, in b, in senses, ref x, out double _, 0));
            senses.Dispose();
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Solve_DimensionMismatch_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var Q = arena.fProxyMat(2, 2); Q[0, 0] = 1f; Q[1, 1] = 1f;
            var c = arena.fProxyVec(3);   // wrong: should be length 2
            var A = arena.fProxyMat(0, 2);
            var b = arena.fProxyVec(0);
            var senses = new NativeArray<ConstraintSense>(0, Allocator.Temp);
            var x = arena.fProxyVec(2);
            Assert.Throws<ArgumentException>(() =>
                QP.solve(in Q, in c, in A, in b, in senses, ref x, out double _, 0));
            senses.Dispose();
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Solve_LowerAboveUpper_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var Q = arena.fProxyMat(2, 2); Q[0, 0] = 1f; Q[1, 1] = 1f;
            var c = arena.fProxyVec(2);
            var A = arena.fProxyMat(0, 2);
            var b = arena.fProxyVec(0);
            var senses = new NativeArray<ConstraintSense>(0, Allocator.Temp);
            var xl = arena.fProxyVec(2); xl[0] = 5f; xl[1] = 0f;
            var xu = arena.fProxyVec(2); xu[0] = 1f; xu[1] = 1f;   // xl[0] > xu[0]
            var x = arena.fProxyVec(2);
            Assert.Throws<ArgumentException>(() =>
                QP.solve(in Q, in c, in A, in b, in senses, in xl, in xu, ref x, out double _, 0));
            senses.Dispose();
        }
        finally { arena.Dispose(); }
    }
    //-skipFor

    // ================================================================================================
    // (5) Ill-conditioned stress through the PUBLIC facade QP.solve (x = garbage on entry -> phase 1
    // finds its own start; this reproduces the exact path the tolerance calibration below was measured
    // on -- see file header criterion (5)).
    // ================================================================================================
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct IllConditionedJob : IJob
    {
        public int N, Seed;
        public double MaxEig;   // minEig fixed at 1 -> condition == MaxEig
        public NativeArray<double> Fail;
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = N, m = n / 2;
            var rng = new Random((uint)Seed | 1u);

            var Q = arena.fProxyMat(n, n);
            Rand.spdInPlace(ref rng, ref Q, 1f, (fProxy)MaxEig);
            var c = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) c[i] = rng.NextFProxy(-1f, 1f);

            var A = arena.fProxyMat(m, n);
            for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) A[i, j] = rng.NextFProxy(0f, 1f);
            var x0 = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x0[i] = rng.NextFProxy(0.2f, 0.8f);
            var Ax0 = arena.fProxyVec(m);
            Blas.dot(in A, in x0, ref Ax0);
            var b = arena.fProxyVec(m);
            for (int i = 0; i < m; i++) b[i] = Ax0[i] + rng.NextFProxy(0.1f, 1f);   // x0 strictly feasible
            var senses = new NativeArray<ConstraintSense>(m, Allocator.Temp);
            for (int i = 0; i < m; i++) senses[i] = ConstraintSense.LessEqual;

            var xl = arena.fProxyVec(n);                                   // 0 (x0 in [0.2,0.8] > 0)
            var xu = arena.fProxyVec(n); for (int i = 0; i < n; i++) xu[i] = n;   // wide box holds x0

            var x = arena.fProxyVec(n); for (int i = 0; i < n; i++) x[i] = -321f;   // garbage; facade ignores it

            var info = QP.solve(in Q, in c, in A, in b, in senses, in xl, in xu, ref x, out double _, 0);

            // feasibility residual: essentially machine-precision-scale regardless of conditioning
            // (tight, non-scaled). stationarity residual: reported ABSOLUTE, scales with ||Q||_inf ==
            // MaxEig, so a RELATIVE tolerance (relTol * MaxEig) -- a fixed absolute tol would
            // spuriously fail at high conditioning.
            H.AssertTrue(Fail, 1, info.status == QPStatus.Optimal);
            H.AssertLE(Fail, 2, info.feasibilityResidual, /*+choose[1e-3|1e-8]*/1e-3/*-choose*/);
            H.AssertLE(Fail, 3, info.stationarityResidual, /*+choose[1e-2|1e-6]*/1e-2/*-choose*/ * MaxEig);

            senses.Dispose(); arena.Dispose();
        }
    }

    static IEnumerable<TestCaseData> IllConditionedCases()
    {
        // condition up to ~1e6 (float) / ~1e12 (double), at n = 16 and n = 32, a couple of seeds.
        double cond1 = /*+choose[1e4|1e8]*/1e4/*-choose*/;
        double cond2 = /*+choose[1e6|1e12]*/1e6/*-choose*/;
        yield return new TestCaseData(16, 101, cond1).SetName($"IllCond_n16_cond{cond1}");
        yield return new TestCaseData(16, 202, cond2).SetName($"IllCond_n16_cond{cond2}");
        yield return new TestCaseData(32, 303, cond2).SetName($"IllCond_n32_cond{cond2}");
    }

    [TestCaseSource(nameof(IllConditionedCases))]
    public void IllConditioned(int n, int seed, double maxEig) =>
        H.Run(fail => new IllConditionedJob { N = n, Seed = seed, MaxEig = maxEig, Fail = fail }.Run());

    // ================================================================================================
    // (6) Heavy degeneracy / no-stall: optimum x0 = (1/2,...) with TWO independent rows each duplicated
    // 40x, all tight at x0; started from the origin. Closed-form optimum obj = -5. See file header.
    // ================================================================================================
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct HeavyDegeneracyJob : IJob
    {
        public NativeArray<double> Fail;
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 8, half = 4, K = 40, m = 2 * K;
            var Q = arena.fProxyMat(n, n); for (int i = 0; i < n; i++) Q[i, i] = 1f;
            var c = arena.fProxyVec(n); for (int i = 0; i < n; i++) c[i] = -1.5f;   // c = -Q x0 - r1 - r2

            // Rows: first K copies of r1 = indicator(first half); next K copies of r2 = indicator(last half).
            // Both LessEqual with b = r.x0 = 2 (x0 = all 0.5). All 2K rows tight at x0.
            var A = arena.fProxyMat(m, n);
            var b = arena.fProxyVec(m);
            var senses = new NativeArray<ConstraintSense>(m, Allocator.Temp);
            for (int r = 0; r < K; r++)
            {
                for (int j = 0; j < half; j++) A[r, j] = 1f;
                b[r] = 2f; senses[r] = ConstraintSense.LessEqual;
            }
            for (int r = 0; r < K; r++)
            {
                for (int j = half; j < n; j++) A[K + r, j] = 1f;
                b[K + r] = 2f; senses[K + r] = ConstraintSense.LessEqual;
            }
            var xl = arena.fProxyVec(n); for (int j = 0; j < n; j++) xl[j] = -1f;
            var xu = arena.fProxyVec(n); for (int j = 0; j < n; j++) xu[j] = 1f;   // x0 = 0.5 interior, bounds inactive
            var x = arena.fProxyVec(n);   // origin: r1.0 = 0 <= 2, r2.0 = 0 <= 2, in box -> feasible, NOT optimal

            var info = QP.qpActiveSetCore(in Q, in c, in A, in b, senses, in xl, in xu, ref x, out double obj, 1000);

            H.AssertTrue(Fail, 1, info.status == QPStatus.Optimal);   // NOT MaxIterations
            H.AssertLE(Fail, 2, math.abs(obj - (-5.0)), /*+choose[5e-3|1e-8]*/5e-3/*-choose*/);
            for (int j = 0; j < n; j++) H.AssertLE(Fail, 3, math.abs((double)x[j] - 0.5), /*+choose[5e-3|1e-6]*/5e-3/*-choose*/);
            H.AssertLE(Fail, 4, info.feasibilityResidual, /*+choose[5e-4|1e-8]*/5e-4/*-choose*/);
            H.AssertLE(Fail, 5, info.iterations, 300);                // far under the 1000 budget

            senses.Dispose(); arena.Dispose();
        }
    }

    [Test] public void HeavyDegeneracy() => H.Run(fail => new HeavyDegeneracyJob { Fail = fail }.Run());

    // ---- shared test-side helpers (Fail[]-array Burst diagnostic pattern, see QPActiveSetTests.fProxy.cs) ----
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
