using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// BASIC smoke tests for Control.lqr / Control.lqrSchedule -- written by the coder agent alongside the
// implementation. The FULL test battery (literature
// vectors, SDA-vs-oracle cross-check, property-based stability/PSD checks, warm-path perturbation
// convergence, redundant-actuator rank flagging, determinism) is the test-writer agent's job; this file
// only checks that a known tiny instance solves, statuses fire (Converged/Diverged), and throws throw.
public class doubleControlTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            ColdConverges,          // double integrator (A=[[1,1],[0,1]], B=[[0],[1]], Q=I, R=[1]):
                                     // classic dlqr example -> Converged, closed-loop |eig(A-BK)| < 1
            ScheduleMatchesOneStep, // N=1 schedule row-block == a single direct RiccatiStep(Qf) call
            ScheduleApproachesInfiniteHorizon, // large-N schedule row 0 (Qf=Q) ~= the infinite-horizon K
            WarmReconvergesFast,    // cold solve w/ state, then warm re-solve on the SAME A/B ->
                                     // Converged in a handful of iterations, same K as the cold solve
            DivergedUnstabilizable, // A=diag(2,.5), B=[0;1]: uncontrollable unstable mode -> Diverged,
                                     // finite iteration count (fails fast, does not hang)
            Determinism,            // two back-to-back cold solves on the same instance -> bit-identical K
        }

        public TestType Type;
        public NativeArray<double> Fail;   // [0]=flag [1]=got [2]=expected/limit [3]=diff/extra

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ColdConverges:
                {
                    var A = new doubleMxN(2, 2, Allocator.Temp);
                    A[0, 0] = (double)1; A[0, 1] = (double)1; A[1, 0] = (double)0; A[1, 1] = (double)1;
                    var B = new doubleMxN(2, 1, Allocator.Temp);
                    B[0, 0] = (double)0; B[1, 0] = (double)1;
                    var Q = new doubleMxN(2, 2, Allocator.Temp);
                    Q[0, 0] = (double)1; Q[1, 1] = (double)1;
                    var R = new doubleMxN(1, 1, Allocator.Temp);
                    R[0, 0] = (double)1;
                    var K = new doubleMxN(1, 2, Allocator.Temp);

                    var info = Control.lqr(in A, in B, in Q, in R, ref K);
                    AssertTrue(info.status == LQRStatus.Converged);

                    // closed-loop stability: max |eig(A - B K)| < 1
                    var Acl = new doubleMxN(2, 2, Allocator.Temp);
                    Blas.dot(in B, in K, ref Acl);
                    for (int i = 0; i < 2; i++)
                        for (int j = 0; j < 2; j++)
                            Acl[i, j] = A[i, j] - Acl[i, j];

                    var er = new doubleN(2, Allocator.Temp);
                    var ei = new doubleN(2, Allocator.Temp);
                    Eigen.valuesQR(ref Acl, ref er, ref ei);
                    double maxMag = 0;
                    for (int i = 0; i < 2; i++)
                    {
                        double mag = math.sqrt(er[i] * er[i] + ei[i] * ei[i]);
                        if (mag > maxMag) maxMag = mag;
                    }
                    AssertLess(maxMag, (double)1);

                    er.Dispose(); ei.Dispose(); Acl.Dispose();
                    A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); K.Dispose();
                    break;
                }

                case TestType.ScheduleMatchesOneStep:
                {
                    // N=1 schedule vs a HAND-computed single Riccati step, via public Blas.dot calls
                    // only (RiccatiStep itself is internal -- not reachable from this template-test
                    // firstpass compile, only from the real generated test assembly's InternalsVisibleTo
                    // grant). m=1 here, so K = (R+BᵀQB)⁻¹BᵀQA collapses to a scalar division -- no CHOP
                    // needed to replicate the formula directly.
                    var A = new doubleMxN(2, 2, Allocator.Temp);
                    A[0, 0] = (double)1; A[0, 1] = (double)1; A[1, 0] = (double)0; A[1, 1] = (double)1;
                    var B = new doubleMxN(2, 1, Allocator.Temp);
                    B[0, 0] = (double)0; B[1, 0] = (double)1;
                    var Q = new doubleMxN(2, 2, Allocator.Temp);
                    Q[0, 0] = (double)1; Q[1, 1] = (double)1;
                    var R = new doubleMxN(1, 1, Allocator.Temp);
                    R[0, 0] = (double)1;

                    var Kschedule = new doubleMxN(1, 2, Allocator.Temp);
                    var info = Control.lqrSchedule(in A, in B, in Q, in R, in Q, 1, ref Kschedule);
                    AssertTrue(info.status == LQRStatus.Converged);
                    AssertEqInt(info.iterations, 1);

                    var QB = new doubleMxN(2, 1, Allocator.Temp);
                    Blas.dot(in Q, in B, ref QB);                              // QB = Q*B
                    var BtQB = new doubleMxN(1, 1, Allocator.Temp);
                    Blas.dot(in B, in QB, ref BtQB, transposeA: true);         // BtQB = BᵀQB
                    double Rbar = R[0, 0] + BtQB[0, 0];
                    var BSA = new doubleMxN(1, 2, Allocator.Temp);
                    Blas.dot(in QB, in A, ref BSA, transposeA: true);          // BSA = (QB)ᵀA = BᵀQA

                    for (int j = 0; j < 2; j++)
                        AssertClose(Kschedule[0, j], BSA[0, j] / Rbar, (double)1e-4);

                    QB.Dispose(); BtQB.Dispose(); BSA.Dispose();
                    A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); Kschedule.Dispose();
                    break;
                }

                case TestType.ScheduleApproachesInfiniteHorizon:
                {
                    var A = new doubleMxN(2, 2, Allocator.Temp);
                    A[0, 0] = (double)1; A[0, 1] = (double)1; A[1, 0] = (double)0; A[1, 1] = (double)1;
                    var B = new doubleMxN(2, 1, Allocator.Temp);
                    B[0, 0] = (double)0; B[1, 0] = (double)1;
                    var Q = new doubleMxN(2, 2, Allocator.Temp);
                    Q[0, 0] = (double)1; Q[1, 1] = (double)1;
                    var R = new doubleMxN(1, 1, Allocator.Temp);
                    R[0, 0] = (double)1;

                    var Kinf = new doubleMxN(1, 2, Allocator.Temp);
                    var infoInf = Control.lqr(in A, in B, in Q, in R, ref Kinf);
                    AssertTrue(infoInf.status == LQRStatus.Converged);

                    var Kschedule = new doubleMxN(60, 2, Allocator.Temp);
                    var info = Control.lqrSchedule(in A, in B, in Q, in R, in Q, 60, ref Kschedule);
                    AssertTrue(info.status == LQRStatus.Converged);

                    for (int j = 0; j < 2; j++)
                        AssertClose(Kschedule[0, j], Kinf[0, j], (double)1e-2);

                    A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); Kinf.Dispose(); Kschedule.Dispose();
                    break;
                }

                case TestType.WarmReconvergesFast:
                {
                    var A = new doubleMxN(2, 2, Allocator.Temp);
                    A[0, 0] = (double)1; A[0, 1] = (double)1; A[1, 0] = (double)0; A[1, 1] = (double)1;
                    var B = new doubleMxN(2, 1, Allocator.Temp);
                    B[0, 0] = (double)0; B[1, 0] = (double)1;
                    var Q = new doubleMxN(2, 2, Allocator.Temp);
                    Q[0, 0] = (double)1; Q[1, 1] = (double)1;
                    var R = new doubleMxN(1, 1, Allocator.Temp);
                    R[0, 0] = (double)1;

                    var state = new doubleLQRState(2, Allocator.Temp);
                    var Kcold = new doubleMxN(1, 2, Allocator.Temp);
                    var coldInfo = Control.lqr(in A, in B, in Q, in R, ref Kcold, ref state);
                    AssertTrue(coldInfo.status == LQRStatus.Converged);
                    AssertTrue(state.populated);

                    var Kwarm = new doubleMxN(1, 2, Allocator.Temp);
                    var warmInfo = Control.lqr(in A, in B, in Q, in R, ref Kwarm, ref state);
                    AssertTrue(warmInfo.status == LQRStatus.Converged);
                    AssertLE(warmInfo.iterations, 5);

                    for (int j = 0; j < 2; j++)
                        AssertClose(Kcold[0, j], Kwarm[0, j], (double)1e-3);

                    state.Dispose();
                    A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); Kcold.Dispose(); Kwarm.Dispose();
                    break;
                }

                case TestType.DivergedUnstabilizable:
                {
                    var A = new doubleMxN(2, 2, Allocator.Temp);
                    A[0, 0] = (double)2; A[1, 1] = (double)0.5;
                    var B = new doubleMxN(2, 1, Allocator.Temp);
                    B[1, 0] = (double)1;
                    var Q = new doubleMxN(2, 2, Allocator.Temp);
                    Q[0, 0] = (double)1; Q[1, 1] = (double)1;
                    var R = new doubleMxN(1, 1, Allocator.Temp);
                    R[0, 0] = (double)1;
                    var K = new doubleMxN(1, 2, Allocator.Temp);

                    var info = Control.lqr(in A, in B, in Q, in R, ref K);
                    AssertTrue(info.status == LQRStatus.Diverged);
                    AssertLE(info.iterations, 50);

                    A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); K.Dispose();
                    break;
                }

                case TestType.Determinism:
                {
                    var A = new doubleMxN(2, 2, Allocator.Temp);
                    A[0, 0] = (double)1; A[0, 1] = (double)1; A[1, 0] = (double)0; A[1, 1] = (double)1;
                    var B = new doubleMxN(2, 1, Allocator.Temp);
                    B[0, 0] = (double)0; B[1, 0] = (double)1;
                    var Q = new doubleMxN(2, 2, Allocator.Temp);
                    Q[0, 0] = (double)1; Q[1, 1] = (double)1;
                    var R = new doubleMxN(1, 1, Allocator.Temp);
                    R[0, 0] = (double)1;

                    var K1 = new doubleMxN(1, 2, Allocator.Temp);
                    var i1 = Control.lqr(in A, in B, in Q, in R, ref K1);
                    var K2 = new doubleMxN(1, 2, Allocator.Temp);
                    var i2 = Control.lqr(in A, in B, in Q, in R, ref K2);

                    AssertEqInt(i1.iterations, i2.iterations);
                    for (int j = 0; j < 2; j++)
                        AssertClose(K1[0, j], K2[0, j], (double)0);

                    A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); K1.Dispose(); K2.Dispose();
                    break;
                }
            }
        }

        void AssertTrue(bool cond)
        {
            if (!cond && Fail[0] == (double)0) { Fail[0] = (double)1; Fail[1] = (double)0; Fail[2] = (double)1; Fail[3] = (double)0; }
            Assert.IsTrue(cond);
        }

        void AssertEqInt(int got, int expected)
        {
            if (got != expected && Fail[0] == (double)0) { Fail[0] = (double)1; Fail[1] = (double)got; Fail[2] = (double)expected; Fail[3] = (double)(got - expected); }
            Assert.IsTrue(got == expected);
        }

        void AssertLE(int got, int limit)
        {
            if (!(got <= limit) && Fail[0] == (double)0) { Fail[0] = (double)1; Fail[1] = (double)got; Fail[2] = (double)limit; Fail[3] = (double)(got - limit); }
            Assert.IsTrue(got <= limit);
        }

        void AssertLess(double got, double limit)
        {
            if (!(got < limit) && Fail[0] == (double)0) { Fail[0] = (double)1; Fail[1] = got; Fail[2] = limit; Fail[3] = got - limit; }
            Assert.IsTrue(got < limit);
        }

        void AssertClose(double a, double b, double precision)
        {
            double diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (double)0) { Fail[0] = (double)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff; }
            Assert.IsTrue(diff <= precision);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void ControlTests(TestJob.TestType type)
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

    // ---- managed-thread argument-validation throw tests (Assert.Catch, same tail pattern as
    //      MIPTests.double.cs's SolveThrowsOnDimensionMismatch) ----

    [Test]
    public void LqrThrowsOnDimensionMismatch()
    {
        var A = new doubleMxN(2, 2, Allocator.Temp);
        A[0, 0] = (double)1; A[0, 1] = (double)1; A[1, 0] = (double)0; A[1, 1] = (double)1;
        var B = new doubleMxN(2, 1, Allocator.Temp);
        B[1, 0] = (double)1;
        var Q = new doubleMxN(2, 2, Allocator.Temp); Q[0, 0] = (double)1; Q[1, 1] = (double)1;
        var R = new doubleMxN(1, 1, Allocator.Temp); R[0, 0] = (double)1;
        var K = new doubleMxN(1, 2, Allocator.Temp);

        var Abad = new doubleMxN(2, 3, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => Control.lqr(in Abad, in B, in Q, in R, ref K));
        var Bbad = new doubleMxN(3, 1, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => Control.lqr(in A, in Bbad, in Q, in R, ref K));
        var Qbad = new doubleMxN(3, 3, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => Control.lqr(in A, in B, in Qbad, in R, ref K));
        var Rbad = new doubleMxN(2, 2, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => Control.lqr(in A, in B, in Q, in Rbad, ref K));
        var Kbad = new doubleMxN(2, 2, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => Control.lqr(in A, in B, in Q, in R, ref Kbad));

        var Rneg = new doubleMxN(1, 1, Allocator.Temp); Rneg[0, 0] = (double)(-1);
        Assert.Catch<ArgumentException>(() => Control.lqr(in A, in B, in Q, in Rneg, ref K));

        A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); K.Dispose();
        Abad.Dispose(); Bbad.Dispose(); Qbad.Dispose(); Rbad.Dispose(); Kbad.Dispose(); Rneg.Dispose();
    }

    [Test]
    public void LqrScheduleThrowsOnBadN()
    {
        var A = new doubleMxN(2, 2, Allocator.Temp);
        A[0, 0] = (double)1; A[0, 1] = (double)1; A[1, 0] = (double)0; A[1, 1] = (double)1;
        var B = new doubleMxN(2, 1, Allocator.Temp);
        B[1, 0] = (double)1;
        var Q = new doubleMxN(2, 2, Allocator.Temp); Q[0, 0] = (double)1; Q[1, 1] = (double)1;
        var R = new doubleMxN(1, 1, Allocator.Temp); R[0, 0] = (double)1;
        var Kschedule = new doubleMxN(2, 2, Allocator.Temp);

        Assert.Catch<ArgumentException>(() => Control.lqrSchedule(in A, in B, in Q, in R, in Q, 0, ref Kschedule));

        var KscheduleBad = new doubleMxN(1, 2, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => Control.lqrSchedule(in A, in B, in Q, in R, in Q, 2, ref KscheduleBad));

        A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); Kschedule.Dispose(); KscheduleBad.Dispose();
    }

    [Test]
    public void LqrWarmThrowsOnUncreatedOrMismatchedState()
    {
        var A = new doubleMxN(2, 2, Allocator.Temp);
        A[0, 0] = (double)1; A[0, 1] = (double)1; A[1, 0] = (double)0; A[1, 1] = (double)1;
        var B = new doubleMxN(2, 1, Allocator.Temp);
        B[1, 0] = (double)1;
        var Q = new doubleMxN(2, 2, Allocator.Temp); Q[0, 0] = (double)1; Q[1, 1] = (double)1;
        var R = new doubleMxN(1, 1, Allocator.Temp); R[0, 0] = (double)1;
        var K = new doubleMxN(1, 2, Allocator.Temp);

        var uncreated = default(doubleLQRState);
        Assert.Catch<ArgumentException>(() => Control.lqr(in A, in B, in Q, in R, ref K, ref uncreated));

        var mismatched = new doubleLQRState(3, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => Control.lqr(in A, in B, in Q, in R, ref K, ref mismatched));

        mismatched.Dispose();
        A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); K.Dispose();
    }
}
