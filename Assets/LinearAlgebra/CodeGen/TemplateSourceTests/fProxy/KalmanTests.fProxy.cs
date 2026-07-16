using System;

using LinearAlgebra;
using LinearAlgebra.Control;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// FULL test battery for the Kalman filter feature (Kalman.fProxy.cs / Kalman.State.fProxy.cs /
// Kalman.Info.cs / KalmanModel.fProxy.cs) plus LQR.lqg. Covers: the linear predict/update filter
// (constant-velocity tracker, exact-symmetry contract), both predict overloads agreeing at u=0, the
// update() indefinite-S failure path (state left untouched), steadyStateGain vs an INDEPENDENT
// fixed-point Riccati oracle (with an orientation-discrimination check), the fixed-gain fast path
// (predictFixed/updateFixed matching the converged general path, never touching P, dim throw), the
// EKF (a nonlinear pendulum model tracked to bounded error, and numericJacobianF/H vs analytic
// Jacobians), and LQR.lqg returning both gains matching the direct lqr / steadyStateGain calls.
//
// Value cases run inside a [BurstCompile] IJob with CompileSynchronously=true and route every numeric
// assertion through Fail[0..3] with IsTrue-style checks (BC1330 forbids enum Assert.AreEqual inside
// Burst) -- the same shape as ControlLQRTests.fProxy.cs / ControlTests.fProxy.cs. Templated (fProxy)
// so codegen emits a float and a double build; per-dtype tolerances via choose-markers (float|double).
// Argument-validation throws (Burst cannot surface an assertable managed exception) are managed [Test]s
// with Assert.Catch. Fixed seeds everywhere for determinism.
public class fProxyKalmanTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            LinearKFHappyPath,          // 1. CV tracker converges near truth; P exactly symmetric + pos diag
            PredictOverloadsAgree,      // 2. predict(A,B,u=0,Q) == predict(A,Q) exactly
            UpdateFailureLeavesState,   // 3. indefinite S (negative R) -> InnovationSolveFailed, x/P untouched
            SteadyStateGainVsOracle,    // 4. Kss == independent Riccati oracle; wrong orientation discriminated
            FixedPathMatchesConverged,  // 5a. predictFixed/updateFixed track the converged general path
            FixedPathNeverTouchesP,     // 5b. the fixed fast path leaves s.P byte-identical
            EKFTracksNonlinear,         // 6a. EKF pendulum tracked to bounded error
            NumericJacobianMatchesAnalytic, // 6b. numericJacobianF/H agree with analytic Jacobians
            LqgReturnsBothGains,        // 7. lqg's Klqr==LQR.lqr, Kkf==Kalman.steadyStateGain, statuses ok
        }

        public TestType Type;
        public NativeArray<fProxy> Fail;   // [0]=flag [1]=got [2]=expected/limit [3]=diff/extra

        public void Execute()
        {
            switch (Type)
            {
                case TestType.LinearKFHappyPath: LinearKFHappyPath(); break;
                case TestType.PredictOverloadsAgree: PredictOverloadsAgree(); break;
                case TestType.UpdateFailureLeavesState: UpdateFailureLeavesState(); break;
                case TestType.SteadyStateGainVsOracle: SteadyStateGainVsOracle(); break;
                case TestType.FixedPathMatchesConverged: FixedPathMatchesConverged(); break;
                case TestType.FixedPathNeverTouchesP: FixedPathNeverTouchesP(); break;
                case TestType.EKFTracksNonlinear: EKFTracksNonlinear(); break;
                case TestType.NumericJacobianMatchesAnalytic: NumericJacobianMatchesAnalytic(); break;
                case TestType.LqgReturnsBothGains: LqgReturnsBothGains(); break;
            }
        }

        // ---- per-dtype tolerances (loose for float, tight for double) ----
        static fProxy CvVelTol() => /*+choose[0.15f|0.15]*/0.15f/*-choose*/;   // CV velocity estimate error
        static fProxy CvPosTol() => /*+choose[0.6f|0.6]*/0.6f/*-choose*/;      // CV position estimate error
        static double GainRelTol() => /*+choose[2e-3|1e-8]*/2e-3/*-choose*/;   // Kss vs oracle, relative Frob
        static double OracleFloorWrong() => 5e-3;                              // correct/wrong oracles differ by >=
        static fProxy FixedTrackTol() => /*+choose[0.1f|0.1]*/0.1f/*-choose*/; // fixed-path vs general-path x
        static fProxy EkfThetaTol() => /*+choose[0.1f|0.1]*/0.1f/*-choose*/;   // EKF angle estimate error
        static fProxy EkfOmegaTol() => /*+choose[0.3f|0.3]*/0.3f/*-choose*/;   // EKF rate estimate error
        static fProxy JacTol() => /*+choose[1e-2f|1e-6]*/1e-2f/*-choose*/;     // numeric vs analytic Jacobian

        // ================================ 1. linear KF happy path ================================
        // Constant-velocity 1D tracker: state [pos, vel], A=[[1,1],[0,1]], H=[[1,0]] (position only).
        // True object moves at constant velocity 0.5; measurements are noisy positions (fixed seed).
        // The filter (started at the WRONG velocity 0) must converge near truth; P must stay EXACTLY
        // symmetric (Joseph+symmetrize contract: P[i,j]==P[j,i] bit-for-bit, not approximately) with a
        // strictly positive diagonal at every step.
        void LinearKFHappyPath()
        {
            BuildCV(out var A, out var H, out var Q, out var R);

            var s = new fProxyKFState(2, 1, Allocator.Temp);
            // wrong initial estimate (velocity 0, truth 0.5) + large initial uncertainty
            s.x[0] = (fProxy)0; s.x[1] = (fProxy)0;
            s.P[0, 0] = (fProxy)5; s.P[1, 1] = (fProxy)5;

            var trueState = new fProxyN(2, Allocator.Temp);
            trueState[0] = (fProxy)0; trueState[1] = (fProxy)0.5;
            var trueNext = new fProxyN(2, Allocator.Temp);

            var z = new fProxyN(1, Allocator.Temp);
            var rng = new Unity.Mathematics.Random(0xC0FFEEu);

            int steps = 60;
            for (int k = 0; k < steps; k++)
            {
                // advance truth: trueState = A trueState (deterministic constant velocity)
                Blas.dot(in A, in trueState, ref trueNext);
                trueState.Data.CopyFrom(trueNext.Data);

                // noisy position measurement
                z[0] = trueState[0] + (fProxy)rng.NextFloat(-0.2f, 0.2f);

                Kalman.predict(ref s, in A, in Q);
                AssertExactSymPosDiag(in s.P);          // predict keeps P symmetric
                var info = Kalman.update(ref s, in H, in R, in z);
                AssertTrue(info.status == KFStatus.Ok);
                AssertExactSymPosDiag(in s.P);          // update keeps P symmetric
            }

            // converged near truth: velocity error must be far below the initial 0.5 error
            AssertClose(s.x[1], (fProxy)0.5, CvVelTol());
            AssertClose(s.x[0], trueState[0], CvPosTol());

            s.Dispose(); A.Dispose(); H.Dispose(); Q.Dispose(); R.Dispose();
            trueState.Dispose(); trueNext.Dispose(); z.Dispose();
        }

        // ================================ 2. predict overloads agree ================================
        // predict(A,B,u,Q) with u=0 must produce EXACTLY the same x and P as the autonomous
        // predict(A,Q): Bu = B*0 is the zero vector and x += 0 is exact in IEEE754, and the covariance
        // path is byte-identical code -- so bit equality, not a tolerance, is the correct assertion.
        void PredictOverloadsAgree()
        {
            BuildCV(out var A, out var H, out var Q, out var R);
            H.Dispose(); R.Dispose();   // not needed here

            var B = new fProxyMxN(2, 1, Allocator.Temp); B[0, 0] = (fProxy)0; B[1, 0] = (fProxy)1;
            var u = new fProxyN(1, Allocator.Temp); u[0] = (fProxy)0;   // zero control

            var sA = new fProxyKFState(2, 1, Allocator.Temp);
            var sB = new fProxyKFState(2, 1, Allocator.Temp);
            SeedState(ref sA);
            SeedState(ref sB);

            Kalman.predict(ref sA, in A, in B, in u, in Q);   // with control input (u=0)
            Kalman.predict(ref sB, in A, in Q);               // autonomous

            for (int i = 0; i < 2; i++)
                AssertTrue(sA.x[i] == sB.x[i]);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    AssertTrue(sA.P[i, j] == sB.P[i, j]);

            sA.Dispose(); sB.Dispose(); A.Dispose(); Q.Dispose(); B.Dispose(); u.Dispose();
        }

        // ================================ 3. update failure path ================================
        // A negative R makes S = HPHᵀ + R indefinite (here scalar -1), which CHOP reports as Indefinite
        // -> KFStatus.InnovationSolveFailed. The contract is that a failed update applies NOTHING: x and
        // P must be byte-identical to their pre-update values.
        void UpdateFailureLeavesState()
        {
            var H = new fProxyMxN(1, 2, Allocator.Temp); H[0, 0] = (fProxy)1; H[0, 1] = (fProxy)0;
            var R = new fProxyMxN(1, 1, Allocator.Temp); R[0, 0] = (fProxy)(-2);   // negative -> S = 1 - 2 = -1
            var z = new fProxyN(1, Allocator.Temp); z[0] = (fProxy)3;

            var s = new fProxyKFState(2, 1, Allocator.Temp);
            s.x[0] = (fProxy)7; s.x[1] = (fProxy)(-4);
            s.P[0, 0] = (fProxy)1; s.P[0, 1] = (fProxy)0; s.P[1, 0] = (fProxy)0; s.P[1, 1] = (fProxy)1;

            var xSnap = new fProxyN(2, Allocator.Temp); xSnap.Data.CopyFrom(s.x.Data);
            var pSnap = new fProxyMxN(in s.P, Allocator.Temp);

            var info = Kalman.update(ref s, in H, in R, in z);

            AssertTrue(info.status == KFStatus.InnovationSolveFailed);
            AssertTrue(!info.Solved);
            for (int i = 0; i < 2; i++)
                AssertTrue(s.x[i] == xSnap[i]);           // x untouched, EXACT
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    AssertTrue(s.P[i, j] == pSnap[i, j]);  // P untouched, EXACT

            s.Dispose(); H.Dispose(); R.Dispose(); z.Dispose(); xSnap.Dispose(); pSnap.Dispose();
        }

        // ================================ 4. steadyStateGain vs oracle ================================
        // steadyStateGain solves the filter predicted-covariance DARE by the LQR/KF duality and extracts
        // Kss. Ground truth is an INDEPENDENT plain fixed-point iteration of that DARE from Sigma0=Q
        // (OracleGain, no SDA/doubling), gain = Sigma Hᵀ (H Sigma Hᵀ + R)⁻¹. Kss must match to tight rel
        // Frobenius. DISCRIMINATION: an oracle run with the A-transpose folded the WRONG way (A vs Aᵀ in
        // the propagation) is a genuinely different problem -- its gain differs from the correct oracle by
        // >> the match tolerance, and Kss is unambiguously closer to the correct one. (In double the gap
        // is many orders of magnitude; in float steadyStateGain's own precision floor narrows it, but Kss
        // is still provably closer to the correctly-oriented oracle than to the wrong one.)
        void SteadyStateGainVsOracle()
        {
            BuildCV(out var A, out var H, out var Q, out var R);

            var Kss = new fProxyMxN(2, 1, Allocator.Temp);
            var info = Kalman.steadyStateGain(in A, in H, in Q, in R, ref Kss);
            AssertTrue(info.status == LQRStatus.Converged);

            var Kcorrect = new fProxyMxN(2, 1, Allocator.Temp);
            OracleGain(in A, in H, in Q, in R, 400, ref Kcorrect);   // correct orientation

            var At = new fProxyMxN(2, 2, Allocator.Temp);
            Blas.trans(in A, ref At);
            var Kwrong = new fProxyMxN(2, 1, Allocator.Temp);
            OracleGain(in At, in H, in Q, in R, 400, ref Kwrong);    // WRONG orientation (A -> Aᵀ)

            double kNorm = math.max(1e-30, FrobNormMat(in Kcorrect));
            double relImplCorrect = FrobDiffMat(in Kss, in Kcorrect) / kNorm;
            double relOraclePair = FrobDiffMat(in Kcorrect, in Kwrong) / kNorm;
            double relImplWrong = FrobDiffMat(in Kss, in Kwrong) / kNorm;

            AssertLEd(relImplCorrect, GainRelTol());          // Kss matches the correct oracle
            AssertGEd(relOraclePair, OracleFloorWrong());     // orientation genuinely matters (non-vacuous)
            AssertGEd(relImplWrong, OracleFloorWrong());      // Kss is NOT the wrong-orientation solution

            A.Dispose(); H.Dispose(); Q.Dispose(); R.Dispose();
            Kss.Dispose(); Kcorrect.Dispose(); Kwrong.Dispose(); At.Dispose();
        }

        // ================================ 5a. fixed path matches converged general path ==============
        // Warm up a general predict/update filter until its covariance (hence its gain) reaches steady
        // state, then run the general path and the fixed-gain fast path (fed the steadyStateGain Kss)
        // side by side on the SAME measurement stream from a common starting estimate. Once the general
        // filter's gain has converged to Kss the two x-trajectories track closely.
        void FixedPathMatchesConverged()
        {
            BuildCV(out var A, out var H, out var Q, out var R);

            var Kss = new fProxyMxN(2, 1, Allocator.Temp);
            var ssInfo = Kalman.steadyStateGain(in A, in H, in Q, in R, ref Kss);
            AssertTrue(ssInfo.status == LQRStatus.Converged);

            var gen = new fProxyKFState(2, 1, Allocator.Temp);
            gen.x[0] = (fProxy)0; gen.x[1] = (fProxy)0;
            gen.P[0, 0] = (fProxy)5; gen.P[1, 1] = (fProxy)5;

            var trueState = new fProxyN(2, Allocator.Temp);
            trueState[0] = (fProxy)0; trueState[1] = (fProxy)0.5;
            var trueNext = new fProxyN(2, Allocator.Temp);
            var z = new fProxyN(1, Allocator.Temp);
            var rng = new Unity.Mathematics.Random(0x5EED01u);

            // warm-up: drive P to steady state on the general path
            for (int k = 0; k < 40; k++)
            {
                Blas.dot(in A, in trueState, ref trueNext); trueState.Data.CopyFrom(trueNext.Data);
                z[0] = trueState[0] + (fProxy)rng.NextFloat(-0.2f, 0.2f);
                Kalman.predict(ref gen, in A, in Q);
                Kalman.update(ref gen, in H, in R, in z);
            }

            // fixed-path state starts from the general path's current estimate
            var fix = new fProxyKFState(2, 1, Allocator.Temp);
            fix.x.Data.CopyFrom(gen.x.Data);

            for (int k = 0; k < 25; k++)
            {
                Blas.dot(in A, in trueState, ref trueNext); trueState.Data.CopyFrom(trueNext.Data);
                z[0] = trueState[0] + (fProxy)rng.NextFloat(-0.2f, 0.2f);

                Kalman.predict(ref gen, in A, in Q);
                Kalman.update(ref gen, in H, in R, in z);

                Kalman.predictFixed(ref fix, in A);
                Kalman.updateFixed(ref fix, in Kss, in H, in z);

                for (int i = 0; i < 2; i++)
                    AssertClose(fix.x[i], gen.x[i], FixedTrackTol());
            }

            gen.Dispose(); fix.Dispose(); A.Dispose(); H.Dispose(); Q.Dispose(); R.Dispose();
            trueState.Dispose(); trueNext.Dispose(); z.Dispose(); Kss.Dispose();
        }

        // ================================ 5b. fixed path never touches P ============================
        // predictFixed/updateFixed do no covariance math. Seed s.P with a distinctive matrix, run both
        // fixed calls, and assert every entry of s.P is byte-identical afterwards.
        void FixedPathNeverTouchesP()
        {
            BuildCV(out var A, out var H, out var Q, out var R);
            Q.Dispose();

            var Kss = new fProxyMxN(2, 1, Allocator.Temp);
            Kss[0, 0] = (fProxy)0.4; Kss[1, 0] = (fProxy)0.2;   // arbitrary fixed gain; P must be inert regardless

            var s = new fProxyKFState(2, 1, Allocator.Temp);
            s.x[0] = (fProxy)1; s.x[1] = (fProxy)(-2);
            s.P[0, 0] = (fProxy)2; s.P[0, 1] = (fProxy)0.5; s.P[1, 0] = (fProxy)0.5; s.P[1, 1] = (fProxy)3;
            var pSnap = new fProxyMxN(in s.P, Allocator.Temp);

            var z = new fProxyN(1, Allocator.Temp); z[0] = (fProxy)0.7;

            Kalman.predictFixed(ref s, in A);
            Kalman.updateFixed(ref s, in Kss, in H, in z);

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    AssertTrue(s.P[i, j] == pSnap[i, j]);

            s.Dispose(); A.Dispose(); H.Dispose(); R.Dispose(); Kss.Dispose(); pSnap.Dispose(); z.Dispose();
        }

        // ================================ 6a. EKF tracks a nonlinear system =========================
        // Pendulum: state [theta, omega], NONLINEAR dynamics theta'=theta+omega*dt,
        // omega'=omega-(g/L)sin(theta)*dt; NONLINEAR measurement h=sin(theta). Both Jacobians analytic.
        // Filter started at the wrong angle must track the true trajectory to bounded error (fixed seed).
        void EKFTracksNonlinear()
        {
            var model = new fProxyPendulumModel { dt = (fProxy)0.05, gOverL = (fProxy)4 };
            var meas = new fProxyPendulumMeas();

            var Q = new fProxyMxN(2, 2, Allocator.Temp); Q[0, 0] = (fProxy)1e-6; Q[1, 1] = (fProxy)1e-6;
            var R = new fProxyMxN(1, 1, Allocator.Temp); R[0, 0] = (fProxy)1e-3;
            var u = new fProxyN(1, Allocator.Temp);   // autonomous; F ignores u

            var s = new fProxyKFState(2, 1, Allocator.Temp);
            s.x[0] = (fProxy)0; s.x[1] = (fProxy)0;    // wrong initial angle (truth 0.3)
            s.P[0, 0] = (fProxy)0.5; s.P[1, 1] = (fProxy)0.5;

            var trueState = new fProxyN(2, Allocator.Temp); trueState[0] = (fProxy)0.3; trueState[1] = (fProxy)0;
            var trueNext = new fProxyN(2, Allocator.Temp);
            var zTrue = new fProxyN(1, Allocator.Temp);
            var z = new fProxyN(1, Allocator.Temp);
            var rng = new Unity.Mathematics.Random(0xECF00Du);

            for (int k = 0; k < 80; k++)
            {
                model.F(in trueState, in u, ref trueNext);   // advance truth (nonlinear, deterministic)
                trueState.Data.CopyFrom(trueNext.Data);

                meas.H(in trueState, ref zTrue);
                z[0] = zTrue[0] + (fProxy)rng.NextFloat(-0.02f, 0.02f);

                Kalman.ekfPredict(ref s, in model, in u, in Q);
                var info = Kalman.ekfUpdate(ref s, in meas, in R, in z);
                AssertTrue(info.status == KFStatus.Ok);
                AssertExactSymPosDiag(in s.P);
            }

            AssertClose(s.x[0], trueState[0], EkfThetaTol());
            AssertClose(s.x[1], trueState[1], EkfOmegaTol());

            s.Dispose(); Q.Dispose(); R.Dispose(); u.Dispose();
            trueState.Dispose(); trueNext.Dispose(); zTrue.Dispose(); z.Dispose();
        }

        // ================================ 6b. numeric vs analytic Jacobians =========================
        // Kalman.numericJacobianF / numericJacobianH (central differences) must agree with the model's
        // analytic Jacobians at several states -- the primitive a user calls when hand-differentiation is
        // impractical must reproduce a correct hand Jacobian.
        void NumericJacobianMatchesAnalytic()
        {
            var model = new fProxyPendulumModel { dt = (fProxy)0.05, gOverL = (fProxy)4 };
            var meas = new fProxyPendulumMeas();

            var x = new fProxyN(2, Allocator.Temp);
            var u = new fProxyN(1, Allocator.Temp);
            var Ja = new fProxyMxN(2, 2, Allocator.Temp);
            var Jn = new fProxyMxN(2, 2, Allocator.Temp);
            var Ha = new fProxyMxN(1, 2, Allocator.Temp);
            var Hn = new fProxyMxN(1, 2, Allocator.Temp);

            int tested = 0;
            for (int c = 0; c < 4; c++)
            {
                fProxy th = c == 0 ? (fProxy)0.1 : c == 1 ? (fProxy)0.3 : c == 2 ? (fProxy)(-0.4) : (fProxy)0;
                fProxy om = c == 0 ? (fProxy)0 : c == 1 ? (fProxy)(-0.2) : c == 2 ? (fProxy)0.5 : (fProxy)0.1;
                x[0] = th; x[1] = om;

                model.JacobianF(in x, in u, ref Ja);
                Kalman.numericJacobianF(in model, in x, in u, ref Jn);
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++)
                        AssertClose(Jn[i, j], Ja[i, j], JacTol());

                meas.JacobianH(in x, ref Ha);
                Kalman.numericJacobianH(in meas, in x, ref Hn);
                for (int j = 0; j < 2; j++)
                    AssertClose(Hn[0, j], Ha[0, j], JacTol());

                tested++;
            }
            AssertTrue(tested == 4);

            x.Dispose(); u.Dispose(); Ja.Dispose(); Jn.Dispose(); Ha.Dispose(); Hn.Dispose();
        }

        // ================================ 7. LQR.lqg ============================================
        // lqg runs BOTH DARE solves (LQR control + KF filter) from the same A and returns two gains. Each
        // must equal the corresponding standalone call bit-for-bit (lqg simply forwards), and both
        // statuses / the aggregate Solved flag must be correct.
        void LqgReturnsBothGains()
        {
            // double integrator dynamics for the LQR side; position measurement for the KF side
            var A = new fProxyMxN(2, 2, Allocator.Temp);
            A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)1; A[1, 0] = (fProxy)0; A[1, 1] = (fProxy)1;
            var B = new fProxyMxN(2, 1, Allocator.Temp); B[0, 0] = (fProxy)0; B[1, 0] = (fProxy)1;
            var H = new fProxyMxN(1, 2, Allocator.Temp); H[0, 0] = (fProxy)1; H[0, 1] = (fProxy)0;
            var Qlqr = Eye(2); var Rlqr = R1(1);
            var Qkf = Eye(2); var Rkf = R1(1);

            var Klqr = new fProxyMxN(1, 2, Allocator.Temp);
            var Kkf = new fProxyMxN(2, 1, Allocator.Temp);
            var info = LQR.lqg(in A, in B, in H, in Qlqr, in Rlqr, in Qkf, in Rkf, ref Klqr, ref Kkf);

            AssertTrue(info.lqrInfo.status == LQRStatus.Converged);
            AssertTrue(info.kfInfo.status == LQRStatus.Converged);
            AssertTrue(info.Solved);

            var KlqrDirect = new fProxyMxN(1, 2, Allocator.Temp);
            var lqrDirect = LQR.lqr(in A, in B, in Qlqr, in Rlqr, ref KlqrDirect);
            AssertTrue(lqrDirect.status == LQRStatus.Converged);

            var KkfDirect = new fProxyMxN(2, 1, Allocator.Temp);
            var kfDirect = Kalman.steadyStateGain(in A, in H, in Qkf, in Rkf, ref KkfDirect);
            AssertTrue(kfDirect.status == LQRStatus.Converged);

            for (int i = 0; i < 1; i++)
                for (int j = 0; j < 2; j++)
                    AssertTrue(Klqr[i, j] == KlqrDirect[i, j]);   // bit-identical
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 1; j++)
                    AssertTrue(Kkf[i, j] == KkfDirect[i, j]);     // bit-identical

            A.Dispose(); B.Dispose(); H.Dispose(); Qlqr.Dispose(); Rlqr.Dispose(); Qkf.Dispose(); Rkf.Dispose();
            Klqr.Dispose(); Kkf.Dispose(); KlqrDirect.Dispose(); KkfDirect.Dispose();
        }

        // ================================ helpers ================================

        // Constant-velocity 1D tracker matrices: A=[[1,1],[0,1]], H=[[1,0]], small process noise,
        // measurement variance 0.05 (covers the U(-0.2,0.2) noise used above).
        static void BuildCV(out fProxyMxN A, out fProxyMxN H, out fProxyMxN Q, out fProxyMxN R)
        {
            A = new fProxyMxN(2, 2, Allocator.Temp);
            A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)1; A[1, 0] = (fProxy)0; A[1, 1] = (fProxy)1;
            H = new fProxyMxN(1, 2, Allocator.Temp); H[0, 0] = (fProxy)1; H[0, 1] = (fProxy)0;
            Q = new fProxyMxN(2, 2, Allocator.Temp); Q[0, 0] = (fProxy)1e-4; Q[1, 1] = (fProxy)1e-4;
            R = new fProxyMxN(1, 1, Allocator.Temp); R[0, 0] = (fProxy)0.05;
        }

        // A distinctive, symmetric starting state used by the predict-overload equality test.
        static void SeedState(ref fProxyKFState s)
        {
            s.x[0] = (fProxy)1; s.x[1] = (fProxy)2;
            s.P[0, 0] = (fProxy)3; s.P[0, 1] = (fProxy)0.5; s.P[1, 0] = (fProxy)0.5; s.P[1, 1] = (fProxy)4;
        }

        // Independent ground truth for steadyStateGain: plain fixed-point iteration of the filter
        // predicted-covariance DARE Sigma = Aeff Sigma Aeffᵀ + Q - Aeff Sigma Hᵀ (H Sigma Hᵀ + R)⁻¹ H Sigma Aeffᵀ
        // from Sigma0=Q (m=1, so the inner "inverse" is a scalar division), then Kss = Sigma Hᵀ / s.
        // Passing Aeff=A gives the correct orientation; passing Aeff=Aᵀ gives the deliberately-wrong one.
        static void OracleGain(in fProxyMxN Aeff, in fProxyMxN H, in fProxyMxN Q, in fProxyMxN R,
                               int iters, ref fProxyMxN Kss)
        {
            int n = Aeff.M_Rows;
            var Sigma = new fProxyMxN(in Q, Allocator.Temp);
            var Snext = new fProxyMxN(n, n, Allocator.Temp);
            var AS = new fProxyMxN(n, n, Allocator.Temp);
            var At = new fProxyMxN(n, n, Allocator.Temp);
            var APAt = new fProxyMxN(n, n, Allocator.Temp);
            var SigHt = new fProxyN(n, Allocator.Temp);
            var ASigHt = new fProxyN(n, Allocator.Temp);
            Blas.trans(in Aeff, ref At);
            fProxy r00 = R[0, 0];

            for (int it = 0; it < iters; it++)
            {
                for (int i = 0; i < n; i++)
                {
                    fProxy acc = 0;
                    for (int kk = 0; kk < n; kk++) acc += Sigma[i, kk] * H[0, kk];
                    SigHt[i] = acc;
                }
                fProxy s = r00;
                for (int kk = 0; kk < n; kk++) s += H[0, kk] * SigHt[kk];

                for (int i = 0; i < n; i++)
                {
                    fProxy acc = 0;
                    for (int kk = 0; kk < n; kk++) acc += Aeff[i, kk] * SigHt[kk];
                    ASigHt[i] = acc;
                }

                Blas.dot(in Aeff, in Sigma, ref AS);
                Blas.dot(in AS, in At, ref APAt);

                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        Snext[i, j] = APAt[i, j] + Q[i, j] - ASigHt[i] * ASigHt[j] / s;

                for (int i = 0; i < n; i++)
                    for (int j = i + 1; j < n; j++)
                    {
                        fProxy avg = (Snext[i, j] + Snext[j, i]) / (fProxy)2;
                        Snext[i, j] = avg; Snext[j, i] = avg;
                    }
                Sigma.Data.CopyFrom(Snext.Data);
            }

            for (int i = 0; i < n; i++)
            {
                fProxy acc = 0;
                for (int kk = 0; kk < n; kk++) acc += Sigma[i, kk] * H[0, kk];
                SigHt[i] = acc;
            }
            fProxy sf = r00;
            for (int kk = 0; kk < n; kk++) sf += H[0, kk] * SigHt[kk];
            for (int i = 0; i < n; i++) Kss[i, 0] = SigHt[i] / sf;

            Sigma.Dispose(); Snext.Dispose(); AS.Dispose(); At.Dispose(); APAt.Dispose();
            SigHt.Dispose(); ASigHt.Dispose();
        }

        static double FrobNormMat(in fProxyMxN M)
        {
            double s = 0;
            for (int i = 0; i < M.M_Rows; i++)
                for (int j = 0; j < M.N_Cols; j++) { double v = (double)M[i, j]; s += v * v; }
            return math.sqrt(s);
        }

        static double FrobDiffMat(in fProxyMxN A, in fProxyMxN B)
        {
            double s = 0;
            for (int i = 0; i < A.M_Rows; i++)
                for (int j = 0; j < A.N_Cols; j++) { double v = (double)A[i, j] - (double)B[i, j]; s += v * v; }
            return math.sqrt(s);
        }

        static fProxyMxN Eye(int n)
        {
            var M = new fProxyMxN(n, n, Allocator.Temp);
            for (int i = 0; i < n; i++) M[i, i] = (fProxy)1;
            return M;
        }

        static fProxyMxN R1(fProxy v)
        {
            var M = new fProxyMxN(1, 1, Allocator.Temp);
            M[0, 0] = v;
            return M;
        }

        // ---- Fail[0..3] diagnostic asserts (same shape as ControlLQRTests.fProxy.cs) ----
        void AssertTrue(bool cond)
        {
            if (!cond && Fail[0] == (fProxy)0) { Fail[0] = (fProxy)1; Fail[1] = (fProxy)0; Fail[2] = (fProxy)1; Fail[3] = (fProxy)0; }
            Assert.IsTrue(cond);
        }

        // Exact-symmetry (P[i,j]==P[j,i] bit-for-bit) + strictly-positive-diagonal covariance contract.
        void AssertExactSymPosDiag(in fProxyMxN P)
        {
            int n = P.M_Rows;
            for (int i = 0; i < n; i++)
            {
                if (!(P[i, i] > (fProxy)0) && Fail[0] == (fProxy)0)
                { Fail[0] = (fProxy)1; Fail[1] = P[i, i]; Fail[2] = (fProxy)0; Fail[3] = P[i, i]; }
                Assert.IsTrue(P[i, i] > (fProxy)0);
                for (int j = i + 1; j < n; j++)
                {
                    bool sym = P[i, j] == P[j, i];
                    if (!sym && Fail[0] == (fProxy)0)
                    { Fail[0] = (fProxy)1; Fail[1] = P[i, j]; Fail[2] = P[j, i]; Fail[3] = P[i, j] - P[j, i]; }
                    Assert.IsTrue(sym);
                }
            }
        }

        void AssertLEd(double got, double limit)
        {
            if (!(got <= limit) && Fail[0] == (fProxy)0) { Fail[0] = (fProxy)1; Fail[1] = (fProxy)got; Fail[2] = (fProxy)limit; Fail[3] = (fProxy)(got - limit); }
            Assert.IsTrue(got <= limit);
        }

        void AssertGEd(double got, double limit)
        {
            if (!(got >= limit) && Fail[0] == (fProxy)0) { Fail[0] = (fProxy)1; Fail[1] = (fProxy)got; Fail[2] = (fProxy)limit; Fail[3] = (fProxy)(got - limit); }
            Assert.IsTrue(got >= limit);
        }

        void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0) { Fail[0] = (fProxy)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff; }
            Assert.IsTrue(diff <= precision);
        }
    }

    // ---- EKF model/measurement struct-functors (nested in the test class so codegen renames the
    //      enclosing generic type per dtype, avoiding a float/double name collision). ----

    // Pendulum dynamics: state [theta, omega], control ignored (autonomous).
    public struct fProxyPendulumModel : IfProxyKFModel
    {
        public fProxy dt;
        public fProxy gOverL;

        public void F(in fProxyN x, in fProxyN u, ref fProxyN xNext)
        {
            fProxy th = x[0], om = x[1];
            xNext[0] = th + om * dt;
            xNext[1] = om - gOverL * math.sin(th) * dt;
        }

        public void JacobianF(in fProxyN x, in fProxyN u, ref fProxyMxN J)
        {
            fProxy th = x[0];
            J[0, 0] = (fProxy)1; J[0, 1] = dt;
            J[1, 0] = -gOverL * math.cos(th) * dt; J[1, 1] = (fProxy)1;
        }
    }

    // Nonlinear measurement h = sin(theta).
    public struct fProxyPendulumMeas : IfProxyKFMeasurement
    {
        public void H(in fProxyN x, ref fProxyN z)
        {
            z[0] = math.sin(x[0]);
        }

        public void JacobianH(in fProxyN x, ref fProxyMxN J)
        {
            J[0, 0] = math.cos(x[0]); J[0, 1] = (fProxy)0;
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void KalmanTests(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    // ---- managed-thread argument-validation throw tests (Burst cannot surface an assertable managed
    //      exception; same tail pattern as ControlTests.fProxy.cs). ----

    [Test]
    public void UpdateFixedThrowsOnWrongMeasurementDim()
    {
        // state constructed for MMax = 1; updateFixed must reject an H whose row count != MMax.
        var s = new fProxyKFState(2, 1, Allocator.Temp);
        var Kss = new fProxyMxN(2, 1, Allocator.Temp);
        var z = new fProxyN(1, Allocator.Temp);

        var Hbad = new fProxyMxN(2, 2, Allocator.Temp);   // M_Rows = 2 != MMax = 1
        Assert.Catch<ArgumentException>(() => Kalman.updateFixed(ref s, in Kss, in Hbad, in z));

        var Hgood = new fProxyMxN(1, 2, Allocator.Temp);
        var KssBad = new fProxyMxN(2, 2, Allocator.Temp);  // n x MMax must be 2 x 1
        Assert.Catch<ArgumentException>(() => Kalman.updateFixed(ref s, in KssBad, in Hgood, in z));

        var zBad = new fProxyN(2, Allocator.Temp);         // z.N must equal MMax = 1
        Assert.Catch<ArgumentException>(() => Kalman.updateFixed(ref s, in Kss, in Hgood, in zBad));

        s.Dispose(); Kss.Dispose(); z.Dispose(); Hbad.Dispose(); Hgood.Dispose(); KssBad.Dispose(); zBad.Dispose();
    }

    [Test]
    public void UpdateThrowsOnDimensionMismatch()
    {
        var s = new fProxyKFState(2, 1, Allocator.Temp);
        var R = new fProxyMxN(1, 1, Allocator.Temp); R[0, 0] = (fProxy)1;
        var z = new fProxyN(1, Allocator.Temp);

        var Hbad = new fProxyMxN(1, 3, Allocator.Temp);    // N_Cols must equal state dim (2)
        Assert.Catch<ArgumentException>(() => Kalman.update(ref s, in Hbad, in R, in z));

        var H = new fProxyMxN(1, 2, Allocator.Temp);
        var Rbad = new fProxyMxN(2, 2, Allocator.Temp);    // R must be m x m (m = H.M_Rows = 1)
        Assert.Catch<ArgumentException>(() => Kalman.update(ref s, in H, in Rbad, in z));

        var zBad = new fProxyN(2, Allocator.Temp);         // z.N must equal H.M_Rows = 1
        Assert.Catch<ArgumentException>(() => Kalman.update(ref s, in H, in R, in zBad));

        s.Dispose(); R.Dispose(); z.Dispose(); Hbad.Dispose(); H.Dispose(); Rbad.Dispose(); zBad.Dispose();
    }

    [Test]
    public void SteadyStateGainThrowsOnBadShapes()
    {
        var A = new fProxyMxN(2, 2, Allocator.Temp);
        A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)1; A[1, 0] = (fProxy)0; A[1, 1] = (fProxy)1;
        var H = new fProxyMxN(1, 2, Allocator.Temp); H[0, 0] = (fProxy)1;
        var Q = new fProxyMxN(2, 2, Allocator.Temp); Q[0, 0] = (fProxy)1; Q[1, 1] = (fProxy)1;
        var R = new fProxyMxN(1, 1, Allocator.Temp); R[0, 0] = (fProxy)1;
        var Kss = new fProxyMxN(2, 1, Allocator.Temp);

        var Abad = new fProxyMxN(2, 3, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => Kalman.steadyStateGain(in Abad, in H, in Q, in R, ref Kss));
        var Hbad = new fProxyMxN(1, 3, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => Kalman.steadyStateGain(in A, in Hbad, in Q, in R, ref Kss));
        var KssBad = new fProxyMxN(2, 2, Allocator.Temp);   // must be n x m = 2 x 1
        Assert.Catch<ArgumentException>(() => Kalman.steadyStateGain(in A, in H, in Q, in R, ref KssBad));

        A.Dispose(); H.Dispose(); Q.Dispose(); R.Dispose(); Kss.Dispose();
        Abad.Dispose(); Hbad.Dispose(); KssBad.Dispose();
    }
}
