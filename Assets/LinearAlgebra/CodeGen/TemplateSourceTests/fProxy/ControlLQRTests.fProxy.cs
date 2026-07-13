using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// FULL test battery for Control.lqr / Control.lqr(warm) / Control.lqrSchedule. The coder's smoke
// tests live in ControlTests.fProxy.cs; this file is the
// exhaustive battery: published literature gains, SDA-vs-recursion cross-check, S-symmetric-PSD +
// closed-loop-stability properties, warm-path perturbation reconvergence, general-m schedule vs a
// hand-computed Riccati step, unstabilizable divergence, semidefinite-R rank flagging, determinism.
//
// FIRSTPASS CONSTRAINT: Control.RiccatiStep/RiccatiIterate are internal and NOT reachable from the
// template-test firstpass compile (only from the generated assembly's InternalsVisibleTo). Everything
// here therefore goes through the PUBLIC API only; the recursion "oracle" is re-implemented from public
// Blas.dot + LU (RiccatiStepPublic below), and the DARE solution S is read back via the warm state's
// public S buffer. Templated (fProxy) so codegen emits a float and a double build; per-dtype tolerances
// via choose-markers (float | double). Every numeric assertion routes through Fail[0..3] like the smoke file.
public class fProxyControlLQRTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            // 1. literature vectors (two, per spec): a published dlqr instance + a hand-derivable scalar case
            LiteratureDoubleIntegrator,
            LiteratureScalarAnalytic,
            // 2. SDA cold solve == the plain Riccati recursion run to convergence, + DARE residual on S
            SdaMatchesRecursionOracle,
            // 3. properties on random stabilizable instances: S symmetric PSD; closed-loop |lambda| < 1
            PropertiesPsdStability,
            // 4. warm path: perturb A ~1e-3, warm re-solve converges fast to the cold-of-perturbed solution
            WarmPerturbation,
            // 5. schedule: N=1 general-m block == a hand-computed single Riccati step; large-N -> infinite K
            ScheduleHandStepGeneralM,
            ScheduleApproachesInfiniteHorizon,
            // 6. failure modes: unstabilizable -> Diverged (prompt); semidefinite R -> rankDeficient flagged
            FailureUnstabilizable,
            FailureRankDeficientR,
            // 7. determinism
            DeterminismRandom,
            WarmDeterminism,
        }

        public TestType Type;
        public NativeArray<fProxy> Fail;   // [0]=flag [1]=got [2]=expected/limit [3]=diff/extra

        public void Execute()
        {
            switch (Type)
            {
                case TestType.LiteratureDoubleIntegrator: LiteratureDoubleIntegrator(); break;
                case TestType.LiteratureScalarAnalytic: LiteratureScalarAnalytic(); break;
                case TestType.SdaMatchesRecursionOracle: SdaMatchesRecursionOracle(); break;
                case TestType.PropertiesPsdStability: PropertiesPsdStability(); break;
                case TestType.WarmPerturbation: WarmPerturbation(); break;
                case TestType.ScheduleHandStepGeneralM: ScheduleHandStepGeneralM(); break;
                case TestType.ScheduleApproachesInfiniteHorizon: ScheduleApproachesInfiniteHorizon(); break;
                case TestType.FailureUnstabilizable: FailureUnstabilizable(); break;
                case TestType.FailureRankDeficientR: FailureRankDeficientR(); break;
                case TestType.DeterminismRandom: DeterminismRandom(); break;
                case TestType.WarmDeterminism: WarmDeterminism(); break;
            }
        }

        // ---- per-dtype tolerances (loose for float, tight for double; float needs ~sqrt(eps) slack) ----
        static fProxy LitKTol() => /*+choose[5e-3f|1e-6]*/5e-3f/*-choose*/;   // published gain, |K|~O(1)
        static fProxy OracleFloor() => /*+choose[1e-4f|1e-11]*/1e-4f/*-choose*/; // oracle-loop stop
        static fProxy CompareTol() => /*+choose[5e-3f|1e-5]*/5e-3f/*-choose*/;   // SDA-vs-oracle rel Frob
        static fProxy DareTol() => /*+choose[5e-3f|1e-6]*/5e-3f/*-choose*/;      // DARE residual on S
        static fProxy PsdEigFloor() => /*+choose[-3e-3f|-1e-9]*/-3e-3f/*-choose*/; // min eig(S) >= floor
        static fProxy SymTol() => /*+choose[1e-4f|1e-9]*/1e-4f/*-choose*/;

        // ============================ 1. literature vectors ============================

        // Published discrete double integrator: A=[[1,1],[0,1]], B=[[0],[1]], Q=I2, R=1. The optimal gain
        // K = [0.42208244, 1.24392885] and S = [[2.94712297,2.36920541],[2.36920541,4.61313426]] were
        // produced by SciPy's reference DARE solver scipy.linalg.solve_discrete_are + the standard
        // K=(R+BᵀSB)⁻¹BᵀSA formula (docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html ,
        // control.dlqr https://python-control.readthedocs.io/en/latest/generated/control.dlqr.html ).
        void LiteratureDoubleIntegrator()
        {
            var A = Mat2(1, 1, 0, 1);
            var B = new fProxyMxN(2, 1, Allocator.Temp); B[0, 0] = (fProxy)0; B[1, 0] = (fProxy)1;
            var Q = Eye(2); var R = R1(1);
            var K = new fProxyMxN(1, 2, Allocator.Temp);
            var state = new fProxyLQRState(2, Allocator.Temp);

            var info = Control.lqr(in A, in B, in Q, in R, ref K, ref state);   // warm overload -> state.S = S
            AssertTrue(info.status == LQRStatus.Converged);

            AssertClose(K[0, 0], (fProxy)0.42208244, LitKTol());
            AssertClose(K[0, 1], (fProxy)1.24392885, LitKTol());
            // S recovered from the warm state matches the published Riccati solution
            AssertClose(state.S[0, 0], (fProxy)2.94712297, (fProxy)5e-2);
            AssertClose(state.S[0, 1], (fProxy)2.36920541, (fProxy)5e-2);
            AssertClose(state.S[1, 1], (fProxy)4.61313426, (fProxy)5e-2);
            AssertLess(ClosedLoopSpecRad(in A, in B, in K), (fProxy)1);

            state.Dispose(); A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); K.Dispose();
        }

        // Scalar system a=2, b=1, q=1, r=1. Scalar DARE s = q + a²s·r/(r+b²s) -> s²-4s-1=0 -> s = 2+√5,
        // and K = b·s·a/(r+b²s) = 2s/(1+s) = (1+√5)/2 (the golden ratio). Both exact, hand-derived (the
        // "quadratic formula solvable in the test" second literature vector the spec allows).
        void LiteratureScalarAnalytic()
        {
            double sqrt5 = math.sqrt(5.0);
            fProxy sExact = (fProxy)(2.0 + sqrt5);
            fProxy kExact = (fProxy)((1.0 + sqrt5) / 2.0);

            var A = R1(2); var B = R1(1); var Q = R1(1); var R = R1(1);
            var K = new fProxyMxN(1, 1, Allocator.Temp);
            var state = new fProxyLQRState(1, Allocator.Temp);

            var info = Control.lqr(in A, in B, in Q, in R, ref K, ref state);
            AssertTrue(info.status == LQRStatus.Converged);
            AssertClose(K[0, 0], kExact, LitKTol());
            AssertClose(state.S[0, 0], sExact, (fProxy)5e-2);
            AssertLess(ClosedLoopSpecRad(in A, in B, in K), (fProxy)1);

            state.Dispose(); A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); K.Dispose();
        }

        // ============ 2. SDA (cold lqr) == plain Riccati recursion to convergence + DARE residual ============
        // Random stabilizable instances over the spec's grid n∈{2,4,8,12}, m∈{1,2,4}. For each: cold lqr
        // gives K_sda and (via the warm state) S_sda; a public-kernel recursion from S0=0 gives the oracle
        // S/K; assert both match (rel Frobenius). Also assert S_sda solves the DARE (residual ~ 0). Any
        // pathological random draw whose cold solve doesn't converge is SKIPPED, not asserted on (per the
        // task's stabilizability guard) -- a tested-count floor keeps the case non-vacuous.
        void SdaMatchesRecursionOracle()
        {
            var rng = new Unity.Mathematics.Random(0xA53Cu);
            int tested = 0;
            for (int ni = 0; ni < 4; ni++)
            {
                int n = ni == 0 ? 2 : ni == 1 ? 4 : ni == 2 ? 8 : 12;
                for (int mi = 0; mi < 3; mi++)
                {
                    int m = mi == 0 ? 1 : mi == 1 ? 2 : 4;
                    BuildRandom(ref rng, n, m, out var A, out var B, out var Q, out var R);

                    var Ksda = new fProxyMxN(m, n, Allocator.Temp);
                    var state = new fProxyLQRState(n, Allocator.Temp);
                    var info = Control.lqr(in A, in B, in Q, in R, ref Ksda, ref state);

                    if (info.status == LQRStatus.Converged)
                    {
                        // oracle: iterate the shared recursion from S=0 to convergence via public kernels
                        var Sor = new fProxyMxN(n, n, Allocator.Temp);
                        var Snx = new fProxyMxN(n, n, Allocator.Temp);
                        var Kor = new fProxyMxN(m, n, Allocator.Temp);
                        double floor = (double)OracleFloor();
                        for (int it = 0; it < 20000; it++)
                        {
                            RiccatiStepPublic(in A, in B, in Q, in R, in Sor, ref Snx, ref Kor);
                            double rel = FrobDiff(in Snx, in Sor) / math.max(1.0, FrobNorm(in Snx));
                            Sor.Data.CopyFrom(Snx.Data);
                            if (rel <= floor) break;
                        }
                        RiccatiStepPublic(in A, in B, in Q, in R, in Sor, ref Snx, ref Kor);

                        double sNorm = math.max(1.0, FrobNorm(in state.S));
                        AssertLEd(FrobDiff(in state.S, in Sor) / sNorm, (double)CompareTol());
                        AssertLEd(FrobDiff(in Ksda, in Kor) / math.max(1.0, FrobNorm(in Kor)), (double)CompareTol());

                        // DARE residual: one more Riccati step on S_sda must return S_sda (fixed point)
                        var Sstep = new fProxyMxN(n, n, Allocator.Temp);
                        var Kstep = new fProxyMxN(m, n, Allocator.Temp);
                        RiccatiStepPublic(in A, in B, in Q, in R, in state.S, ref Sstep, ref Kstep);
                        AssertLEd(FrobDiff(in Sstep, in state.S) / sNorm, (double)DareTol());

                        Sor.Dispose(); Snx.Dispose(); Kor.Dispose(); Sstep.Dispose(); Kstep.Dispose();
                        tested++;
                    }

                    state.Dispose(); Ksda.Dispose();
                    A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose();
                }
            }
            AssertTrue(tested >= 6);   // most of the 12 draws must be genuinely tested
        }

        // ============ 3. S symmetric PSD + closed-loop stability on random stabilizable instances ============
        void PropertiesPsdStability()
        {
            var rng = new Unity.Mathematics.Random(0x1234u);
            int tested = 0;
            for (int s = 0; s < 8; s++)
            {
                int n = 2 + (s % 4) * 2;   // 2,4,6,8,2,4,6,8
                int m = 1 + (s % 3);       // 1,2,3,1,2,3,1,2
                BuildRandom(ref rng, n, m, out var A, out var B, out var Q, out var R);

                var K = new fProxyMxN(m, n, Allocator.Temp);
                var state = new fProxyLQRState(n, Allocator.Temp);
                var info = Control.lqr(in A, in B, in Q, in R, ref K, ref state);

                if (info.status == LQRStatus.Converged)
                {
                    // S symmetric
                    fProxy maxAsym = 0;
                    for (int i = 0; i < n; i++)
                        for (int j = i + 1; j < n; j++)
                        {
                            fProxy d = math.abs(state.S[i, j] - state.S[j, i]);
                            if (d > maxAsym) maxAsym = d;
                        }
                    AssertClose(maxAsym, (fProxy)0, SymTol());

                    // S PSD: min eigenvalue >= floor (S symmetric -> real spectrum)
                    AssertGE(MinEig(in state.S), PsdEigFloor());

                    // closed-loop stability
                    AssertLess(ClosedLoopSpecRad(in A, in B, in K), (fProxy)1);
                    tested++;
                }

                state.Dispose(); K.Dispose(); A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose();
            }
            AssertTrue(tested >= 4);
        }

        // ============ 4. warm path: 1e-3 A-perturbation reconverges fast to the cold-of-perturbed K ============
        // Base = double integrator (deterministic). Cold solve populates the state; perturb A[0,1] by 1e-3
        // relative; warm re-solve must Converge in a small iteration count and land on a fresh cold solve
        // of the perturbed system (measured warm ~2 float / ~8 double, cold-recursion ~7/~13 -- generous
        // absolute bound, since warm-vs-cold step counts are code-path-sensitive, not exactly pinnable).
        void WarmPerturbation()
        {
            var A = Mat2(1, 1, 0, 1);
            var B = new fProxyMxN(2, 1, Allocator.Temp); B[0, 0] = (fProxy)0; B[1, 0] = (fProxy)1;
            var Q = Eye(2); var R = R1(1);

            var state = new fProxyLQRState(2, Allocator.Temp);
            var Kcold = new fProxyMxN(1, 2, Allocator.Temp);
            var coldInfo = Control.lqr(in A, in B, in Q, in R, ref Kcold, ref state);
            AssertTrue(coldInfo.status == LQRStatus.Converged);
            AssertTrue(state.populated);

            var Ap = Mat2(1, (fProxy)(1.0 * (1.0 + 1e-3)), 0, 1);   // perturb the (0,1) entry by 1e-3 rel

            var Kwarm = new fProxyMxN(1, 2, Allocator.Temp);
            var warmInfo = Control.lqr(in Ap, in B, in Q, in R, ref Kwarm, ref state);
            AssertTrue(warmInfo.status == LQRStatus.Converged);
            AssertLE(warmInfo.iterations, 30);   // warm ≪ any cap; generous absolute margin

            // fresh cold solve of the perturbed system
            var Kpert = new fProxyMxN(1, 2, Allocator.Temp);
            var pInfo = Control.lqr(in Ap, in B, in Q, in R, ref Kpert);
            AssertTrue(pInfo.status == LQRStatus.Converged);
            for (int j = 0; j < 2; j++) AssertClose(Kwarm[0, j], Kpert[0, j], (fProxy)5e-3);

            state.Dispose(); A.Dispose(); Ap.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose();
            Kcold.Dispose(); Kwarm.Dispose(); Kpert.Dispose();
        }

        // ============ 5a. schedule N=1 (general m) == a hand-computed single Riccati step from Qf ============
        // n=3, m=2 random full-rank instance. lqrSchedule(N=1, Qf=Q) does exactly one RiccatiStep from
        // S=Qf; RiccatiStepPublic replicates it (LU on the full-rank R+BᵀSB gives the same K as production
        // CHOP). This exercises the m>1 matrix solve the smoke test's m=1 scalar case cannot.
        void ScheduleHandStepGeneralM()
        {
            var rng = new Unity.Mathematics.Random(0x5151u);
            int n = 3, m = 2;
            BuildRandom(ref rng, n, m, out var A, out var B, out var Q, out var R);

            var Ksched = new fProxyMxN(m, n, Allocator.Temp);
            var info = Control.lqrSchedule(in A, in B, in Q, in R, in Q, 1, ref Ksched);
            AssertTrue(info.status == LQRStatus.Converged);
            AssertEqInt(info.iterations, 1);

            var Shand = new fProxyMxN(n, n, Allocator.Temp);
            var Khand = new fProxyMxN(m, n, Allocator.Temp);
            RiccatiStepPublic(in A, in B, in Q, in R, in Q, ref Shand, ref Khand);   // one step from S=Qf=Q

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    AssertClose(Ksched[i, j], Khand[i, j], (fProxy)5e-3);

            Shand.Dispose(); Khand.Dispose(); Ksched.Dispose();
            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose();
        }

        // ============ 5b. large-N schedule, Qf=Q -> row-block 0 approaches the infinite-horizon K ============
        // Different instance from the smoke test's double integrator: n=3, m=2 random stabilizable.
        void ScheduleApproachesInfiniteHorizon()
        {
            var rng = new Unity.Mathematics.Random(0x9001u);
            int n = 3, m = 2;
            BuildRandom(ref rng, n, m, out var A, out var B, out var Q, out var R);

            var Kinf = new fProxyMxN(m, n, Allocator.Temp);
            var infoInf = Control.lqr(in A, in B, in Q, in R, ref Kinf);
            AssertTrue(infoInf.status == LQRStatus.Converged);

            int N = 80;
            var Ksched = new fProxyMxN(N * m, n, Allocator.Temp);
            var info = Control.lqrSchedule(in A, in B, in Q, in R, in Q, N, ref Ksched);
            AssertTrue(info.status == LQRStatus.Converged);
            AssertLEd(info.residual, (double)DareTol());   // deeply converged at k=0

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    AssertClose(Ksched[i, j], Kinf[i, j], (fProxy)5e-3);

            Kinf.Dispose(); Ksched.Dispose();
            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose();
        }

        // ============ 6a. unstabilizable -> Diverged, promptly (uncontrollable unstable mode) ============
        // A=diag(2,.5), B=[0;1]: the mode with eigenvalue 2 is uncontrollable, so no feedback stabilizes.
        // SDA's H_k blows through the data-scaled threshold within a handful of doublings.
        void FailureUnstabilizable()
        {
            var A = Mat2(2, 0, 0, (fProxy)0.5);
            var B = new fProxyMxN(2, 1, Allocator.Temp); B[0, 0] = (fProxy)0; B[1, 0] = (fProxy)1;
            var Q = Eye(2); var R = R1(1);
            var K = new fProxyMxN(1, 2, Allocator.Temp);

            var info = Control.lqr(in A, in B, in Q, in R, ref K);
            AssertTrue(info.status == LQRStatus.Diverged);
            AssertLE(info.iterations, 15);   // fails fast (hand estimate ~5), well under the 50 cap
            AssertTrue(info.residual == double.PositiveInfinity);

            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); K.Dispose();
        }

        // ============ 6b. semidefinite R (redundant actuator) -> rankDeficientControl flagged, K stabilizes ==
        // Double-integrator dynamics with a DUPLICATED control column, B=[b b], and R=diag(1,0): the second
        // actuator is a redundant copy with zero cost. R itself is rank 1 < 2, so the COLD SDA path's CHOP
        // on R while forming G0=BR⁺Bᵀ reports rank-deficiency and flags rankDeficientControl (the only
        // entry that sees a raw semidefinite R -- R+BᵀSB is full rank here, so the warm/schedule paths
        // would NOT flag it, hence targeting the cold lqr). The returned min-norm K still stabilizes.
        void FailureRankDeficientR()
        {
            var A = Mat2(1, 1, 0, 1);
            var B = new fProxyMxN(2, 2, Allocator.Temp);
            B[0, 0] = (fProxy)0; B[1, 0] = (fProxy)1; B[0, 1] = (fProxy)0; B[1, 1] = (fProxy)1;   // duplicate column
            var Q = Eye(2);
            var R = new fProxyMxN(2, 2, Allocator.Temp); R[0, 0] = (fProxy)1; R[1, 1] = (fProxy)0;   // semidefinite
            var K = new fProxyMxN(2, 2, Allocator.Temp);

            var info = Control.lqr(in A, in B, in Q, in R, ref K);
            AssertTrue(info.status == LQRStatus.Converged);
            AssertTrue(info.rankDeficientControl);
            AssertLess(ClosedLoopSpecRad(in A, in B, in K), (fProxy)1);   // still stabilizing

            A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); K.Dispose();
        }

        // ============ 7. determinism ============

        // Two back-to-back cold solves on the same random n=6/m=2 instance -> bit-identical K and equal
        // iteration count (single-threaded, RNG-free solve path).
        void DeterminismRandom()
        {
            var rng = new Unity.Mathematics.Random(0x7777u);
            int n = 6, m = 2;
            BuildRandom(ref rng, n, m, out var A, out var B, out var Q, out var R);

            var K1 = new fProxyMxN(m, n, Allocator.Temp);
            var i1 = Control.lqr(in A, in B, in Q, in R, ref K1);
            var K2 = new fProxyMxN(m, n, Allocator.Temp);
            var i2 = Control.lqr(in A, in B, in Q, in R, ref K2);

            AssertTrue(i1.status == LQRStatus.Converged);
            AssertEqInt(i1.iterations, i2.iterations);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    AssertClose(K1[i, j], K2[i, j], (fProxy)0);   // exact

            K1.Dispose(); K2.Dispose(); A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose();
        }

        // Warm path determinism: two independent (fresh state -> cold -> warm) sequences on the same
        // inputs produce bit-identical warm gains.
        void WarmDeterminism()
        {
            var A = Mat2(1, 1, 0, 1);
            var B = new fProxyMxN(2, 1, Allocator.Temp); B[0, 0] = (fProxy)0; B[1, 0] = (fProxy)1;
            var Q = Eye(2); var R = R1(1);

            var Kw1 = WarmSequence(in A, in B, in Q, in R, out int it1);
            var Kw2 = WarmSequence(in A, in B, in Q, in R, out int it2);

            AssertEqInt(it1, it2);
            for (int j = 0; j < 2; j++) AssertClose(Kw1[0, j], Kw2[0, j], (fProxy)0);

            Kw1.Dispose(); Kw2.Dispose(); A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose();
        }

        fProxyMxN WarmSequence(in fProxyMxN A, in fProxyMxN B, in fProxyMxN Q, in fProxyMxN R, out int warmIters)
        {
            var state = new fProxyLQRState(2, Allocator.Temp);
            var Kcold = new fProxyMxN(1, 2, Allocator.Temp);
            Control.lqr(in A, in B, in Q, in R, ref Kcold, ref state);
            var Kwarm = new fProxyMxN(1, 2, Allocator.Temp);
            var wi = Control.lqr(in A, in B, in Q, in R, ref Kwarm, ref state);
            warmIters = wi.iterations;
            Kcold.Dispose(); state.Dispose();
            return Kwarm;   // caller disposes
        }

        // ================================ helpers ================================

        // One Riccati/DARE step S⁻ = Q + AᵀSA - AᵀSB(R+BᵀSB)⁻¹BᵀSA, K = (R+BᵀSB)⁻¹BᵀSA via PUBLIC kernels
        // (Blas.dot + LU). Mirrors internal Control.RiccatiStep; LU on the full-rank SPD (R+BᵀSB) yields
        // the same K as production's CHOP. Snext is re-symmetrized like production.
        static void RiccatiStepPublic(in fProxyMxN A, in fProxyMxN B, in fProxyMxN Q, in fProxyMxN R,
                                      in fProxyMxN S, ref fProxyMxN Snext, ref fProxyMxN K)
        {
            int n = A.M_Rows, m = B.N_Cols;

            var SB = new fProxyMxN(n, m, Allocator.Temp);
            Blas.dot(in S, in B, ref SB);                       // SB = S*B
            var Rbar = new fProxyMxN(m, m, Allocator.Temp);
            Blas.dot(in B, in SB, ref Rbar, transposeA: true);  // BᵀSB
            for (int i = 0; i < m; i++)
                for (int j = 0; j < m; j++) Rbar[i, j] += R[i, j];
            var BSA = new fProxyMxN(m, n, Allocator.Temp);
            Blas.dot(in SB, in A, ref BSA, transposeA: true);   // (SB)ᵀA = BᵀSA

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) K[i, j] = BSA[i, j];
            var P = new Pivot(m, Allocator.Temp);
            LU.decompInPlace(ref Rbar, ref P);
            LU.decompSolve(ref Rbar, in P, ref K);              // K = Rbar⁻¹ BSA

            var SA = new fProxyMxN(n, n, Allocator.Temp);
            Blas.dot(in S, in A, ref SA);                       // S*A
            var AtSA = new fProxyMxN(n, n, Allocator.Temp);
            Blas.dot(in A, in SA, ref AtSA, transposeA: true);  // AᵀSA
            var AtSBK = new fProxyMxN(n, n, Allocator.Temp);
            Blas.dot(in BSA, in K, ref AtSBK, transposeA: true); // (BᵀSA)ᵀK = AᵀSB*K

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) Snext[i, j] = Q[i, j] + AtSA[i, j] - AtSBK[i, j];
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    fProxy avg = (Snext[i, j] + Snext[j, i]) / (fProxy)2;
                    Snext[i, j] = avg; Snext[j, i] = avg;
                }

            SB.Dispose(); Rbar.Dispose(); BSA.Dispose(); P.Dispose();
            SA.Dispose(); AtSA.Dispose(); AtSBK.Dispose();
        }

        // Random stabilizable instance: A random then scaled to spectral radius ~1.05 (stabilizability
        // then holds generically for a random B), random B, Q=I, R=I.
        static void BuildRandom(ref Unity.Mathematics.Random rng, int n, int m,
                                out fProxyMxN A, out fProxyMxN B, out fProxyMxN Q, out fProxyMxN R)
        {
            A = new fProxyMxN(n, n, Allocator.Temp);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) A[i, j] = (fProxy)rng.NextFloat(-1f, 1f);
            fProxy sr = SpectralRadius(in A);
            fProxy scale = sr > (fProxy)1e-6 ? (fProxy)1.05 / sr : (fProxy)1;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) A[i, j] *= scale;

            B = new fProxyMxN(n, m, Allocator.Temp);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++) B[i, j] = (fProxy)rng.NextFloat(-1f, 1f);
            Q = Eye(n);
            R = new fProxyMxN(m, m, Allocator.Temp);
            for (int i = 0; i < m; i++) R[i, i] = (fProxy)1;
        }

        static fProxy SpectralRadius(in fProxyMxN M)
        {
            int n = M.M_Rows;
            var C = new fProxyMxN(in M, Allocator.Temp);   // valuesQRInPlace is destructive
            var er = new fProxyN(n, Allocator.Temp);
            var ei = new fProxyN(n, Allocator.Temp);
            Eigen.valuesQRInPlace(ref C, ref er, ref ei);
            fProxy mx = 0;
            for (int i = 0; i < n; i++)
            {
                fProxy mg = math.sqrt(er[i] * er[i] + ei[i] * ei[i]);
                if (mg > mx) mx = mg;
            }
            C.Dispose(); er.Dispose(); ei.Dispose();
            return mx;
        }

        static fProxy ClosedLoopSpecRad(in fProxyMxN A, in fProxyMxN B, in fProxyMxN K)
        {
            int n = A.M_Rows;
            var Acl = new fProxyMxN(n, n, Allocator.Temp);
            Blas.dot(in B, in K, ref Acl);                       // B*K
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) Acl[i, j] = A[i, j] - Acl[i, j];
            fProxy sr = SpectralRadius(in Acl);
            Acl.Dispose();
            return sr;
        }

        // Minimum eigenvalue of a symmetric matrix (real spectrum) via valuesQRInPlace.
        static fProxy MinEig(in fProxyMxN M)
        {
            int n = M.M_Rows;
            var C = new fProxyMxN(in M, Allocator.Temp);
            var er = new fProxyN(n, Allocator.Temp);
            var ei = new fProxyN(n, Allocator.Temp);
            Eigen.valuesQRInPlace(ref C, ref er, ref ei);
            fProxy mn = er[0];
            for (int i = 1; i < n; i++) if (er[i] < mn) mn = er[i];
            C.Dispose(); er.Dispose(); ei.Dispose();
            return mn;
        }

        static double FrobNorm(in fProxyMxN M)
        {
            double s = 0;
            for (int i = 0; i < M.M_Rows; i++)
                for (int j = 0; j < M.N_Cols; j++) { double v = (double)M[i, j]; s += v * v; }
            return math.sqrt(s);
        }

        static double FrobDiff(in fProxyMxN A, in fProxyMxN B)
        {
            double s = 0;
            for (int i = 0; i < A.M_Rows; i++)
                for (int j = 0; j < A.N_Cols; j++) { double v = (double)A[i, j] - (double)B[i, j]; s += v * v; }
            return math.sqrt(s);
        }

        static fProxyMxN Mat2(fProxy a, fProxy b, fProxy c, fProxy d)
        {
            var M = new fProxyMxN(2, 2, Allocator.Temp);
            M[0, 0] = a; M[0, 1] = b; M[1, 0] = c; M[1, 1] = d;
            return M;
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

        // ---- Fail[0..3] diagnostic asserts (same shape as ControlTests.fProxy.cs) ----
        void AssertTrue(bool cond)
        {
            if (!cond && Fail[0] == (fProxy)0) { Fail[0] = (fProxy)1; Fail[1] = (fProxy)0; Fail[2] = (fProxy)1; Fail[3] = (fProxy)0; }
            Assert.IsTrue(cond);
        }

        void AssertEqInt(int got, int expected)
        {
            if (got != expected && Fail[0] == (fProxy)0) { Fail[0] = (fProxy)1; Fail[1] = (fProxy)got; Fail[2] = (fProxy)expected; Fail[3] = (fProxy)(got - expected); }
            Assert.IsTrue(got == expected);
        }

        void AssertLE(int got, int limit)
        {
            if (!(got <= limit) && Fail[0] == (fProxy)0) { Fail[0] = (fProxy)1; Fail[1] = (fProxy)got; Fail[2] = (fProxy)limit; Fail[3] = (fProxy)(got - limit); }
            Assert.IsTrue(got <= limit);
        }

        void AssertLEd(double got, double limit)
        {
            if (!(got <= limit) && Fail[0] == (fProxy)0) { Fail[0] = (fProxy)1; Fail[1] = (fProxy)got; Fail[2] = (fProxy)limit; Fail[3] = (fProxy)(got - limit); }
            Assert.IsTrue(got <= limit);
        }

        void AssertLess(fProxy got, fProxy limit)
        {
            if (!(got < limit) && Fail[0] == (fProxy)0) { Fail[0] = (fProxy)1; Fail[1] = got; Fail[2] = limit; Fail[3] = got - limit; }
            Assert.IsTrue(got < limit);
        }

        void AssertGE(fProxy got, fProxy limit)
        {
            if (!(got >= limit) && Fail[0] == (fProxy)0) { Fail[0] = (fProxy)1; Fail[1] = got; Fail[2] = limit; Fail[3] = got - limit; }
            Assert.IsTrue(got >= limit);
        }

        void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0) { Fail[0] = (fProxy)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff; }
            Assert.IsTrue(diff <= precision);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void ControlLQRTests(TestJob.TestType type)
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

    // ---- extra managed-thread validation throws not already in ControlTests.fProxy.cs ----

    [Test]
    public void LqrThrowsOnNonFiniteR()
    {
        var A = new fProxyMxN(2, 2, Allocator.Temp);
        A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)1; A[1, 0] = (fProxy)0; A[1, 1] = (fProxy)1;
        var B = new fProxyMxN(2, 1, Allocator.Temp); B[1, 0] = (fProxy)1;
        var Q = new fProxyMxN(2, 2, Allocator.Temp); Q[0, 0] = (fProxy)1; Q[1, 1] = (fProxy)1;
        var K = new fProxyMxN(1, 2, Allocator.Temp);

        var Rnan = new fProxyMxN(1, 1, Allocator.Temp); Rnan[0, 0] = (fProxy)float.NaN;
        Assert.Catch<ArgumentException>(() => Control.lqr(in A, in B, in Q, in Rnan, ref K));

        A.Dispose(); B.Dispose(); Q.Dispose(); K.Dispose(); Rnan.Dispose();
    }

    [Test]
    public void LqrScheduleThrowsOnNonSquareQf()
    {
        var A = new fProxyMxN(2, 2, Allocator.Temp);
        A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)1; A[1, 0] = (fProxy)0; A[1, 1] = (fProxy)1;
        var B = new fProxyMxN(2, 1, Allocator.Temp); B[1, 0] = (fProxy)1;
        var Q = new fProxyMxN(2, 2, Allocator.Temp); Q[0, 0] = (fProxy)1; Q[1, 1] = (fProxy)1;
        var R = new fProxyMxN(1, 1, Allocator.Temp); R[0, 0] = (fProxy)1;
        var Ksched = new fProxyMxN(2, 2, Allocator.Temp);

        var QfBad = new fProxyMxN(2, 3, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => Control.lqrSchedule(in A, in B, in Q, in R, in QfBad, 1, ref Ksched));

        A.Dispose(); B.Dispose(); Q.Dispose(); R.Dispose(); Ksched.Dispose(); QfBad.Dispose();
    }
}
