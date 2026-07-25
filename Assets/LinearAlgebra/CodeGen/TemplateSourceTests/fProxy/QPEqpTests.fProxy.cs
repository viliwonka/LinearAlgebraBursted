using System;
using System.Collections.Generic;

using BULA;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

// Acceptance battery for the equality-constrained QP (EQP) kernel (QP.eqpSolve /
// QP.eqpNullSpaceStep). Both entry points are `internal` -- reached
// here via the assembly-wide grant InternalsVisibleTo("BurstLinearAlgebra.TemplateSource.Tests-firstpass")
// on TemplateSource/AssemblyInfo.cs (the same grant fProxyChooseMarkerDemo already relies on) for this
// firstpass compile, and via InternalsVisibleTo("BurstLinearAlgebra.Tests") for the generated
// float/double test assembly.
//
// Burst execution: compute runs inside [BurstCompile(CompileSynchronously = true)] IJob structs; NUnit
// Assert.IsTrue with == only inside the job, first failure recorded into a Fail[] diagnostic array read
// back on the managed side.
//
// ---- The acceptance oracle ----
//
// SPD Q: assemble and solve the full KKT saddle system directly with the library LU on the SAME
// instance and assert eqpSolve's (x, lambda) agree componentwise to factor tolerance. The KKT system
// (sign convention VERIFIED against the kernel, which recovers lambda from A_Wᵀlambda = Qx+c) is
//
//     [ Q   -A_Wᵀ ] [ x ]   [ -c  ]
//     [ A_W   0   ] [ λ ] = [ b_W ]
//
// The saddle matrix is nonsingular whenever Q is PD on null(A_W) and A_W has full row rank, so LU
// with partial pivoting solves it and x is UNIQUE -- componentwise comparison is well-defined.
//
// PSD-singular Q (Q = LᵀL, rank r = n/2): the reduced Hessian ZᵀQZ can be singular (kernel takes its
// regularized-Cholesky retry) and the full KKT matrix is then singular too, so the LU oracle does NOT
// apply and the constrained minimizer x is NOT unique. Instead the instance is made WELL-POSED (not
// genuinely unbounded -- Stage-1 Unbounded detection is documented-weak and must not be exercised
// here) by choosing c = -Q x_feasible with x_feasible = LQ.minNormSolve(A_W x = b_W): then x_feasible
// is a global unconstrained minimizer (gradient Qx_f + c = 0) that also happens to be feasible, so the
// constrained optimum has the KNOWN objective -½ x_fᵀQ x_f regardless of which minimizer is returned.
// Verification there: status Optimal, feasibility ~0, objective == the known optimum, and the KKT
// stationarity/multiplier residual ‖Qx + c - A_Wᵀλ‖∞ ~ 0 (all oracle-free, minimizer-independent).
//
// Both modes additionally assert the QPInfo.stationarityResidual / .feasibilityResidual diagnostics
// are ~0 on Optimal (handoff requirement) and cross-check them against an independently recomputed
// KKT residual.
//
// Coverage (handoff): n up to 64; k in {1, 2, n/4, n/2, n-1, n} (k=2 is MANDATORY -- it caught a real
// FormNullSpaceBasis reflector bug in development that k=1 cannot expose; k=n exercises the fully
// determined nz==0 branch; tiny n=1/n=2 cover the 1x1 / minimal cases); several fixed seeds. k=0 (empty
// working set) must throw ArgumentException.
public class fProxyQPEqpTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EqpJob : IJob
    {
        public int Mode;   // 0 = random SPD Q (KKT-LU oracle); 1 = PSD-singular Q (objective oracle)
        public int N, K, Seed;

        // [0]=failure flag, [1]=checkId, [2]=got, [3]=limit/expected, [4]=diff
        public NativeArray<double> Fail;

        public void Execute()
        {
            var rng = new Random((uint)Seed | 1u);
            int n = N, k = K;

            var Q = new fProxyMxN(n, n, Allocator.Temp);
            var c = new fProxyN(n, Allocator.Temp);
            var A = new fProxyMxN(k, n, Allocator.Temp);
            var b = new fProxyN(k, Allocator.Temp);
            var x = new fProxyN(n, Allocator.Temp, true);
            var lam = new fProxyN(k, Allocator.Temp, true);

            // A_W = the first k rows of a Haar-uniform random n x n orthogonal matrix: random and
            // independent as the spec asks, but well-conditioned (all singular values 1), so the
            // multiplier recovery A_Wᵀλ = g is stable and the componentwise LU-oracle comparison stays
            // faithful even at k -> n. (A random *Gaussian* square A_W is ill-conditioned near k = n,
            // which makes λ recovered via QR vs LU diverge by cond(A_W)·eps though both still satisfy
            // the KKT residual -- an oracle-sensitivity artifact, not a kernel error.)
            var Qo = new fProxyMxN(n, n, Allocator.Temp);
            Rand.orthogonalInPlace(ref rng, ref Qo);
            for (int i = 0; i < k; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = Qo[i, j];
            for (int i = 0; i < k; i++) b[i] = rng.NextFProxy(-1f, 1f);

            double objStar = 0;
            if (Mode == 0)
            {
                Rand.spdInPlace(ref rng, ref Q, 1f, 10f);        // eigenvalues in [1,10] -> cond ~ 10
                for (int i = 0; i < n; i++) c[i] = rng.NextFProxy(-1f, 1f);
            }
            else
            {
                int r = math.max(1, n / 2);
                var L = new fProxyMxN(r, n, Allocator.Temp);
                for (int i = 0; i < r; i++)
                    for (int j = 0; j < n; j++)
                        L[i, j] = rng.NextFProxy(-1f, 1f);
                Blas.dot(in L, in L, ref Q, transposeA: true);   // Q = LᵀL, symmetric PSD, rank r

                var xf = new fProxyN(n, Allocator.Temp);
                LQ.minNormSolve(in A, in b, ref xf);             // min-norm feasible point = kernel's start
                // Target a DIFFERENT feasible optimum xt = xf + w, w in null(A_W), so the null-space step
                // is nontrivial and the regularized-Cholesky path actually runs. A_W has orthonormal rows,
                // so the null projector is w = r - A_Wᵀ(A_W r) for a random r. Setting c = -Q xt makes xt
                // a global unconstrained minimizer (grad Qxt + c = 0) that is also feasible => the
                // constrained optimum, with known objective -½ xtᵀQ xt (independent of which minimizer is
                // returned when Q is singular along null(A_W)).
                var rr = new fProxyN(n, Allocator.Temp);
                for (int i = 0; i < n; i++) rr[i] = rng.NextFProxy(-1f, 1f);
                var Ar = new fProxyN(k, Allocator.Temp);
                Blas.dot(in A, in rr, ref Ar);                   // A_W r
                var xt = new fProxyN(n, Allocator.Temp);
                for (int i = 0; i < n; i++)
                {
                    double atr = 0;
                    for (int j = 0; j < k; j++) atr += (double)A[j, i] * (double)Ar[j];
                    xt[i] = xf[i] + rr[i] - (fProxy)atr;          // xf + (I - A_Wᵀ A_W) r
                }
                var Qxt = new fProxyN(n, Allocator.Temp);
                Blas.dot(in Q, in xt, ref Qxt);
                for (int i = 0; i < n; i++) c[i] = -Qxt[i];       // c = -Q xt
                double q = 0;
                for (int i = 0; i < n; i++) q += (double)xt[i] * (double)Qxt[i];
                objStar = -0.5 * q;                               // known optimal objective
            }

            var info = QP.eqpSolve(in Q, in c, in A, in b, ref x, ref lam);

            AssertTrue(1, info.status == QPStatus.Optimal);
            if (info.status != QPStatus.Optimal) { return; }

            double normQ = (double)Norms.LInf(in Q);
            double scale = 1.0 + normQ;

            // ---- Independently recomputed KKT residuals (both modes, oracle-free) ----
            var Qx = new fProxyN(n, Allocator.Temp);
            Blas.dot(in Q, in x, ref Qx);
            double statRes = 0;
            for (int i = 0; i < n; i++)
            {
                double gi = (double)Qx[i] + (double)c[i];
                double atl = 0;
                for (int j = 0; j < k; j++) atl += (double)A[j, i] * (double)lam[j];
                statRes = math.max(statRes, math.abs(gi - atl));
            }
            var Ax = new fProxyN(k, Allocator.Temp);
            Blas.dot(in A, in x, ref Ax);
            double feasRes = 0;
            for (int i = 0; i < k; i++) feasRes = math.max(feasRes, math.abs((double)Ax[i] - (double)b[i]));

            // Tolerances set just above the residuals observed across all fixed-seed cases (FloatMode.
            // Default; small margins for cross-arch float drift). Observed maxima: feas ~6e-7 (float)/
            // 1e-15 (double); Mode0 stationarity ~5.6e-6 (float)/1e-14 (double); Mode1 regularized
            // stationarity ~4e-3 (float)/1.4e-7 (double) -- the sqrt(eps)·‖Q‖ price of the regularized-
            // Cholesky retry, so it is ~‖Q‖·sqrt(eps) larger than the exact Mode0 step and scaled by
            // (1+‖Q‖∞) accordingly.
            double feasTol = /*+choose[5e-6|1e-12]*/5e-6/*-choose*/;
            double statTol = (Mode == 0 ? /*+choose[3e-6|1e-14]*/3e-6/*-choose*/ : /*+choose[3e-3|1e-7]*/3e-3/*-choose*/) * scale;

            AssertLE(2, feasRes, feasTol);                                  // recomputed feasibility
            AssertLE(3, statRes, statTol);                                  // recomputed KKT stationarity (checks lambda)
            AssertLE(4, info.feasibilityResidual, feasTol);                 // QPInfo diagnostic ~ 0
            AssertLE(5, info.stationarityResidual, statTol);                // QPInfo diagnostic ~ 0

            if (Mode == 0)
            {
                // ---- KKT-LU oracle: componentwise (x, lambda) agreement ----
                int m = n + k;
                var Kk = new fProxyMxN(m, m, Allocator.Temp);          // zero-initialized (bottom-right 0 block stays 0)
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        Kk[i, j] = Q[i, j];
                for (int j = 0; j < k; j++)
                    for (int i = 0; i < n; i++)
                    {
                        Kk[i, n + j] = -A[j, i];        // top-right -A_Wᵀ
                        Kk[n + j, i] = A[j, i];         // bottom-left A_W
                    }
                var rhs = new fProxyN(m, Allocator.Temp);
                for (int i = 0; i < n; i++) rhs[i] = -c[i];
                for (int j = 0; j < k; j++) rhs[n + j] = b[j];
                var piv = new Pivot(m, Allocator.Temp);
                var lu = LU.solveInPlace(ref Kk, ref piv, ref rhs);
                AssertTrue(6, lu.Solved);

                double dx = 0, dl = 0;
                for (int i = 0; i < n; i++) dx = math.max(dx, math.abs((double)x[i] - (double)rhs[i]));
                for (int j = 0; j < k; j++) dl = math.max(dl, math.abs((double)lam[j] - (double)rhs[n + j]));
                // observed ~7e-7 (float)/1.3e-15 (double) for x; ~9e-6 (float)/1.4e-14 (double) for lambda
                AssertLE(7, dx, /*+choose[1e-4|1e-13]*/1e-4/*-choose*/);
                AssertLE(8, dl, /*+choose[1e-4|1e-13]*/1e-4/*-choose*/);
            }
            else
            {
                // ---- objective oracle (minimizer-independent) ----
                // observed |Δobj| ~9e-6 (float)/2.4e-14 (double)
                double objTol = /*+choose[5e-5|1e-12]*/5e-5/*-choose*/ + /*+choose[1e-5|1e-11]*/1e-5/*-choose*/ * (1.0 + math.abs(objStar));
                AssertCloseD(9, info.objective, objStar, objTol);
                AssertTrue(10, math.isfinite((fProxy)info.objective));
            }
        }

        void RecordFail(int id, double got, double limit, double diff)
        {
            if (Fail[0] == 0) { Fail[0] = 1; Fail[1] = id; Fail[2] = got; Fail[3] = limit; Fail[4] = diff; }
        }
        void AssertTrue(int id, bool cond)
        {
            if (!cond) RecordFail(id, 0, 1, 0);
            Assert.IsTrue(cond);
        }
        void AssertLE(int id, double val, double limit)
        {
            bool ok = val <= limit;
            if (!ok) RecordFail(id, val, limit, val - limit);
            Assert.IsTrue(ok);
        }
        void AssertCloseD(int id, double a, double b, double tol)
        {
            double diff = math.abs(a - b);
            bool ok = diff <= tol;
            if (!ok) RecordFail(id, a, b, diff);
            Assert.IsTrue(ok);
        }
    }

    // ---- case sources ----

    // SPD (Mode 0): the KKT-LU componentwise oracle. n in {1,2,8,16,32,64}, k in {1,2,n/4,n/2,n-1,n}
    // (deduped, clamped to 1..n). Three seeds for the mid sizes; one for the tiny and the largest.
    static IEnumerable<TestCaseData> SpdCases()
    {
        int[] seeds = { 12345, 67890, 20260709 };
        int[] ns = { 1, 2, 8, 16, 32, 64 };
        foreach (int n in ns)
        {
            var ks = new SortedSet<int> { 1, 2, n / 4, n / 2, n - 1, n };
            foreach (int k in ks)
            {
                if (k < 1 || k > n) continue;
                int nSeeds = (n <= 2 || n >= 64) ? 1 : seeds.Length;
                for (int s = 0; s < nSeeds; s++)
                    yield return new TestCaseData(n, k, seeds[s]).SetName($"Spd_n{n}_k{k}_s{seeds[s]}");
            }
        }
    }

    // PSD-singular (Mode 1): objective + KKT-residual oracle. k chosen to straddle the reduced-Hessian
    // singularity boundary nz = n-k vs r = n/2: k < n/2 -> reduced Hessian singular (regularized-Cholesky
    // retry fires); k >= n/2 -> nonsingular. Two seeds each.
    static IEnumerable<TestCaseData> SingularCases()
    {
        int[] seeds = { 12345, 67890 };
        (int n, int k)[] combos =
        {
            (8, 2), (8, 4), (8, 6),
            (16, 2), (16, 8), (16, 12),
            (32, 2), (32, 8), (32, 24),
        };
        foreach (var (n, k) in combos)
            foreach (int s in seeds)
                yield return new TestCaseData(n, k, s).SetName($"Singular_n{n}_k{k}_s{s}");
    }

    [TestCaseSource(nameof(SpdCases))]
    public void Eqp_Spd(int n, int k, int seed) => RunEqp(0, n, k, seed);

    [TestCaseSource(nameof(SingularCases))]
    public void Eqp_Singular(int n, int k, int seed) => RunEqp(1, n, k, seed);

    static void RunEqp(int mode, int n, int k, int seed)
    {
        var fail = new NativeArray<double>(5, Allocator.TempJob);
        try
        {
            new EqpJob { Mode = mode, N = n, K = k, Seed = seed, Fail = fail }.Run();
            if (fail[0] != 0)
                Assert.Fail($"check {fail[1]}: got {fail[2]:G6}, limit/expected {fail[3]:G6}, diff {fail[4]:G6}");
        }
        finally { fail.Dispose(); }
    }

    // ---- k = 0 (empty working set): eqpNullSpaceStep must throw ArgumentException. Managed thread
    // (like LPTests' SolveThrowsOnDimensionMismatch) so the exception propagates to NUnit cleanly. x is
    // trivially feasible (no constraints); the guard throws before any factorization runs. ----

    [Test]
    public void Eqp_ThrowsOnEmptyWorkingSet()
    {
        int n = 4;
        var Q = new fProxyMxN(n, n, Allocator.Temp);
        var c = new fProxyN(n, Allocator.Temp);
        var A = new fProxyMxN(0, n, Allocator.Temp);   // k = 0
        var b = new fProxyN(0, Allocator.Temp);
        var x = new fProxyN(n, Allocator.Temp);      // zero -> feasible for an empty working set
        var lam = new fProxyN(0, Allocator.Temp);

        Assert.Catch<ArgumentException>(() => QP.eqpNullSpaceStep(in Q, in c, in A, in b, ref x, ref lam));
    }
}
