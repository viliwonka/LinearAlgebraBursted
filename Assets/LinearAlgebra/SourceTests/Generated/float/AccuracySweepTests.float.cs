using System;

using LinearAlgebra;
using LinearAlgebra.Gallery;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

// ============================================================================================
// ACCURACY / PRECISION SWEEP for the level-3 blocked kernels (QR, LQ, Cholesky, LU).
//
// Purpose: prove the blocked (compact-WY / TRSM+SYRK / GEMM-trailing-update) rewrites preserved
// numerical accuracy vs the classic level-2 forms. Blocked and unblocked differ ONLY in the
// summation order of the trailing updates, so backward-stability bounds are unchanged in theory;
// this file confirms it empirically at sizes where the blocked path is actually engaged
// (QR N_Cols>=64, LQ M_Rows>=512, Cholesky n>=256, LU n>=256), on both well- and ill-conditioned
// inputs, and PRINTS the residual magnitudes so they are on record (grep the test log for
// "[AccuracySweep]").
//
// KEY NUMERICAL POINT (why the tolerances are c*n*eps and NOT kappa*eps):
//   The reconstruction residual ‖A − Q·R‖/‖A‖, orthogonality ‖QᵀQ − I‖, ‖A − L·Lᵀ‖/‖A‖,
//   ‖P·A − L·U‖/‖A‖ and the normwise solve backward error ‖A·x − b‖/(‖A‖‖x‖+‖b‖) are all
//   BACKWARD errors. For a backward-stable factorization these are O(c·n·ε) INDEPENDENT of the
//   condition number κ(A). κ amplifies the FORWARD error ‖x − x_true‖, not the residuals above.
//   That is exactly why an ill-conditioned input (Hilbert, κ≫1/ε; Lehmer, κ~n²) must STILL show a
//   tiny reconstruction residual — a "large" residual there would be a real accuracy regression,
//   not a conditioning artifact. So we assert c·n·ε bounds and additionally LOG a cheap κ proxy
//   (max/min factor-diagonal ratio) per case so a reviewer can see the input really was ill-cond.
//
//   The forward solve error IS κ-amplified, so for the solve round-trips we log ‖x−x_true‖/‖x_true‖
//   for the record but only HARD-assert the κ-independent quantities (residual + normwise backward
//   error).
//
// Landmine honored: every Frobenius norm / residual is accumulated in DOUBLE even for the float
// kernels (the reconstruction products are recomputed in double from the float factors), so the
// yardstick is ~1e-16 clean and never as noisy as the float thing being measured.
// ============================================================================================
public class floatAccuracySweepTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct AccuracyJob : IJob
    {
        public enum TestType
        {
            // QR: reconstruction + orthogonality. Blocked path at N_Cols >= 2*QR_BLOCK = 64.
            QR_Well_256,
            QR_Well_512,
            QR_Hilbert_256,   // κ(Hilbert_256) ≫ 1/ε yet backward residual stays O(n·ε).

            // LQ: reconstruction + orthonormal-rows. Blocked path at M_Rows >= 512. Wide (m<=n).
            LQ_Well_512x640,
            LQ_Well_640x800,
            LQ_IllScaled_512x640,   // rows geometrically scaled -> ill-conditioned L, κ ~ 1e3.

            // Cholesky: reconstruction, well SPD (MᵀM + nI) and ill SPD (Lehmer, κ<4n²).
            // Blocked path at n >= 256.
            Chol_Well_256,
            Chol_Well_512,
            Chol_Lehmer_256,
            Chol_Lehmer_512,

            // LU: P·A = L·U residual, well- and ill-conditioned. Blocked path at n >= 256.
            LU_Well_256,
            LU_Well_300,       // non-block-aligned last panel (300 = 9*32 + 12).
            LU_Lehmer_256,
            // LU solve round-trips: residual + normwise backward error (hard) and forward error (log).
            LU_Solve_Well_256,
            LU_Solve_Lehmer_256,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<float> Fail;

        // Residual report carried out to the managed side for logging (all accumulated in double).
        // Slots are per-kernel (logged as r0..r2 + kappa; interpret via this legend):
        //   QR        : r0 = ‖A−QR‖/‖A‖ (blocked), r1 = ‖QᵀQ−I‖ (blocked), r2 = ‖QᵀQ−I‖ (unblocked ref)
        //   LQ        : r0 = ‖A−LQ‖/‖A‖,           r1 = ‖QQᵀ−I‖,           r2 = 0
        //   Cholesky  : r0 = ‖A−LLᵀ‖/‖A‖,          r1 = 0,                 r2 = 0
        //   LU        : r0 = ‖PA−LU‖/‖A‖,          r1 = 0,                 r2 = 0
        //   LU solve  : r0 = ‖Ax−b‖/‖b‖,           r1 = normwise backward, r2 = forward error ‖x−x*‖/‖x*‖
        //   kappa (r3): cheap κ proxy = max/min |factor diagonal| (squared for Cholesky), diagnostic only.
        public NativeArray<double> Res;

        // Machine epsilon of the current expansion, as a double (float ~1.19e-7, double ~2.22e-16).
        double Eps() => (double)Consts.floatEpsilon;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.QR_Well_256:          RunQR(256, 256, 30256, false); break;
                case TestType.QR_Well_512:          RunQR(512, 512, 30512, false); break;
                case TestType.QR_Hilbert_256:       RunQR(256, 256, 0, true);      break;

                case TestType.LQ_Well_512x640:      RunLQ(512, 640, 40512, 0); break;
                case TestType.LQ_Well_640x800:      RunLQ(640, 800, 40640, 0); break;
                case TestType.LQ_IllScaled_512x640: RunLQ(512, 640, 40513, 1); break;

                case TestType.Chol_Well_256:        RunChol(256, 50256, false); break;
                case TestType.Chol_Well_512:        RunChol(512, 50512, false); break;
                case TestType.Chol_Lehmer_256:      RunChol(256, 0, true);      break;
                case TestType.Chol_Lehmer_512:      RunChol(512, 0, true);      break;

                case TestType.LU_Well_256:          RunLU(256, 60256, false); break;
                case TestType.LU_Well_300:          RunLU(300, 60300, false); break;
                case TestType.LU_Lehmer_256:        RunLU(256, 0, true);      break;

                case TestType.LU_Solve_Well_256:    RunLUSolve(256, 61256, false); break;
                case TestType.LU_Solve_Lehmer_256:  RunLUSolve(256, 0, true);      break;
            }
        }

        // ---------------------------------------------------------------------------------------
        // QR: A (m×n, m>=n) -> Q (m×n, orthonormal cols) · R (n×n, upper). Blocked at n>=64.
        // ---------------------------------------------------------------------------------------
        void RunQR(int m, int n, uint seed, bool hilbert)
        {
            var arena = new Arena(Allocator.Persistent);

            floatMxN A;
            if (hilbert)
                A = arena.floatHilbert(n);                  // SPD, totally positive, κ ≫ 1/ε.
            else
                A = WellCondSquare(ref arena, n, seed);      // diagonally dominant, κ ~ O(1).

            // Blocked path: the allocating overload routes to the compact-WY level-3 core at n>=64.
            var Qb = A.Copy();
            var Rb = arena.floatMat(n, n);
            QR.qrDecomposition(ref Qb, ref Rb);
            Assert.IsFalse(Analysis.isAnyNan(in Qb));
            Assert.IsFalse(Analysis.isAnyNan(in Rb));

            // Unblocked reference: the zero-alloc (ref u) overload is deliberately NOT blocked (see the
            // QR.float.cs comment) — it runs the classic rank-1 Householder sweep, giving an independent
            // in-test oracle at the SAME size without any production "force unblocked" flag. This is the
            // direct blocked-vs-unblocked accuracy comparison the sweep is really after.
            var Qr = A.Copy();
            var Rr = arena.floatMat(n, n);
            var u  = arena.floatVec(m);
            QR.qrDecomposition(ref Qr, ref Rr, ref u);

            double recon    = ReconResidual2(in A, in Qb, in Rb); // ‖A − Q·R‖_F / ‖A‖_F (blocked), double.
            double orth     = OrthoErrorCols(in Qb);              // ‖QᵀQ − I‖_F (blocked), double.
            double reconRef = ReconResidual2(in A, in Qr, in Rr); // same, unblocked reference.
            double orthRef  = OrthoErrorCols(in Qr);
            double kappa    = DiagCond(in Rb, false);             // κ proxy = max/min |R_ii|.

            Res[0] = recon; Res[1] = orth; Res[2] = orthRef; Res[3] = kappa;

            // Blocking must not MATERIALLY worsen the backward error vs the unblocked reference. Two-part
            // bound (mirrors LUTests' AssertResidualNotWorse): an O(n·ε) floor so well-conditioned cases
            // — where the reference orthogonality is ~1e-15 — are not held to an impossible ratio, plus a
            // 16× ratio guard that trips loudly on a genuine blocked regression.
            //
            // Householder QR orthogonality is famously κ-INDEPENDENT (~n·ε) for well-conditioned A, but on
            // a numerically rank-deficient input (Hilbert_256: κ≈1e19, far past 1/ε_double) BOTH the
            // blocked and unblocked forms lose orthogonality to ~2e-6 (see the logged r1≈r2 for
            // QR_Hilbert_256). That is a property of the INPUT (its trailing columns are pure rounding
            // noise), not a defect of blocking — and this reference comparison is what proves it, instead
            // of a fixed tiny bound that would false-fail exactly as the spec warned.
            AssertLE(recon, math.max(ReconBound(n), 16.0 * reconRef));
            AssertLE(orth,  math.max(ReconBound(n), 16.0 * orthRef));

            arena.Dispose();
        }

        // ---------------------------------------------------------------------------------------
        // LQ: A (m×n, m<=n) -> L (m×m, lower) · Q (m×n, orthonormal ROWS). Blocked at m>=512.
        // mode 0 = well-conditioned (boosted diagonal); mode 1 = row-scaled (ill-conditioned L).
        // ---------------------------------------------------------------------------------------
        void RunLQ(int m, int n, uint seed, int mode)
        {
            var arena = new Arena(Allocator.Persistent);

            var random = new Unity.Mathematics.Random(seed);
            var A = arena.floatRandomMat(m, n, -5f, 5f, seed);
            for (int d = 0; d < m; d++)
                A[d, d] += 5.1f + 10f * random.NextFloat();

            if (mode == 1)
            {
                // Geometrically scale each row so ‖row i‖ spans ~[1e-3, 1] -> L becomes ill-conditioned
                // (κ ~ 1e3) while the reconstruction stays a backward-stable O(n·ε) problem.
                for (int i = 0; i < m; i++)
                {
                    float s = (float)math.pow(10.0, -3.0 * i / (m - 1));
                    for (int j = 0; j < n; j++)
                        A[i, j] *= s;
                }
            }

            var origA = A.Copy();
            var L = arena.floatMat(m, m);
            var Q = arena.floatMat(m, n);

            LQ.lqDecomposition(ref A, ref L, ref Q);

            Assert.IsFalse(Analysis.isAnyNan(in L));
            Assert.IsFalse(Analysis.isAnyNan(in Q));

            double recon = ReconResidualLQ(in origA, in L, in Q); // ‖A − L·Q‖_F / ‖A‖_F, double.
            double orth  = OrthoErrorRows(in Q);                  // ‖Q·Qᵀ − I_m‖_F, double.
            double kappa = DiagCond(in L, false);                // κ proxy = max/min |L_ii|.

            Res[0] = recon; Res[1] = orth; Res[2] = 0; Res[3] = kappa;

            // Reconstruction (right-multiply GEMM) is backward stable; orthonormal-rows error tiny.
            AssertLE(recon, ReconBound(n));
            AssertLE(orth,  ReconBound(m));

            arena.Dispose();
        }

        // ---------------------------------------------------------------------------------------
        // Cholesky: SPD A -> L (lower), A = L·Lᵀ. Blocked at n>=256.
        // ---------------------------------------------------------------------------------------
        void RunChol(int n, uint seed, bool lehmer)
        {
            var arena = new Arena(Allocator.Persistent);

            floatMxN A = lehmer ? arena.floatLehmer(n)     // SPD, κ < 4n² (~2.6e5 at n=256).
                                 : BuildSPD(ref arena, n, seed);

            var L = arena.floatMat(n, n);

            bool ok = Cholesky.choleskyDecomposition(in A, ref L);
            Assert.IsTrue(ok);                               // Lehmer stays numerically PD.
            Assert.IsFalse(Analysis.isAnyNan(in L));

            double recon = ReconResidualLLt(in A, in L);     // ‖A − L·Lᵀ‖_F / ‖A‖_F, double.
            double kappa = DiagCond(in L, true);             // κ proxy = (max/min L_ii)².

            Res[0] = recon; Res[1] = 0; Res[2] = 0; Res[3] = kappa;

            // Cholesky is backward stable for SPD A: residual O(n·ε) even for ill-cond Lehmer.
            AssertLE(recon, ReconBound(n));

            arena.Dispose();
        }

        // ---------------------------------------------------------------------------------------
        // LU: A -> P·A = L·U (partial pivoting). Blocked at n>=256.
        // ---------------------------------------------------------------------------------------
        void RunLU(int n, uint seed, bool lehmer)
        {
            var arena = new Arena(Allocator.Persistent);

            floatMxN A = lehmer ? arena.floatLehmer(n)     // κ ~ 1e5, LU hits no zero pivot.
                                 : WellCondLU(ref arena, n, seed);

            var U = A.Copy();
            var L = arena.floatIdentityMat(n);
            var pivot = new Pivot(n, Allocator.Temp);

            bool ok = LU.luDecomposition(ref U, ref L, ref pivot);
            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in U));
            Assert.IsFalse(Analysis.isAnyNan(in L));

            double recon = ReconResidualPALU(in A, in L, in U, in pivot); // ‖P·A − L·U‖_F/‖A‖_F.
            double kappa = DiagCond(in U, false);                         // κ proxy = max/min |U_ii|.

            Res[0] = recon; Res[1] = 0; Res[2] = 0; Res[3] = kappa;

            // LU with partial pivoting is backward stable (growth factor bounded): residual O(n·ε).
            AssertLE(recon, ReconBound(n));

            pivot.Dispose();
            arena.Dispose();
        }

        // ---------------------------------------------------------------------------------------
        // LU solve round-trip: b = A·x_true, solve, report residual + backward error + forward error.
        // ---------------------------------------------------------------------------------------
        void RunLUSolve(int n, uint seed, bool lehmer)
        {
            var arena = new Arena(Allocator.Persistent);

            floatMxN A = lehmer ? arena.floatLehmer(n)
                                 : WellCondLU(ref arena, n, seed);

            var xTrue = arena.floatRandomVec(n, 1f, 10f, seed == 0 ? 424242u : seed + 1u);
            var b = Blas.dot(A, xTrue);

            var U = A.Copy();
            var L = arena.floatIdentityMat(n);
            var pivot = new Pivot(n, Allocator.Temp);

            bool ok = LU.luDecomposition(ref U, ref L, ref pivot);
            Assert.IsTrue(ok);

            var x = b.Copy();
            LU.luSolve(ref L, ref U, in pivot, ref x);
            Assert.IsFalse(Analysis.isAnyNan(in x));

            // Residual r = A·x − b, all in double.
            double frobA = FrobA(in A);
            double normX = 0, normB = 0, normR = 0, fwdNum = 0, fwdDen = 0;
            for (int i = 0; i < n; i++)
            {
                double xi = (double)x[i];
                normX += xi * xi;
                double bi = (double)b[i];
                normB += bi * bi;

                double axi = 0;
                for (int k = 0; k < n; k++)
                    axi += (double)A[i, k] * (double)x[k];
                double ri = axi - bi;
                normR += ri * ri;

                double dx = (double)x[i] - (double)xTrue[i];
                fwdNum += dx * dx;
                double xt = (double)xTrue[i];
                fwdDen += xt * xt;
            }
            normX = math.sqrt(normX);
            normB = math.sqrt(normB);
            normR = math.sqrt(normR);

            double solveRes = normR / normB;                         // ‖Ax−b‖ / ‖b‖ (spec metric).
            double bwd      = normR / (frobA * normX + normB);       // normwise backward error.
            double fwd      = math.sqrt(fwdNum) / math.sqrt(fwdDen); // κ-amplified forward error.

            Res[0] = solveRes; Res[1] = bwd; Res[2] = fwd; Res[3] = 0;

            // HARD assert only the κ-independent backward error (O(n·ε)); forward error is logged
            // for the record because it legitimately grows with κ and is NOT a stability signal.
            AssertLE(bwd, ReconBound(n));

            pivot.Dispose();
            arena.Dispose();
        }

        // ===== builders ========================================================================

        // Diagonally-dominant square random -> well-conditioned (κ ~ O(1)).
        static floatMxN WellCondSquare(ref Arena arena, int n, uint seed)
        {
            var A = arena.floatRandomMat(n, n, -1f, 1f, seed);
            for (int d = 0; d < n; d++)
                A[d, d] += (float)(2 * n);
            return A;
        }

        // Random with a boosted diagonal and a modest spread (well-conditioned, needs some pivoting).
        static floatMxN WellCondLU(ref Arena arena, int n, uint seed)
        {
            var A = arena.floatRandomMat(n, n, -10f, 10f, seed);
            for (int d = 0; d < n; d++)
            {
                A[d, d] *= 2f;
                if (math.abs(A[d, d]) < 0.01f)
                    A[d, d] *= 10f;
            }
            return A;
        }

        // SPD via A = MᵀM + n·I (strictly PD, diagonally dominant -> Cholesky must succeed).
        static floatMxN BuildSPD(ref Arena arena, int n, uint seed)
        {
            var M = arena.floatRandomMat(n, n, -1f, 1f, seed);
            var A = Blas.dot(M, M, true);   // Mᵀ·M
            for (int d = 0; d < n; d++)
                A[d, d] += n;
            return A;
        }

        // ===== double-accurate residual helpers ================================================

        static double FrobA(in floatMxN A)
        {
            double s = 0;
            for (int i = 0; i < A.Length; i++)
            {
                double v = (double)A[i];
                s += v * v;
            }
            return math.sqrt(s);
        }

        // ‖A − Q·R‖_F / ‖A‖_F, Q(m×n), R(n×n), accumulated in double.
        static double ReconResidual2(in floatMxN A, in floatMxN Q, in floatMxN R)
        {
            int m = A.M_Rows, n = A.N_Cols;
            double num = 0, den = 0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int k = 0; k < n; k++)
                        acc += (double)Q[i, k] * (double)R[k, j];
                    double d = (double)A[i, j] - acc;
                    num += d * d;
                    double a = (double)A[i, j];
                    den += a * a;
                }
            return math.sqrt(num) / math.sqrt(den);
        }

        // ‖A − L·Q‖_F / ‖A‖_F, L(m×m) lower, Q(m×n), accumulated in double.
        static double ReconResidualLQ(in floatMxN A, in floatMxN L, in floatMxN Q)
        {
            int m = A.M_Rows, n = A.N_Cols;
            double num = 0, den = 0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int k = 0; k <= i && k < m; k++)   // L lower-triangular: k<=i.
                        acc += (double)L[i, k] * (double)Q[k, j];
                    double d = (double)A[i, j] - acc;
                    num += d * d;
                    double a = (double)A[i, j];
                    den += a * a;
                }
            return math.sqrt(num) / math.sqrt(den);
        }

        // ‖A − L·Lᵀ‖_F / ‖A‖_F, L(n×n) lower, accumulated in double.
        static double ReconResidualLLt(in floatMxN A, in floatMxN L)
        {
            int n = A.M_Rows;
            double num = 0, den = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    int kmax = math.min(i, j);
                    for (int k = 0; k <= kmax; k++)         // (L·Lᵀ)[i,j] = Σ_k L[i,k]·L[j,k].
                        acc += (double)L[i, k] * (double)L[j, k];
                    double d = (double)A[i, j] - acc;
                    num += d * d;
                    double a = (double)A[i, j];
                    den += a * a;
                }
            return math.sqrt(num) / math.sqrt(den);
        }

        // ‖P·A − L·U‖_F / ‖A‖_F, P applied via inverse-row on a copy of A, accumulated in double.
        double ReconResidualPALU(in floatMxN A, in floatMxN L, in floatMxN U, in Pivot P)
        {
            int n = A.M_Rows;
            var PA = A.Copy();
            P.ApplyInverseRow(ref PA);   // PA = P·A (matches AssertLU's PA = LU convention).

            double num = 0, den = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    // L unit-lower (k<=i, L[i,i]=1), U upper (k<=j). (L·U)[i,j] = Σ_k L[i,k]·U[k,j].
                    int kmax = math.min(i, j);
                    for (int k = 0; k <= kmax; k++)
                        acc += (double)L[i, k] * (double)U[k, j];
                    double d = (double)PA[i, j] - acc;
                    num += d * d;
                    double a = (double)A[i, j];
                    den += a * a;
                }
            return math.sqrt(num) / math.sqrt(den);
        }

        // ‖QᵀQ − I‖_F for Q with orthonormal COLUMNS (n×n Gram), accumulated in double.
        static double OrthoErrorCols(in floatMxN Q)
        {
            int m = Q.M_Rows, n = Q.N_Cols;
            double s = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int k = 0; k < m; k++)
                        acc += (double)Q[k, i] * (double)Q[k, j];
                    double e = acc - (i == j ? 1.0 : 0.0);
                    s += e * e;
                }
            return math.sqrt(s);
        }

        // ‖Q·Qᵀ − I_m‖_F for Q with orthonormal ROWS (m×m Gram), accumulated in double.
        static double OrthoErrorRows(in floatMxN Q)
        {
            int m = Q.M_Rows, n = Q.N_Cols;
            double s = 0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < m; j++)
                {
                    double acc = 0;
                    for (int k = 0; k < n; k++)
                        acc += (double)Q[i, k] * (double)Q[j, k];
                    double e = acc - (i == j ? 1.0 : 0.0);
                    s += e * e;
                }
            return math.sqrt(s);
        }

        // Cheap conditioning proxy: ratio of largest to smallest |diagonal| of the triangular factor.
        // For a triangular factor this is a lower bound on / order estimate of κ. squared=true for
        // Cholesky where κ(A) ~ (max/min L_ii)². Purely diagnostic (logged, never asserted on).
        static double DiagCond(in floatMxN T, bool squared)
        {
            int n = math.min(T.M_Rows, T.N_Cols);
            double mx = 0, mn = double.MaxValue;
            for (int i = 0; i < n; i++)
            {
                double d = math.abs((double)T[i, i]);
                if (d > mx) mx = d;
                if (d < mn) mn = d;
            }
            if (mn <= 0) return double.PositiveInfinity;
            double r = mx / mn;
            return squared ? r * r : r;
        }

        // Backward-error ceiling: c·n·ε. c=256 gives ~1-2 orders of headroom over the observed
        // O(n·ε) residuals (see the logged magnitudes) while still tripping loudly on any gross
        // blocked-path regression (a broken trailing update lands the residual at O(1), not O(n·ε)).
        double ReconBound(int n) => 256.0 * n * Eps();

        // Fail layout: [0]=flag, [1]=value, [2]=limit, [3]=excess. value/limit stored as float for
        // the shared harness; the precise double lives in Res and is what gets logged.
        void AssertLE(double value, double limit)
        {
            if (!(value <= limit) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = (float)value;
                Fail[2] = (float)limit;
                Fail[3] = (float)(value - limit);
            }
            Assert.IsTrue(value <= limit);
        }
    }

    // true only when float expands to double.
    private static bool IsDouble() => (double)Consts.floatEpsilon < 1e-10;
    private static string DType() => IsDouble() ? "double" : "float";

    public static Array GetEnums() => Enum.GetValues(typeof(AccuracyJob.TestType));

    [TestCaseSource("GetEnums")]
    public void AccuracySweep(AccuracyJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        var res  = new NativeArray<double>(4, Allocator.TempJob);
        try
        {
            new AccuracyJob { Type = type, Fail = fail, Res = res }.Run();

            // Residual report — grep the EditMode test log for "[AccuracySweep]". See the Res legend
            // on AccuracyJob for what r0/r1/r2 mean per kernel.
            Debug.Log($"[AccuracySweep] {type,-24} {DType(),-6} " +
                      $"r0={res[0]:E3}  r1={res[1]:E3}  r2={res[2]:E3}  kappa~={res[3]:E3}");

            // Under Burst a failed in-job assert logs+aborts without throwing to the caller — surface
            // the recorded diagnostics here too.
            if (fail[0] != (float)0)
                Assert.Fail($"{type} [{DType()}]: got {fail[1]}, limit {fail[2]}, excess {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (float)0)
                Assert.Fail($"{type} [{DType()}]: got {fail[1]}, limit {fail[2]}, excess {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
            res.Dispose();
        }
    }
}
