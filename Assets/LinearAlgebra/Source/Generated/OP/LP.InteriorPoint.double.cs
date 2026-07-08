#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class LP
    {
        // ============================================================================================
        // Mehrotra predictor-corrector primal-dual interior point (the LPMethod.InteriorPoint backend).
        //
        // Converts the canonical primal form (min cᵀx s.t. Ax {≤,=,≥} b, x ≥ 0) to standard form
        // min cᵀz s.t. Aₛ z = b, z ≥ 0 by adding one non-negative slack/surplus per inequality (no
        // artificials -- the interior-point method starts from a strictly interior point, not a basic
        // feasible vertex). Each iteration forms the normal-equation matrix M = Aₛ D Aₛᵀ (D = Z S⁻¹),
        // Cholesky-factors it ONCE (reusing the library's CHO), and reuses that factor for both the
        // affine-scaling predictor solve and the centering-corrector solve. Step lengths keep z, s > 0.
        //
        // Detects only Optimal vs MaxIterations: unlike the simplex backend it does not emit exact
        // infeasibility / unboundedness certificates (that needs a homogeneous self-dual embedding).
        // For those certificates, or for exact vertex solutions, use LPMethod.Simplex.
        //
        // Job-safe: all scratch is Allocator.Temp and disposed before returning.
        // ============================================================================================
        static LPInfo interiorCore(in doubleMxN A, in doubleN b, in doubleN c,
                                   in NativeArray<ConstraintSense> senses,
                                   ref doubleN x, out double objective, int maxIter)
        {
            int m = A.M_Rows, n = A.N_Cols;

            // --- standard form: add a slack column per inequality ( ≤ → +1, ≥ → −1 ) ---
            int nSlack = 0;
            for (int i = 0; i < m; i++) if (senses[i] != ConstraintSense.Equal) nSlack++;
            int nv = n + nSlack;

            var As = new doubleMxN(m, nv, Allocator.Temp);   // zero-initialized
            var cs = new doubleN(nv, Allocator.Temp);
            for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) As[i, j] = A[i, j];
            {
                int sc = n;
                for (int i = 0; i < m; i++)
                {
                    if (senses[i] == ConstraintSense.Equal) continue;
                    As[i, sc] = senses[i] == ConstraintSense.LessEqual ? (double)1 : (double)(-1);
                    sc++;
                }
            }
            for (int j = 0; j < n; j++) cs[j] = c[j];   // slack costs stay 0

            // --- scratch ---
            var z = new doubleN(nv, Allocator.Temp); var s = new doubleN(nv, Allocator.Temp);
            var y = new doubleN(m, Allocator.Temp);
            var dz = new doubleN(nv, Allocator.Temp); var ds = new doubleN(nv, Allocator.Temp);
            var dzA = new doubleN(nv, Allocator.Temp); var dsA = new doubleN(nv, Allocator.Temp);
            var dy = new doubleN(m, Allocator.Temp);
            var d = new doubleN(nv, Allocator.Temp);        // d_k = z_k / s_k
            var rc = new doubleN(nv, Allocator.Temp);       // dual residual  Aᵀy + s − c
            var g = new doubleN(nv, Allocator.Temp);
            var tmpNV = new doubleN(nv, Allocator.Temp);
            var rp = new doubleN(m, Allocator.Temp);        // primal residual  Az − b
            var ADrd = new doubleN(m, Allocator.Temp);
            var tmpM = new doubleN(m, Allocator.Temp);
            var rhsY = new doubleN(m, Allocator.Temp);
            var M = new doubleMxN(m, m, Allocator.Temp);
            var L = new doubleMxN(m, m, Allocator.Temp);

            double reg = Consts.doubleZeroThreshold;
            double BIG = (double)1e30;
            double eta = (double)0.99;
            double tol = 100.0 * (double)Consts.doubleEpsilon;

            double bNorm = 0, cNorm = 0;
            for (int i = 0; i < m; i++) bNorm += (double)b[i] * (double)b[i];
            for (int j = 0; j < nv; j++) cNorm += (double)cs[j] * (double)cs[j];
            bNorm = math.sqrt(bNorm); cNorm = math.sqrt(cNorm);

            // --- Mehrotra starting point: least-norm x̃, dual ỹ from (Aₛ Aₛᵀ)⁻¹, then shift interior ---
            for (int j = 0; j < nv; j++) d[j] = (double)1;
            BuildNormal(As, d, M, m, nv, reg);
            bool ok = CHO.decomp(in M, ref L);
            if (ok)
            {
                for (int i = 0; i < m; i++) tmpM[i] = b[i];
                CHO.decompSolve(ref L, ref tmpM);          // tmpM = (AAᵀ)⁻¹ b
                ATmul(As, tmpM, z, m, nv);                 // z̃ = Aᵀ (AAᵀ)⁻¹ b
                Amul(As, cs, tmpM, m, nv);                 // tmpM = A c
                CHO.decompSolve(ref L, ref tmpM);          // ỹ = (AAᵀ)⁻¹ A c
                for (int i = 0; i < m; i++) y[i] = tmpM[i];
                ATmul(As, y, s, m, nv);
                for (int j = 0; j < nv; j++) s[j] = cs[j] - s[j];   // s̃ = c − Aᵀ ỹ
            }
            else
            {
                for (int j = 0; j < nv; j++) { z[j] = (double)1; s[j] = (double)1; }
                for (int i = 0; i < m; i++) y[i] = (double)0;
            }
            // shift z, s into the strict interior
            {
                double minZ = BIG, minS = BIG;
                for (int j = 0; j < nv; j++) { minZ = math.min(minZ, z[j]); minS = math.min(minS, s[j]); }
                double dxs = math.max(-(double)1.5 * minZ, (double)0);
                double dss = math.max(-(double)1.5 * minS, (double)0);
                double sumZ = 0, sumS = 0, dotZS = 0;
                for (int j = 0; j < nv; j++)
                {
                    z[j] += dxs; s[j] += dss;
                    sumZ += (double)z[j]; sumS += (double)s[j];
                }
                for (int j = 0; j < nv; j++) dotZS += (double)z[j] * (double)s[j];
                double pdz = (double)(0.5 * dotZS / math.max(sumS, 1e-30));
                double pds = (double)(0.5 * dotZS / math.max(sumZ, 1e-30));
                for (int j = 0; j < nv; j++)
                {
                    z[j] = math.max(z[j] + pdz, (double)reg);
                    s[j] = math.max(s[j] + pds, (double)reg);
                }
            }

            int budget = maxIter > 0 ? maxIter : 100;
            LPStatus status = LPStatus.MaxIterations;
            int iters = 0;

            while (iters < budget)
            {
                // residuals & duality measure
                Amul(As, z, rp, m, nv);
                for (int i = 0; i < m; i++) rp[i] -= b[i];                 // rp = Az − b
                ATmul(As, y, rc, m, nv);
                for (int j = 0; j < nv; j++) rc[j] += s[j] - cs[j];        // rc = Aᵀy + s − c
                double mu = 0, objz = 0;
                for (int j = 0; j < nv; j++) { mu += (double)z[j] * (double)s[j]; objz += (double)cs[j] * (double)z[j]; }
                mu /= nv;

                double rpN = 0, rcN = 0;
                for (int i = 0; i < m; i++) rpN += (double)rp[i] * (double)rp[i];
                for (int j = 0; j < nv; j++) rcN += (double)rc[j] * (double)rc[j];
                rpN = math.sqrt(rpN); rcN = math.sqrt(rcN);

                if (rpN / (1.0 + bNorm) < tol && rcN / (1.0 + cNorm) < tol && mu / (1.0 + math.abs(objz)) < tol)
                { status = LPStatus.Optimal; break; }

                // normal matrix M = A D Aᵀ, D = Z S⁻¹, factored once and reused this iteration
                for (int j = 0; j < nv; j++) d[j] = z[j] / s[j];
                BuildNormal(As, d, M, m, nv, reg);
                if (!CHO.decomp(in M, ref L)) { status = LPStatus.MaxIterations; break; }

                // ---- affine predictor:  rhsY = b − A(D rc) ----
                for (int j = 0; j < nv; j++) tmpNV[j] = d[j] * rc[j];
                Amul(As, tmpNV, ADrd, m, nv);
                for (int i = 0; i < m; i++) { rhsY[i] = b[i] - ADrd[i]; tmpM[i] = rhsY[i]; }
                CHO.decompSolve(ref L, ref tmpM);                         // tmpM = Δy_aff
                ATmul(As, tmpM, tmpNV, m, nv);                            // tmpNV = Aᵀ Δy_aff
                for (int j = 0; j < nv; j++) dsA[j] = -rc[j] - tmpNV[j];  // Δs = −rc − AᵀΔy
                for (int j = 0; j < nv; j++) dzA[j] = -d[j] * dsA[j] - z[j];

                double apA = math.min((double)1, MaxStep(z, dzA, nv, BIG));
                double adA = math.min((double)1, MaxStep(s, dsA, nv, BIG));
                double muAff = 0;
                for (int j = 0; j < nv; j++)
                    muAff += (double)(z[j] + apA * dzA[j]) * (double)(s[j] + adA * dsA[j]);
                muAff /= nv;
                double ratio = muAff / math.max(mu, 1e-30);
                double sigma = ratio * ratio * ratio;

                // ---- centering corrector:  g = (σμ − Δz_aff∘Δs_aff)/s,  rhsY −= A g ----
                for (int j = 0; j < nv; j++)
                    g[j] = (double)((sigma * mu - (double)dzA[j] * (double)dsA[j]) / (double)s[j]);
                Amul(As, g, tmpM, m, nv);
                for (int i = 0; i < m; i++) tmpM[i] = rhsY[i] - tmpM[i];
                CHO.decompSolve(ref L, ref tmpM);                         // tmpM = Δy
                for (int i = 0; i < m; i++) dy[i] = tmpM[i];
                ATmul(As, dy, tmpNV, m, nv);                              // tmpNV = Aᵀ Δy
                for (int j = 0; j < nv; j++) ds[j] = -rc[j] - tmpNV[j];
                for (int j = 0; j < nv; j++) dz[j] = -d[j] * ds[j] - z[j] + g[j];

                double ap = math.min((double)1, eta * MaxStep(z, dz, nv, BIG));
                double ad = math.min((double)1, eta * MaxStep(s, ds, nv, BIG));

                for (int j = 0; j < nv; j++) { z[j] += ap * dz[j]; s[j] += ad * ds[j]; }
                for (int i = 0; i < m; i++) y[i] += ad * dy[i];
                iters++;
            }

            // extract structural x and objective
            for (int j = 0; j < n; j++) x[j] = z[j];
            double obj = 0;
            for (int j = 0; j < n; j++) obj += (double)c[j] * (double)z[j];
            objective = obj;

            As.Dispose(); cs.Dispose(); z.Dispose(); s.Dispose(); y.Dispose();
            dz.Dispose(); ds.Dispose(); dzA.Dispose(); dsA.Dispose(); dy.Dispose();
            d.Dispose(); rc.Dispose(); g.Dispose(); tmpNV.Dispose();
            rp.Dispose(); ADrd.Dispose(); tmpM.Dispose(); rhsY.Dispose(); M.Dispose(); L.Dispose();

            return new LPInfo { status = status, iterations = iters, objective = obj };
        }

        // out[i] = Σ_k A[i,k] v[k]   (A m×nv, v length nv, out length m)
        static void Amul(doubleMxN A, doubleN v, doubleN outv, int m, int nv)
        {
            for (int i = 0; i < m; i++)
            {
                double acc = (double)0;
                for (int k = 0; k < nv; k++) acc += A[i, k] * v[k];
                outv[i] = acc;
            }
        }

        // out[j] = Σ_i A[i,j] v[i]   (v length m, out length nv)
        static void ATmul(doubleMxN A, doubleN v, doubleN outv, int m, int nv)
        {
            for (int j = 0; j < nv; j++) outv[j] = (double)0;
            for (int i = 0; i < m; i++)
            {
                double vi = v[i];
                for (int j = 0; j < nv; j++) outv[j] += A[i, j] * vi;
            }
        }

        // M = A diag(d) Aᵀ + reg·I  (symmetric, m×m). Upper triangle then mirrored.
        static void BuildNormal(doubleMxN A, doubleN d, doubleMxN M, int m, int nv, double reg)
        {
            for (int i = 0; i < m; i++)
            {
                for (int j = i; j < m; j++)
                {
                    double acc = (double)0;
                    for (int k = 0; k < nv; k++) acc += A[i, k] * d[k] * A[j, k];
                    if (i == j) acc += reg;
                    M[i, j] = acc;
                    M[j, i] = acc;
                }
            }
        }

        // Largest α ≥ 0 with v + α·dv ≥ 0 (uncapped; BIG if dv has no negative entry). Caller caps at 1.
        static double MaxStep(doubleN v, doubleN dv, int nv, double BIG)
        {
            double a = BIG;
            for (int k = 0; k < nv; k++)
                if (dv[k] < (double)0) a = math.min(a, -v[k] / dv[k]);
            return a;
        }
    }
}
