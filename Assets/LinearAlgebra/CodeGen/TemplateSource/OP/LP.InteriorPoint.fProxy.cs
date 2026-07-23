using System;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

using LinearAlgebra.Internal;

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
        // feasible vertex). Each iteration forms the normal-equation matrix M = Aₛ D Aₛᵀ (D = Z S⁻¹)
        // structure-aware (single-nonzero slack/identity columns only touch M's diagonal -- see
        // BuildNormalStructured), Cholesky-factors it ONCE (reusing the library's CHO), and reuses that
        // factor for both the affine-scaling predictor solve and the centering-corrector solve. Step
        // lengths keep z, s > 0.
        //
        // Detects only Optimal vs MaxIterations: unlike the simplex backend it does not emit exact
        // infeasibility / unboundedness certificates (that needs a homogeneous self-dual embedding).
        // For those certificates, or for exact vertex solutions, use LPMethod.RevisedSimplex or
        // LPMethod.DualSimplex.
        //
        // Job-safe: all scratch is Allocator.Temp and disposed before returning.
        // ============================================================================================
        static LPInfo interiorCore(in fProxyMxN A, in fProxyN b, in fProxyN c,
                                   in NativeArray<ConstraintSense> senses,
                                   ref fProxyN x, out double objective, int maxIter)
        {
            int m = A.M_Rows, n = A.N_Cols;

            // --- standard form: add a slack column per inequality ( ≤ → +1, ≥ → −1 ) ---
            int nSlack = 0;
            for (int i = 0; i < m; i++) if (senses[i] != ConstraintSense.Equal) nSlack++;
            int nv = n + nSlack;

            var As = new fProxyMxN(m, nv, Allocator.Temp);   // zero-initialized
            var cs = new fProxyN(nv, Allocator.Temp);
            for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) As[i, j] = A[i, j];
            {
                int sc = n;
                for (int i = 0; i < m; i++)
                {
                    if (senses[i] == ConstraintSense.Equal) continue;
                    As[i, sc] = senses[i] == ConstraintSense.LessEqual ? (fProxy)1 : (fProxy)(-1);
                    sc++;
                }
            }
            for (int j = 0; j < n; j++) cs[j] = c[j];   // slack costs stay 0

            // --- exploit standard-form structure: a column of Aₛ with a SINGLE nonzero (every slack
            // column here, and the ±I blocks a LAD caller bakes into A) contributes only v²·d[k] to the
            // DIAGONAL of M = Aₛ D Aₛᵀ, so the O(m²·nv) BuildNormal never needs to see it. Classify the
            // columns once (the structure is fixed), compact the multi-nonzero ones into Ad, and add the
            // single-entry diagonal terms analytically each iteration. For the LAD form [A|−A|−I|I]
            // (nv = 2n+2m) this cuts normal-matrix formation from O(m³) to O(m²·n). ---
            var colRow = new NativeArray<int>(nv, Allocator.Temp);      // single-nonzero row, −1 = multi, −2 = empty
            var colVal = new NativeArray<fProxy>(nv, Allocator.Temp);   // that nonzero's value
            int nDense = 0;
            for (int k = 0; k < nv; k++)
            {
                int cnt = 0, row = -1; fProxy val = (fProxy)0;
                for (int i = 0; i < m && cnt < 2; i++)
                {
                    fProxy a = As[i, k];
                    if (a != (fProxy)0) { cnt++; row = i; val = a; }
                }
                if (cnt >= 2) { colRow[k] = -1; nDense++; }
                else if (cnt == 1) { colRow[k] = row; colVal[k] = val; }
                else colRow[k] = -2;
            }
            var denseK = new NativeArray<int>(math.max(nDense, 1), Allocator.Temp);   // dense slot -> original column
            var Ad = new fProxyMxN(m, math.max(nDense, 1), Allocator.Temp);           // compacted multi-nonzero columns
            var dd = new fProxyN(math.max(nDense, 1), Allocator.Temp);                // their d[k] slice
            {
                int cc = 0;
                for (int k = 0; k < nv; k++)
                {
                    if (colRow[k] != -1) continue;
                    denseK[cc] = k;
                    for (int i = 0; i < m; i++) Ad[i, cc] = As[i, k];
                    cc++;
                }
            }

            // --- scratch ---
            var z = new fProxyN(nv, Allocator.Temp); var s = new fProxyN(nv, Allocator.Temp);
            var y = new fProxyN(m, Allocator.Temp);
            var dz = new fProxyN(nv, Allocator.Temp); var ds = new fProxyN(nv, Allocator.Temp);
            var dzA = new fProxyN(nv, Allocator.Temp); var dsA = new fProxyN(nv, Allocator.Temp);
            var dy = new fProxyN(m, Allocator.Temp);
            var d = new fProxyN(nv, Allocator.Temp);        // d_k = z_k / s_k
            var rc = new fProxyN(nv, Allocator.Temp);       // dual residual  Aᵀy + s − c
            var g = new fProxyN(nv, Allocator.Temp);
            var tmpNV = new fProxyN(nv, Allocator.Temp);
            var rp = new fProxyN(m, Allocator.Temp);        // primal residual  Az − b
            var ADrd = new fProxyN(m, Allocator.Temp);
            var tmpM = new fProxyN(m, Allocator.Temp);
            var rhsY = new fProxyN(m, Allocator.Temp);
            var M = new fProxyMxN(m, m, Allocator.Temp);
            var L = new fProxyMxN(m, m, Allocator.Temp);
            var zBest = new fProxyN(nv, Allocator.Temp);

            fProxy reg = Consts.fProxyZeroThreshold;
            fProxy BIG = (fProxy)1e30;
            fProxy eta = (fProxy)0.99;
            double tol = 100.0 * (double)Consts.fProxyEpsilon;

            double bNorm = 0, cNorm = 0;
            for (int i = 0; i < m; i++) bNorm += (double)b[i] * (double)b[i];
            for (int j = 0; j < nv; j++) cNorm += (double)cs[j] * (double)cs[j];
            bNorm = math.sqrt(bNorm); cNorm = math.sqrt(cNorm);

            // --- Mehrotra starting point: least-norm x̃, dual ỹ from (Aₛ Aₛᵀ)⁻¹, then shift interior ---
            for (int j = 0; j < nv; j++) d[j] = (fProxy)1;
            BuildNormalStructured(Ad, d, dd, denseK, colRow, colVal, M, m, nv, nDense, reg);
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
                for (int j = 0; j < nv; j++) { z[j] = (fProxy)1; s[j] = (fProxy)1; }
                for (int i = 0; i < m; i++) y[i] = (fProxy)0;
            }
            // shift z, s into the strict interior
            {
                fProxy minZ = BIG, minS = BIG;
                for (int j = 0; j < nv; j++) { minZ = math.min(minZ, z[j]); minS = math.min(minS, s[j]); }
                fProxy dxs = math.max(-(fProxy)1.5 * minZ, (fProxy)0);
                fProxy dss = math.max(-(fProxy)1.5 * minS, (fProxy)0);
                double sumZ = 0, sumS = 0, dotZS = 0;
                for (int j = 0; j < nv; j++)
                {
                    z[j] += dxs; s[j] += dss;
                    sumZ += (double)z[j]; sumS += (double)s[j];
                }
                for (int j = 0; j < nv; j++) dotZS += (double)z[j] * (double)s[j];
                fProxy pdz = (fProxy)(0.5 * dotZS / math.max(sumS, 1e-30));
                fProxy pds = (fProxy)(0.5 * dotZS / math.max(sumZ, 1e-30));
                for (int j = 0; j < nv; j++)
                {
                    z[j] = math.max(z[j] + pdz, (fProxy)reg);
                    s[j] = math.max(s[j] + pds, (fProxy)reg);
                }
            }

            int budget = maxIter > 0 ? maxIter : 100;
            LPStatus status = LPStatus.MaxIterations;
            int iters = 0;

            // Best-iterate safeguard (mirrors the sparse standardFormInterior): a failed late
            // factorization or a float blow-up must not poison the answer -- keep the best-scoring
            // iterate seen and extract x from it, not from whatever z the loop stopped on.
            for (int j = 0; j < nv; j++) zBest[j] = z[j];
            double bestScore = double.MaxValue;

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

                double score = rpN / (1.0 + bNorm) + rcN / (1.0 + cNorm) + mu / (1.0 + math.abs(objz));
                if (!(score < 1e300)) break;                               // NaN/Inf blow-up -> keep zBest
                if (score < bestScore) { bestScore = score; for (int j = 0; j < nv; j++) zBest[j] = z[j]; }

                if (rpN / (1.0 + bNorm) < tol && rcN / (1.0 + cNorm) < tol && mu / (1.0 + math.abs(objz)) < tol)
                { status = LPStatus.Optimal; break; }

                // normal matrix M = A D Aᵀ, D = Z S⁻¹, factored once and reused this iteration
                for (int j = 0; j < nv; j++) d[j] = z[j] / s[j];
                BuildNormalStructured(Ad, d, dd, denseK, colRow, colVal, M, m, nv, nDense, reg);
                // Near a degenerate optimum M goes numerically indefinite (float especially: reg sits
                // far below M's scale once d = z/s spreads) long before the iterate stops improving.
                // Rather than giving up on the first failed Cholesky, bump the diagonal regularization
                // a few decades and refactor -- the damped step still makes progress; if M is beyond
                // saving, bail out and return the best iterate.
                {
                    bool okM = CHO.decomp(in M, ref L);
                    fProxy bump = reg;
                    for (int t = 0; !okM && t < 4; t++)
                    {
                        bump *= (fProxy)1e3;
                        for (int i = 0; i < m; i++) M[i, i] += bump;
                        okM = CHO.decomp(in M, ref L);
                    }
                    if (!okM) break;                                       // status stays MaxIterations
                }

                // ---- affine predictor:  rhsY = b − A(D rc) ----
                for (int j = 0; j < nv; j++) tmpNV[j] = d[j] * rc[j];
                Amul(As, tmpNV, ADrd, m, nv);
                for (int i = 0; i < m; i++) { rhsY[i] = b[i] - ADrd[i]; tmpM[i] = rhsY[i]; }
                CHO.decompSolve(ref L, ref tmpM);                         // tmpM = Δy_aff
                ATmul(As, tmpM, tmpNV, m, nv);                            // tmpNV = Aᵀ Δy_aff
                for (int j = 0; j < nv; j++) dsA[j] = -rc[j] - tmpNV[j];  // Δs = −rc − AᵀΔy
                for (int j = 0; j < nv; j++) dzA[j] = -d[j] * dsA[j] - z[j];

                fProxy apA = math.min((fProxy)1, MaxStep(z, dzA, nv, BIG));
                fProxy adA = math.min((fProxy)1, MaxStep(s, dsA, nv, BIG));
                double muAff = 0;
                for (int j = 0; j < nv; j++)
                    muAff += (double)(z[j] + apA * dzA[j]) * (double)(s[j] + adA * dsA[j]);
                muAff /= nv;
                double ratio = muAff / math.max(mu, 1e-30);
                double sigma = ratio * ratio * ratio;

                // ---- centering corrector:  g = (σμ − Δz_aff∘Δs_aff)/s,  rhsY −= A g ----
                for (int j = 0; j < nv; j++)
                    g[j] = (fProxy)((sigma * mu - (double)dzA[j] * (double)dsA[j]) / (double)s[j]);
                Amul(As, g, tmpM, m, nv);
                for (int i = 0; i < m; i++) tmpM[i] = rhsY[i] - tmpM[i];
                CHO.decompSolve(ref L, ref tmpM);                         // tmpM = Δy
                for (int i = 0; i < m; i++) dy[i] = tmpM[i];
                ATmul(As, dy, tmpNV, m, nv);                              // tmpNV = Aᵀ Δy
                for (int j = 0; j < nv; j++) ds[j] = -rc[j] - tmpNV[j];
                for (int j = 0; j < nv; j++) dz[j] = -d[j] * ds[j] - z[j] + g[j];

                fProxy ap = math.min((fProxy)1, eta * MaxStep(z, dz, nv, BIG));
                fProxy ad = math.min((fProxy)1, eta * MaxStep(s, ds, nv, BIG));

                for (int j = 0; j < nv; j++) { z[j] += ap * dz[j]; s[j] += ad * ds[j]; }
                for (int i = 0; i < m; i++) y[i] += ad * dy[i];
                iters++;
            }

            // extract structural x and objective (from the best-scoring iterate, not the last z)
            for (int j = 0; j < n; j++) x[j] = zBest[j];
            double obj = 0;
            for (int j = 0; j < n; j++) obj += (double)c[j] * (double)zBest[j];
            objective = obj;

            As.Dispose(); cs.Dispose(); z.Dispose(); s.Dispose(); y.Dispose();
            dz.Dispose(); ds.Dispose(); dzA.Dispose(); dsA.Dispose(); dy.Dispose();
            d.Dispose(); rc.Dispose(); g.Dispose(); tmpNV.Dispose();
            rp.Dispose(); ADrd.Dispose(); tmpM.Dispose(); rhsY.Dispose(); M.Dispose(); L.Dispose();
            zBest.Dispose(); colRow.Dispose(); colVal.Dispose(); denseK.Dispose(); Ad.Dispose(); dd.Dispose();

            return new LPInfo { status = status, iterations = iters, objective = obj };
        }

        // out[i] = Σ_k A[i,k] v[k]   (A m×nv, v length nv, out length m). This is a plain row-major GEMV
        // -- exactly UnsafeOP.matVecDot's shape -- so it is routed through that kernel instead of a
        // naive single-accumulator scalar dot per row. matVecDot ACCUMULATES (y[r] += ...), so outv is
        // zeroed first to preserve this function's own "assign, not accumulate" contract. On the hot
        // path for both LP.InteriorPoint's interiorCore (2 calls/iteration) and LP.FrischNewton's
        // ladFrischNewtonCore (2 calls/iteration, dominant once LP.lad's hybrid default routes large-m
        // LAD to ladFN -- see LAD_HYBRID_THRESHOLD). matVecDot's SIMD fold sums each row in a
        // different order than strict left-to-right scalar accumulation would -- rounding-only, not
        // bitwise-identical.
        static unsafe void Amul(fProxyMxN A, fProxyN v, fProxyN outv, int m, int nv)
        {
            UnsafeUtility.MemClear(outv.Data.Ptr, (long)m * UnsafeUtility.SizeOf<fProxy>());
            UnsafeOP.matVecDot(A.Data.Ptr, v.Data.Ptr, outv.Data.Ptr, m, nv);
        }

        // out[j] = Σ_i A[i,j] v[i]   (v length m, out length nv). Routed through UnsafeOP.vecMatDot
        // (i outer / j inner, outv[j] += A[i,j]*v[i] -- a row-axpy accumulation, zeroed first), the
        // vectorising [NoAlias] pointer path.
        static unsafe void ATmul(fProxyMxN A, fProxyN v, fProxyN outv, int m, int nv)
        {
            UnsafeOP.vecMatDot(v.Data.Ptr, A.Data.Ptr, outv.Data.Ptr, m, nv);
        }

        // M = A diag(d) Aᵀ + reg·I  (symmetric, m×m). Upper triangle then mirrored.
        static void BuildNormal(fProxyMxN A, fProxyN d, fProxyMxN M, int m, int nv, fProxy reg)
        {
            for (int i = 0; i < m; i++)
            {
                for (int j = i; j < m; j++)
                {
                    fProxy acc = (fProxy)0;
                    for (int k = 0; k < nv; k++) acc += A[i, k] * d[k] * A[j, k];
                    if (i == j) acc += reg;
                    M[i, j] = acc;
                    M[j, i] = acc;
                }
            }
        }

        // M = Aₛ diag(d) Aₛᵀ + reg·I exploiting the standard-form structure captured by colRow/colVal:
        // only the compacted multi-nonzero columns Ad (original indices in denseK) run through the
        // O(m²·nDense) kernel; each single-nonzero column k (value v in row r) adds (v·d[k])·v straight
        // to M[r,r]. Accumulation order matches the naive kernel on the [multi | single] column layouts
        // interiorCore builds (LAD: [A|−A|−I|I]; general LP: [A|slack]), so the result is bit-identical.
        static void BuildNormalStructured(fProxyMxN Ad, fProxyN d, fProxyN dd, NativeArray<int> denseK,
                                          NativeArray<int> colRow, NativeArray<fProxy> colVal,
                                          fProxyMxN M, int m, int nv, int nDense, fProxy reg)
        {
            for (int c = 0; c < nDense; c++) dd[c] = d[denseK[c]];
            BuildNormal(Ad, dd, M, m, nDense, (fProxy)0);
            for (int k = 0; k < nv; k++)
            {
                int r = colRow[k];
                if (r >= 0) M[r, r] += colVal[k] * d[k] * colVal[k];
            }
            for (int i = 0; i < m; i++) M[i, i] += reg;
        }

        // Largest α ≥ 0 with v + α·dv ≥ 0 (uncapped; BIG if dv has no negative entry). Caller caps at 1.
        static fProxy MaxStep(fProxyN v, fProxyN dv, int nv, fProxy BIG)
        {
            fProxy a = BIG;
            for (int k = 0; k < nv; k++)
                if (dv[k] < (fProxy)0) a = math.min(a, -v[k] / dv[k]);
            return a;
        }
    }
}
