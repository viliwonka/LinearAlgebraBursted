#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using Unity.Collections;
using Unity.Mathematics;

using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class LP
    {
        // ============================================================================================
        // SPARSE least absolute deviation (L1 regression) over a block-sparse (BSR) design matrix.
        //
        // Same reformulation as the dense LP.lad -- minimize ‖A x − b‖₁ becomes the all-equality LP
        //   min Σ(uᵢ + vᵢ)  s.t.  A(x⁺−x⁻) − u + v = b,   [x⁺|x⁻|u|v] ≥ 0
        // -- but solved by a MATRIX-FREE Mehrotra interior point: the per-iteration normal equations
        // M Δy = rhs (M = Aₛ D Aₛᵀ, SPD) are solved with the library's Krylov.pcg over a
        // doubleNormalOperator (M is never formed) preconditioned by a doubleNormalJacobi (diagonal of
        // M, computed matrix-free). The constraint operator Aₛ = [A|−A|−I|I] is doubleLadOperator, which
        // only ever calls spMV/spMVT on the sparse A. This is the regime where a dense LP is not an
        // option -- A stays sparse throughout, nothing scales with a dense N².
        //
        // Interior point reports Optimal / MaxIterations only (no exact infeasibility/unboundedness
        // certificate). Job-safe: all scratch is Allocator.Temp, disposed before return.
        // ============================================================================================

        /// <summary>
        /// Sparse least absolute deviation: minimize ‖A x − b‖₁ over a free x ∈ ℝⁿ, with A a block-sparse
        /// <see cref="doubleBSR"/>. Matrix-free interior point (see the file header). <paramref name="x"/>
        /// (length A.N_Cols) is overwritten with the coefficients; <paramref name="objective"/> returns
        /// the L1 residual ‖A x − b‖₁. For small dense problems use the dense <see cref="lad(in doubleMxN,
        /// in doubleN, ref doubleN, out double, LPMethod, int)"/> instead.
        /// </summary>
        public static LPInfo lad(in doubleBSR A, in doubleN b, ref doubleN x, out double objective, int maxIter = 0)
        {
            int m = A.M_Rows, n = A.N_Cols;
            if (b.N != m) throw new System.ArgumentException("LP.lad(BSR): b.N must equal A.M_Rows");
            if (x.N != n) throw new System.ArgumentException("LP.lad(BSR): x.N must equal A.N_Cols");

            int nv = 2 * n + 2 * m;

            // standard-form scratch (length nv)
            var z = new doubleN(nv, Allocator.Temp); var s = new doubleN(nv, Allocator.Temp);
            var dz = new doubleN(nv, Allocator.Temp); var ds = new doubleN(nv, Allocator.Temp);
            var dzA = new doubleN(nv, Allocator.Temp); var dsA = new doubleN(nv, Allocator.Temp);
            var d = new doubleN(nv, Allocator.Temp); var rc = new doubleN(nv, Allocator.Temp);
            var g = new doubleN(nv, Allocator.Temp); var tmpNV = new doubleN(nv, Allocator.Temp);
            var cvec = new doubleN(nv, Allocator.Temp); var normNV = new doubleN(nv, Allocator.Temp);
            // length-m scratch
            var y = new doubleN(m, Allocator.Temp); var dy = new doubleN(m, Allocator.Temp);
            var rp = new doubleN(m, Allocator.Temp); var rhsY = new doubleN(m, Allocator.Temp);
            var ADrc = new doubleN(m, Allocator.Temp); var tmpM = new doubleN(m, Allocator.Temp);
            var diagM = new doubleN(m, Allocator.Temp); var invDiag = new doubleN(m, Allocator.Temp);
            // pcg scratch (length m)
            var pr = new doubleN(m, Allocator.Temp); var pp = new doubleN(m, Allocator.Temp);
            var pAp = new doubleN(m, Allocator.Temp); var pz = new doubleN(m, Allocator.Temp);
            // LAD operator scratch
            var ladSp = new doubleN(n, Allocator.Temp); var ladTm = new doubleN(m, Allocator.Temp);
            var ladAtr = new doubleN(n, Allocator.Temp);

            for (int i = 0; i < 2 * m; i++) cvec[2 * n + i] = (double)1;   // cost 1 on every u, v

            // Primal-dual regularization M := Aₛ D Aₛᵀ + reg·I. The normal matrix becomes numerically
            // singular as the interior point approaches the (degenerate) L1 optimum; without reg the
            // inexact PCG returns a huge Δy that overflows. reg bounds M's smallest eigenvalue.
            double reg = math.max(Consts.doubleZeroThreshold, (double)1e-8);
            var lad = new doubleLadOperator(in A, in ladSp, in ladTm, in ladAtr);
            var Mop = new doubleNormalOperator<doubleLadOperator>(in lad, in d, in normNV, reg);
            var Jac = new doubleNormalJacobi(in invDiag);

            double BIG = (double)1e30;
            double eta = (double)0.99;
            double floorPos = Consts.doubleZeroThreshold;
            double tol = 100.0 * (double)Consts.doubleEpsilon;
            double pcgTol = Consts.doubleSqrtEps;
            int pcgMaxIter = math.min(2 * m + 20, 500);

            double bNorm = 0, cNorm = 0;
            for (int i = 0; i < m; i++) bNorm += (double)b[i] * (double)b[i];
            for (int j = 0; j < nv; j++) cNorm += (double)cvec[j] * (double)cvec[j];
            bNorm = math.sqrt(bNorm); cNorm = math.sqrt(cNorm);

            // --- Mehrotra starting point (D = 1): z̃ = Aₛᵀ(Aₛ Aₛᵀ)⁻¹ b, ỹ = (Aₛ Aₛᵀ)⁻¹(Aₛ c) ---
            for (int j = 0; j < nv; j++) d[j] = (double)1;
            lad.NormalDiagonal(in d, ref diagM);
            for (int i = 0; i < m; i++) invDiag[i] = (double)1 / (diagM[i] + reg);

            for (int i = 0; i < m; i++) dy[i] = (double)0;
            Krylov.pcg(in Mop, in Jac, in b, ref dy, ref pr, ref pp, ref pAp, ref pz, pcgMaxIter, pcgTol);
            lad.ApplyT(in dy, ref z);                                    // z = z̃

            lad.Apply(in cvec, ref tmpM);                               // tmpM = Aₛ c
            for (int i = 0; i < m; i++) y[i] = (double)0;
            Krylov.pcg(in Mop, in Jac, in tmpM, ref y, ref pr, ref pp, ref pAp, ref pz, pcgMaxIter, pcgTol);
            lad.ApplyT(in y, ref s);
            for (int j = 0; j < nv; j++) s[j] = cvec[j] - s[j];         // s̃ = c − Aₛᵀ ỹ

            // shift z, s strictly interior (identical heuristic to the dense interiorCore)
            {
                double minZ = BIG, minS = BIG;
                for (int j = 0; j < nv; j++) { minZ = math.min(minZ, z[j]); minS = math.min(minS, s[j]); }
                double dxs = math.max(-(double)1.5 * minZ, (double)0);
                double dss = math.max(-(double)1.5 * minS, (double)0);
                double sumZ = 0, sumS = 0, dotZS = 0;
                for (int j = 0; j < nv; j++) { z[j] += dxs; s[j] += dss; sumZ += (double)z[j]; sumS += (double)s[j]; }
                for (int j = 0; j < nv; j++) dotZS += (double)z[j] * (double)s[j];
                double pdz = (double)(0.5 * dotZS / math.max(sumS, 1e-30));
                double pds = (double)(0.5 * dotZS / math.max(sumZ, 1e-30));
                for (int j = 0; j < nv; j++)
                {
                    z[j] = math.max(z[j] + pdz, floorPos);
                    s[j] = math.max(s[j] + pds, floorPos);
                }
            }

            int budget = maxIter > 0 ? maxIter : 100;
            LPStatus status = LPStatus.MaxIterations;
            int iters = 0;

            // Inexact PCG directions can diverge near the degenerate L1 optimum (M → singular). Keep the
            // best-scoring iterate seen and recover x from it, so a late blow-up can't poison the result.
            var bestZ = new doubleN(nv, Allocator.Temp, false);
            for (int j = 0; j < nv; j++) bestZ[j] = z[j];
            double bestScore = double.MaxValue;

            while (iters < budget)
            {
                lad.Apply(in z, ref rp);
                for (int i = 0; i < m; i++) rp[i] -= b[i];                       // rp = Aₛz − b
                lad.ApplyT(in y, ref rc);
                for (int j = 0; j < nv; j++) rc[j] += s[j] - cvec[j];            // rc = Aₛᵀy + s − c
                double mu = 0, objz = 0;
                for (int j = 0; j < nv; j++) { mu += (double)z[j] * (double)s[j]; objz += (double)cvec[j] * (double)z[j]; }
                mu /= nv;

                double rpN = 0, rcN = 0;
                for (int i = 0; i < m; i++) rpN += (double)rp[i] * (double)rp[i];
                for (int j = 0; j < nv; j++) rcN += (double)rc[j] * (double)rc[j];
                rpN = math.sqrt(rpN); rcN = math.sqrt(rcN);

                double score = rpN / (1.0 + bNorm) + rcN / (1.0 + cNorm) + mu / (1.0 + math.abs(objz));
                if (!(score < 1e300)) break;                                     // NaN/Inf blow-up -> keep bestZ
                if (score < bestScore) { bestScore = score; for (int j = 0; j < nv; j++) bestZ[j] = z[j]; }

                if (rpN / (1.0 + bNorm) < tol && rcN / (1.0 + cNorm) < tol && mu / (1.0 + math.abs(objz)) < tol)
                { status = LPStatus.Optimal; break; }

                for (int j = 0; j < nv; j++) d[j] = z[j] / s[j];                 // D = Z S⁻¹
                lad.NormalDiagonal(in d, ref diagM);
                for (int i = 0; i < m; i++) invDiag[i] = (double)1 / (diagM[i] + reg);

                // ---- affine predictor:  M Δy_aff = b − Aₛ(D rc) ----
                for (int j = 0; j < nv; j++) tmpNV[j] = d[j] * rc[j];
                lad.Apply(in tmpNV, ref ADrc);
                for (int i = 0; i < m; i++) rhsY[i] = b[i] - ADrc[i];
                Krylov.pcg(in Mop, in Jac, in rhsY, ref dy, ref pr, ref pp, ref pAp, ref pz, pcgMaxIter, pcgTol);
                lad.ApplyT(in dy, ref tmpNV);                                    // tmpNV = Aₛᵀ Δy_aff
                for (int j = 0; j < nv; j++) { dsA[j] = -rc[j] - tmpNV[j]; dzA[j] = -d[j] * dsA[j] - z[j]; }

                double apA = math.min((double)1, MaxStep(z, dzA, nv, BIG));
                double adA = math.min((double)1, MaxStep(s, dsA, nv, BIG));
                double muAff = 0;
                for (int j = 0; j < nv; j++) muAff += (double)(z[j] + apA * dzA[j]) * (double)(s[j] + adA * dsA[j]);
                muAff /= nv;
                double sig = muAff / math.max(mu, 1e-30); sig = sig * sig * sig;

                // ---- centering corrector:  g = (σμ − Δz_aff∘Δs_aff)/s,  rhs −= Aₛ g ----
                for (int j = 0; j < nv; j++) g[j] = (double)((sig * mu - (double)dzA[j] * (double)dsA[j]) / (double)s[j]);
                lad.Apply(in g, ref tmpM);
                for (int i = 0; i < m; i++) rhsY[i] -= tmpM[i];
                Krylov.pcg(in Mop, in Jac, in rhsY, ref dy, ref pr, ref pp, ref pAp, ref pz, pcgMaxIter, pcgTol);
                lad.ApplyT(in dy, ref tmpNV);
                for (int j = 0; j < nv; j++) { ds[j] = -rc[j] - tmpNV[j]; dz[j] = -d[j] * ds[j] - z[j] + g[j]; }

                double ap = math.min((double)1, eta * MaxStep(z, dz, nv, BIG));
                double ad = math.min((double)1, eta * MaxStep(s, ds, nv, BIG));
                for (int j = 0; j < nv; j++) { z[j] += ap * dz[j]; s[j] += ad * ds[j]; }
                for (int i = 0; i < m; i++) y[i] += ad * dy[i];
                iters++;
            }

            // extract x = x⁺ − x⁻ (from the best-scoring iterate) and report ‖A x − b‖₁
            for (int j = 0; j < n; j++) x[j] = bestZ[j] - bestZ[n + j];
            BSR.spMV(in A, in x, ref tmpM);
            double l1 = 0;
            for (int i = 0; i < m; i++) l1 += math.abs((double)tmpM[i] - (double)b[i]);
            objective = l1;

            z.Dispose(); s.Dispose(); dz.Dispose(); ds.Dispose(); dzA.Dispose(); dsA.Dispose();
            d.Dispose(); rc.Dispose(); g.Dispose(); tmpNV.Dispose(); cvec.Dispose(); normNV.Dispose();
            y.Dispose(); dy.Dispose(); rp.Dispose(); rhsY.Dispose(); ADrc.Dispose(); tmpM.Dispose();
            diagM.Dispose(); invDiag.Dispose(); pr.Dispose(); pp.Dispose(); pAp.Dispose(); pz.Dispose();
            ladSp.Dispose(); ladTm.Dispose(); ladAtr.Dispose(); bestZ.Dispose();

            return new LPInfo { status = status, iterations = iters, objective = l1 };
        }
    }
}
