#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using Unity.Collections;
using Unity.Mathematics;

using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class LP
    {
        // ============================================================================================
        // SPARSE linear programming & least absolute deviation over a block-sparse (BSR) constraint
        // matrix, solved by a MATRIX-FREE Mehrotra predictor-corrector interior point.
        //
        // Both entry points reduce to the same standard form  min cᵀz  s.t.  Aₛ z = b,  z ≥ 0, and share
        // one generic interior-point loop (standardFormInterior). The per-iteration normal equations
        // M Δy = rhs (M = Aₛ D Aₛᵀ, SPD) are solved with the library's Krylov.pcg over a
        // floatNormalOperator (M is never formed) preconditioned by a floatNormalJacobi (diagonal of M,
        // computed matrix-free). Only the standard-form constraint operator Aₛ differs:
        //   * LP.lad  -> floatLadOperator            Aₛ = [A | −A | −I | I]   (LAD is all-equality)
        //   * LP.solve -> floatSlackAugmentedOperator Aₛ = [A | ±slack cols]  (one per inequality row)
        // Every Aₛ only ever calls spMV/spMVT on the sparse A, so nothing scales with a dense N². This is
        // the regime where a dense LP is not an option.
        //
        // Interior point reports Optimal / MaxIterations only (no exact infeasibility/unboundedness
        // certificate -- that needs a homogeneous self-dual embedding; use the dense simplex for small
        // problems needing those). Job-safe: all scratch is Allocator.Temp, disposed before return.
        // ============================================================================================

        /// <summary>
        /// Sparse least absolute deviation: minimize ‖A x − b‖₁ over a free x ∈ ℝⁿ, with A a block-sparse
        /// <see cref="floatBSR"/>. Matrix-free interior point (see the file header). <paramref name="x"/>
        /// (length A.N_Cols) is overwritten with the coefficients; <paramref name="objective"/> returns
        /// the L1 residual ‖A x − b‖₁. For small dense problems use the dense <see cref="lad(in floatMxN,
        /// in floatN, ref floatN, out double, LPMethod, int)"/> instead.
        /// </summary>
        public static LPInfo lad(in floatBSR A, in floatN b, ref floatN x, out double objective, int maxIter = 0)
        {
            int m = A.M_Rows, n = A.N_Cols;
            if (b.N != m) throw new System.ArgumentException("LP.lad(BSR): b.N must equal A.M_Rows");
            if (x.N != n) throw new System.ArgumentException("LP.lad(BSR): x.N must equal A.N_Cols");

            int nv = 2 * n + 2 * m;

            var ladSp = new floatN(n, Allocator.Temp); var ladTm = new floatN(m, Allocator.Temp);
            var ladAtr = new floatN(n, Allocator.Temp);
            var cvec = new floatN(nv, Allocator.Temp);
            var zBest = new floatN(nv, Allocator.Temp, false);
            var tmpM = new floatN(m, Allocator.Temp);

            for (int i = 0; i < 2 * m; i++) cvec[2 * n + i] = (float)1;   // cost 1 on every u, v

            var lad = new floatLadOperator(in A, in ladSp, in ladTm, in ladAtr);
            var info = standardFormInterior(in lad, in b, in cvec, ref zBest, maxIter);

            // extract x = x⁺ − x⁻ (from the best-scoring iterate) and report the true L1 residual ‖A x − b‖₁
            for (int j = 0; j < n; j++) x[j] = zBest[j] - zBest[n + j];
            BSR.spMV(in A, in x, ref tmpM);
            double l1 = 0;
            for (int i = 0; i < m; i++) l1 += math.abs((double)tmpM[i] - (double)b[i]);
            objective = l1;
            info.objective = l1;

            ladSp.Dispose(); ladTm.Dispose(); ladAtr.Dispose();
            cvec.Dispose(); zBest.Dispose(); tmpM.Dispose();
            return info;
        }

        /// <summary>
        /// General sparse linear program: minimize cᵀx s.t. A x {≤,=,≥} b (per-row
        /// <see cref="ConstraintSense"/>), x ≥ 0, with A a block-sparse <see cref="floatBSR"/>. Matrix-
        /// free interior point (see the file header): each inequality row gets one non-negative
        /// slack/surplus, expressed by a <see cref="Sparse.floatSlackAugmentedOperator"/> that never
        /// materializes the slack columns. <paramref name="x"/> (length A.N_Cols) is overwritten with the
        /// solution; <paramref name="objective"/> returns cᵀx. Interior point only -- no simplex for
        /// sparse -- so it reports Optimal / MaxIterations only (no infeasibility/unboundedness
        /// certificate; use the dense <see cref="solve(in floatMxN, in floatN, in floatN, in
        /// NativeArray{ConstraintSense}, ref floatN, out double, LPMethod, int)"/> simplex for those).
        /// </summary>
        public static LPInfo solve(in floatBSR A, in floatN b, in floatN c,
                                   in NativeArray<ConstraintSense> senses,
                                   ref floatN x, out double objective, int maxIter = 0)
        {
            int m = A.M_Rows, n = A.N_Cols;
            if (b.N != m) throw new System.ArgumentException("LP.solve(BSR): b.N must equal A.M_Rows");
            if (c.N != n) throw new System.ArgumentException("LP.solve(BSR): c.N must equal A.N_Cols");
            if (x.N != n) throw new System.ArgumentException("LP.solve(BSR): x.N must equal A.N_Cols");
            if (senses.Length != m) throw new System.ArgumentException("LP.solve(BSR): senses.Length must equal A.M_Rows");

            // one slack/surplus column per inequality ( ≤ → +1, ≥ → −1 ); equalities get none
            int nSlack = 0;
            for (int i = 0; i < m; i++) if (senses[i] != ConstraintSense.Equal) nSlack++;
            int nv = n + nSlack;

            var slackRow = new NativeArray<int>(nSlack, Allocator.Temp);
            var slackSign = new NativeArray<float>(nSlack, Allocator.Temp);
            {
                int k = 0;
                for (int i = 0; i < m; i++)
                {
                    if (senses[i] == ConstraintSense.Equal) continue;
                    slackRow[k] = i;
                    slackSign[k] = senses[i] == ConstraintSense.LessEqual ? (float)1 : (float)(-1);
                    k++;
                }
            }

            var opSp = new floatN(n, Allocator.Temp); var opAtr = new floatN(n, Allocator.Temp);
            var cvec = new floatN(nv, Allocator.Temp);
            var zBest = new floatN(nv, Allocator.Temp, false);
            for (int j = 0; j < n; j++) cvec[j] = c[j];   // slack costs stay 0

            var op = new floatSlackAugmentedOperator(in A, nSlack, in slackRow, in slackSign, in opSp, in opAtr);
            var info = standardFormInterior(in op, in b, in cvec, ref zBest, maxIter);

            double obj = 0;
            for (int j = 0; j < n; j++) { x[j] = zBest[j]; obj += (double)c[j] * (double)zBest[j]; }
            objective = obj;
            info.objective = obj;

            slackRow.Dispose(); slackSign.Dispose(); opSp.Dispose(); opAtr.Dispose();
            cvec.Dispose(); zBest.Dispose();
            return info;
        }

        // --------------------------------------------------------------------------------------------
        // Shared matrix-free Mehrotra predictor-corrector on standard form  min cᵀz s.t. Aₛ z = b, z ≥ 0.
        // Generic over the standard-form constraint operator (the ONLY thing that differs between LAD and
        // general LP); everything below -- infeasible-start point, normal-equations PCG solve, affine +
        // centering steps, best-iterate safeguard -- is identical. On return zBest (length Aₛ.Cols) holds
        // the best-scoring iterate; the caller extracts x and fills LPInfo.objective. reg regularizes the
        // (near-singular near the optimum) normal matrix M := Aₛ D Aₛᵀ + reg·I.
        // --------------------------------------------------------------------------------------------
        static LPInfo standardFormInterior<TOp>(in TOp aS, in floatN b, in floatN cvec,
                                                ref floatN zBest, int maxIter)
            where TOp : struct, IfloatStandardFormOperator
        {
            int m = aS.Rows, nv = aS.Cols;

            // standard-form scratch (length nv)
            var z = new floatN(nv, Allocator.Temp); var s = new floatN(nv, Allocator.Temp);
            var dz = new floatN(nv, Allocator.Temp); var ds = new floatN(nv, Allocator.Temp);
            var dzA = new floatN(nv, Allocator.Temp); var dsA = new floatN(nv, Allocator.Temp);
            var d = new floatN(nv, Allocator.Temp); var rc = new floatN(nv, Allocator.Temp);
            var g = new floatN(nv, Allocator.Temp); var tmpNV = new floatN(nv, Allocator.Temp);
            var normNV = new floatN(nv, Allocator.Temp);
            // length-m scratch
            var y = new floatN(m, Allocator.Temp); var dy = new floatN(m, Allocator.Temp);
            var rp = new floatN(m, Allocator.Temp); var rhsY = new floatN(m, Allocator.Temp);
            var ADrc = new floatN(m, Allocator.Temp); var tmpM = new floatN(m, Allocator.Temp);
            var diagM = new floatN(m, Allocator.Temp); var invDiag = new floatN(m, Allocator.Temp);
            // pcg scratch (length m)
            var pr = new floatN(m, Allocator.Temp); var pp = new floatN(m, Allocator.Temp);
            var pAp = new floatN(m, Allocator.Temp); var pz = new floatN(m, Allocator.Temp);

            // Primal-dual regularization M := Aₛ D Aₛᵀ + reg·I. The normal matrix becomes numerically
            // singular as the interior point approaches the (often degenerate) optimum; without reg the
            // inexact PCG returns a huge Δy that overflows. reg bounds M's smallest eigenvalue.
            float reg = math.max(Consts.floatZeroThreshold, (float)1e-8);
            var Mop = new floatNormalOperator<TOp>(in aS, in d, in normNV, reg);
            var Jac = new floatNormalJacobi(in invDiag);

            float BIG = (float)1e30;
            float eta = (float)0.99;
            float floorPos = Consts.floatZeroThreshold;
            double tol = 100.0 * (double)Consts.floatEpsilon;
            float pcgTol = Consts.floatSqrtEps;
            int pcgMaxIter = math.min(2 * m + 20, 500);

            double bNorm = 0, cNorm = 0;
            for (int i = 0; i < m; i++) bNorm += (double)b[i] * (double)b[i];
            for (int j = 0; j < nv; j++) cNorm += (double)cvec[j] * (double)cvec[j];
            bNorm = math.sqrt(bNorm); cNorm = math.sqrt(cNorm);

            // --- Mehrotra starting point (D = 1): z̃ = Aₛᵀ(Aₛ Aₛᵀ)⁻¹ b, ỹ = (Aₛ Aₛᵀ)⁻¹(Aₛ c) ---
            for (int j = 0; j < nv; j++) d[j] = (float)1;
            aS.NormalDiagonal(in d, ref diagM);
            for (int i = 0; i < m; i++) invDiag[i] = (float)1 / (diagM[i] + reg);

            for (int i = 0; i < m; i++) dy[i] = (float)0;
            Krylov.pcg(in Mop, in Jac, in b, ref dy, ref pr, ref pp, ref pAp, ref pz, pcgMaxIter, pcgTol);
            aS.ApplyT(in dy, ref z);                                    // z = z̃

            aS.Apply(in cvec, ref tmpM);                               // tmpM = Aₛ c
            for (int i = 0; i < m; i++) y[i] = (float)0;
            Krylov.pcg(in Mop, in Jac, in tmpM, ref y, ref pr, ref pp, ref pAp, ref pz, pcgMaxIter, pcgTol);
            aS.ApplyT(in y, ref s);
            for (int j = 0; j < nv; j++) s[j] = cvec[j] - s[j];         // s̃ = c − Aₛᵀ ỹ

            // shift z, s strictly interior
            {
                float minZ = BIG, minS = BIG;
                for (int j = 0; j < nv; j++) { minZ = math.min(minZ, z[j]); minS = math.min(minS, s[j]); }
                float dxs = math.max(-(float)1.5 * minZ, (float)0);
                float dss = math.max(-(float)1.5 * minS, (float)0);
                double sumZ = 0, sumS = 0, dotZS = 0;
                for (int j = 0; j < nv; j++) { z[j] += dxs; s[j] += dss; sumZ += (double)z[j]; sumS += (double)s[j]; }
                for (int j = 0; j < nv; j++) dotZS += (double)z[j] * (double)s[j];
                float pdz = (float)(0.5 * dotZS / math.max(sumS, 1e-30));
                float pds = (float)(0.5 * dotZS / math.max(sumZ, 1e-30));
                for (int j = 0; j < nv; j++)
                {
                    z[j] = math.max(z[j] + pdz, floorPos);
                    s[j] = math.max(s[j] + pds, floorPos);
                }
            }

            int budget = maxIter > 0 ? maxIter : 100;
            LPStatus status = LPStatus.MaxIterations;
            int iters = 0;

            // Inexact PCG directions can diverge near a degenerate optimum (M → singular). Keep the
            // best-scoring iterate seen (in the caller's zBest) and recover x from it, so a late blow-up
            // can't poison the result.
            for (int j = 0; j < nv; j++) zBest[j] = z[j];
            double bestScore = double.MaxValue;

            while (iters < budget)
            {
                aS.Apply(in z, ref rp);
                for (int i = 0; i < m; i++) rp[i] -= b[i];                       // rp = Aₛz − b
                aS.ApplyT(in y, ref rc);
                for (int j = 0; j < nv; j++) rc[j] += s[j] - cvec[j];            // rc = Aₛᵀy + s − c
                double mu = 0, objz = 0;
                for (int j = 0; j < nv; j++) { mu += (double)z[j] * (double)s[j]; objz += (double)cvec[j] * (double)z[j]; }
                mu /= nv;

                double rpN = 0, rcN = 0;
                for (int i = 0; i < m; i++) rpN += (double)rp[i] * (double)rp[i];
                for (int j = 0; j < nv; j++) rcN += (double)rc[j] * (double)rc[j];
                rpN = math.sqrt(rpN); rcN = math.sqrt(rcN);

                double score = rpN / (1.0 + bNorm) + rcN / (1.0 + cNorm) + mu / (1.0 + math.abs(objz));
                if (!(score < 1e300)) break;                                     // NaN/Inf blow-up -> keep zBest
                if (score < bestScore) { bestScore = score; for (int j = 0; j < nv; j++) zBest[j] = z[j]; }

                if (rpN / (1.0 + bNorm) < tol && rcN / (1.0 + cNorm) < tol && mu / (1.0 + math.abs(objz)) < tol)
                { status = LPStatus.Optimal; break; }

                for (int j = 0; j < nv; j++) d[j] = z[j] / s[j];                 // D = Z S⁻¹
                aS.NormalDiagonal(in d, ref diagM);
                for (int i = 0; i < m; i++) invDiag[i] = (float)1 / (diagM[i] + reg);

                // ---- affine predictor:  M Δy_aff = b − Aₛ(D rc) ----
                for (int j = 0; j < nv; j++) tmpNV[j] = d[j] * rc[j];
                aS.Apply(in tmpNV, ref ADrc);
                for (int i = 0; i < m; i++) rhsY[i] = b[i] - ADrc[i];
                Krylov.pcg(in Mop, in Jac, in rhsY, ref dy, ref pr, ref pp, ref pAp, ref pz, pcgMaxIter, pcgTol);
                aS.ApplyT(in dy, ref tmpNV);                                    // tmpNV = Aₛᵀ Δy_aff
                for (int j = 0; j < nv; j++) { dsA[j] = -rc[j] - tmpNV[j]; dzA[j] = -d[j] * dsA[j] - z[j]; }

                float apA = math.min((float)1, MaxStep(z, dzA, nv, BIG));
                float adA = math.min((float)1, MaxStep(s, dsA, nv, BIG));
                double muAff = 0;
                for (int j = 0; j < nv; j++) muAff += (double)(z[j] + apA * dzA[j]) * (double)(s[j] + adA * dsA[j]);
                muAff /= nv;
                double sig = muAff / math.max(mu, 1e-30); sig = sig * sig * sig;

                // ---- centering corrector:  g = (σμ − Δz_aff∘Δs_aff)/s,  rhs −= Aₛ g ----
                for (int j = 0; j < nv; j++) g[j] = (float)((sig * mu - (double)dzA[j] * (double)dsA[j]) / (double)s[j]);
                aS.Apply(in g, ref tmpM);
                for (int i = 0; i < m; i++) rhsY[i] -= tmpM[i];
                Krylov.pcg(in Mop, in Jac, in rhsY, ref dy, ref pr, ref pp, ref pAp, ref pz, pcgMaxIter, pcgTol);
                aS.ApplyT(in dy, ref tmpNV);
                for (int j = 0; j < nv; j++) { ds[j] = -rc[j] - tmpNV[j]; dz[j] = -d[j] * ds[j] - z[j] + g[j]; }

                float ap = math.min((float)1, eta * MaxStep(z, dz, nv, BIG));
                float ad = math.min((float)1, eta * MaxStep(s, ds, nv, BIG));
                for (int j = 0; j < nv; j++) { z[j] += ap * dz[j]; s[j] += ad * ds[j]; }
                for (int i = 0; i < m; i++) y[i] += ad * dy[i];
                iters++;
            }

            z.Dispose(); s.Dispose(); dz.Dispose(); ds.Dispose(); dzA.Dispose(); dsA.Dispose();
            d.Dispose(); rc.Dispose(); g.Dispose(); tmpNV.Dispose(); normNV.Dispose();
            y.Dispose(); dy.Dispose(); rp.Dispose(); rhsY.Dispose(); ADrc.Dispose(); tmpM.Dispose();
            diagM.Dispose(); invDiag.Dispose(); pr.Dispose(); pp.Dispose(); pAp.Dispose(); pz.Dispose();

            return new LPInfo { status = status, iterations = iters, objective = 0 };
        }
    }
}
