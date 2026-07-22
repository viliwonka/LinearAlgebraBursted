using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov
    {
        /// <summary>
        /// GCRO-DR (Morgan 2002 / Parks-de Sturler-Mackey-Johnson-Maiti 2006): restarted GMRES(m)
        /// that RECYCLES a k-dimensional approximate invariant subspace (harmonic Ritz vectors)
        /// across restart cycles, for a general (nonsymmetric) square A x = b. Generic over both the
        /// operator (<see cref="IfProxyLinearOperator"/>) and the preconditioner
        /// (<see cref="IfProxyPreconditioner"/>) -- the SINGLE body behind the plain and the
        /// right-preconditioned entry points.
        ///
        /// Each cycle: project the recycled subspace out of the residual and fold its correction into
        /// x, run an m-step Arnoldi (projected against the recycled C so the Krylov directions stay
        /// recycle-orthogonal) with the SAME Hessenberg/Givens least-squares machinery as
        /// <see cref="gmres{TOp,TPre}"/>, then rebuild the k recycled vectors from a small dense
        /// harmonic-Ritz eigenproblem over the combined (old-recycle + this-cycle-Krylov) subspace.
        /// The harmonic Ritz VALUES come from <see cref="Eigen.valuesQRInPlace"/> (general nonsymmetric
        /// eigenvalues); each selected value's REFINED vector (the minimizer of ‖(A-θI)v‖ over the
        /// combined subspace) comes from <see cref="Eigen.symmetricInPlace"/> on a small symmetric
        /// matrix -- this library has no general nonsymmetric eigenVECTOR solver, so the refined-vector
        /// route reuses the two eigensolvers that exist instead of hand-rolling a new one.
        ///
        /// With <see cref="fProxyIdentityPreconditioner"/> the IsIdentity fold matches
        /// <see cref="gmres{TOp,TPre}"/>'s: no M⁻¹ apply, no separate preconditioned-basis storage.
        /// recycle = 0 disables recycling entirely and is bit-identical to <see cref="gmres{TOp,TPre}"/>.
        ///
        /// x is a warm-startable initial guess, overwritten with the solution; tol is relative
        /// (‖b − Ax‖ ≤ tol·‖b‖); maxIter counts TOTAL inner Arnoldi iterations across restarts (the
        /// per-cycle residual/recycle-projection matvecs are not counted, matching gmres's convention).
        /// recycle must be in [0, restart). Allocates its workspace (Arnoldi + recycle + small dense
        /// deflation buffers) from the Temp allocator. Returns the shared <see cref="SolveInfo"/>;
        /// rnorm on a Converged exit is a freshly recomputed ‖b−Ax‖, not the raw Arnoldi/Givens
        /// estimate (which only measures the C-orthogonal, recycle-projected residual) -- a failed
        /// verify falls through to another cycle instead of a false Converged. Status: Converged /
        /// MaxIterations / Breakdown (a collapsed Hessenberg pivot in the least-squares
        /// back-substitution) -- never NaN, never a false Converged. A failed or ill-conditioned
        /// per-cycle deflation update degrades gracefully (keeps the previous cycle's recycled
        /// subspace) rather than aborting the solve.
        /// </summary>
        public static SolveInfo gcrodr<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x,
                                     int restart, int recycle, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols) throw new ArgumentException("gcrodr: A must be square");
            if (b.N != A.Rows) throw new ArgumentException("gcrodr: b.N must equal A.Rows");
            if (x.N != A.Rows) throw new ArgumentException("gcrodr: x.N must equal A.Rows");
            if (restart < 1) throw new ArgumentException("gcrodr: restart must be >= 1");
            if (recycle < 0) throw new ArgumentException("gcrodr: recycle must be >= 0");
            if (recycle >= restart) throw new ArgumentException("gcrodr: recycle must be < restart");
            if (maxIter < 1) throw new ArgumentException("gcrodr: maxIter must be >= 1");
            if (!M.IsConstant)
                throw new ArgumentException("Krylov.gcrodr: requires a constant (non-flexible) preconditioner (M.IsConstant == false — e.g. an AMG K-cycle). Use the flexible variant (fcg / fgmres).");

            int n = A.Rows;
            int m = restart;
            bool flexible = !M.IsIdentity;
            bool recycling = recycle > 0;

            fProxy bb = Blas.dot(b, b);
            if (bb == (fProxy)0)
            {
                x.CopyFrom(in b);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }
            fProxy bnorm = math.sqrt(bb);
            fProxy thresh = tol * bnorm;
            // hMaxAbs tracks the running max |H| entry (raw Arnoldi, pre-Givens) seen so far in this
            // solve -- an ||A||-scaled reference for the Arnoldi/back-substitution pivot guards below,
            // instead of the ||b||-scaled bnorm (which under- or over-guards whenever ||A|| and ||b||
            // differ in magnitude). The Ru guard is scaled separately, per use, by max |Ru[i,i]|.
            fProxy hMaxAbs = (fProxy)0;
            fProxy hPivotGuard = (fProxy)0;

            // ---- Arnoldi workspace (Temp), fixed max size, mirrors gmres/fgmres ----
            var V = new UnsafeList<fProxyN>(m + 1, Allocator.Temp);
            for (int i = 0; i <= m; i++) V.Add(new fProxyN(n));
            UnsafeList<fProxyN> Zv = default;
            if (flexible)
            {
                Zv = new UnsafeList<fProxyN>(m, Allocator.Temp);
                for (int i = 0; i < m; i++) Zv.Add(new fProxyN(n));
            }
            UnsafeList<fProxyN> AV = default;
            if (recycling)
            {
                AV = new UnsafeList<fProxyN>(m, Allocator.Temp);
                for (int i = 0; i < m; i++) AV.Add(new fProxyN(n));
            }
            var H = new fProxyMxN(m + 1, m, Allocator.Temp, false);
            var cs = new fProxyN(m);
            var sn = new fProxyN(m);
            var g = new fProxyN(m + 1);
            var y = new fProxyN(m);
            var w = new fProxyN(n);

            // ---- recycled-subspace state (Temp), fixed max size `recycle`, active prefix `kcur` ----
            UnsafeList<fProxyN> U = default, C = default;
            fProxyMxN Ru = default;   // recycle x recycle, active top-left kcur x kcur upper-triangular
            fProxyMxN Bmat = default; // recycle x m, active kcur x p projection coefficients
            fProxyN ctr = default;
            fProxyN zproj = default;
            int kcur = 0;
            if (recycling)
            {
                U = new UnsafeList<fProxyN>(recycle, Allocator.Temp);
                C = new UnsafeList<fProxyN>(recycle, Allocator.Temp);
                for (int i = 0; i < recycle; i++) { U.Add(new fProxyN(n)); C.Add(new fProxyN(n)); }
                Ru = new fProxyMxN(recycle, recycle, Allocator.Temp, false);
                Bmat = new fProxyMxN(recycle, m, Allocator.Temp, false);
                ctr = new fProxyN(recycle);
                zproj = new fProxyN(recycle);
            }

            int total = 0;
            fProxy resnorm = bnorm;
            bool converged = false;
            bool breakdown = false;

            while (total < maxIter && !converged && !breakdown)
            {
                // ---- residual, then recycled-subspace projection (fresh matvec both times: never
                // trust an incremental residual update through a possibly near-singular Ru solve) ----
                fProxyN v0 = V[0];
                A.Apply(in x, ref w);
                v0.CopyFrom(in b);
                v0.addScaledInPlace((fProxy)(-1), w);

                if (kcur > 0)
                {
                    for (int i = 0; i < kcur; i++) ctr[i] = Blas.dot(C[i], v0);

                    fProxy ruMaxAbs = (fProxy)0;
                    for (int i = 0; i < kcur; i++) ruMaxAbs = math.max(ruMaxAbs, math.abs(Ru[i, i]));
                    fProxy ruPivotGuard = Consts.fProxyEpsilon * (fProxy)100 * ruMaxAbs;
                    for (int i = kcur - 1; i >= 0; i--)
                    {
                        fProxy s = ctr[i];
                        for (int l = i + 1; l < kcur; l++) s -= Ru[i, l] * zproj[l];
                        fProxy diag = Ru[i, i];
                        zproj[i] = math.abs(diag) > ruPivotGuard ? s / diag : (fProxy)0;
                    }
                    for (int i = 0; i < kcur; i++) x.addScaledInPlace(zproj[i], U[i]);

                    A.Apply(in x, ref w);
                    v0.CopyFrom(in b);
                    v0.addScaledInPlace((fProxy)(-1), w);
                }

                fProxy beta = math.sqrt(Blas.dot(v0, v0));
                resnorm = beta;
                if (beta <= thresh) { converged = true; break; }

                fProxy invBeta = (fProxy)1 / beta;
                for (int i = 0; i < n; i++) v0[i] *= invBeta;
                for (int i = 0; i <= m; i++) g[i] = (fProxy)0;
                g[0] = beta;

                int p = 0;
                for (int j = 0; j < m && total < maxIter; j++)
                {
                    fProxyN vj = V[j];
                    if (flexible)
                    {
                        fProxyN zj = Zv[j];
                        M.Apply(in vj, ref zj);
                        A.Apply(in zj, ref w);
                    }
                    else
                    {
                        A.Apply(in vj, ref w);
                    }

                    if (recycling) AV[j].CopyFrom(in w);   // raw A*(this-cycle basis vector), pre-projection

                    if (kcur > 0)
                    {
                        for (int i = 0; i < kcur; i++)
                        {
                            fProxy bij = Blas.dot(C[i], w);
                            Bmat[i, j] = bij;
                            w.addScaledInPlace(-bij, C[i]);
                        }
                    }

                    // Modified Gram-Schmidt against v_0..v_j.
                    for (int i = 0; i <= j; i++)
                    {
                        fProxyN vi = V[i];
                        fProxy hij = Blas.dot(w, vi);
                        H[i, j] = hij;
                        hMaxAbs = math.max(hMaxAbs, math.abs(hij));
                        w.addScaledInPlace(-hij, vi);
                    }
                    fProxy hj1 = math.sqrt(Blas.dot(w, w));
                    H[j + 1, j] = hj1;
                    hPivotGuard = Consts.fProxyEpsilon * (fProxy)100 * hMaxAbs;
                    bool arnoldiDone = hj1 <= hPivotGuard;
                    hMaxAbs = math.max(hMaxAbs, hj1);
                    if (!arnoldiDone)
                    {
                        fProxyN vj1 = V[j + 1];
                        fProxy invh = (fProxy)1 / hj1;
                        vj1.CopyFrom(in w);
                        for (int i = 0; i < n; i++) vj1[i] *= invh;
                    }

                    // Apply previous Givens rotations to column j of H.
                    for (int i = 0; i < j; i++)
                    {
                        fProxy t0 = cs[i] * H[i, j] + sn[i] * H[i + 1, j];
                        H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j];
                        H[i, j] = t0;
                    }

                    // New Givens rotation zeroing H[j+1,j].
                    fProxy a0 = H[j, j], b0 = H[j + 1, j];
                    fProxy rr = math.sqrt(a0 * a0 + b0 * b0);
                    fProxy c, s;
                    if (rr > (fProxy)0) { c = a0 / rr; s = b0 / rr; }
                    else { c = (fProxy)1; s = (fProxy)0; }
                    cs[j] = c; sn[j] = s;
                    H[j, j] = rr;
                    H[j + 1, j] = (fProxy)0;

                    fProxy gj = g[j];
                    g[j] = c * gj;
                    g[j + 1] = -s * gj;

                    resnorm = math.abs(g[j + 1]);
                    total++;
                    p = j + 1;

                    if (resnorm <= thresh) { converged = true; break; }
                    if (arnoldiDone) break;
                }

                // Back-substitute H[0..p-1,0..p-1] y = g[0..p-1]; a collapsed pivot is an honest
                // Breakdown (never divide by a near-zero H[i,i]). hPivotGuard uses the full running
                // hMaxAbs (including the last column's subdiagonal), the freshest ||A||-scale estimate.
                hPivotGuard = Consts.fProxyEpsilon * (fProxy)100 * hMaxAbs;
                for (int i = p - 1; i >= 0; i--)
                {
                    fProxy sum = g[i];
                    for (int l = i + 1; l < p; l++) sum -= H[i, l] * y[l];
                    fProxy diag = H[i, i];
                    if (math.abs(diag) <= hPivotGuard) { breakdown = true; break; }
                    y[i] = sum / diag;
                }
                if (breakdown)
                {
                    // x was never updated this cycle -- report the TRUE fresh residual at the
                    // returned x, not the stale (and here misleadingly small) Arnoldi/Givens estimate.
                    A.Apply(in x, ref w);
                    fProxy rr2 = (fProxy)0;
                    for (int i = 0; i < n; i++) { fProxy d = b[i] - w[i]; rr2 += d * d; }
                    resnorm = math.sqrt(rr2);
                    break;
                }

                // x += (this-cycle basis)·y  [identity: V itself; flexible: the stored M⁻¹-applied Zv]
                // then the recycle correction x -= U·(Ru⁻¹ (B[:,:p]·y)) that keeps the C-component of
                // the new residual exactly zero (see the folder DEVLOG for the derivation).
                for (int i = 0; i < p; i++)
                {
                    fProxyN bi = flexible ? Zv[i] : V[i];
                    x.addScaledInPlace(y[i], bi);
                }

                if (kcur > 0 && p > 0)
                {
                    for (int i = 0; i < kcur; i++)
                    {
                        fProxy s = (fProxy)0;
                        for (int j = 0; j < p; j++) s += Bmat[i, j] * y[j];
                        ctr[i] = s;
                    }
                    fProxy ruMaxAbs = (fProxy)0;
                    for (int i = 0; i < kcur; i++) ruMaxAbs = math.max(ruMaxAbs, math.abs(Ru[i, i]));
                    fProxy ruPivotGuard = Consts.fProxyEpsilon * (fProxy)100 * ruMaxAbs;
                    for (int i = kcur - 1; i >= 0; i--)
                    {
                        fProxy s = ctr[i];
                        for (int l = i + 1; l < kcur; l++) s -= Ru[i, l] * zproj[l];
                        fProxy diag = Ru[i, i];
                        zproj[i] = math.abs(diag) > ruPivotGuard ? s / diag : (fProxy)0;
                    }
                    for (int i = 0; i < kcur; i++) x.addScaledInPlace(-zproj[i], U[i]);
                }

                if (converged)
                {
                    // Verify-at-exit: |g[j+1]| only measures the C-orthogonal (recycle-projected)
                    // residual, and a clamped Ru back-solve above can leave the C-component
                    // un-cancelled -- recompute the TRUE residual at the just-updated x. V[0] and w
                    // are both about to be fully overwritten at the top of the next cycle regardless,
                    // so they're free scratch here; a failed verify falls through to the normal
                    // not-converged path below (including the deflation update).
                    fProxyN v0v = V[0];
                    fProxy trueRR = VerifyTrueResidual(in A, in b, in x, ref w, ref v0v);
                    resnorm = math.sqrt(trueRR);
                    converged = resnorm <= thresh;
                }

                // ---- deflation update: rebuild the recycled subspace from the combined
                // (old-recycle + this-cycle Krylov) space via a small dense harmonic-Ritz eigenproblem.
                // Skipped (old U/C/Ru kept as-is) on convergence, on a degenerate zero-step cycle, or
                // whenever a numerical guard below trips -- recycling is an accelerator, not a
                // correctness requirement, so any failure here degrades gracefully.
                if (recycling && !converged && p > 0)
                {
                    int d = kcur + p;
                    int kcurAtEntry = kcur;
                    bool hadOldU = kcurAtEntry > 0;

                    UnsafeList<fProxyN> AU = default;
                    if (hadOldU)
                    {
                        AU = new UnsafeList<fProxyN>(kcurAtEntry, Allocator.Temp);
                        for (int i = 0; i < kcurAtEntry; i++)
                        {
                            var col = new fProxyN(n);
                            for (int l = 0; l < kcurAtEntry; l++) col.addScaledInPlace(Ru[l, i], C[l]);
                            AU.Add(col);
                        }
                    }

                    var Fmat = new fProxyMxN(d, d, Allocator.Temp, false);
                    var Gmat = new fProxyMxN(d, d, Allocator.Temp, false);
                    var Pgram = new fProxyMxN(d, d, Allocator.Temp, false);

                    for (int ai = 0; ai < d; ai++)
                    {
                        fProxyN APa = ai < kcurAtEntry ? AU[ai] : AV[ai - kcurAtEntry];
                        fProxyN Pa = ai < kcurAtEntry ? U[ai] : (flexible ? Zv[ai - kcurAtEntry] : V[ai - kcurAtEntry]);

                        for (int bi = ai; bi < d; bi++)
                        {
                            fProxyN APbv = bi < kcurAtEntry ? AU[bi] : AV[bi - kcurAtEntry];
                            fProxyN Pbv = bi < kcurAtEntry ? U[bi] : (flexible ? Zv[bi - kcurAtEntry] : V[bi - kcurAtEntry]);

                            fProxy fval = Blas.dot(APa, APbv);
                            Fmat[ai, bi] = fval; Fmat[bi, ai] = fval;

                            fProxy pval = Blas.dot(Pa, Pbv);
                            Pgram[ai, bi] = pval; Pgram[bi, ai] = pval;
                        }

                        for (int bi = 0; bi < d; bi++)
                        {
                            fProxyN Pbv = bi < kcurAtEntry ? U[bi] : (flexible ? Zv[bi - kcurAtEntry] : V[bi - kcurAtEntry]);
                            Gmat[ai, bi] = Blas.dot(APa, Pbv);
                        }
                    }

                    var GmatLU = new fProxyMxN(Gmat, Allocator.Temp);
                    var Xsol = new fProxyMxN(Fmat, Allocator.Temp);
                    var piv = new Pivot(d, Allocator.Temp);
                    var luInfo = LU.solveInPlace(ref GmatLU, ref piv, ref Xsol);
                    piv.Dispose();
                    GmatLU.Dispose();

                    var evReal = new fProxyN(d);
                    var evImag = new fProxyN(d);
                    bool haveEig = false;
                    if (luInfo.Solved)
                    {
                        var eiInfo = Eigen.valuesQRInPlace(ref Xsol, ref evReal, ref evImag);
                        haveEig = eiInfo.status == IterativeSolveStatus.Converged;
                    }

                    fProxy huge = (fProxy)1e30;
                    var keys = new fProxyN(d);
                    for (int i = 0; i < d; i++)
                    {
                        if (!haveEig) { keys[i] = huge; continue; }
                        fProxy re = evReal[i], im = evImag[i];
                        fProxy imagTol = Consts.fProxyZeroThreshold * (math.abs(re) + (fProxy)1);
                        keys[i] = math.abs(im) <= imagTol ? math.abs(re) : huge;
                    }

                    int target = math.min(recycle, d);
                    var selIdx = new UnsafeList<int>(math.max(target, 1), Allocator.Temp);
                    for (int sIt = 0; sIt < target; sIt++)
                    {
                        int best = -1;
                        fProxy bestKey = huge;
                        for (int i = 0; i < d; i++)
                            if (keys[i] < bestKey) { bestKey = keys[i]; best = i; }
                        if (best < 0 || bestKey >= huge) break;
                        selIdx.Add(best);
                        keys[best] = huge;
                    }
                    int kNew = selIdx.Length;

                    int allocK = math.max(kNew, 1);
                    var Zsel = new fProxyMxN(d, allocK, Allocator.Temp, false);
                    var Ntheta = new fProxyMxN(d, d, Allocator.Temp, false);
                    var evTmp = new fProxyN(d);
                    var Vtmp = new fProxyMxN(d, d, Allocator.Temp, false);

                    bool eigOk = kNew > 0;
                    for (int sIt = 0; sIt < kNew && eigOk; sIt++)
                    {
                        int idx = selIdx[sIt];
                        fProxy th = evReal[idx];
                        for (int ai = 0; ai < d; ai++)
                            for (int bi = 0; bi < d; bi++)
                                Ntheta[ai, bi] = Fmat[ai, bi] - th * (Gmat[ai, bi] + Gmat[bi, ai]) + th * th * Pgram[ai, bi];

                        var symInfo = Eigen.symmetricInPlace(ref Ntheta, ref evTmp, ref Vtmp);
                        if (symInfo.status != IterativeSolveStatus.Converged) { eigOk = false; break; }
                        for (int r = 0; r < d; r++) Zsel[r, sIt] = Vtmp[r, d - 1];
                    }

                    if (eigOk && kNew > 0)
                    {
                        var Unew = new UnsafeList<fProxyN>(kNew, Allocator.Temp);
                        var AUraw = new fProxyMxN(n, kNew, Allocator.Temp, false);

                        for (int sIt = 0; sIt < kNew; sIt++)
                        {
                            var ucol = new fProxyN(n);
                            var aucol = new fProxyN(n);
                            for (int l = 0; l < d; l++)
                            {
                                fProxy zl = Zsel[l, sIt];
                                fProxyN Pl = l < kcurAtEntry ? U[l] : (flexible ? Zv[l - kcurAtEntry] : V[l - kcurAtEntry]);
                                fProxyN APl = l < kcurAtEntry ? AU[l] : AV[l - kcurAtEntry];
                                ucol.addScaledInPlace(zl, Pl);
                                aucol.addScaledInPlace(zl, APl);
                            }
                            Unew.Add(ucol);
                            for (int r = 0; r < n; r++) AUraw[r, sIt] = aucol[r];
                            aucol.Dispose();
                        }

                        var Rnew = new fProxyMxN(kNew, kNew, Allocator.Temp, false);
                        QR.decompInPlace(ref AUraw, ref Rnew);

                        fProxy rGuard = Consts.fProxyEpsilon * (fProxy)100 * (math.abs(Rnew[0, 0]) + (fProxy)1);
                        int kSafe = 0;
                        while (kSafe < kNew && math.abs(Rnew[kSafe, kSafe]) > rGuard) kSafe++;

                        if (kSafe > 0)
                        {
                            for (int i = 0; i < kSafe; i++)
                            {
                                U[i].CopyFrom(Unew[i]);
                                for (int r = 0; r < n; r++) C[i][r] = AUraw[r, i];
                                for (int l = 0; l < kSafe; l++) Ru[i, l] = i <= l ? Rnew[i, l] : (fProxy)0;
                            }
                            kcur = kSafe;
                        }

                        Rnew.Dispose();
                        AUraw.Dispose();
                        for (int sIt = 0; sIt < kNew; sIt++) Unew[sIt].Dispose();
                        Unew.Dispose();
                    }

                    Vtmp.Dispose(); evTmp.Dispose(); Ntheta.Dispose(); Zsel.Dispose();
                    selIdx.Dispose(); keys.Dispose();
                    evImag.Dispose(); evReal.Dispose();
                    Xsol.Dispose(); Pgram.Dispose(); Gmat.Dispose(); Fmat.Dispose();
                    if (hadOldU)
                    {
                        for (int i = 0; i < kcurAtEntry; i++) AU[i].Dispose();
                        AU.Dispose();
                    }
                }
            }

            for (int i = 0; i <= m; i++) V[i].Dispose();
            V.Dispose();
            if (flexible) { for (int i = 0; i < m; i++) Zv[i].Dispose(); Zv.Dispose(); }
            if (recycling)
            {
                for (int i = 0; i < m; i++) AV[i].Dispose();
                AV.Dispose();
                for (int i = 0; i < recycle; i++) { U[i].Dispose(); C[i].Dispose(); }
                U.Dispose(); C.Dispose();
                Ru.Dispose(); Bmat.Dispose(); ctr.Dispose(); zproj.Dispose();
            }
            H.Dispose(); cs.Dispose(); sn.Dispose(); g.Dispose(); y.Dispose(); w.Dispose();

            var status = breakdown ? IterativeSolveStatus.Breakdown
                       : converged ? IterativeSolveStatus.Converged
                       : IterativeSolveStatus.MaxIterations;
            return MakeSolveInfo(status, total, resnorm);
        }

        /// <summary>
        /// Unpreconditioned GCRO-DR -- forwards into the merged
        /// <see cref="gcrodr{TOp, TPre}(in TOp, in TPre, in fProxyN, ref fProxyN, int, int, int, fProxy)"/>
        /// with the identity preconditioner.
        /// </summary>
        public static SolveInfo gcrodr<TOp>(in TOp A, in fProxyN b, ref fProxyN x, int restart, int recycle, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            return gcrodr(in A, default(fProxyIdentityPreconditioner), in b, ref x, restart, recycle, maxIter, tol);
        }

        static int fProxyGcrodrDefaultRecycle(int restart) => math.min(10, math.max(0, restart - 1));

        /// <summary>GCRO-DR over a dense <see cref="fProxyMxN"/>. Forwards via fProxyDenseOperator.</summary>
        public static SolveInfo gcrodr(in fProxyMxN A, in fProxyN b, ref fProxyN x, int restart, int recycle, int maxIter, fProxy tol)
            => gcrodr(new fProxyDenseOperator(in A), in b, ref x, restart, recycle, maxIter, tol);

        /// <summary>GCRO-DR over a dense matrix with defaults (restart = min(30, N), recycle = min(10, restart-1), maxIter = N, tol = sqrtEps).</summary>
        public static SolveInfo gcrodr(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            int r = math.min(30, A.M_Rows);
            return gcrodr(new fProxyDenseOperator(in A), in b, ref x, r, fProxyGcrodrDefaultRecycle(r), A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>GCRO-DR over a block-sparse (BSR) matrix. Forwards via fProxyBSROperator.</summary>
        public static SolveInfo gcrodr(in fProxyBSR A, in fProxyN b, ref fProxyN x, int restart, int recycle, int maxIter, fProxy tol)
            => gcrodr(new fProxyBSROperator(in A), in b, ref x, restart, recycle, maxIter, tol);

        /// <summary>GCRO-DR over a BSR matrix with defaults (restart = min(30, N), recycle = min(10, restart-1), maxIter = N, tol = sqrtEps).</summary>
        public static SolveInfo gcrodr(in fProxyBSR A, in fProxyN b, ref fProxyN x)
        {
            int r = math.min(30, A.M_Rows);
            return gcrodr(new fProxyBSROperator(in A), in b, ref x, r, fProxyGcrodrDefaultRecycle(r), A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>Right-preconditioned GCRO-DR over a BSR matrix with ANY <see cref="IfProxyPreconditioner"/> (ILU0).</summary>
        public static SolveInfo gcrodr<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x, int restart, int recycle, int maxIter, fProxy tol)
            where TPre : struct, IfProxyPreconditioner
            => gcrodr(new fProxyBSROperator(in A), in M, in b, ref x, restart, recycle, maxIter, tol);

        /// <summary>Right-preconditioned GCRO-DR over a BSR matrix with ANY <see cref="IfProxyPreconditioner"/> (ILU0), with defaults (restart = min(30, N)).</summary>
        public static SolveInfo gcrodr<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x)
            where TPre : struct, IfProxyPreconditioner
        {
            int r = math.min(30, A.M_Rows);
            return gcrodr(new fProxyBSROperator(in A), in M, in b, ref x, r, fProxyGcrodrDefaultRecycle(r), A.M_Rows, Consts.fProxySqrtEps);
        }
    }
}
