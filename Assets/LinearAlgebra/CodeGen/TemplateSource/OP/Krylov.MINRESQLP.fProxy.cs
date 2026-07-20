using System;
using Unity.Mathematics;
using LinearAlgebra.Internal;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

        /// <summary>
        /// MINRES-QLP (Choi, Paige &amp; Saunders 2011) for symmetric (possibly indefinite,
        /// singular, or ill-conditioned) systems A x = b, generic over BOTH the operator
        /// (<see cref="IfProxyLinearOperator"/>) and the preconditioner (<see
        /// cref="IfProxyPreconditioner"/>). Structural sibling of <see cref="minres{TOp,TPre}"/>:
        /// same symmetric-Lanczos base and left-Givens (MINRES) reflection, plus a second,
        /// right-Givens (QLP) reflection that regularizes the near-singular tail of the
        /// tridiagonal system. Where plain MINRES can diverge or stall on a singular/rank-deficient
        /// A, this returns the MINIMUM-LENGTH solution of the compatible system, or of the
        /// associated least-squares problem when A x = b is incompatible.
        ///
        /// A MUST be symmetric; a real preconditioner M MUST be SPD (caller precondition, not
        /// verified beyond the NaN-safe breakdown guards). x is a warm-startable initial guess,
        /// overwritten with the solution. tol is the relative-residual tolerance (reference's
        /// RTOL); maxIter bounds the Lanczos step count.
        ///
        /// Caller provides x and nine scratch vectors (v, r1, r2, r3, w, wl, wl2, xl2, t1; all
        /// length A.Rows) -- see the folder DEVLOG for the buffer-reuse/aliasing plan. Returns a
        /// <see cref="SolveInfo"/>: Converged for every reference outcome that reports a usable x
        /// (compatible solve, min-length least-squares solve, or lucky eigenvector solve),
        /// MaxIterations when the step budget was exhausted, Breakdown when xnorm/Acond exceeded
        /// their safety bounds, the recurrence went singular, or a non-SPD M was detected -- see
        /// the folder DEVLOG for the full flag-to-status mapping and the reference's own flag
        /// semantics.
        /// </summary>
        public static SolveInfo minresQLP<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x,
                                       ref fProxyN v, ref fProxyN r1, ref fProxyN r2, ref fProxyN r3,
                                       ref fProxyN w, ref fProxyN wl, ref fProxyN wl2, ref fProxyN xl2,
                                       ref fProxyN t1, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("minresQLP: A must be square");

            if (b.N != A.Rows) throw new ArgumentException("minresQLP: b.N must equal A.Rows");
            if (x.N != A.Rows) throw new ArgumentException("minresQLP: x.N must equal A.Rows");
            if (v.N != A.Rows) throw new ArgumentException("minresQLP: v.N must equal A.Rows");
            if (r1.N != A.Rows) throw new ArgumentException("minresQLP: r1.N must equal A.Rows");
            if (r2.N != A.Rows) throw new ArgumentException("minresQLP: r2.N must equal A.Rows");
            if (r3.N != A.Rows) throw new ArgumentException("minresQLP: r3.N must equal A.Rows");
            if (w.N != A.Rows) throw new ArgumentException("minresQLP: w.N must equal A.Rows");
            if (wl.N != A.Rows) throw new ArgumentException("minresQLP: wl.N must equal A.Rows");
            if (wl2.N != A.Rows) throw new ArgumentException("minresQLP: wl2.N must equal A.Rows");
            if (xl2.N != A.Rows) throw new ArgumentException("minresQLP: xl2.N must equal A.Rows");
            if (t1.N != A.Rows) throw new ArgumentException("minresQLP: t1.N must equal A.Rows");

            if (maxIter < 1)
                throw new ArgumentException("minresQLP: maxIter must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[11];
                ptrs[0] = (long)v.Data.Ptr;   ptrs[1] = (long)r1.Data.Ptr;  ptrs[2] = (long)r2.Data.Ptr;
                ptrs[3] = (long)r3.Data.Ptr;  ptrs[4] = (long)w.Data.Ptr;   ptrs[5] = (long)wl.Data.Ptr;
                ptrs[6] = (long)wl2.Data.Ptr; ptrs[7] = (long)xl2.Data.Ptr; ptrs[8] = (long)t1.Data.Ptr;
                ptrs[9] = (long)x.Data.Ptr;   ptrs[10] = (long)b.Data.Ptr;
                RequireDistinctBuffers("minresQLP: v/r1/r2/r3/w/wl/wl2/xl2/t1/x/b must be distinct", ptrs, 11);
            }

            fProxy bb = Blas.dot(b, b);
            if (bb == (fProxy)0)
            {
                x.CopyFrom(in b);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }

            // r2 = r0 = b - A x (warm start: r1 is temp scratch here, overwritten before it is
            // ever read as Lanczos history).
            A.Apply(in x, ref r1);
            r2.CopyFrom(in b);
            r2.addScaledInPlace((fProxy)(-1), r1);

            fProxy beta1;
            if (M.IsIdentity)
            {
                beta1 = math.sqrt(Blas.dot(r2, r2));
            }
            else
            {
                M.Apply(in r2, ref r3);
                fProxy beta1Sq = Blas.dot(r2, r3);
                if (!(beta1Sq > (fProxy)0))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, 0, math.sqrt(Blas.dot(r2, r2)));
                beta1 = math.sqrt(beta1Sq);
            }
            if (beta1 == (fProxy)0)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);

            fProxy eps = Consts.fProxyEpsilon;
            fProxy tiny = eps * eps;
            fProxy maxxnorm = (fProxy)1e7;
            fProxy Acondlim = (fProxy)1e15;
            fProxy TranCond = (fProxy)1e7;

            const int flag0 = -2;
            int flag = flag0;
            int QLPiter = 0;
            fProxy beta = (fProxy)0;
            fProxy betan = beta1;
            fProxy tau = (fProxy)0, taul = (fProxy)0;
            fProxy phi = beta1;
            fProxy cs = (fProxy)(-1), sn = (fProxy)0;
            fProxy cr1 = (fProxy)(-1), sr1 = (fProxy)0;
            fProxy cr2 = (fProxy)(-1), sr2 = (fProxy)0;
            fProxy dltan = (fProxy)0, eplnn = (fProxy)0;
            fProxy gama = (fProxy)0, gamal = (fProxy)0, gamal2 = (fProxy)0;
            fProxy eta = (fProxy)0, etal = (fProxy)0, etal2 = (fProxy)0;
            fProxy vepln = (fProxy)0, veplnl = (fProxy)0, veplnl2 = (fProxy)0;
            fProxy ul3 = (fProxy)0, ul2 = (fProxy)0, ul = (fProxy)0, u = (fProxy)0;
            fProxy gmin = (fProxy)0, gminl = (fProxy)0, gminl2 = (fProxy)0;
            fProxy rnorm = beta1;
            fProxy xnorm = (fProxy)0, xl2norm = (fProxy)0, Axnorm = (fProxy)0;
            fProxy Anorm = (fProxy)0, Acond = (fProxy)1;
            fProxy relres = (fProxy)1;
            fProxy gamalQLP = (fProxy)0, veplnQLP = (fProxy)0, gamaQLP = (fProxy)0, ulQLP = (fProxy)0, uQLP = (fProxy)0;

            // 3-term search-direction history and the lagged solution accumulator start at 0.
            w.zeroInPlace(); wl.zeroInPlace(); wl2.zeroInPlace(); xl2.zeroInPlace();

            int iters;
            for (iters = 1; iters <= maxIter; iters++)
            {
                // ---- symmetric Lanczos step ----
                fProxy betal = beta;
                beta = betan;

                if (M.IsIdentity) Blas.scaledCopy(1 / beta, r2, ref v);
                else              Blas.scaledCopy(1 / beta, r3, ref v);

                A.Apply(in v, ref r3);
                if (iters > 1) r3.addScaledInPlace(-(beta / betal), r1);
                fProxy alfa = Blas.dot(r3, v);
                r3.addScaledInPlace(-(alfa / beta), r2);

                { fProxyN tmp = r1; r1 = r2; r2 = r3; r3 = tmp; }

                fProxy betanSq;
                if (M.IsIdentity)
                {
                    betanSq = Blas.dot(r2, r2);
                }
                else
                {
                    M.Apply(in r2, ref r3);
                    betanSq = Blas.dot(r2, r3);
                    if (!(betanSq >= (fProxy)0)) { flag = -3; break; }
                }
                betan = math.sqrt(betanSq);

                if (iters == 1 && betan == (fProxy)0)
                {
                    // v is an exact eigenvector of A (Av = alfa v): either r0 has no component in
                    // range(A) (x already optimal, flag 0) or the correction r0/alfa solves exactly
                    // (flag -1).
                    if (alfa == (fProxy)0) flag = 0;
                    else { x.addScaledInPlace(1 / alfa, r1); flag = -1; }
                    break;
                }

                fProxy pnorm = math.sqrt(betal * betal + alfa * alfa + betan * betan);

                // ---- previous left rotation Q_{k-1} ----
                fProxy dbar = dltan;
                fProxy dlta = cs * dbar + sn * alfa;
                fProxy epln = eplnn;
                fProxy gbar = sn * dbar - cs * alfa;
                eplnn = sn * betan;
                dltan = -cs * betan;
                fProxy dltaQLP = dlta;

                // ---- current left plane rotation Q_k ----
                fProxy gamal3 = gamal2;
                gamal2 = gamal;
                gamal = gama;
                SymGivens(gbar, betan, out cs, out sn, out gama);
                fProxy gamaTmp = gama;
                fProxy taul2 = taul;
                taul = tau;
                tau = cs * phi;
                Axnorm = math.sqrt(Axnorm * Axnorm + tau * tau);
                phi = sn * phi;

                // ---- previous right plane rotation P_{k-2,k} ----
                if (iters > 2)
                {
                    veplnl2 = veplnl;
                    etal2 = etal;
                    etal = eta;
                    fProxy dltaTmp = sr2 * vepln - cr2 * dlta;
                    veplnl = cr2 * vepln + sr2 * dlta;
                    dlta = dltaTmp;
                    eta = sr2 * gama;
                    gama = -cr2 * gama;
                }

                // ---- current right plane rotation P_{k-1,k} ----
                if (iters > 1)
                {
                    SymGivens(gamal, dlta, out cr1, out sr1, out gamal);
                    vepln = sr1 * gama;
                    gama = -cr1 * gama;
                }

                // ---- xnorm update / maxxnorm-and-degeneracy guard ----
                fProxy ul4 = ul3;
                ul3 = ul2;
                if (iters > 2) ul2 = (taul2 - etal2 * ul4 - veplnl2 * ul3) / gamal2;
                if (iters > 1) ul = (taul - etal * ul3 - veplnl * ul2) / gamal;
                fProxy xnormTmp = math.sqrt(xl2norm * xl2norm + ul2 * ul2 + ul * ul);
                if (math.abs(gama) > tiny && xnormTmp < maxxnorm)
                {
                    u = (tau - eta * ul2 - vepln * ul) / gama;
                    if (math.sqrt(xnormTmp * xnormTmp + u * u) > maxxnorm) { u = (fProxy)0; flag = 6; }
                }
                else
                {
                    u = (fProxy)0;
                    flag = 9;
                }
                xl2norm = math.sqrt(xl2norm * xl2norm + ul2 * ul2);
                xnorm = math.sqrt(xl2norm * xl2norm + ul * ul + u * u);

                // ---- update w & x: conventional MINRES step, or the MINRES-QLP right-reflection ----
                if (Acond < TranCond && flag != flag0 && QLPiter == 0)
                {
                    fProxyN tmp = wl2; wl2 = wl; wl = w; w = tmp;
                    Blas.combine3(ref w, v, -epln, wl2, -dltaQLP, wl, 1 / gamaTmp);
                    if (xnorm < maxxnorm) x.addScaledInPlace(tau, w);
                    else flag = 6;
                }
                else
                {
                    QLPiter += 1;
                    if (QLPiter == 1)
                    {
                        xl2.zeroInPlace();
                        if (iters > 1)
                        {
                            if (iters > 3)
                            {
                                Blas.scaledCopy(gamal3, wl2, ref t1);
                                t1.addScaledInPlace(veplnl2, wl);
                                t1.addScaledInPlace(etal, w);
                                { fProxyN buf = wl2; wl2 = t1; t1 = buf; }
                            }
                            if (iters > 2)
                            {
                                Blas.scaledCopy(gamalQLP, wl, ref t1);
                                t1.addScaledInPlace(veplnQLP, w);
                                { fProxyN buf = wl; wl = t1; t1 = buf; }
                            }
                            w.mulInPlace(gamaQLP);
                            xl2.CopyFrom(in x);
                            xl2.addScaledInPlace(-ulQLP, wl);
                            xl2.addScaledInPlace(-uQLP, w);
                        }
                    }

                    if (iters == 1)
                    {
                        fProxyN buf = wl2;
                        wl2 = wl;
                        Blas.scaledCopy(sr1, v, ref buf);
                        wl = buf;
                        Blas.scaledCopy(-cr1, v, ref w);
                    }
                    else if (iters == 2)
                    {
                        fProxyN buf = wl2;
                        wl2 = wl;
                        Blas.scaledCopy(cr1, w, ref buf);
                        buf.addScaledInPlace(sr1, v);
                        w.mulInPlace(sr1);
                        w.addScaledInPlace(-cr1, v);
                        wl = buf;
                    }
                    else
                    {
                        fProxyN wl2Free = wl2;
                        Blas.scaledCopy(cr2, wl, ref wl2Free);
                        wl2Free.addScaledInPlace(sr2, v);
                        v.mulInPlace(-cr2);
                        v.addScaledInPlace(sr2, wl);
                        Blas.scaledCopy(cr1, w, ref t1);
                        t1.addScaledInPlace(sr1, v);
                        w.mulInPlace(sr1);
                        w.addScaledInPlace(-cr1, v);
                        fProxyN wlFree = wl;
                        wl2 = wl2Free;
                        wl = t1;
                        t1 = wlFree;
                    }

                    xl2.addScaledInPlace(ul2, wl2);
                    x.CopyFrom(in xl2);
                    x.addScaledInPlace(ul, wl);
                    x.addScaledInPlace(u, w);
                }

                // ---- next right plane rotation P_{k-1,k+1}; snapshot for the next QLP transition ----
                fProxy gamalTmp2 = gamal;
                SymGivens(gamal, eplnn, out cr2, out sr2, out gamal);
                gamalQLP = gamalTmp2;
                veplnQLP = vepln;
                gamaQLP = gama;
                ulQLP = ul;
                uQLP = u;

                // ---- norm / condition estimates ----
                fProxy absGama = math.abs(gama);
                Anorm = math.max(math.max(Anorm, pnorm), math.max(gamal, absGama));
                if (iters == 1)
                {
                    gmin = gama;
                    gminl = gmin;
                }
                else
                {
                    gminl2 = gminl;
                    gminl = gmin;
                    gmin = math.min(math.min(gminl2, gamal), absGama);
                }
                fProxy Acondl = Acond;
                Acond = Anorm / gmin;
                fProxy rnorml = rnorm;
                fProxy relresl = relres;
                if (flag != 9) rnorm = phi;
                relres = rnorm / (Anorm * xnorm + beta1);
                fProxy rootl = math.sqrt(gbar * gbar + dltan * dltan);
                fProxy relAresl = rootl / Anorm;

                // ---- stopping tests (relative residual, ‖Ar‖, xnorm/Acond safety bounds) ----
                fProxy epsx = Anorm * xnorm * eps;
                if (flag == flag0 || flag == 9)
                {
                    fProxy chk1 = 1 + relres;
                    fProxy chk2 = 1 + relAresl;
                    if (iters >= maxIter) flag = 8;
                    if (Acond >= Acondlim) flag = 7;
                    if (xnorm >= maxxnorm) flag = 6;
                    if (epsx >= beta1) flag = 5;
                    if (chk2 <= 1) flag = 4;
                    if (chk1 <= 1) flag = 3;
                    if (relAresl <= tol) flag = 2;
                    if (relres <= tol) flag = 1;
                }

                // flag in {2,4,6,7}: this step's boundary trip is not yet trustworthy -- report the
                // pre-step diagnostics (x/w/wl/wl2 themselves are NOT rolled back).
                if (flag == 2 || flag == 4 || flag == 6 || flag == 7)
                {
                    iters -= 1;
                    Acond = Acondl;
                    rnorm = rnorml;
                    relres = relresl;
                }

                if (flag != flag0) break;
            }

            if (flag == flag0) { flag = 8; iters = maxIter; }

            // Final true residual (fresh, regardless of exit reason -- r3/r1 are idle by now).
            fProxy finalRnorm = math.sqrt(VerifyTrueResidual(in A, in b, in x, ref r3, ref r1));

            IterativeSolveStatus status;
            if (flag == 8) status = IterativeSolveStatus.MaxIterations;
            else if (flag == 6 || flag == 7 || flag == 9 || flag == -3) status = IterativeSolveStatus.Breakdown;
            else status = IterativeSolveStatus.Converged;

            // Honesty guard: a QLP-flagged convergence whose FRESH true residual is not
            // small on the RAW relative scale ‖b-Ax‖/‖b‖ is downgraded to an honest
            // MaxIterations, never a false Converged. The flag metric divides by
            // (Anorm*xnorm + beta1), which a near-breakdown's large Anorm*xnorm can inflate
            // enough to mask a big true residual; beta1 (= ‖b‖) is the un-inflated scale.
            // Factor is generous so genuine convergence (a few x the QLP metric) is kept.
            if (status == IterativeSolveStatus.Converged &&
                finalRnorm > (fProxy)64 * tol * beta1)
            {
                status = IterativeSolveStatus.MaxIterations;
            }

            return MakeSolveInfo(status, iters, finalRnorm);
        }

        /// <summary>
        /// Stable symmetric Givens rotation (Golub &amp; Van Loan): returns c, s, r such that
        /// [[c, s], [-s, c]] · [a, b]ᵀ = [r, 0]ᵀ, with r = ±‖(a,b)‖ chosen for cancellation safety.
        /// Zero -> +1 sign convention (matches <see cref="Helpers.signOrOne"/>).
        /// </summary>
        static void SymGivens(fProxy a, fProxy b, out fProxy c, out fProxy s, out fProxy r)
        {
            if (b == (fProxy)0)
            {
                c = Helpers.signOrOne(a);
                s = (fProxy)0;
                r = math.abs(a);
            }
            else if (a == (fProxy)0)
            {
                c = (fProxy)0;
                s = Helpers.signOrOne(b);
                r = math.abs(b);
            }
            else if (math.abs(b) > math.abs(a))
            {
                fProxy t = a / b;
                s = Helpers.signOrOne(b) / math.sqrt((fProxy)1 + t * t);
                c = s * t;
                r = b / s;
            }
            else
            {
                fProxy t = b / a;
                c = Helpers.signOrOne(a) / math.sqrt((fProxy)1 + t * t);
                s = c * t;
                r = a / c;
            }
        }

        /// <summary>
        /// Unpreconditioned MINRES-QLP -- forwards into the merged
        /// <see cref="minresQLP{TOp, TPre}(in TOp, in TPre, in fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, int, fProxy)"/>
        /// with the identity preconditioner.
        /// </summary>
        public static SolveInfo minresQLP<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                       ref fProxyN v, ref fProxyN r1, ref fProxyN r2, ref fProxyN r3,
                                       ref fProxyN w, ref fProxyN wl, ref fProxyN wl2, ref fProxyN xl2,
                                       ref fProxyN t1, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            return minresQLP(in A, default(fProxyIdentityPreconditioner), in b, ref x, ref v, ref r1, ref r2, ref r3, ref w, ref wl, ref wl2, ref xl2, ref t1, maxIter, tol);
        }

        /// <summary>
        /// MINRES-QLP over a dense <see cref="fProxyMxN"/> -- zero-alloc primitive. Forwards into
        /// <see cref="minresQLP{TOp}"/> via <see cref="fProxyDenseOperator"/>.
        /// </summary>
        public static SolveInfo minresQLP(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                  ref fProxyN v, ref fProxyN r1, ref fProxyN r2, ref fProxyN r3,
                                  ref fProxyN w, ref fProxyN wl, ref fProxyN wl2, ref fProxyN xl2,
                                  ref fProxyN t1, int maxIter, fProxy tol)
        {
            return minresQLP(new fProxyDenseOperator(in A), in b, ref x, ref v, ref r1, ref r2, ref r3, ref w, ref wl, ref wl2, ref xl2, ref t1, maxIter, tol);
        }

        /// <summary>MINRES-QLP over a dense matrix -- allocates nine scratch vectors from the arena.</summary>
        public static SolveInfo minresQLP(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN v   = b.fProxyTempVec(A.M_Rows);
            fProxyN r1  = b.fProxyTempVec(A.M_Rows);
            fProxyN r2  = b.fProxyTempVec(A.M_Rows);
            fProxyN r3  = b.fProxyTempVec(A.M_Rows);
            fProxyN w   = b.fProxyTempVec(A.M_Rows);
            fProxyN wl  = b.fProxyTempVec(A.M_Rows);
            fProxyN wl2 = b.fProxyTempVec(A.M_Rows);
            fProxyN xl2 = b.fProxyTempVec(A.M_Rows);
            fProxyN t1  = b.fProxyTempVec(A.M_Rows);
            return minresQLP(in A, in b, ref x, ref v, ref r1, ref r2, ref r3, ref w, ref wl, ref wl2, ref xl2, ref t1, maxIter, tol);
        }

        /// <summary>MINRES-QLP over a dense matrix with default maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).</summary>
        public static SolveInfo minresQLP(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return minresQLP(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// MINRES-QLP over a symmetric block-sparse (BSR) matrix -- zero-alloc primitive. Forwards
        /// into <see cref="minresQLP{TOp}"/> via <c>fProxyBSROperator</c>.
        /// </summary>
        public static SolveInfo minresQLP(in fProxyBSR A, in fProxyN b, ref fProxyN x,
                                  ref fProxyN v, ref fProxyN r1, ref fProxyN r2, ref fProxyN r3,
                                  ref fProxyN w, ref fProxyN wl, ref fProxyN wl2, ref fProxyN xl2,
                                  ref fProxyN t1, int maxIter, fProxy tol)
        {
            return minresQLP(new fProxyBSROperator(in A), in b, ref x, ref v, ref r1, ref r2, ref r3, ref w, ref wl, ref wl2, ref xl2, ref t1, maxIter, tol);
        }

        /// <summary>MINRES-QLP over a BSR matrix -- allocates nine scratch vectors from the arena.</summary>
        public static SolveInfo minresQLP(in fProxyBSR A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN v   = b.fProxyTempVec(A.M_Rows);
            fProxyN r1  = b.fProxyTempVec(A.M_Rows);
            fProxyN r2  = b.fProxyTempVec(A.M_Rows);
            fProxyN r3  = b.fProxyTempVec(A.M_Rows);
            fProxyN w   = b.fProxyTempVec(A.M_Rows);
            fProxyN wl  = b.fProxyTempVec(A.M_Rows);
            fProxyN wl2 = b.fProxyTempVec(A.M_Rows);
            fProxyN xl2 = b.fProxyTempVec(A.M_Rows);
            fProxyN t1  = b.fProxyTempVec(A.M_Rows);
            return minresQLP(in A, in b, ref x, ref v, ref r1, ref r2, ref r3, ref w, ref wl, ref wl2, ref xl2, ref t1, maxIter, tol);
        }

        /// <summary>MINRES-QLP over a BSR matrix with default maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).</summary>
        public static SolveInfo minresQLP(in fProxyBSR A, in fProxyN b, ref fProxyN x)
        {
            return minresQLP(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned MINRES-QLP solver -- allocates nine scratch vectors from the arena and
        /// calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo minresQLP<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x,
                                          int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            fProxyN v   = b.fProxyTempVec(A.Rows);
            fProxyN r1  = b.fProxyTempVec(A.Rows);
            fProxyN r2  = b.fProxyTempVec(A.Rows);
            fProxyN r3  = b.fProxyTempVec(A.Rows);
            fProxyN w   = b.fProxyTempVec(A.Rows);
            fProxyN wl  = b.fProxyTempVec(A.Rows);
            fProxyN wl2 = b.fProxyTempVec(A.Rows);
            fProxyN xl2 = b.fProxyTempVec(A.Rows);
            fProxyN t1  = b.fProxyTempVec(A.Rows);
            return minresQLP(in A, in M, in b, ref x, ref v, ref r1, ref r2, ref r3, ref w, ref wl, ref wl2, ref xl2, ref t1, maxIter, tol);
        }

        /// <summary>
        /// Preconditioned MINRES-QLP solver with default maxIter (A.Rows) and tol
        /// (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo minresQLP<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            return minresQLP(in A, in M, in b, ref x, A.Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned MINRES-QLP over a block-sparse (BSR) matrix with its matching
        /// block-Jacobi preconditioner. Forwards into <see cref="minresQLP{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c>.
        /// </summary>
        public static SolveInfo minresQLP(in fProxyBSR A, in fProxyBlockJacobi M, in fProxyN b, ref fProxyN x,
                               ref fProxyN v, ref fProxyN r1, ref fProxyN r2, ref fProxyN r3,
                               ref fProxyN w, ref fProxyN wl, ref fProxyN wl2, ref fProxyN xl2,
                               ref fProxyN t1, int maxIter, fProxy tol)
        {
            return minresQLP(new fProxyBSROperator(in A), in M, in b, ref x, ref v, ref r1, ref r2, ref r3, ref w, ref wl, ref wl2, ref xl2, ref t1, maxIter, tol);
        }

        /// <summary>
        /// Block-Jacobi Preconditioned MINRES-QLP over a BSR matrix -- allocates nine scratch
        /// vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo minresQLP(in fProxyBSR A, in fProxyBlockJacobi M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN v   = b.fProxyTempVec(A.M_Rows);
            fProxyN r1  = b.fProxyTempVec(A.M_Rows);
            fProxyN r2  = b.fProxyTempVec(A.M_Rows);
            fProxyN r3  = b.fProxyTempVec(A.M_Rows);
            fProxyN w   = b.fProxyTempVec(A.M_Rows);
            fProxyN wl  = b.fProxyTempVec(A.M_Rows);
            fProxyN wl2 = b.fProxyTempVec(A.M_Rows);
            fProxyN xl2 = b.fProxyTempVec(A.M_Rows);
            fProxyN t1  = b.fProxyTempVec(A.M_Rows);
            return minresQLP(in A, in M, in b, ref x, ref v, ref r1, ref r2, ref r3, ref w, ref wl, ref wl2, ref xl2, ref t1, maxIter, tol);
        }

        /// <summary>
        /// Block-Jacobi Preconditioned MINRES-QLP over a BSR matrix, with default maxIter
        /// (A.M_Rows) and tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo minresQLP(in fProxyBSR A, in fProxyBlockJacobi M, in fProxyN b, ref fProxyN x)
        {
            return minresQLP(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }
    }
}
