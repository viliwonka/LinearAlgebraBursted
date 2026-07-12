using System;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // ================================================================================================
    // Nonlinear least squares via Levenberg-Marquardt with Nielsen damping. Algorithm reference:
    // Madsen, Nielsen & Tingleff, "Methods for Non-Linear Least Squares Problems" (2nd ed., 2004),
    // Algorithm 3.16 -- the gain-ratio damping update (mu *= max(1/3, 1-(2*rho-1)^3) on accept,
    // mu *= nu, nu *= 2 on reject) and the convergence structure (the step-size test on the PROPOSED
    // step, before it is evaluated against the objective, so a reject/reject spiral trips SmallStep
    // instead of escalating mu without bound -- verified in the from-scratch numpy prototype against
    // a NIST StRD case that false-failed under a naive post-accept-only step check). The per-column
    // Marquardt scaling (D = diag of column norms of J, floored at its own running max) is the
    // well-known MINPACK convention, not transcribed from MINPACK source -- Math.NET Numerics'
    // LevenbergMarquardtMinimizer.cs (MIT) was read as an independent C# structural reference.
    // Robust-loss row rescaling (nlsApplyRobustScale) is scipy `optimize._lsq` (BSD-3):
    // least_squares.py's huber/cauchy loss definitions and common.py's scale_for_robust_loss_function,
    // verified line-by-line against the installed scipy source. Tukey biweight has no scipy
    // precedent (scipy ships no redescending loss); its Rho/RhoPrime/RhoPrime2 are the standard
    // robust-statistics biweight identities, derived independently (see Interfaces/DEVLOG.md).
    // ================================================================================================

    public static partial class Optimize
    {
        // ---- shared engine helpers (no TF/TLoss generics -- pure buffer arithmetic, reused by both
        // the numeric- and analytic-Jacobian cores) ---------------------------------------------------

        // D_j = max(running D_j, effective_j), effective_j = column-norm_j(J) for any column with a
        // genuine (however small) sensitivity, or the LARGEST real column norm this iteration for a
        // column at-or-below flatThresh -- see the file DEVLOG for why a plain per-type absolute
        // epsilon test (never scaled by the matrix's own magnitude) is required here, and why the
        // flat-column floor is the max REAL column norm rather than a small constant. d must be
        // pre-zeroed before the first call so that call also seeds D from J's own initial column
        // norms. colNorms is scratch sized to J.N_Cols (caches each column norm so the flat-floor
        // pass does not recompute it). Returns the max real (above-flatThresh) column norm seen this
        // call; 0 means every column is at-or-below flatThresh, so d was left unchanged by this call
        // -- on the very first call (d still all-zero) the caller must treat that as already
        // stationary rather than dividing by a zero scale.
        private static fProxy nlsUpdateScale(ref fProxyN d, in fProxyMxN J, fProxy flatThresh, ref fProxyN colNorms)
        {
            int m = J.M_Rows, n = J.N_Cols;

            fProxy maxRealColNorm = (fProxy)0;
            for (int j = 0; j < n; j++)
            {
                fProxy s = (fProxy)0;
                for (int i = 0; i < m; i++) { fProxy v = J[i, j]; s += v * v; }
                fProxy cn = math.sqrt(s);
                colNorms[j] = cn;
                if (cn > flatThresh && cn > maxRealColNorm) maxRealColNorm = cn;
            }

            for (int j = 0; j < n; j++)
            {
                fProxy cn = colNorms[j];
                fProxy effective = cn <= flatThresh ? maxRealColNorm : cn;
                if (effective > d[j]) d[j] = effective;
            }

            return maxRealColNorm;
        }

        private static fProxy nlsMaxD2(in fProxyN d)
        {
            fProxy best = (fProxy)0;
            for (int j = 0; j < d.N; j++)
            {
                fProxy v = d[j] * d[j];
                if (v > best) best = v;
            }
            return best;
        }

        // scipy optimize._lsq common.py `scale_for_robust_loss_function` (BSD-3): rescales (r, J) so
        // an ordinary (unweighted) damped least-squares step on the rescaled system reproduces one
        // IRLS step of the robust loss. J_scale is floored at Consts.fProxyEpsilon (not zero) -- a
        // residual whose rho'' term drives rho'+2*rho''*s negative (a redescending loss past its
        // own transition) must not divide by a non-positive number.
        private static void nlsApplyRobustScale<TLoss>(in fProxyN r, in fProxyMxN J, ref fProxyN rs, ref fProxyMxN Js, in TLoss loss)
            where TLoss : struct, IfProxyRobustLoss
        {
            int m = J.M_Rows, n = J.N_Cols;
            fProxy floorEps = Consts.fProxyEpsilon;
            for (int i = 0; i < m; i++)
            {
                fProxy ri = r[i];
                fProxy s = ri * ri;
                fProxy rp = loss.RhoPrime(s);
                fProxy rp2 = loss.RhoPrime2(s);
                fProxy jscale = math.sqrt(math.max(rp + (fProxy)2 * rp2 * s, floorEps));
                rs[i] = ri * (rp / jscale);
                for (int j = 0; j < n; j++)
                    Js[i, j] = J[i, j] * jscale;
            }
        }

        // Forward/central finite-difference Jacobian. Per-parameter relative step h_j =
        // sqrt(max(epsfcn, eps)) * max(|p_j|, 1) -- MINPACK's own numerical-differencing convention.
        private static void nlsNumericJacobian<TF>(ref TF f, in fProxyN p, in fProxyN r0, ref fProxyMxN J,
            NLSJacobianMode mode, fProxy epsfcn, ref fProxyN pPert, ref fProxyN rPert, ref fProxyN rPert2)
            where TF : struct, IfProxyResidualFunction
        {
            int n = p.N, m = r0.N;
            fProxy hEps = math.sqrt(math.max(epsfcn, Consts.fProxyEpsilon));

            for (int j = 0; j < n; j++)
            {
                fProxy pj = p[j];
                fProxy step = hEps * math.max(math.abs(pj), (fProxy)1);

                for (int k = 0; k < n; k++) pPert[k] = p[k];
                pPert[j] = pj + step;
                f.Residuals(in pPert, ref rPert);

                if (mode == NLSJacobianMode.Central)
                {
                    pPert[j] = pj - step;
                    f.Residuals(in pPert, ref rPert2);
                    fProxy inv = (fProxy)1 / ((fProxy)2 * step);
                    for (int i = 0; i < m; i++) J[i, j] = (rPert[i] - rPert2[i]) * inv;
                }
                else
                {
                    fProxy inv = (fProxy)1 / step;
                    for (int i = 0; i < m; i++) J[i, j] = (rPert[i] - r0[i]) * inv;
                }
            }
        }

        // Damped augmented least squares: [Js; sqrt(mu)*diag(d)] h = [-rs; 0], solved via QR.solveInPlace
        // (m+n x n, tall by construction -- full column rank as long as mu > 0, since d is floored
        // away from zero). Returns false only if h itself comes back non-finite (mu overflow).
        private static bool nlsSolveStep(in fProxyMxN Js, in fProxyN rs, in fProxyN d, fProxy mu,
            ref fProxyMxN Aaug, ref fProxyN baug, ref fProxyN h, ref fProxyN u, ref fProxyN w)
        {
            int m = Js.M_Rows, n = Js.N_Cols;

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    Aaug[i, j] = Js[i, j];

            fProxy sq = math.sqrt(mu);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                    Aaug[m + i, j] = (fProxy)0;
                Aaug[m + i, i] = sq * d[i];
            }

            for (int i = 0; i < m; i++) baug[i] = -rs[i];
            for (int i = 0; i < n; i++) baug[m + i] = (fProxy)0;

            QR.solveInPlace(ref Aaug, ref baug, ref h, ref u, ref w);

            for (int j = 0; j < n; j++)
                if (!math.isfinite(h[j])) return false;
            return true;
        }

        // Convergence bookkeeping (cost, gain ratio, gradient/step norms) accumulates in double --
        // same idiom as ladIRLS's own dx/xn convergence check -- while J, r, h, d, mu stay fProxy
        // (the actual factorized system QR.solveInPlace consumes).
        private static double nlsCost(in fProxyN r)
        {
            double s = 0;
            for (int i = 0; i < r.N; i++) { double v = r[i]; s += v * v; }
            return 0.5 * s;
        }

        private static double nlsCostRobust<TLoss>(in fProxyN r, in TLoss loss)
            where TLoss : struct, IfProxyRobustLoss
        {
            double s = 0;
            for (int i = 0; i < r.N; i++)
            {
                fProxy ri = r[i];
                s += (double)loss.Rho(ri * ri);
            }
            return 0.5 * s;
        }

        private static double nlsScaledGradNorm(in fProxyN g, in fProxyN d)
        {
            double best = 0;
            for (int j = 0; j < g.N; j++)
            {
                double v = math.abs((double)g[j] / (double)d[j]);
                if (v > best) best = v;
            }
            return best;
        }

        private static double nlsL2Double(in fProxyN v)
        {
            double s = 0;
            for (int i = 0; i < v.N; i++) { double x = v[i]; s += x * x; }
            return math.sqrt(s);
        }

        private static double nlsPredictedReduction(in fProxyN h, in fProxyN d, fProxy mu, in fProxyN g)
        {
            double s = 0;
            for (int j = 0; j < h.N; j++)
            {
                double dj = d[j];
                double term = (double)mu * dj * dj * (double)h[j] - (double)g[j];
                s += (double)h[j] * term;
            }
            return 0.5 * s;
        }

        // ---- numeric-Jacobian core (shared by the plain and robust-loss overloads: fProxyL2Loss is
        // the identity element of nlsApplyRobustScale, so the unweighted overloads route through here
        // too instead of duplicating the loop) ----------------------------------------------------------

        private static NLSInfo nlsSolveNumericCore<TF, TLoss>(ref TF f, ref fProxyN p, int m, in TLoss loss,
            fProxy gradTol, fProxy stepTol, int maxIter, NLSJacobianMode jacobianMode, fProxy epsfcn)
            where TF : struct, IfProxyResidualFunction
            where TLoss : struct, IfProxyRobustLoss
        {
            int n = p.N;
            if (m <= 0) throw new ArgumentException("nlsSolve: m must be positive");
            if (n <= 0) throw new ArgumentException("nlsSolve: p.N must be positive");
            if (maxIter < 1) throw new ArgumentException("nlsSolve: maxIter must be >= 1");

            var r = new fProxyN(m, Allocator.Temp, true);
            var J = new fProxyMxN(m, n, Allocator.Temp, true);
            var rs = new fProxyN(m, Allocator.Temp, true);
            var Js = new fProxyMxN(m, n, Allocator.Temp, true);
            var d = new fProxyN(n, Allocator.Temp);
            var colNorms = new fProxyN(n, Allocator.Temp, true);
            var g = new fProxyN(n, Allocator.Temp, true);
            var h = new fProxyN(n, Allocator.Temp, true);
            var Aaug = new fProxyMxN(m + n, n, Allocator.Temp, true);
            var baug = new fProxyN(m + n, Allocator.Temp, true);
            var u = new fProxyN(m + n, Allocator.Temp, true);
            var w = new fProxyN(n, Allocator.Temp, true);
            var pTrial = new fProxyN(n, Allocator.Temp, true);
            var rTrial = new fProxyN(m, Allocator.Temp, true);
            var pPert = new fProxyN(n, Allocator.Temp, true);
            var rPert = new fProxyN(m, Allocator.Temp, true);
            var rPert2 = new fProxyN(m, Allocator.Temp, true);

            f.Residuals(in p, ref r);
            nlsNumericJacobian(ref f, in p, in r, ref J, jacobianMode, epsfcn, ref pPert, ref rPert, ref rPert2);

            double cost = nlsCostRobust(in r, in loss);

            NLSStatus status;
            int it = 0;
            double gnorm;

            // flatThresh classifies a column as flat/negligible -- NOT Consts.fProxySqrtEps *
            // LInfJ0 (a threshold scaled by the WHOLE Jacobian's magnitude cross-contaminates
            // columns of genuinely different natural scale: a parameter with a small-but-real
            // derivative gets misclassified as flat and floored up to another parameter's much
            // larger scale, destroying its own gradient signal; found empirically on a NIST StRD
            // case whose two parameters' sensitivities differ by ~1e6x). A plain per-type epsilon
            // (no matrix-scale multiplier) only classifies a column as flat when it is actually
            // at-or-below machine precision for ITS OWN norm -- see nlsUpdateScale for what a
            // flat column is floored TO (not this threshold itself).
            fProxy flatThresh = Consts.fProxyEpsilon;
            fProxy maxRealColNorm0 = nlsUpdateScale(ref d, in J, flatThresh, ref colNorms);

            if (maxRealColNorm0 <= (fProxy)0)
            {
                // Every column is at-or-below flatThresh (includes the literal-zero Jacobian): d is
                // left all-zero and unusable as a scale -- already stationary, nothing to move.
                status = NLSStatus.Converged;
                gnorm = 0;
            }
            else
            {
                nlsApplyRobustScale(in r, in J, ref rs, ref Js, in loss);
                Blas.dot(in rs, in Js, ref g);

                double gnorm0 = nlsScaledGradNorm(in g, in d);
                double floorAbs = math.sqrt((double)Consts.fProxyEpsilon);
                fProxy mu = (fProxy)1e-3 * nlsMaxD2(in d);
                fProxy nu = (fProxy)2;
                fProxy muMax = (fProxy)1 / (Consts.fProxyEpsilon * Consts.fProxyEpsilon);

                gnorm = gnorm0;
                bool stop = gnorm <= gradTol * math.max(gnorm0, floorAbs);
                status = stop ? NLSStatus.Converged : NLSStatus.MaxIterations;

                while (!stop && it < maxIter)
                {
                    bool solveOk = nlsSolveStep(in Js, in rs, in d, mu, ref Aaug, ref baug, ref h, ref u, ref w);
                    if (!solveOk) { status = NLSStatus.FailedLinearSolve; break; }

                    double pNorm = nlsL2Double(in p);
                    double hNorm = nlsL2Double(in h);
                    if (hNorm <= (double)stepTol * (pNorm + (double)stepTol))
                    {
                        status = NLSStatus.SmallStep;
                        stop = true;
                        break;
                    }

                    for (int j = 0; j < n; j++) pTrial[j] = p[j] + h[j];
                    f.Residuals(in pTrial, ref rTrial);
                    double costNew = nlsCostRobust(in rTrial, in loss);

                    double predicted = nlsPredictedReduction(in h, in d, mu, in g);
                    double actual = cost - costNew;
                    double rhoGain = predicted > 0 ? actual / predicted : -1.0;

                    it++;
                    if (rhoGain > 0)
                    {
                        for (int j = 0; j < n; j++) p[j] = pTrial[j];
                        r.CopyFrom(in rTrial);
                        cost = costNew;

                        nlsNumericJacobian(ref f, in p, in r, ref J, jacobianMode, epsfcn, ref pPert, ref rPert, ref rPert2);
                        nlsUpdateScale(ref d, in J, flatThresh, ref colNorms);
                        nlsApplyRobustScale(in r, in J, ref rs, ref Js, in loss);
                        Blas.dot(in rs, in Js, ref g);

                        double factor = math.max(1.0 / 3.0, 1.0 - (2.0 * rhoGain - 1.0) * (2.0 * rhoGain - 1.0) * (2.0 * rhoGain - 1.0));
                        mu = mu * (fProxy)factor;
                        nu = (fProxy)2;

                        gnorm = nlsScaledGradNorm(in g, in d);
                        if (gnorm <= gradTol * math.max(gnorm0, floorAbs))
                        {
                            status = NLSStatus.Converged;
                            stop = true;
                        }
                    }
                    else
                    {
                        mu = mu * nu;
                        nu = nu * (fProxy)2;
                    }

                    if (!math.isfinite(mu) || mu > muMax)
                    {
                        status = NLSStatus.FailedLinearSolve;
                        break;
                    }
                }

                gnorm = nlsScaledGradNorm(in g, in d);
            }

            double rnorm = nlsL2Double(in r);

            rPert2.Dispose(); rPert.Dispose(); pPert.Dispose();
            rTrial.Dispose(); pTrial.Dispose();
            w.Dispose(); u.Dispose(); baug.Dispose(); Aaug.Dispose();
            h.Dispose(); g.Dispose(); colNorms.Dispose(); d.Dispose(); Js.Dispose(); rs.Dispose(); J.Dispose(); r.Dispose();

            return new NLSInfo { status = status, iterations = it, objective = cost, residualNorm = rnorm, gradientNorm = gnorm };
        }

        // ---- analytic-Jacobian core (no robust-loss overload in v1 -- see file DEVLOG) -----------------

        private static NLSInfo nlsSolveAnalyticCore<TF>(ref TF f, ref fProxyN p, int m,
            fProxy gradTol, fProxy stepTol, int maxIter)
            where TF : struct, IfProxyResidualJacobian
        {
            int n = p.N;
            if (m <= 0) throw new ArgumentException("nlsSolve: m must be positive");
            if (n <= 0) throw new ArgumentException("nlsSolve: p.N must be positive");
            if (maxIter < 1) throw new ArgumentException("nlsSolve: maxIter must be >= 1");

            var r = new fProxyN(m, Allocator.Temp, true);
            var J = new fProxyMxN(m, n, Allocator.Temp, true);
            var d = new fProxyN(n, Allocator.Temp);
            var colNorms = new fProxyN(n, Allocator.Temp, true);
            var g = new fProxyN(n, Allocator.Temp, true);
            var h = new fProxyN(n, Allocator.Temp, true);
            var Aaug = new fProxyMxN(m + n, n, Allocator.Temp, true);
            var baug = new fProxyN(m + n, Allocator.Temp, true);
            var u = new fProxyN(m + n, Allocator.Temp, true);
            var w = new fProxyN(n, Allocator.Temp, true);
            var pTrial = new fProxyN(n, Allocator.Temp, true);
            var rTrial = new fProxyN(m, Allocator.Temp, true);

            f.Residuals(in p, ref r);
            f.Jacobian(in p, ref J);

            double cost = nlsCost(in r);

            NLSStatus status;
            int it = 0;
            double gnorm;

            // flatThresh classifies a column as flat/negligible -- NOT Consts.fProxySqrtEps *
            // LInfJ0 (a threshold scaled by the WHOLE Jacobian's magnitude cross-contaminates
            // columns of genuinely different natural scale: a parameter with a small-but-real
            // derivative gets misclassified as flat and floored up to another parameter's much
            // larger scale, destroying its own gradient signal; found empirically on a NIST StRD
            // case whose two parameters' sensitivities differ by ~1e6x). A plain per-type epsilon
            // (no matrix-scale multiplier) only classifies a column as flat when it is actually
            // at-or-below machine precision for ITS OWN norm -- see nlsUpdateScale for what a
            // flat column is floored TO (not this threshold itself).
            fProxy flatThresh = Consts.fProxyEpsilon;
            fProxy maxRealColNorm0 = nlsUpdateScale(ref d, in J, flatThresh, ref colNorms);

            if (maxRealColNorm0 <= (fProxy)0)
            {
                // Every column is at-or-below flatThresh (includes the literal-zero Jacobian): d is
                // left all-zero and unusable as a scale -- already stationary, nothing to move.
                status = NLSStatus.Converged;
                gnorm = 0;
            }
            else
            {
                Blas.dot(in r, in J, ref g);

                double gnorm0 = nlsScaledGradNorm(in g, in d);
                double floorAbs = math.sqrt((double)Consts.fProxyEpsilon);
                fProxy mu = (fProxy)1e-3 * nlsMaxD2(in d);
                fProxy nu = (fProxy)2;
                fProxy muMax = (fProxy)1 / (Consts.fProxyEpsilon * Consts.fProxyEpsilon);

                gnorm = gnorm0;
                bool stop = gnorm <= gradTol * math.max(gnorm0, floorAbs);
                status = stop ? NLSStatus.Converged : NLSStatus.MaxIterations;

                while (!stop && it < maxIter)
                {
                    bool solveOk = nlsSolveStep(in J, in r, in d, mu, ref Aaug, ref baug, ref h, ref u, ref w);
                    if (!solveOk) { status = NLSStatus.FailedLinearSolve; break; }

                    double pNorm = nlsL2Double(in p);
                    double hNorm = nlsL2Double(in h);
                    if (hNorm <= (double)stepTol * (pNorm + (double)stepTol))
                    {
                        status = NLSStatus.SmallStep;
                        stop = true;
                        break;
                    }

                    for (int j = 0; j < n; j++) pTrial[j] = p[j] + h[j];
                    f.Residuals(in pTrial, ref rTrial);
                    double costNew = nlsCost(in rTrial);

                    double predicted = nlsPredictedReduction(in h, in d, mu, in g);
                    double actual = cost - costNew;
                    double rhoGain = predicted > 0 ? actual / predicted : -1.0;

                    it++;
                    if (rhoGain > 0)
                    {
                        for (int j = 0; j < n; j++) p[j] = pTrial[j];
                        r.CopyFrom(in rTrial);
                        cost = costNew;

                        f.Jacobian(in p, ref J);
                        nlsUpdateScale(ref d, in J, flatThresh, ref colNorms);
                        Blas.dot(in r, in J, ref g);

                        double factor = math.max(1.0 / 3.0, 1.0 - (2.0 * rhoGain - 1.0) * (2.0 * rhoGain - 1.0) * (2.0 * rhoGain - 1.0));
                        mu = mu * (fProxy)factor;
                        nu = (fProxy)2;

                        gnorm = nlsScaledGradNorm(in g, in d);
                        if (gnorm <= gradTol * math.max(gnorm0, floorAbs))
                        {
                            status = NLSStatus.Converged;
                            stop = true;
                        }
                    }
                    else
                    {
                        mu = mu * nu;
                        nu = nu * (fProxy)2;
                    }

                    if (!math.isfinite(mu) || mu > muMax)
                    {
                        status = NLSStatus.FailedLinearSolve;
                        break;
                    }
                }

                gnorm = nlsScaledGradNorm(in g, in d);
            }

            double rnorm = nlsL2Double(in r);

            rTrial.Dispose(); pTrial.Dispose();
            w.Dispose(); u.Dispose(); baug.Dispose(); Aaug.Dispose();
            h.Dispose(); g.Dispose(); colNorms.Dispose(); d.Dispose(); J.Dispose(); r.Dispose();

            return new NLSInfo { status = status, iterations = it, objective = cost, residualNorm = rnorm, gradientNorm = gnorm };
        }

        // ==== public entry points =======================================================================

        /// <summary>
        /// Levenberg-Marquardt (Nielsen damping) nonlinear least squares: minimizes 0.5·‖r(p)‖²,
        /// r given by <typeparamref name="TF"/>. Numeric (finite-difference) Jacobian.
        /// <paramref name="p"/> is both the initial guess and the overwritten result (length n, the
        /// parameter count). <paramref name="m"/> is the residual count (r.N). Converged once the
        /// scaled gradient infinity-norm falls to <paramref name="gradTol"/> times its own value at
        /// the start (never an assumed O(1) scale); SmallStep once the proposed step falls to
        /// <paramref name="stepTol"/>·(‖p‖+stepTol). Job-safe (Allocator.Temp scratch, disposed
        /// before returning).
        /// </summary>
        public static NLSInfo nlsSolve<TF>(ref TF f, ref fProxyN p, int m,
            fProxy gradTol, fProxy stepTol, int maxIter, NLSJacobianMode jacobianMode, fProxy epsfcn)
            where TF : struct, IfProxyResidualFunction
        {
            var loss = new fProxyL2Loss();
            return nlsSolveNumericCore(ref f, ref p, m, in loss, gradTol, stepTol, maxIter, jacobianMode, epsfcn);
        }

        /// <summary>nlsSolve with a default noise floor (epsfcn = 0, i.e. machine epsilon only).</summary>
        public static NLSInfo nlsSolve<TF>(ref TF f, ref fProxyN p, int m,
            fProxy gradTol, fProxy stepTol, int maxIter, NLSJacobianMode jacobianMode)
            where TF : struct, IfProxyResidualFunction
            => nlsSolve(ref f, ref p, m, gradTol, stepTol, maxIter, jacobianMode, (fProxy)0);

        /// <summary>nlsSolve with default tolerances (gradTol = Consts.fProxySqrtEps, stepTol =
        /// Consts.fProxyEpsilon), maxIter (200), Jacobian mode (Forward) and epsfcn (0).</summary>
        public static NLSInfo nlsSolve<TF>(ref TF f, ref fProxyN p, int m)
            where TF : struct, IfProxyResidualFunction
            => nlsSolve(ref f, ref p, m, Consts.fProxySqrtEps, Consts.fProxyEpsilon, 200, NLSJacobianMode.Forward, (fProxy)0);

        /// <summary>
        /// Levenberg-Marquardt nonlinear least squares with an ANALYTIC Jacobian
        /// (<typeparamref name="TF"/> : <see cref="IfProxyResidualJacobian"/>) -- otherwise identical
        /// contract to the numeric-Jacobian overload above.
        /// </summary>
        public static NLSInfo nlsSolve<TF>(ref TF f, ref fProxyN p, int m,
            fProxy gradTol, fProxy stepTol, int maxIter)
            where TF : struct, IfProxyResidualJacobian
            => nlsSolveAnalyticCore(ref f, ref p, m, gradTol, stepTol, maxIter);

        /// <summary>
        /// Levenberg-Marquardt nonlinear least squares under a robust <typeparamref name="TLoss"/>
        /// (scipy-style per-iteration row rescale of r/J, see <see cref="IfProxyRobustLoss"/>) --
        /// otherwise identical contract to the plain numeric-Jacobian overload above. Numeric
        /// Jacobian only (no analytic-Jacobian + robust-loss combination in v1).
        /// </summary>
        public static NLSInfo nlsSolve<TF, TLoss>(ref TF f, ref fProxyN p, int m, in TLoss loss,
            fProxy gradTol, fProxy stepTol, int maxIter, NLSJacobianMode jacobianMode, fProxy epsfcn)
            where TF : struct, IfProxyResidualFunction
            where TLoss : struct, IfProxyRobustLoss
            => nlsSolveNumericCore(ref f, ref p, m, in loss, gradTol, stepTol, maxIter, jacobianMode, epsfcn);

        /// <summary>nlsSolve (robust loss) with a default Jacobian mode (Forward) and epsfcn (0).</summary>
        public static NLSInfo nlsSolve<TF, TLoss>(ref TF f, ref fProxyN p, int m, in TLoss loss,
            fProxy gradTol, fProxy stepTol, int maxIter)
            where TF : struct, IfProxyResidualFunction
            where TLoss : struct, IfProxyRobustLoss
            => nlsSolve(ref f, ref p, m, in loss, gradTol, stepTol, maxIter, NLSJacobianMode.Forward, (fProxy)0);

        /// <summary>nlsSolve (robust loss) with default tolerances/maxIter (see the 3-arg overload).</summary>
        public static NLSInfo nlsSolve<TF, TLoss>(ref TF f, ref fProxyN p, int m, in TLoss loss)
            where TF : struct, IfProxyResidualFunction
            where TLoss : struct, IfProxyRobustLoss
            => nlsSolve(ref f, ref p, m, in loss, Consts.fProxySqrtEps, Consts.fProxyEpsilon, 200);

        // ==== curveFit facade ===========================================================================

        // r_i = model.Eval(x_i, p) - y_i. Internal -- not part of the public API surface, only the
        // vehicle curveFit uses to reach nlsSolve<TF>'s numeric-Jacobian path.
        private struct fProxyCurveFitResidual<TModel> : IfProxyResidualFunction
            where TModel : struct, IfProxyCurveModel
        {
            public TModel Model;
            public fProxyN X;
            public fProxyN Y;

            public void Residuals(in fProxyN p, ref fProxyN r)
            {
                for (int i = 0; i < r.N; i++)
                    r[i] = Model.Eval(X[i], in p) - Y[i];
            }
        }

        // r_i = (model.Eval(x_i, p) - y_i) / sigma_i -- standard chi-square weighting (sigma =
        // per-point standard deviation, not a raw weight).
        private struct fProxyCurveFitWeightedResidual<TModel> : IfProxyResidualFunction
            where TModel : struct, IfProxyCurveModel
        {
            public TModel Model;
            public fProxyN X;
            public fProxyN Y;
            public fProxyN Sigma;

            public void Residuals(in fProxyN p, ref fProxyN r)
            {
                for (int i = 0; i < r.N; i++)
                    r[i] = (Model.Eval(X[i], in p) - Y[i]) / Sigma[i];
            }
        }

        /// <summary>
        /// Curve-fit facade over <see cref="nlsSolve{TF}(ref TF, ref fProxyN, int)"/>: fits
        /// <typeparamref name="TModel"/>'s y = f(x; p) to (xdata, ydata) by nonlinear least squares,
        /// minimizing 0.5·Σ(f(x_i; p) - y_i)². <paramref name="p"/> is both the initial guess and the
        /// overwritten result. Numeric Jacobian (forward difference) only.
        /// </summary>
        public static NLSInfo curveFit<TModel>(in fProxyN xdata, in fProxyN ydata, ref TModel model, ref fProxyN p,
            fProxy gradTol, fProxy stepTol, int maxIter)
            where TModel : struct, IfProxyCurveModel
        {
            if (xdata.N != ydata.N)
                throw new ArgumentException("curveFit: xdata.N must equal ydata.N");

            var residual = new fProxyCurveFitResidual<TModel> { Model = model, X = xdata, Y = ydata };
            return nlsSolve(ref residual, ref p, xdata.N, gradTol, stepTol, maxIter, NLSJacobianMode.Forward, (fProxy)0);
        }

        /// <summary>curveFit with default tolerances (gradTol = Consts.fProxySqrtEps, stepTol =
        /// Consts.fProxyEpsilon) and maxIter (200).</summary>
        public static NLSInfo curveFit<TModel>(in fProxyN xdata, in fProxyN ydata, ref TModel model, ref fProxyN p)
            where TModel : struct, IfProxyCurveModel
            => curveFit(in xdata, in ydata, ref model, ref p, Consts.fProxySqrtEps, Consts.fProxyEpsilon, 200);

        /// <summary>
        /// curveFit weighted by a per-point standard deviation <paramref name="sigma"/>: minimizes
        /// 0.5·Σ((f(x_i; p) - y_i)/sigma_i)² (standard chi-square fitting).
        /// </summary>
        public static NLSInfo curveFit<TModel>(in fProxyN xdata, in fProxyN ydata, in fProxyN sigma, ref TModel model, ref fProxyN p,
            fProxy gradTol, fProxy stepTol, int maxIter)
            where TModel : struct, IfProxyCurveModel
        {
            if (xdata.N != ydata.N)
                throw new ArgumentException("curveFit: xdata.N must equal ydata.N");
            if (sigma.N != xdata.N)
                throw new ArgumentException("curveFit: sigma.N must equal xdata.N");

            var residual = new fProxyCurveFitWeightedResidual<TModel> { Model = model, X = xdata, Y = ydata, Sigma = sigma };
            return nlsSolve(ref residual, ref p, xdata.N, gradTol, stepTol, maxIter, NLSJacobianMode.Forward, (fProxy)0);
        }

        /// <summary>Weighted curveFit with default tolerances/maxIter (see the unweighted 4-arg overload).</summary>
        public static NLSInfo curveFit<TModel>(in fProxyN xdata, in fProxyN ydata, in fProxyN sigma, ref TModel model, ref fProxyN p)
            where TModel : struct, IfProxyCurveModel
            => curveFit(in xdata, in ydata, in sigma, ref model, ref p, Consts.fProxySqrtEps, Consts.fProxyEpsilon, 200);
    }
}
