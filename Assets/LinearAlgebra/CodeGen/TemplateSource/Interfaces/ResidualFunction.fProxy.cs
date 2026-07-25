using System;
using Unity.Mathematics;

namespace BULA
{
    /// <summary>
    /// A vector-valued residual r(p) (length m) of a parameter vector p (length n), the nonlinear
    /// least squares analogue of <see cref="IfProxyScalarFunction"/> -- the Burst struct-functor
    /// <see cref="Optimize.nlsSolve{TF}(ref TF, ref fProxyN, int)"/> minimizes 0.5·‖r(p)‖².
    /// Implement on a small struct holding only blittable fields.
    /// </summary>
    public interface IfProxyResidualFunction
    {
        /// <summary>r = residuals at p. r must be distinct from p; r.N is the fixed residual count m.</summary>
        void Residuals(in fProxyN p, ref fProxyN r);
    }

    /// <summary>
    /// <see cref="IfProxyResidualFunction"/> with an analytic Jacobian, for
    /// <see cref="Optimize.nlsSolve{TF}(ref TF, ref fProxyN, int, fProxy, fProxy, int)"/>'s
    /// no-finite-difference overload.
    /// </summary>
    public interface IfProxyResidualJacobian : IfProxyResidualFunction
    {
        /// <summary>J = dr/dp at p (m x n, J.M_Rows == r.N, J.N_Cols == p.N).</summary>
        void Jacobian(in fProxyN p, ref fProxyMxN J);
    }

    /// <summary>
    /// A scalar M-estimator loss rho(s) of a squared residual s = r², for
    /// <see cref="Optimize.nlsSolve{TF, TLoss}(ref TF, ref fProxyN, int, in TLoss)"/>'s robust-loss
    /// overload and (shared, standalone) a future linear IRLS facade. Rho, RhoPrime (dRho/ds), and
    /// RhoPrime2 (d²Rho/ds²) follow the scipy <c>optimize._lsq</c> convention: the total objective is
    /// 0.5·Σrho(s_i), so <see cref="fProxyL2Loss"/> (rho(s) = s) reproduces plain least squares
    /// exactly. Implement on a small, ideally-readonly struct holding only blittable fields.
    /// </summary>
    public interface IfProxyRobustLoss
    {
        fProxy Rho(fProxy s);
        fProxy RhoPrime(fProxy s);
        fProxy RhoPrime2(fProxy s);
    }

    /// <summary>Plain least squares: rho(s) = s. The identity element of <see cref="IfProxyRobustLoss"/> --
    /// every unweighted <see cref="Optimize.nlsSolve{TF}(ref TF, ref fProxyN, int)"/> overload is this
    /// loss applied through the shared robust-loss engine.</summary>
    public readonly struct fProxyL2Loss : IfProxyRobustLoss
    {
        public fProxy Rho(fProxy s) => s;
        public fProxy RhoPrime(fProxy s) => (fProxy)1;
        public fProxy RhoPrime2(fProxy s) => (fProxy)0;
    }

    /// <summary>
    /// L1 (least-absolute-deviation) loss: rho(s) = sqrt(s), so the objective is 0.5·Σ|r_i|.
    /// <see cref="Floor"/> is the residual magnitude below which the weight stops growing --
    /// RhoPrime is 1/(2·|r|), unbounded as |r| → 0, so a floor is REQUIRED for a finite weight; a
    /// zero-initialized instance (<c>default</c> / <c>new fProxyL1Loss()</c>) uses
    /// <see cref="Consts.fProxySqrtEps"/>. Set it to the noise floor of your data when you have one.
    ///
    /// This is IRLS-approximate L1, not exact. For an exact L1 fit of a LINEAR model use
    /// <see cref="LP.lad(in fProxyMxN, in fProxyN, ref fProxyN, out double, int)"/>, which is a
    /// finite algorithm reaching the true optimum; this loss exists so the same metric can be
    /// applied to fits that have no exact combinatorial solver (orthogonal / geometric fits).
    /// </summary>
    public readonly struct fProxyL1Loss : IfProxyRobustLoss
    {
        public readonly fProxy Floor;

        /// <param name="floor">Residual magnitude below which the IRLS weight saturates; must be
        /// positive.</param>
        public fProxyL1Loss(fProxy floor)
        {
            if (!(floor > (fProxy)0))
                throw new ArgumentException("fProxyL1Loss: floor must be positive");
            Floor = floor;
        }

        fProxy FloorSq => Floor > (fProxy)0 ? Floor * Floor : Consts.fProxySqrtEps * Consts.fProxySqrtEps;

        public fProxy Rho(fProxy s) => math.sqrt(math.max(s, FloorSq));

        public fProxy RhoPrime(fProxy s) => (fProxy)0.5 / math.sqrt(math.max(s, FloorSq));

        public fProxy RhoPrime2(fProxy s)
        {
            fProxy t = math.max(s, FloorSq);
            return (fProxy)(-0.25) / (t * math.sqrt(t));
        }
    }

    /// <summary>
    /// Huber loss: quadratic (rho(s) = s) for |r| &lt;= <see cref="Scale"/>, linear beyond it.
    /// <see cref="Scale"/> is the transition residual magnitude (the classic Huber "delta").
    /// </summary>
    public readonly struct fProxyHuberLoss : IfProxyRobustLoss
    {
        public readonly fProxy Scale;

        /// <param name="scale">Transition residual magnitude; must be positive.</param>
        public fProxyHuberLoss(fProxy scale)
        {
            if (!(scale > (fProxy)0))
                throw new ArgumentException("fProxyHuberLoss: scale must be positive");
            Scale = scale;
        }

        public fProxy Rho(fProxy s)
        {
            fProxy c2 = Scale * Scale;
            return s <= c2 ? s : (fProxy)2 * Scale * math.sqrt(s) - c2;
        }

        public fProxy RhoPrime(fProxy s)
        {
            fProxy c2 = Scale * Scale;
            return s <= c2 ? (fProxy)1 : Scale / math.sqrt(s);
        }

        public fProxy RhoPrime2(fProxy s)
        {
            fProxy c2 = Scale * Scale;
            return s <= c2 ? (fProxy)0 : (fProxy)(-0.5) * Scale / (s * math.sqrt(s));
        }
    }

    /// <summary>
    /// Cauchy (Lorentzian) loss: rho(s) = Scale²·log(1 + s/Scale²). Convex reweighting like
    /// <see cref="fProxyHuberLoss"/>, but down-weights large residuals more aggressively.
    /// <see cref="Scale"/> is the tuning constant c; must be positive.
    /// </summary>
    public readonly struct fProxyCauchyLoss : IfProxyRobustLoss
    {
        public readonly fProxy Scale;

        public fProxyCauchyLoss(fProxy scale)
        {
            if (!(scale > (fProxy)0))
                throw new ArgumentException("fProxyCauchyLoss: scale must be positive");
            Scale = scale;
        }

        public fProxy Rho(fProxy s)
        {
            fProxy c2 = Scale * Scale;
            return c2 * DetMath.Log((fProxy)1 + s / c2);
        }

        public fProxy RhoPrime(fProxy s)
        {
            fProxy c2 = Scale * Scale;
            return c2 / (c2 + s);
        }

        public fProxy RhoPrime2(fProxy s)
        {
            fProxy c2 = Scale * Scale;
            fProxy denom = c2 + s;
            return -c2 / (denom * denom);
        }
    }

    /// <summary>
    /// Tukey biweight loss: a REDESCENDING M-estimator -- rho flattens to a constant and RhoPrime
    /// reaches exactly zero once |r| &gt;= <see cref="Scale"/>, fully rejecting residuals beyond it
    /// (unlike <see cref="fProxyHuberLoss"/> / <see cref="fProxyCauchyLoss"/>, which only shrink
    /// their influence). Because of this, an <see cref="Optimize.nlsSolve{TF, TLoss}"/> call whose
    /// STARTING point puts every residual beyond <see cref="Scale"/> sees a uniformly-zero weighted
    /// gradient and reports false-converged at iteration 0 -- choose <see cref="Scale"/> comfortably
    /// larger than the expected residual spread at the start point (or warm-start from an
    /// <see cref="fProxyHuberLoss"/>/plain fit), the standard caveat for any redescending estimator.
    /// </summary>
    public readonly struct fProxyTukeyLoss : IfProxyRobustLoss
    {
        public readonly fProxy Scale;

        public fProxyTukeyLoss(fProxy scale)
        {
            if (!(scale > (fProxy)0))
                throw new ArgumentException("fProxyTukeyLoss: scale must be positive");
            Scale = scale;
        }

        public fProxy Rho(fProxy s)
        {
            fProxy c2 = Scale * Scale;
            fProxy u = s / c2;
            if (u >= (fProxy)1) return c2 / (fProxy)3;
            fProxy w = (fProxy)1 - u;
            return (c2 / (fProxy)3) * ((fProxy)1 - w * w * w);
        }

        public fProxy RhoPrime(fProxy s)
        {
            fProxy c2 = Scale * Scale;
            fProxy u = s / c2;
            if (u >= (fProxy)1) return (fProxy)0;
            fProxy w = (fProxy)1 - u;
            return w * w;
        }

        public fProxy RhoPrime2(fProxy s)
        {
            fProxy c2 = Scale * Scale;
            fProxy u = s / c2;
            if (u >= (fProxy)1) return (fProxy)0;
            fProxy w = (fProxy)1 - u;
            return (fProxy)(-2) * w / c2;
        }
    }

    /// <summary>
    /// A scalar curve model y = f(x; p), for <see cref="Optimize.curveFit{TModel}(in fProxyN, in fProxyN, ref TModel, ref fProxyN)"/>.
    /// Implement on a small struct holding only blittable fields.
    /// </summary>
    public interface IfProxyCurveModel
    {
        fProxy Eval(fProxy x, in fProxyN p);
    }
}
