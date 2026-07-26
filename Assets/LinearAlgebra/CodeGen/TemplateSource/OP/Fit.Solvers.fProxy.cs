using System;

using Unity.Collections;
using Unity.Mathematics;

//+deleteThis
using fProxy3 = Unity.Mathematics.float3;
//-deleteThis

namespace BULA
{
    // ================================================================================================
    // Shape-agnostic solvers. Each is generic over the shape, so a new shape works with all of them
    // the moment it satisfies the corresponding interface -- no solver changes.
    //
    //   irls    reweight and refit until the weights settle      (IfProxyWeighted3)
    //   ransac  consensus over random minimal samples            (IfProxyEstimable3)
    //   nls     Levenberg-Marquardt on the packed parameters      (IfProxyParametric3)
    //
    // The DRIVERS own the invariants the shapes must not have to remember: weight initialisation, the
    // all-weights-zero collapse guard, convergence, and the iteration budget. That division is the
    // point of the refactor -- the same collapse bug appeared twice while those loops were duplicated
    // per shape, because the invariant lived in a comment rather than in one place.
    //
    // Job-safe: scratch is Allocator.Temp, disposed before returning.
    // ================================================================================================
    public static partial class Fit
    {
        /// <summary>
        /// Iteratively reweighted least squares: refit under weights, recompute residuals, reweight,
        /// repeat. <paramref name="model"/> is overwritten with the result.
        ///
        /// <paramref name="warmStart"/> decides whether the INCOMING model matters. Left false, the
        /// first pass is an unweighted fit over every point and the starting model is ignored
        /// entirely. Set it when the caller already holds a model worth trusting -- the consensus set
        /// from <see cref="ransac"/> being the case this library recommends -- and the initial weights
        /// come from that model's own residuals instead, so a redescending loss starts in the basin
        /// that model found rather than in whatever basin the contaminated unweighted fit lands in.
        ///
        /// <paramref name="priorW"/> is an OPTIONAL per-point weight (uncreated = none) multiplied
        /// into every iteration's weight -- measurement confidence, inverse variance, anything the
        /// caller knows independently of the residual. With <see cref="fProxyL2Loss"/> and a prior,
        /// this is ordinary weighted least squares in one pass, because that loss's weights never
        /// move; without either it is a plain fit. No special-cased path for those.
        ///
        /// False means the fit collapsed: a redescending loss can drive every weight to zero, and
        /// solving from that yields NaN, so it stops rather than certify garbage.
        /// </summary>
        public static bool irls<TModel, TLoss>(NativeArray<fProxy3> points, ref TModel model,
                                               in TLoss loss, in fProxyN priorW, int maxIter = 0,
                                               bool warmStart = false)
            where TModel : struct, IfProxyWeighted3
            where TLoss : struct, IfProxyRobustLoss
        {
            int n = points.Length;
            if (n < model.MinimalSamples)
                throw new ArgumentException("Fit.irls: fewer points than the shape's MinimalSamples");
            if (maxIter <= 0) maxIter = DefaultIrlsIter;

            bool hasPrior = priorW.IsCreated;
            if (hasPrior && priorW.N != n)
                throw new ArgumentException("Fit.irls: priorW.N must equal points.Length");

            var w = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++)
            {
                fProxy prior = hasPrior ? priorW[i] : (fProxy)1;
                if (!warmStart) { w[i] = prior; continue; }

                // Weights from the incoming model's own residuals. Refit runs before Distance is ever
                // read below, so without this the starting model has no influence whatsoever.
                fProxy d = model.Distance(points[i]);
                w[i] = loss.RhoPrime(d * d) * prior;
            }

            bool ok = false;
            for (int it = 0; it < maxIter; it++)
            {
                fProxy sw = (fProxy)0;
                for (int i = 0; i < n; i++) sw += w[i];
                if (!(sw > (fProxy)0)) { ok = false; break; }

                ok = model.Refit(points, in w);
                if (!ok) break;

                fProxy maxDelta = (fProxy)0, maxW = (fProxy)0;
                for (int i = 0; i < n; i++)
                {
                    fProxy d = model.Distance(points[i]);
                    fProxy wNew = loss.RhoPrime(d * d);
                    if (hasPrior) wNew *= priorW[i];
                    maxDelta = math.max(maxDelta, math.abs(wNew - w[i]));
                    maxW = math.max(maxW, wNew);
                    w[i] = wNew;
                }

                // Relative to the largest weight: an absolute test is unreachable once weights are
                // large (an inverse-variance prior, or L1's 0.5/floor), so the loop would silently
                // always run its full budget.
                if (maxDelta <= Consts.fProxySqrtEps * math.max(maxW, (fProxy)1)) break;
            }

            w.Dispose();
            return ok;
        }

        /// <summary>IRLS with no prior weights. See the prior-weighted overload.</summary>
        public static bool irls<TModel, TLoss>(NativeArray<fProxy3> points, ref TModel model,
                                               in TLoss loss, int maxIter = 0, bool warmStart = false)
            where TModel : struct, IfProxyWeighted3
            where TLoss : struct, IfProxyRobustLoss
            => irls(points, ref model, in loss, default(fProxyN), maxIter, warmStart);

        /// <summary>
        /// Levenberg-Marquardt on the shape's packed parameters, minimizing its own distance function
        /// over every point. <paramref name="model"/> is the initial guess and is overwritten --
        /// these solves are LOCAL, so seeding from a previous frame is both faster and likelier to
        /// find the right minimum than any fresh guess.
        /// </summary>
        public static bool nls<TModel, TLoss>(NativeArray<fProxy3> points, ref TModel model,
                                              in TLoss loss)
            where TModel : struct, IfProxyParametric3
            where TLoss : struct, IfProxyRobustLoss
        {
            int n = points.Length;
            var p = new fProxyN(model.ParamCount, Allocator.Temp);
            model.Pack(ref p);

            var f = new fProxyShapeResidual<TModel> { Points = points, Model = model };
            var info = Optimize.nlsSolve(ref f, ref p, n, in loss);

            model.Unpack(in p);
            p.Dispose();
            return info;
        }

        /// <summary>Levenberg-Marquardt with plain least squares. See the loss overload.</summary>
        public static bool nls<TModel>(NativeArray<fProxy3> points, ref TModel model)
            where TModel : struct, IfProxyParametric3
        {
            var l2 = new fProxyL2Loss();
            return nls(points, ref model, in l2);
        }

        /// <summary>
        /// Adapts any <see cref="IfProxyParametric3"/> shape to the residual functor
        /// <see cref="Optimize.nlsSolve{TF,TLoss}"/> expects: unpack the parameter vector into the
        /// shape, then read off its own distance to each point. ONE adapter serves every shape, which
        /// is what removes the per-shape residual structs.
        /// </summary>
        public struct fProxyShapeResidual<TModel> : IfProxyResidualFunction
            where TModel : struct, IfProxyParametric3
        {
            public NativeArray<fProxy3> Points;
            public TModel Model;

            public void Residuals(in fProxyN p, ref fProxyN r)
            {
                var m = Model;
                m.Unpack(in p);
                for (int i = 0; i < Points.Length; i++) r[i] = m.Distance(Points[i]);
            }
        }
    }
}
