#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // IfProxyScalarFunction (the shared Burst "lambda" functor used by optimizers AND generators)
    // lives in Interfaces/ScalarFunction.fProxy.cs; the derivative/gradient interfaces below are optimizer-specific.

    public interface IfProxyScalarDerivativeFunction : IfProxyScalarFunction {
        fProxy Derivative(fProxy x);
    }

    public interface IfProxyGradientFunction {
        fProxy Eval(in fProxyN x);
        void Gradient(in fProxyN x, ref fProxyN g);
    }

    public static partial class Optimize_OP {

        /// <summary>
        /// Bracketing root find. Requires f(lo) and f(hi) to have opposite signs; returns false if not bracketed (root = the better endpoint).
        /// Converges when (hi - lo) &lt;= xTol + rTol * |mid| or f(mid) == 0. root = final midpoint. Returns true on convergence.
        /// xTol is absolute, rTol relative to the current midpoint; rTol &gt;= a few * Consts.fProxyEpsilon
        /// keeps the criterion reachable regardless of the root's magnitude.
        /// </summary>
        public static bool bisection<F>(ref F f, fProxy lo, fProxy hi, out fProxy root,
                                        fProxy xTol, fProxy rTol, int maxIter)
            where F : struct, IfProxyScalarFunction
        {
            if (maxIter < 1)
                throw new ArgumentException("bisection: maxIter must be >= 1");

            fProxy flo = f.Eval(lo);
            fProxy fhi = f.Eval(hi);

            if (flo == (fProxy)0) { root = lo; return true; }
            if (fhi == (fProxy)0) { root = hi; return true; }

            // Same sign → not bracketed (product test replaced to avoid underflow/NaN misjudgement)
            if ((flo > (fProxy)0) == (fhi > (fProxy)0)) {
                root = (math.abs(flo) <= math.abs(fhi)) ? lo : hi;
                return false;
            }

            for (int i = 0; i < maxIter; i++) {
                fProxy mid = lo + (hi - lo) * (fProxy)0.5;
                fProxy fmid = f.Eval(mid);

                if (fmid == (fProxy)0) { root = mid; return true; }
                if ((hi - lo) <= xTol + rTol * math.abs(mid)) { root = mid; return true; }

                // Sign changes between lo and mid → root in [lo, mid]
                if ((flo > (fProxy)0) != (fmid > (fProxy)0)) {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fmid;
                }
            }

            root = lo + (hi - lo) * (fProxy)0.5;
            return (hi - lo) <= xTol + rTol * math.abs(root);
        }

        /// <summary>bisection with absolute tolerance only (rTol = 0).</summary>
        public static bool bisection<F>(ref F f, fProxy lo, fProxy hi, out fProxy root,
                                        fProxy xTol, int maxIter)
            where F : struct, IfProxyScalarFunction
            => bisection(ref f, lo, hi, out root, xTol, (fProxy)0, maxIter);

        /// <summary>bisection with default maxIter (200).</summary>
        public static bool bisection<F>(ref F f, fProxy lo, fProxy hi, out fProxy root,
                                        fProxy xTol)
            where F : struct, IfProxyScalarFunction
            => bisection(ref f, lo, hi, out root, xTol, (fProxy)0, 200);

        /// <summary>bisection with default xTol (Consts.fProxyZeroThreshold), rTol (4 * Consts.fProxyEpsilon) and maxIter (200).</summary>
        public static bool bisection<F>(ref F f, fProxy lo, fProxy hi, out fProxy root)
            where F : struct, IfProxyScalarFunction
            => bisection(ref f, lo, hi, out root, Consts.fProxyZeroThreshold, (fProxy)4 * Consts.fProxyEpsilon, 200);

        /// <summary>
        /// Newton root find. Converged when |f(x)| &lt;= fTol. Returns false if |f'(x)| &lt; Consts.fProxyZeroThreshold (flat/badly-scaled, absolute guard) or maxIter exhausted; root holds last iterate.
        /// </summary>
        public static bool newtonRoot<F>(ref F f, fProxy x0, out fProxy root,
                                         fProxy fTol, int maxIter)
            where F : struct, IfProxyScalarDerivativeFunction
        {
            if (maxIter < 1)
                throw new ArgumentException("newtonRoot: maxIter must be >= 1");

            fProxy x = x0;

            for (int i = 0; i < maxIter; i++) {
                fProxy fx = f.Eval(x);
                if (math.abs(fx) <= fTol) { root = x; return true; }

                fProxy d = f.Derivative(x);
                if (math.abs(d) < Consts.fProxyZeroThreshold) { root = x; return false; }

                x = x - fx / d;
            }

            root = x;
            return math.abs(f.Eval(x)) <= fTol;
        }

        /// <summary>newtonRoot with default maxIter (100).</summary>
        public static bool newtonRoot<F>(ref F f, fProxy x0, out fProxy root,
                                         fProxy fTol)
            where F : struct, IfProxyScalarDerivativeFunction
            => newtonRoot(ref f, x0, out root, fTol, 100);

        /// <summary>newtonRoot with default fTol (Consts.fProxyZeroThreshold) and maxIter (100).</summary>
        public static bool newtonRoot<F>(ref F f, fProxy x0, out fProxy root)
            where F : struct, IfProxyScalarDerivativeFunction
            => newtonRoot(ref f, x0, out root, Consts.fProxyZeroThreshold, 100);

        /// <summary>
        /// Golden-section minimization of unimodal f on [a, b]. xMin = midpoint of final bracket. Returns true when (b - a) &lt;= xTol within maxIter.
        /// xTol is an absolute tolerance on the bracket width, rTol relative to the bracket midpoint.
        /// Note: a smooth minimum can only be localized to ~|xMin| * sqrt(machine eps)
        /// (Consts.fProxySqrtEps), no matter how small the tolerances are.
        /// </summary>
        public static bool goldenSection<F>(ref F f, fProxy a, fProxy b, out fProxy xMin,
                                            fProxy xTol, fProxy rTol, int maxIter)
            where F : struct, IfProxyScalarFunction
        {
            if (maxIter < 1)
                throw new ArgumentException("goldenSection: maxIter must be >= 1");

            if (a > b) {
                fProxy tmp = a;
                a = b;
                b = tmp;
            }

            fProxy invphi = (math.sqrt((fProxy)5) - (fProxy)1) * (fProxy)0.5;

            fProxy c = b - invphi * (b - a);
            fProxy d = a + invphi * (b - a);
            fProxy fc = f.Eval(c);
            fProxy fd = f.Eval(d);

            for (int i = 0; i < maxIter; i++) {
                if ((b - a) <= xTol + rTol * math.abs(a + (b - a) * (fProxy)0.5)) break;

                if (fc < fd) {
                    b = d;
                    d = c;
                    fd = fc;
                    c = b - invphi * (b - a);
                    fc = f.Eval(c);
                } else {
                    a = c;
                    c = d;
                    fc = fd;
                    d = a + invphi * (b - a);
                    fd = f.Eval(d);
                }
            }

            xMin = a + (b - a) * (fProxy)0.5;
            return (b - a) <= xTol + rTol * math.abs(xMin);
        }

        /// <summary>goldenSection with absolute tolerance only (rTol = 0).</summary>
        public static bool goldenSection<F>(ref F f, fProxy a, fProxy b, out fProxy xMin,
                                            fProxy xTol, int maxIter)
            where F : struct, IfProxyScalarFunction
            => goldenSection(ref f, a, b, out xMin, xTol, (fProxy)0, maxIter);

        /// <summary>goldenSection with default maxIter (200).</summary>
        public static bool goldenSection<F>(ref F f, fProxy a, fProxy b, out fProxy xMin,
                                            fProxy xTol)
            where F : struct, IfProxyScalarFunction
            => goldenSection(ref f, a, b, out xMin, xTol, (fProxy)0, 200);

        /// <summary>goldenSection with default xTol (Consts.fProxyZeroThreshold), rTol (3 * Consts.fProxySqrtEps) and maxIter (200).</summary>
        public static bool goldenSection<F>(ref F f, fProxy a, fProxy b, out fProxy xMin)
            where F : struct, IfProxyScalarFunction
            => goldenSection(ref f, a, b, out xMin, Consts.fProxyZeroThreshold, (fProxy)3 * Consts.fProxySqrtEps, 200);

        // Fixed-step gradient descent, in-place on x (x -= learningRate * g until L2(g) <= gradTol or
        // maxIter). g is caller-provided scratch (length x.N); zero-alloc. iterations = performed count.
        public static bool gradientDescent<F>(ref F f, ref fProxyN x, ref fProxyN g,
                                              fProxy learningRate, fProxy gradTol, int maxIter,
                                              out int iterations)
            where F : struct, IfProxyGradientFunction
        {
            if (g.N != x.N)
                throw new ArgumentException("gradientDescent: g.N must equal x.N");

            if (maxIter < 1)
                throw new ArgumentException("gradientDescent: maxIter must be >= 1");

            iterations = 0;

            for (int i = 0; i < maxIter; i++) {
                f.Gradient(in x, ref g);

                if (fProxyNorms_OP.L2(in g) <= gradTol)
                    return true;

                for (int j = 0; j < x.N; j++)
                    x[j] -= learningRate * g[j];

                iterations++;
            }

            // Final convergence check: compute fresh gradient at returned x
            f.Gradient(in x, ref g);
            return fProxyNorms_OP.L2(in g) <= gradTol;
        }
    }
}
