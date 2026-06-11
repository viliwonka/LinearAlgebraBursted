#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra
{
    public interface IfProxyScalarFunction {
        fProxy Eval(fProxy x);
    }

    public interface IfProxyScalarDerivativeFunction : IfProxyScalarFunction {
        fProxy Derivative(fProxy x);
    }

    public interface IfProxyGradientFunction {
        fProxy Eval(in fProxyN x);
        void Gradient(in fProxyN x, ref fProxyN g);
    }

    public static partial class Optimize {

        /// <summary>
        /// Bracketing root find. Requires f(lo) and f(hi) to have opposite signs; returns false if not bracketed (root = the better endpoint).
        /// Converges when (hi - lo) &lt;= xTol or f(mid) == 0. root = final midpoint. Returns true on convergence.
        /// Note: xTol is an absolute tolerance on the interval width.
        /// </summary>
        public static bool bisection<F>(ref F f, fProxy lo, fProxy hi, out fProxy root,
                                        fProxy xTol = Consts.fProxyZeroTreshold, int maxIter = 200)
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
                if ((hi - lo) <= xTol) { root = mid; return true; }

                // Sign changes between lo and mid → root in [lo, mid]
                if ((flo > (fProxy)0) != (fmid > (fProxy)0)) {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fmid;
                }
            }

            root = lo + (hi - lo) * (fProxy)0.5;
            return (hi - lo) <= xTol;
        }

        /// <summary>
        /// Newton root find. Converged when |f(x)| &lt;= fTol. Returns false if |f'(x)| &lt; Consts.fProxyZeroTreshold (flat/badly-scaled, absolute guard) or maxIter exhausted; root holds last iterate.
        /// </summary>
        public static bool newtonRoot<F>(ref F f, fProxy x0, out fProxy root,
                                         fProxy fTol = Consts.fProxyZeroTreshold, int maxIter = 100)
            where F : struct, IfProxyScalarDerivativeFunction
        {
            if (maxIter < 1)
                throw new ArgumentException("newtonRoot: maxIter must be >= 1");

            fProxy x = x0;

            for (int i = 0; i < maxIter; i++) {
                fProxy fx = f.Eval(x);
                if (math.abs(fx) <= fTol) { root = x; return true; }

                fProxy d = f.Derivative(x);
                if (math.abs(d) < Consts.fProxyZeroTreshold) { root = x; return false; }

                x = x - fx / d;
            }

            root = x;
            return math.abs(f.Eval(x)) <= fTol;
        }

        /// <summary>
        /// Golden-section minimization of unimodal f on [a, b]. xMin = midpoint of final bracket. Returns true when (b - a) &lt;= xTol within maxIter.
        /// Note: xTol is an absolute tolerance on the bracket width.
        /// </summary>
        public static bool goldenSection<F>(ref F f, fProxy a, fProxy b, out fProxy xMin,
                                            fProxy xTol = Consts.fProxyZeroTreshold, int maxIter = 200)
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
                if ((b - a) <= xTol) break;

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
            return (b - a) <= xTol;
        }

        // Fixed-step gradient descent, in-place on x. g is caller-provided scratch (length x.N). Does NOT allocate.
        // Iterates x -= learningRate * g until L2(g) <= gradTol or maxIter. Returns true if gradTol reached; iterations = performed count.
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

                if (fProxyNormsOP.L2(in g) <= gradTol)
                    return true;

                for (int j = 0; j < x.N; j++)
                    x[j] -= learningRate * g[j];

                iterations++;
            }

            // Final convergence check: compute fresh gradient at returned x
            f.Gradient(in x, ref g);
            return fProxyNormsOP.L2(in g) <= gradTol;
        }
    }
}
