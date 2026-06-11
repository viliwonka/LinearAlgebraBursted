#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra
{
    public interface IfloatScalarFunction {
        float Eval(float x);
    }

    public interface IfloatScalarDerivativeFunction : IfloatScalarFunction {
        float Derivative(float x);
    }

    public interface IfloatGradientFunction {
        float Eval(in floatN x);
        void Gradient(in floatN x, ref floatN g);
    }

    public static partial class Optimize {

        /// <summary>
        /// Bracketing root find. Requires f(lo) and f(hi) to have opposite signs; returns false if not bracketed (root = the better endpoint).
        /// Converges when (hi - lo) &lt;= xTol or f(mid) == 0. root = final midpoint. Returns true on convergence.
        /// Note: xTol is an absolute tolerance on the interval width.
        /// </summary>
        public static bool bisection<F>(ref F f, float lo, float hi, out float root,
                                        float xTol = Consts.floatZeroTreshold, int maxIter = 200)
            where F : struct, IfloatScalarFunction
        {
            if (maxIter < 1)
                throw new ArgumentException("bisection: maxIter must be >= 1");

            float flo = f.Eval(lo);
            float fhi = f.Eval(hi);

            if (flo == (float)0) { root = lo; return true; }
            if (fhi == (float)0) { root = hi; return true; }

            // Same sign → not bracketed (product test replaced to avoid underflow/NaN misjudgement)
            if ((flo > (float)0) == (fhi > (float)0)) {
                root = (math.abs(flo) <= math.abs(fhi)) ? lo : hi;
                return false;
            }

            for (int i = 0; i < maxIter; i++) {
                float mid = lo + (hi - lo) * (float)0.5;
                float fmid = f.Eval(mid);

                if (fmid == (float)0) { root = mid; return true; }
                if ((hi - lo) <= xTol) { root = mid; return true; }

                // Sign changes between lo and mid → root in [lo, mid]
                if ((flo > (float)0) != (fmid > (float)0)) {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fmid;
                }
            }

            root = lo + (hi - lo) * (float)0.5;
            return (hi - lo) <= xTol;
        }

        /// <summary>
        /// Newton root find. Converged when |f(x)| &lt;= fTol. Returns false if |f'(x)| &lt; Consts.floatZeroTreshold (flat/badly-scaled, absolute guard) or maxIter exhausted; root holds last iterate.
        /// </summary>
        public static bool newtonRoot<F>(ref F f, float x0, out float root,
                                         float fTol = Consts.floatZeroTreshold, int maxIter = 100)
            where F : struct, IfloatScalarDerivativeFunction
        {
            if (maxIter < 1)
                throw new ArgumentException("newtonRoot: maxIter must be >= 1");

            float x = x0;

            for (int i = 0; i < maxIter; i++) {
                float fx = f.Eval(x);
                if (math.abs(fx) <= fTol) { root = x; return true; }

                float d = f.Derivative(x);
                if (math.abs(d) < Consts.floatZeroTreshold) { root = x; return false; }

                x = x - fx / d;
            }

            root = x;
            return math.abs(f.Eval(x)) <= fTol;
        }

        /// <summary>
        /// Golden-section minimization of unimodal f on [a, b]. xMin = midpoint of final bracket. Returns true when (b - a) &lt;= xTol within maxIter.
        /// Note: xTol is an absolute tolerance on the bracket width.
        /// </summary>
        public static bool goldenSection<F>(ref F f, float a, float b, out float xMin,
                                            float xTol = Consts.floatZeroTreshold, int maxIter = 200)
            where F : struct, IfloatScalarFunction
        {
            if (maxIter < 1)
                throw new ArgumentException("goldenSection: maxIter must be >= 1");

            if (a > b) {
                float tmp = a;
                a = b;
                b = tmp;
            }

            float invphi = (math.sqrt((float)5) - (float)1) * (float)0.5;

            float c = b - invphi * (b - a);
            float d = a + invphi * (b - a);
            float fc = f.Eval(c);
            float fd = f.Eval(d);

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

            xMin = a + (b - a) * (float)0.5;
            return (b - a) <= xTol;
        }

        // Fixed-step gradient descent, in-place on x. g is caller-provided scratch (length x.N). Does NOT allocate.
        // Iterates x -= learningRate * g until L2(g) <= gradTol or maxIter. Returns true if gradTol reached; iterations = performed count.
        public static bool gradientDescent<F>(ref F f, ref floatN x, ref floatN g,
                                              float learningRate, float gradTol, int maxIter,
                                              out int iterations)
            where F : struct, IfloatGradientFunction
        {
            if (g.N != x.N)
                throw new ArgumentException("gradientDescent: g.N must equal x.N");

            if (maxIter < 1)
                throw new ArgumentException("gradientDescent: maxIter must be >= 1");

            iterations = 0;

            for (int i = 0; i < maxIter; i++) {
                f.Gradient(in x, ref g);

                if (floatNormsOP.L2(in g) <= gradTol)
                    return true;

                for (int j = 0; j < x.N; j++)
                    x[j] -= learningRate * g[j];

                iterations++;
            }

            // Final convergence check: compute fresh gradient at returned x
            f.Gradient(in x, ref g);
            return floatNormsOP.L2(in g) <= gradTol;
        }
    }
}
