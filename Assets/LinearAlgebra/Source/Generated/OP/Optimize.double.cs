#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // IdoubleScalarFunction (the shared scalar-curve functor — the Burst "lambda" used by the
    // optimizers AND the generators) lives in Interfaces/ScalarFunction.double.cs. The derivative /
    // gradient functor interfaces below are optimizer-specific and stay here.

    public interface IdoubleScalarDerivativeFunction : IdoubleScalarFunction {
        double Derivative(double x);
    }

    public interface IdoubleGradientFunction {
        double Eval(in doubleN x);
        void Gradient(in doubleN x, ref doubleN g);
    }

    public static partial class Optimize {

        /// <summary>
        /// Bracketing root find. Requires f(lo) and f(hi) to have opposite signs; returns false if not bracketed (root = the better endpoint).
        /// Converges when (hi - lo) &lt;= xTol + rTol * |mid| or f(mid) == 0. root = final midpoint. Returns true on convergence.
        /// xTol is absolute, rTol relative to the current midpoint; rTol &gt;= a few * Consts.doubleEpsilon
        /// keeps the criterion reachable regardless of the root's magnitude.
        /// </summary>
        public static bool bisection<F>(ref F f, double lo, double hi, out double root,
                                        double xTol, double rTol, int maxIter)
            where F : struct, IdoubleScalarFunction
        {
            if (maxIter < 1)
                throw new ArgumentException("bisection: maxIter must be >= 1");

            double flo = f.Eval(lo);
            double fhi = f.Eval(hi);

            if (flo == (double)0) { root = lo; return true; }
            if (fhi == (double)0) { root = hi; return true; }

            // Same sign → not bracketed (product test replaced to avoid underflow/NaN misjudgement)
            if ((flo > (double)0) == (fhi > (double)0)) {
                root = (math.abs(flo) <= math.abs(fhi)) ? lo : hi;
                return false;
            }

            for (int i = 0; i < maxIter; i++) {
                double mid = lo + (hi - lo) * (double)0.5;
                double fmid = f.Eval(mid);

                if (fmid == (double)0) { root = mid; return true; }
                if ((hi - lo) <= xTol + rTol * math.abs(mid)) { root = mid; return true; }

                // Sign changes between lo and mid → root in [lo, mid]
                if ((flo > (double)0) != (fmid > (double)0)) {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fmid;
                }
            }

            root = lo + (hi - lo) * (double)0.5;
            return (hi - lo) <= xTol + rTol * math.abs(root);
        }

        /// <summary>bisection with absolute tolerance only (rTol = 0).</summary>
        public static bool bisection<F>(ref F f, double lo, double hi, out double root,
                                        double xTol, int maxIter)
            where F : struct, IdoubleScalarFunction
            => bisection(ref f, lo, hi, out root, xTol, (double)0, maxIter);

        /// <summary>bisection with default maxIter (200).</summary>
        public static bool bisection<F>(ref F f, double lo, double hi, out double root,
                                        double xTol)
            where F : struct, IdoubleScalarFunction
            => bisection(ref f, lo, hi, out root, xTol, (double)0, 200);

        /// <summary>bisection with default xTol (Consts.doubleZeroTreshold), rTol (4 * Consts.doubleEpsilon) and maxIter (200).</summary>
        public static bool bisection<F>(ref F f, double lo, double hi, out double root)
            where F : struct, IdoubleScalarFunction
            => bisection(ref f, lo, hi, out root, Consts.doubleZeroTreshold, (double)4 * Consts.doubleEpsilon, 200);

        /// <summary>
        /// Newton root find. Converged when |f(x)| &lt;= fTol. Returns false if |f'(x)| &lt; Consts.doubleZeroTreshold (flat/badly-scaled, absolute guard) or maxIter exhausted; root holds last iterate.
        /// </summary>
        public static bool newtonRoot<F>(ref F f, double x0, out double root,
                                         double fTol, int maxIter)
            where F : struct, IdoubleScalarDerivativeFunction
        {
            if (maxIter < 1)
                throw new ArgumentException("newtonRoot: maxIter must be >= 1");

            double x = x0;

            for (int i = 0; i < maxIter; i++) {
                double fx = f.Eval(x);
                if (math.abs(fx) <= fTol) { root = x; return true; }

                double d = f.Derivative(x);
                if (math.abs(d) < Consts.doubleZeroTreshold) { root = x; return false; }

                x = x - fx / d;
            }

            root = x;
            return math.abs(f.Eval(x)) <= fTol;
        }

        /// <summary>newtonRoot with default maxIter (100).</summary>
        public static bool newtonRoot<F>(ref F f, double x0, out double root,
                                         double fTol)
            where F : struct, IdoubleScalarDerivativeFunction
            => newtonRoot(ref f, x0, out root, fTol, 100);

        /// <summary>newtonRoot with default fTol (Consts.doubleZeroTreshold) and maxIter (100).</summary>
        public static bool newtonRoot<F>(ref F f, double x0, out double root)
            where F : struct, IdoubleScalarDerivativeFunction
            => newtonRoot(ref f, x0, out root, Consts.doubleZeroTreshold, 100);

        /// <summary>
        /// Golden-section minimization of unimodal f on [a, b]. xMin = midpoint of final bracket. Returns true when (b - a) &lt;= xTol within maxIter.
        /// xTol is an absolute tolerance on the bracket width, rTol relative to the bracket midpoint.
        /// Note: a smooth minimum can only be localized to ~|xMin| * sqrt(machine eps)
        /// (Consts.doubleSqrtEps), no matter how small the tolerances are.
        /// </summary>
        public static bool goldenSection<F>(ref F f, double a, double b, out double xMin,
                                            double xTol, double rTol, int maxIter)
            where F : struct, IdoubleScalarFunction
        {
            if (maxIter < 1)
                throw new ArgumentException("goldenSection: maxIter must be >= 1");

            if (a > b) {
                double tmp = a;
                a = b;
                b = tmp;
            }

            double invphi = (math.sqrt((double)5) - (double)1) * (double)0.5;

            double c = b - invphi * (b - a);
            double d = a + invphi * (b - a);
            double fc = f.Eval(c);
            double fd = f.Eval(d);

            for (int i = 0; i < maxIter; i++) {
                if ((b - a) <= xTol + rTol * math.abs(a + (b - a) * (double)0.5)) break;

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

            xMin = a + (b - a) * (double)0.5;
            return (b - a) <= xTol + rTol * math.abs(xMin);
        }

        /// <summary>goldenSection with absolute tolerance only (rTol = 0).</summary>
        public static bool goldenSection<F>(ref F f, double a, double b, out double xMin,
                                            double xTol, int maxIter)
            where F : struct, IdoubleScalarFunction
            => goldenSection(ref f, a, b, out xMin, xTol, (double)0, maxIter);

        /// <summary>goldenSection with default maxIter (200).</summary>
        public static bool goldenSection<F>(ref F f, double a, double b, out double xMin,
                                            double xTol)
            where F : struct, IdoubleScalarFunction
            => goldenSection(ref f, a, b, out xMin, xTol, (double)0, 200);

        /// <summary>goldenSection with default xTol (Consts.doubleZeroTreshold), rTol (3 * Consts.doubleSqrtEps) and maxIter (200).</summary>
        public static bool goldenSection<F>(ref F f, double a, double b, out double xMin)
            where F : struct, IdoubleScalarFunction
            => goldenSection(ref f, a, b, out xMin, Consts.doubleZeroTreshold, (double)3 * Consts.doubleSqrtEps, 200);

        // Fixed-step gradient descent, in-place on x. g is caller-provided scratch (length x.N). Does NOT allocate.
        // Iterates x -= learningRate * g until L2(g) <= gradTol or maxIter. Returns true if gradTol reached; iterations = performed count.
        public static bool gradientDescent<F>(ref F f, ref doubleN x, ref doubleN g,
                                              double learningRate, double gradTol, int maxIter,
                                              out int iterations)
            where F : struct, IdoubleGradientFunction
        {
            if (g.N != x.N)
                throw new ArgumentException("gradientDescent: g.N must equal x.N");

            if (maxIter < 1)
                throw new ArgumentException("gradientDescent: maxIter must be >= 1");

            iterations = 0;

            for (int i = 0; i < maxIter; i++) {
                f.Gradient(in x, ref g);

                if (doubleNormsOP.L2(in g) <= gradTol)
                    return true;

                for (int j = 0; j < x.N; j++)
                    x[j] -= learningRate * g[j];

                iterations++;
            }

            // Final convergence check: compute fresh gradient at returned x
            f.Gradient(in x, ref g);
            return doubleNormsOP.L2(in g) <= gradTol;
        }
    }
}
