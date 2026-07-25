using Unity.Mathematics;

namespace BULA
{
    /// <summary>
    /// Easing curves as tiny Burst struct-functors (each : IfProxyScalarFunction). Map t∈[0,1]→[0,1]
    /// (the Back/Elastic variants overshoot outside [0,1]). Use standalone
    /// (<c>new fProxyEasing.SmoothStep().Eval((fProxy)0.3)</c>) or bake a LUT via
    /// <c>Generate.sample</c>. fProxy-only.
    /// </summary>
    public static partial class fProxyEasing
    {
        public struct Linear : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t) => t;
        }

        public struct SmoothStep : IfProxyScalarFunction
        {
            // 3t² - 2t³ (Hermite), zero first derivative at both ends.
            public fProxy Eval(fProxy t) => t * t * ((fProxy)3 - (fProxy)2 * t);
        }

        public struct SmootherStep : IfProxyScalarFunction
        {
            // 6t⁵ - 15t⁴ + 10t³ (Ken Perlin), zero first AND second derivative at both ends.
            public fProxy Eval(fProxy t) => t * t * t * (t * (t * (fProxy)6 - (fProxy)15) + (fProxy)10);
        }

        // ---- Quadratic ----
        public struct EaseInQuad : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t) => t * t;
        }
        public struct EaseOutQuad : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t) => t * ((fProxy)2 - t); // 1-(1-t)²
        }
        public struct EaseInOutQuad : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                if (t < (fProxy)0.5) return (fProxy)2 * t * t;
                fProxy u = (fProxy)(-2) * t + (fProxy)2;
                return (fProxy)1 - u * u * (fProxy)0.5;
            }
        }

        // ---- Cubic ----
        public struct EaseInCubic : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t) => t * t * t;
        }
        public struct EaseOutCubic : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                fProxy u = (fProxy)1 - t;
                return (fProxy)1 - u * u * u;
            }
        }
        public struct EaseInOutCubic : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                if (t < (fProxy)0.5) return (fProxy)4 * t * t * t;
                fProxy u = (fProxy)(-2) * t + (fProxy)2;
                return (fProxy)1 - u * u * u * (fProxy)0.5;
            }
        }

        // ---- Quartic ----
        public struct EaseInQuart : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t) => t * t * t * t;
        }
        public struct EaseOutQuart : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                fProxy u = (fProxy)1 - t;
                return (fProxy)1 - u * u * u * u;
            }
        }
        public struct EaseInOutQuart : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                if (t < (fProxy)0.5) return (fProxy)8 * t * t * t * t;
                fProxy u = (fProxy)(-2) * t + (fProxy)2;
                return (fProxy)1 - u * u * u * u * (fProxy)0.5;
            }
        }

        // ---- Sine ----
        public struct EaseInSine : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t) => (fProxy)1 - DetMath.Cos(t * (fProxy)(System.Math.PI * 0.5));
        }
        public struct EaseOutSine : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t) => DetMath.Sin(t * (fProxy)(System.Math.PI * 0.5));
        }
        public struct EaseInOutSine : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t) => (fProxy)(-0.5) * (DetMath.Cos((fProxy)System.Math.PI * t) - (fProxy)1);
        }

        // ---- Exponential ----
        public struct EaseInExpo : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                if (t <= (fProxy)0) return (fProxy)0;
                return DetMath.Pow((fProxy)2, (fProxy)10 * t - (fProxy)10);
            }
        }
        public struct EaseOutExpo : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                if (t >= (fProxy)1) return (fProxy)1;
                return (fProxy)1 - DetMath.Pow((fProxy)2, (fProxy)(-10) * t);
            }
        }
        public struct EaseInOutExpo : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                if (t <= (fProxy)0) return (fProxy)0;
                if (t >= (fProxy)1) return (fProxy)1;
                if (t < (fProxy)0.5)
                    return (fProxy)0.5 * DetMath.Pow((fProxy)2, (fProxy)20 * t - (fProxy)10);
                return (fProxy)1 - (fProxy)0.5 * DetMath.Pow((fProxy)2, (fProxy)(-20) * t + (fProxy)10);
            }
        }

        // ---- Bounce / Elastic / Back (overshoot) ----
        public struct EaseOutBounce : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                fProxy n1 = (fProxy)7.5625;
                fProxy d1 = (fProxy)2.75;

                if (t < (fProxy)1 / d1)
                    return n1 * t * t;
                if (t < (fProxy)2 / d1)
                {
                    t -= (fProxy)1.5 / d1;
                    return n1 * t * t + (fProxy)0.75;
                }
                if (t < (fProxy)2.5 / d1)
                {
                    t -= (fProxy)2.25 / d1;
                    return n1 * t * t + (fProxy)0.9375;
                }
                t -= (fProxy)2.625 / d1;
                return n1 * t * t + (fProxy)0.984375;
            }
        }

        public struct EaseInElastic : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                if (t <= (fProxy)0) return (fProxy)0;
                if (t >= (fProxy)1) return (fProxy)1;
                fProxy c4 = (fProxy)(2.0 * System.Math.PI / 3.0);
                return -DetMath.Pow((fProxy)2, (fProxy)10 * t - (fProxy)10)
                       * DetMath.Sin(((fProxy)10 * t - (fProxy)10.75) * c4);
            }
        }
        public struct EaseOutElastic : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                if (t <= (fProxy)0) return (fProxy)0;
                if (t >= (fProxy)1) return (fProxy)1;
                fProxy c4 = (fProxy)(2.0 * System.Math.PI / 3.0);
                return DetMath.Pow((fProxy)2, (fProxy)(-10) * t)
                       * DetMath.Sin(((fProxy)10 * t - (fProxy)0.75) * c4) + (fProxy)1;
            }
        }

        public struct EaseInBack : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                fProxy c1 = (fProxy)1.70158;
                fProxy c3 = c1 + (fProxy)1;
                return c3 * t * t * t - c1 * t * t;
            }
        }
        public struct EaseOutBack : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                fProxy c1 = (fProxy)1.70158;
                fProxy c3 = c1 + (fProxy)1;
                fProxy u = t - (fProxy)1;
                return (fProxy)1 + c3 * u * u * u + c1 * u * u;
            }
        }

        // EaseOutBounce reflected: easeInBounce(t) = 1 - easeOutBounce(1 - t).
        public struct EaseInBounce : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t) => (fProxy)1 - new EaseOutBounce().Eval((fProxy)1 - t);
        }
        public struct EaseInOutBounce : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                var ob = new EaseOutBounce();
                if (t < (fProxy)0.5)
                    return ((fProxy)1 - ob.Eval((fProxy)1 - (fProxy)2 * t)) * (fProxy)0.5;
                return ((fProxy)1 + ob.Eval((fProxy)2 * t - (fProxy)1)) * (fProxy)0.5;
            }
        }

        public struct EaseInOutElastic : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                if (t <= (fProxy)0) return (fProxy)0;
                if (t >= (fProxy)1) return (fProxy)1;
                fProxy c5 = (fProxy)(2.0 * System.Math.PI / 4.5);
                fProxy s = DetMath.Sin(((fProxy)20 * t - (fProxy)11.125) * c5);
                if (t < (fProxy)0.5)
                    return -(DetMath.Pow((fProxy)2, (fProxy)20 * t - (fProxy)10) * s) * (fProxy)0.5;
                return (DetMath.Pow((fProxy)2, (fProxy)(-20) * t + (fProxy)10) * s) * (fProxy)0.5 + (fProxy)1;
            }
        }

        public struct EaseInOutBack : IfProxyScalarFunction
        {
            public fProxy Eval(fProxy t)
            {
                fProxy c1 = (fProxy)1.70158;
                fProxy c2 = c1 * (fProxy)1.525;
                if (t < (fProxy)0.5)
                {
                    fProxy u = (fProxy)2 * t;
                    return (u * u * ((c2 + (fProxy)1) * u - c2)) * (fProxy)0.5;
                }
                else
                {
                    fProxy u = (fProxy)2 * t - (fProxy)2;
                    return (u * u * ((c2 + (fProxy)1) * u + c2) + (fProxy)2) * (fProxy)0.5;
                }
            }
        }
    }
}
