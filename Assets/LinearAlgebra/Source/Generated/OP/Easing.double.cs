#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using Unity.Mathematics;

namespace LinearAlgebra
{
    /// <summary>
    /// Easing curves as tiny Burst struct-functors (each : IdoubleScalarFunction). Map t∈[0,1]→[0,1]
    /// (the Back/Elastic variants overshoot outside [0,1]). Use standalone
    /// (<c>new doubleEasing.SmoothStep().Eval(0.3f)</c>) or bake a LUT via
    /// <c>doubleGenOP.sample</c> / <c>arena.doubleEasingLUT</c>. double-only.
    /// </summary>
    public static partial class doubleEasing
    {
        public struct Linear : IdoubleScalarFunction
        {
            public double Eval(double t) => t;
        }

        public struct SmoothStep : IdoubleScalarFunction
        {
            // 3t² - 2t³ (Hermite), zero first derivative at both ends.
            public double Eval(double t) => t * t * ((double)3 - (double)2 * t);
        }

        public struct SmootherStep : IdoubleScalarFunction
        {
            // 6t⁵ - 15t⁴ + 10t³ (Ken Perlin), zero first AND second derivative at both ends.
            public double Eval(double t) => t * t * t * (t * (t * (double)6 - (double)15) + (double)10);
        }

        // ---- Quadratic ----
        public struct EaseInQuad : IdoubleScalarFunction
        {
            public double Eval(double t) => t * t;
        }
        public struct EaseOutQuad : IdoubleScalarFunction
        {
            public double Eval(double t) => t * ((double)2 - t); // 1-(1-t)²
        }
        public struct EaseInOutQuad : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                if (t < (double)0.5) return (double)2 * t * t;
                double u = (double)(-2) * t + (double)2;
                return (double)1 - u * u * (double)0.5;
            }
        }

        // ---- Cubic ----
        public struct EaseInCubic : IdoubleScalarFunction
        {
            public double Eval(double t) => t * t * t;
        }
        public struct EaseOutCubic : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                double u = (double)1 - t;
                return (double)1 - u * u * u;
            }
        }
        public struct EaseInOutCubic : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                if (t < (double)0.5) return (double)4 * t * t * t;
                double u = (double)(-2) * t + (double)2;
                return (double)1 - u * u * u * (double)0.5;
            }
        }

        // ---- Quartic ----
        public struct EaseInQuart : IdoubleScalarFunction
        {
            public double Eval(double t) => t * t * t * t;
        }
        public struct EaseOutQuart : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                double u = (double)1 - t;
                return (double)1 - u * u * u * u;
            }
        }
        public struct EaseInOutQuart : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                if (t < (double)0.5) return (double)8 * t * t * t * t;
                double u = (double)(-2) * t + (double)2;
                return (double)1 - u * u * u * u * (double)0.5;
            }
        }

        // ---- Sine ----
        public struct EaseInSine : IdoubleScalarFunction
        {
            public double Eval(double t) => (double)1 - math.cos(t * (double)(System.Math.PI * 0.5));
        }
        public struct EaseOutSine : IdoubleScalarFunction
        {
            public double Eval(double t) => math.sin(t * (double)(System.Math.PI * 0.5));
        }
        public struct EaseInOutSine : IdoubleScalarFunction
        {
            public double Eval(double t) => (double)(-0.5) * (math.cos((double)System.Math.PI * t) - (double)1);
        }

        // ---- Exponential ----
        public struct EaseInExpo : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                if (t <= (double)0) return (double)0;
                return math.pow((double)2, (double)10 * t - (double)10);
            }
        }
        public struct EaseOutExpo : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                if (t >= (double)1) return (double)1;
                return (double)1 - math.pow((double)2, (double)(-10) * t);
            }
        }
        public struct EaseInOutExpo : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                if (t <= (double)0) return (double)0;
                if (t >= (double)1) return (double)1;
                if (t < (double)0.5)
                    return (double)0.5 * math.pow((double)2, (double)20 * t - (double)10);
                return (double)1 - (double)0.5 * math.pow((double)2, (double)(-20) * t + (double)10);
            }
        }

        // ---- Bounce / Elastic / Back (overshoot) ----
        public struct EaseOutBounce : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                double n1 = (double)7.5625;
                double d1 = (double)2.75;

                if (t < (double)1 / d1)
                    return n1 * t * t;
                if (t < (double)2 / d1)
                {
                    t -= (double)1.5 / d1;
                    return n1 * t * t + (double)0.75;
                }
                if (t < (double)2.5 / d1)
                {
                    t -= (double)2.25 / d1;
                    return n1 * t * t + (double)0.9375;
                }
                t -= (double)2.625 / d1;
                return n1 * t * t + (double)0.984375;
            }
        }

        public struct EaseInElastic : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                if (t <= (double)0) return (double)0;
                if (t >= (double)1) return (double)1;
                double c4 = (double)(2.0 * System.Math.PI / 3.0);
                return -math.pow((double)2, (double)10 * t - (double)10)
                       * math.sin(((double)10 * t - (double)10.75) * c4);
            }
        }
        public struct EaseOutElastic : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                if (t <= (double)0) return (double)0;
                if (t >= (double)1) return (double)1;
                double c4 = (double)(2.0 * System.Math.PI / 3.0);
                return math.pow((double)2, (double)(-10) * t)
                       * math.sin(((double)10 * t - (double)0.75) * c4) + (double)1;
            }
        }

        public struct EaseInBack : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                double c1 = (double)1.70158;
                double c3 = c1 + (double)1;
                return c3 * t * t * t - c1 * t * t;
            }
        }
        public struct EaseOutBack : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                double c1 = (double)1.70158;
                double c3 = c1 + (double)1;
                double u = t - (double)1;
                return (double)1 + c3 * u * u * u + c1 * u * u;
            }
        }

        // EaseOutBounce reflected: easeInBounce(t) = 1 - easeOutBounce(1 - t).
        public struct EaseInBounce : IdoubleScalarFunction
        {
            public double Eval(double t) => (double)1 - new EaseOutBounce().Eval((double)1 - t);
        }
        public struct EaseInOutBounce : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                var ob = new EaseOutBounce();
                if (t < (double)0.5)
                    return ((double)1 - ob.Eval((double)1 - (double)2 * t)) * (double)0.5;
                return ((double)1 + ob.Eval((double)2 * t - (double)1)) * (double)0.5;
            }
        }

        public struct EaseInOutElastic : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                if (t <= (double)0) return (double)0;
                if (t >= (double)1) return (double)1;
                double c5 = (double)(2.0 * System.Math.PI / 4.5);
                double s = math.sin(((double)20 * t - (double)11.125) * c5);
                if (t < (double)0.5)
                    return -(math.pow((double)2, (double)20 * t - (double)10) * s) * (double)0.5;
                return (math.pow((double)2, (double)(-20) * t + (double)10) * s) * (double)0.5 + (double)1;
            }
        }

        public struct EaseInOutBack : IdoubleScalarFunction
        {
            public double Eval(double t)
            {
                double c1 = (double)1.70158;
                double c2 = c1 * (double)1.525;
                if (t < (double)0.5)
                {
                    double u = (double)2 * t;
                    return (u * u * ((c2 + (double)1) * u - c2)) * (double)0.5;
                }
                else
                {
                    double u = (double)2 * t - (double)2;
                    return (u * u * ((c2 + (double)1) * u + c2) + (double)2) * (double)0.5;
                }
            }
        }
    }
}
