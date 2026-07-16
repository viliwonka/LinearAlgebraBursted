using System.Runtime.CompilerServices;

using Unity.Burst;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // Deterministic transcendental functions. Every routine uses only + - * / , math.floor,
    // math.sqrt, math.select and integer/bit reinterpretation — all IEEE correctly-rounded and
    // non-reassociating under Burst FloatMode.Strict — so results are bit-identical across CPU
    // architectures for a fixed Burst version, unlike math.exp/sin/... Accuracy is a few ULP.
    // Argument reduction is Cody-Waite: exp/log cover the whole finite range; sin/cos/tan are
    // accurate for |x| up to the point Payne-Hanek would be required (~1e7 float / ~1e15 double).
    // Edge cases are total and branch-free (computed then masked, preserving auto-vectorization).
    public static partial class DetMath
    {
        // exp argument reduction: x = n·ln2 + r, |r| <= ln2/2, ln2 split hi/lo so n·hi is exact.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void ExpReduce(fProxy x, out fProxy n, out fProxy r)
        {
            //+skipFor[double]
            const float INV_LN2 = 1.4426950408889634f;
            const float HI = 0.693359375f;
            const float LO = -2.12194440e-4f;
            //-skipFor
            //+emitFor[double]
            //!const double INV_LN2 = 1.4426950408889634;
            //!const double HI = 6.93147180369123816490e-01;
            //!const double LO = 1.90821492927058770002e-10;
            //-emitFor
            n = math.floor(INV_LN2 * x + (fProxy)0.5);
            r = x - n * HI;
            r = r - n * LO;
        }

        // 2^n assembled from the IEEE exponent field (ldexp as integer bit ops).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static fProxy Ldexp(fProxy mant, fProxy n)
        {
            //+skipFor[double]
            int e = (int)n;
            return mant * math.asfloat((e + 127) << 23);
            //-skipFor
            //+emitFor[double]
            //!long e = (long)n;
            //!return mant * math.asdouble((e + 1023L) << 52);
            //-emitFor
        }

        // exp overflow/underflow guard: x > OV -> +inf (also x = +inf), x < UN -> 0 (also x = -inf),
        // NaN -> NaN.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static fProxy ExpGuard(fProxy x, fProxy y)
        {
            //+skipFor[double]
            const float OV = 88.72284f, UN = -87.33655f, INF = float.PositiveInfinity, NANV = float.NaN;
            //-skipFor
            //+emitFor[double]
            //!const double OV = 709.782712893384, UN = -708.3964185322641, INF = double.PositiveInfinity, NANV = double.NaN;
            //-emitFor
            y = math.select(y, (fProxy)INF,  x > OV);
            y = math.select(y, (fProxy)0,    x < UN);
            y = math.select(y, (fProxy)NANV, x != x);
            return y;
        }

        // e^x. Total: overflow -> +inf, underflow -> 0, NaN -> NaN.
        public static fProxy Exp(fProxy x)
        {
            ExpReduce(x, out fProxy n, out fProxy r);
            //+skipFor[double]
            float p = 1.38368463709141461e-03f;
            p = p * r + 8.37481579955782172e-03f;
            p = p * r + 4.16682255624844372e-02f;
            p = p * r + 1.66664201699263076e-01f;
            p = p * r + 4.99999920798497477e-01f;
            p = p * r + 1.00000003632318291e+00f;
            p = p * r + 1.00000000055416338e+00f;
            //-skipFor
            //+emitFor[double]
            //!double p = 2.4994305002802721e-08;
            //!p = p * r + 2.7632293277020833e-07;
            //!p = p * r + 2.7557622530418205e-06;
            //!p = p * r + 2.4801486521436422e-05;
            //!p = p * r + 1.9841269432679901e-04;
            //!p = p * r + 1.3888888951223976e-03;
            //!p = p * r + 8.3333333335592709e-03;
            //!p = p * r + 4.1666666666492767e-02;
            //!p = p * r + 1.6666666666666169e-01;
            //!p = p * r + 5.0000000000000177e-01;
            //!p = p * r + 1.0000000000000000e+00;
            //!p = p * r + 1.0000000000000000e+00;
            //-emitFor
            return ExpGuard(x, Ldexp(p, n));
        }

        // 2^x.
        public static fProxy Exp2(fProxy x)  => Exp(x * (fProxy)0.6931471805599453);
        // 10^x.
        public static fProxy Exp10(fProxy x) => Exp(x * (fProxy)2.302585092994046);

        // log domain guard: log(0) = -inf, log(+inf) = +inf, x < 0 -> NaN, NaN -> NaN.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static fProxy LogGuard(fProxy x, fProxy y)
        {
            //+skipFor[double]
            const float NINF = float.NegativeInfinity, PINF = float.PositiveInfinity, NANV = float.NaN;
            //-skipFor
            //+emitFor[double]
            //!const double NINF = double.NegativeInfinity, PINF = double.PositiveInfinity, NANV = double.NaN;
            //-emitFor
            y = math.select(y, (fProxy)NINF, x == (fProxy)0);
            y = math.select(y, (fProxy)PINF, x == (fProxy)PINF);
            y = math.select(y, (fProxy)NANV, x < (fProxy)0);
            y = math.select(y, (fProxy)NANV, x != x);
            return y;
        }

        // Natural log. Total: log(0) = -inf, log(+inf) = +inf, x < 0 or NaN -> NaN. Subnormal inputs
        // are scaled into the normal range first (branch-free).
        public static fProxy Log(fProxy x)
        {
            //+skipFor[double]
            const float SQRT2 = 1.4142135623730951f;
            const float LN2_HI = 0.693115234375f;
            const float LN2_LO = 3.194618329871446e-05f;
            bool sub = x < 1.1754944e-38f;                           // below smallest normal (0/subnormal)
            float xs = math.select(x, x * 8388608f, sub);           // scale by 2^23 into normal range
            int i = math.asint(xs);
            int e = ((i >> 23) & 0xFF) - 127 - math.select(0, 23, sub);
            float m = math.asfloat((i & 0x007FFFFF) | 0x3F800000);   // mantissa in [1,2)
            float adj = math.select(0f, 1f, m > SQRT2);
            m = m - adj * (m * 0.5f);                                // if m>sqrt2: halve -> [sqrt2/2,sqrt2)
            float ef = (float)e + adj;
            float f = m - 1f;
            float s = f / (2f + f);
            float z = s * s;
            float b = 1.49762394462376722e-1f;
            b = b * z + 1.99868990328974250e-1f;
            b = b * z + 3.33334124209226285e-1f;
            b = b * z + 9.99999999255582188e-1f;
            float logm = (2f * s) * b;
            float y = ef * LN2_HI + (logm + ef * LN2_LO);
            return LogGuard(x, y);
            //-skipFor
            //+emitFor[double]
            //!const double SQRT2 = 1.4142135623730951;
            //!const double LN2_HI = 0.6931471675634384;
            //!const double LN2_LO = 1.2996506893889889e-08;
            //!bool sub = x < 2.2250738585072014e-308;               // below smallest normal (0/subnormal)
            //!double xs = math.select(x, x * 4503599627370496.0, sub); // scale by 2^52 into normal range
            //!long l = math.aslong(xs);
            //!int e = (int)((l >> 52) & 0x7FF) - 1023 - math.select(0, 52, sub);
            //!double m = math.asdouble((l & 0x000FFFFFFFFFFFFFL) | 0x3FF0000000000000L);
            //!double adj = math.select(0.0, 1.0, m > SQRT2);
            //!m = m - adj * (m * 0.5);
            //!double ef = (double)e + adj;
            //!double f = m - 1.0;
            //!double s = f / (2.0 + f);
            //!double z = s * s;
            //!double b = 7.42033651975721971e-02;
            //!b = b * z + 7.65476597014954508e-02;
            //!b = b * z + 9.09187247444205753e-02;
            //!b = b * z + 1.11110974736367899e-01;
            //!b = b * z + 1.42857143903260792e-01;
            //!b = b * z + 1.99999999996063942e-01;
            //!b = b * z + 3.33333333333338971e-01;
            //!b = b * z + 9.99999999999999999e-01;
            //!double logm = (2.0 * s) * b;
            //!double y = ef * LN2_HI + (logm + ef * LN2_LO);
            //!return LogGuard(x, y);
            //-emitFor
        }

        // log2(x) / log10(x). Same domain as Log.
        public static fProxy Log2(fProxy x)  => Log(x) * (fProxy)1.4426950408889634;
        public static fProxy Log10(fProxy x) => Log(x) * (fProxy)0.4342944819032518;

        // x^y for x > 0, via exp(y·log x). x = 0 gives 0 for y > 0, +inf for y < 0; x < 0 gives NaN.
        // y = 0 gives 1 for ANY x (including 0^0 = 1), matching IEEE pow.
        public static fProxy Pow(fProxy x, fProxy y) => math.select(Exp(y * Log(x)), (fProxy)1, y == (fProxy)0);

        // x^n for INTEGER n, via exponentiation by squaring (+ - * / only, deterministic). Exact
        // reduction order; handles negative x (sign follows the multiplies). n < 0 gives 1/x^|n|.
        public static fProxy Pow(fProxy x, int n)
        {
            bool neg = n < 0;
            uint u = (uint)(neg ? -n : n);
            fProxy r = (fProxy)1;
            fProxy p = x;
            while (u != 0u)
            {
                if ((u & 1u) != 0u) r *= p;
                p *= p;
                u >>= 1;
            }
            return neg ? (fProxy)1 / r : r;
        }

        // trig non-finite guard: sin/cos of +-inf or NaN -> NaN.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static fProxy TrigGuard(fProxy x, fProxy y)
        {
            //+skipFor[double]
            const float INF = float.PositiveInfinity, NANV = float.NaN;
            //-skipFor
            //+emitFor[double]
            //!const double INF = double.PositiveInfinity, NANV = double.NaN;
            //-emitFor
            return math.select(y, (fProxy)NANV, (x != x) | (math.abs(x) == (fProxy)INF));
        }

        // trig reduction: x = q·(pi/2) + r, |r| <= pi/4, via 3-word Cody-Waite pi/2. quad = q & 3.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void ReduceTrig(fProxy x, out fProxy r, out int quad)
        {
            //+skipFor[double]
            const float T2_PI = 0.6366197723675814f;
            const float P1 = 1.5703125f;
            const float P2 = 4.83751297e-04f;
            const float P3 = 7.5497901e-08f;
            //-skipFor
            //+emitFor[double]
            //!const double T2_PI = 0.6366197723675814;
            //!const double P1 = 1.5707963267341256;
            //!const double P2 = 6.07710050359346e-11;
            //!const double P3 = 2.912732056093356e-20;
            //-emitFor
            fProxy kf = math.floor(x * T2_PI + (fProxy)0.5);
            r = x - kf * P1;
            r = r - kf * P2;
            r = r - kf * P3;
            quad = (int)kf & 3;
        }

        // sin(r), |r| <= pi/4: odd minimax sin(r) = r·P(r^2).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static fProxy SinPoly(fProxy r)
        {
            fProxy u = r * r;
            //+skipFor[double]
            float p = -1.95018220122293565e-4f;
            p = p * u + 8.33201645305208673e-3f;
            p = p * u + -1.66666502242394270e-1f;
            p = p * u + 9.99999996761798119e-1f;
            //-skipFor
            //+emitFor[double]
            //!double p = 1.58941363709740611e-10;
            //!p = p * u + -2.50507058463590353e-08;
            //!p = p * u + 2.75573132990149026e-06;
            //!p = p * u + -1.98412698284021293e-04;
            //!p = p * u + 8.33333333332000244e-03;
            //!p = p * u + -1.66666666666666148e-01;
            //!p = p * u + 9.99999999999999997e-01;
            //-emitFor
            return r * p;
        }

        // cos(r), |r| <= pi/4: even minimax cos(r) = Q(r^2).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static fProxy CosPoly(fProxy r)
        {
            fProxy u = r * r;
            //+skipFor[double]
            float q = 2.43726791799261521e-5f;
            q = q * u + -1.38865291471812224e-3f;
            q = q * u + 4.16666132334750667e-2f;
            q = q * u + -4.99999995715568754e-1f;
            q = q * u + 9.99999999943937325e-1f;
            //-skipFor
            //+emitFor[double]
            //!double q = -1.13521232066215503e-11;
            //!q = q * u + 2.08755551456743361e-09;
            //!q = q * u + -2.75573128656958583e-07;
            //!q = q * u + 2.48015872828994611e-05;
            //!q = q * u + -1.38888888888589604e-03;
            //!q = q * u + 4.16666666666664303e-02;
            //!q = q * u + -4.99999999999999993e-01;
            //!q = q * u + 1.00000000000000000e+00;
            //-emitFor
            return q;
        }

        // sin(x). |x| beyond ~1e7 (float) / ~1e15 (double) loses reduction accuracy; +-inf/NaN -> NaN.
        public static fProxy Sin(fProxy x)
        {
            ReduceTrig(x, out fProxy r, out int quad);
            fProxy s = SinPoly(r);
            fProxy c = CosPoly(r);
            fProxy sw = (fProxy)(quad & 1);                       // q odd -> use cos poly
            fProxy baseS = s + sw * (c - s);
            fProxy sign = (fProxy)1 - (fProxy)2 * (fProxy)((quad >> 1) & 1);
            return TrigGuard(x, sign * baseS);
        }

        // cos(x). Same range/edge behavior as Sin.
        public static fProxy Cos(fProxy x)
        {
            ReduceTrig(x, out fProxy r, out int quad);
            fProxy s = SinPoly(r);
            fProxy c = CosPoly(r);
            fProxy sw = (fProxy)(quad & 1);
            fProxy baseC = c + sw * (s - c);
            fProxy sign = (fProxy)1 - (fProxy)2 * (fProxy)(((quad + 1) >> 1) & 1);
            return TrigGuard(x, sign * baseC);
        }

        // sin and cos of x from ONE shared reduction (cheaper than Sin(x) + Cos(x)).
        public static void SinCos(fProxy x, out fProxy sinx, out fProxy cosx)
        {
            ReduceTrig(x, out fProxy r, out int quad);
            fProxy s = SinPoly(r);
            fProxy c = CosPoly(r);
            fProxy sw = (fProxy)(quad & 1);
            fProxy baseS = s + sw * (c - s);
            fProxy baseC = c + sw * (s - c);
            fProxy sSign = (fProxy)1 - (fProxy)2 * (fProxy)((quad >> 1) & 1);
            fProxy cSign = (fProxy)1 - (fProxy)2 * (fProxy)(((quad + 1) >> 1) & 1);
            sinx = TrigGuard(x, sSign * baseS);
            cosx = TrigGuard(x, cSign * baseC);
        }

        // tan(x) = sin(x)/cos(x). Poles (odd multiples of pi/2) return +-inf/large per the divide.
        public static fProxy Tan(fProxy x) { SinCos(x, out fProxy s, out fProxy c); return s / c; }

        // atan(x), any real x (total: +-inf -> +-pi/2, NaN -> NaN).
        public static fProxy Atan(fProxy x)
        {
            //+skipFor[double]
            const float TAN_PI_8 = 0.41421356237309503f;
            const float PI_4 = 0.7853981633974483f;
            const float PI_2 = 1.5707963267948966f;
            //-skipFor
            //+emitFor[double]
            //!const double TAN_PI_8 = 0.41421356237309503;
            //!const double PI_4 = 0.7853981633974483;
            //!const double PI_2 = 1.5707963267948966;
            //-emitFor
            fProxy ax = math.abs(x);
            bool big = ax > (fProxy)1;
            fProxy xx = math.select(ax, (fProxy)1 / ax, big);          // fold to [0,1]
            bool s1 = xx > TAN_PI_8;
            fProxy xr = math.select(xx, (xx - (fProxy)1) / (xx + (fProxy)1), s1);
            fProxy segbase = math.select((fProxy)0, PI_4, s1);
            fProxy u = xr * xr;
            //+skipFor[double]
            float p = 7.96193782326451162e-02f;
            p = p * u + -1.38446124780312687e-01f;
            p = p * u + 1.99737806986236897e-01f;
            p = p * u + -3.33327794114148066e-01f;
            p = p * u + 9.99999981144094623e-01f;
            //-skipFor
            //+emitFor[double]
            //!double p = 2.10096493006257428e-02;
            //!p = p * u + -4.33790372071600680e-02;
            //!p = p * u + 5.68488398630815686e-02;
            //!p = p * u + -6.63958241019000515e-02;
            //!p = p * u + 7.68987994100881424e-02;
            //!p = p * u + -9.09076797385649291e-02;
            //!p = p * u + 1.11111059670532823e-01;
            //!p = p * u + -1.42857141759054819e-01;
            //!p = p * u + 1.99999999987947714e-01;
            //!p = p * u + -3.33333333333281651e-01;
            //!p = p * u + 9.99999999999999963e-01;
            //-emitFor
            fProxy inner = segbase + xr * p;
            fProxy res = math.select(inner, PI_2 - inner, big);        // undo the 1/x fold
            return math.select(res, -res, x < (fProxy)0);              // odd function
        }

        // atan2(y, x): angle in (-pi, pi]. Branch-free quadrant fix-up; x = 0 (y != 0) folds via
        // +-inf into Atan; the origin (0,0) returns 0.
        public static fProxy Atan2(fProxy y, fProxy x)
        {
            fProxy PI = (fProxy)3.141592653589793;
            fProxy a = Atan(y / x);
            a = math.select(a, a + PI, (x < (fProxy)0) & (y >= (fProxy)0));
            a = math.select(a, a - PI, (x < (fProxy)0) & (y <  (fProxy)0));
            a = math.select(a, (fProxy)0, (x == (fProxy)0) & (y == (fProxy)0));  // 0/0 -> NaN -> 0
            return a;
        }

        // asin(x) / acos(x), |x| <= 1.
        public static fProxy Asin(fProxy x) => Atan(x / math.sqrt((fProxy)1 - x * x));
        public static fProxy Acos(fProxy x) => (fProxy)1.5707963267948966 - Asin(x);

        // sinh and cosh from ONE exp + one reciprocal.
        public static void SinhCosh(fProxy x, out fProxy sh, out fProxy ch)
        {
            fProxy e = Exp(x);
            fProxy er = (fProxy)1 / e;
            sh = (e - er) * (fProxy)0.5;
            ch = (e + er) * (fProxy)0.5;
        }
        public static fProxy Sinh(fProxy x) { SinhCosh(x, out fProxy sh, out fProxy ch); return sh; }
        public static fProxy Cosh(fProxy x) { SinhCosh(x, out fProxy sh, out fProxy ch); return ch; }
        // tanh(x) via e^{2|x|}; saturates to +-1 for large |x| (no inf/inf), odd. Total: NaN -> NaN.
        public static fProxy Tanh(fProxy x)
        {
            fProxy ax = math.abs(x);
            fProxy e = Exp((fProxy)2 * ax);                      // >= 1; overflows to +inf for large |x|
            fProxy t = (fProxy)1 - (fProxy)2 / (e + (fProxy)1);  // 2/(inf+1) = 0 -> t = 1 (no inf/inf)
            return math.select(t, -t, x < (fProxy)0);            // odd
        }
        // acosh(x), x >= 1.
        public static fProxy Acosh(fProxy x) => Log(x + math.sqrt(x * x - (fProxy)1));
    }
}
