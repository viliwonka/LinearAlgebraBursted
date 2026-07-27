using System.Globalization;
using System.Runtime.CompilerServices;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using BULA;

#pragma warning disable 1718 // x != x is the NaN test, mirroring DetMath's own edge-case contract

namespace BULA.Benchmarks
{
    // GENERATED per-dtype half of DetMathBenchmark (prototype exp + timed IJobs + row builders). The
    // dtype-agnostic harness (size, section, headers) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/DetMathBenchmark.cs.

    // Prototype in-house exp — benchmark-only, NOT shipped. Deterministic by construction: Cody-Waite
    // argument reduction (x = n·ln2 + r), a polynomial for exp(r) on |r| <= ln2/2, and 2^n assembled
    // directly from the IEEE exponent bits. Only + - * and integer/bit ops — no libm/math.exp in the
    // path, so it is bit-identical across CPU architectures under FloatMode.Strict. Overflow/underflow
    // edge cases are NOT handled (inputs are bounded); a shipping DetMath.Exp would add them.
    internal static class DetMathProtoFProxy
    {
        // Cody-Waite argument reduction: x = n·ln2 + r, |r| <= ln2/2, with ln2 split hi/lo so
        // x - n·hi - n·lo carries no cancellation error. Float hi (0.693359375) and double hi (fdlibm
        // 0x3FE62E42FEE00000) both have their low mantissa bits zero, so n·hi is exact for the n here.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void Reduce(fProxy x, out fProxy n, out fProxy r)
        {
            //+skipFor[double]
            const float INV_LN2 = 1.4426950408889634f;
            const float HI = 0.693359375f;
            const float LO = -2.12194440e-4f;      // HI + LO == ln2
            //-skipFor
            //+emitFor[double]
            //!const double INV_LN2 = 1.4426950408889634;
            //!const double HI = 6.93147180369123816490e-01;
            //!const double LO = 1.90821492927058770002e-10;
            //-emitFor
            n = math.floor(INV_LN2 * x + (fProxy)0.5);
            r = x - n * HI;
            //+skipFor[double]
            r = r - n * LO;
            //-skipFor
            //+emitFor[double]
            //!r = r - n * LO;
            //-emitFor
        }

        // 2^n assembled directly from the IEEE exponent field (ldexp as integer bit ops). No edge-case
        // clamp — inputs are bounded so n stays well inside the exponent range.
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

        // Branch-free overflow/underflow guard shared by all exp variants. x > OV -> +inf (also handles
        // x = +inf), x < UN -> 0 (also x = -inf). NaN takes neither branch and propagates from the poly.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static fProxy ExpGuard(fProxy x, fProxy y)
        {
            //+skipFor[double]
            const float OV = 88.72284f, UN = -87.33655f, INF = float.PositiveInfinity;
            //-skipFor
            //+emitFor[double]
            //!const double OV = 709.782712893384, UN = -708.3964185322641, INF = double.PositiveInfinity;
            //-emitFor
            y = math.select(y, (fProxy)INF, x > OV);
            y = math.select(y, (fProxy)0,   x < UN);
            return y;
        }

        // Accurate, Horner: minimax poly for exp(r) directly (float degree 6 ~0.03 ULP fit; double
        // degree 11 ~0.03 ULP fit). Sequential Horner — one long dependency chain (throughput-friendly
        // once vectorized, latency-bound scalar).
        public static fProxy ExpAcc(fProxy x)
        {
            Reduce(x, out fProxy n, out fProxy r);
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

        // Accurate, Estrin: SAME minimax coefficients, but the polynomial is regrouped into a balanced
        // tree of independent sub-expressions (Estrin's scheme). Shorter dependency chain than Horner
        // → exposes instruction-level parallelism, so it wins on per-call latency. Different rounding
        // order than Horner (still a fixed +-* sequence → deterministic), so a shipping DetMath must
        // pick ONE canonical scheme.
        public static fProxy ExpAccEstrin(fProxy x)
        {
            Reduce(x, out fProxy n, out fProxy r);
            fProxy r2 = r * r;
            fProxy r4 = r2 * r2;
            //+skipFor[double]
            // deg 6: p = (c0+c1 r) + r2(c2+c3 r) + r4((c4+c5 r) + c6 r2)
            float a0 = 1.00000000055416338e+00f + 1.00000003632318291e+00f * r;
            float a1 = 4.99999920798497477e-01f + 1.66664201699263076e-01f * r;
            float a2 = 4.16682255624844372e-02f + 8.37481579955782172e-03f * r;
            float hi = a2 + 1.38368463709141461e-03f * r2;
            float p  = (a0 + a1 * r2) + hi * r4;
            //-skipFor
            //+emitFor[double]
            //!double r8 = r4 * r4;
            //!// deg 11: pairs a_i = c_{2i}+c_{2i+1} r; p = (a0+a1 r2) + r4(a2+a3 r2) + r8(a4+a5 r2)
            //!double a0 = 1.0000000000000000e+00 + 1.0000000000000000e+00 * r;
            //!double a1 = 5.0000000000000177e-01 + 1.6666666666666169e-01 * r;
            //!double a2 = 4.1666666666492767e-02 + 8.3333333335592709e-03 * r;
            //!double a3 = 1.3888888951223976e-03 + 1.9841269432679901e-04 * r;
            //!double a4 = 2.4801486521436422e-05 + 2.7557622530418205e-06 * r;
            //!double a5 = 2.7632293277020833e-07 + 2.4994305002802721e-08 * r;
            //!double lo  = a0 + a1 * r2;
            //!double mid = a2 + a3 * r2;
            //!double hi  = a4 + a5 * r2;
            //!double p   = (lo + mid * r4) + hi * r8;
            //-emitFor
            return ExpGuard(x, Ldexp(p, n));
        }

        // Fast: fewer terms (float degree 3 ~6e-4; double degree 6 ~1e-7). "Slightly inaccurate" tier.
        public static fProxy ExpFast(fProxy x)
        {
            Reduce(x, out fProxy n, out fProxy r);
            //+skipFor[double]
            float mant = 1.6666666e-1f;
            mant = mant * r + 0.5f;
            mant = mant * r + 1.0f;
            mant = mant * r + 1.0f;               // 1 + r + r^2/2 + r^3/6
            //-skipFor
            //+emitFor[double]
            //!double mant = 1.3888888888888889e-03;   // 1/6!
            //!mant = mant * r + 8.3333333333333332e-03;  // 1/5!
            //!mant = mant * r + 4.1666666666666664e-02;  // 1/4!
            //!mant = mant * r + 1.6666666666666666e-01;  // 1/3!
            //!mant = mant * r + 5.0000000000000000e-01;  // 1/2!
            //!mant = mant * r + 1.0;
            //!mant = mant * r + 1.0;
            //-emitFor
            return ExpGuard(x, Ldexp(mant, n));
        }

        // Branch-free trig guard: sin/cos of ±inf or NaN → NaN (the floored reduction produces garbage
        // for non-finite x). Computed then masked so the loop still vectorizes.
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

        // Trig reduction: x = q·(π/2) + r, |r| <= π/4, via q = round(x·2/π) and a 2-part Cody-Waite
        // π/2 split. Bounded-argument (matches the exp caveat): fine for |x| up to ~10; large |x|
        // needs Payne-Hanek. quadrant = q & 3 selects sin/cos poly + sign below.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void ReduceTrig(fProxy x, out fProxy r, out int quad)
        {
            // 3-word Cody-Waite: accurate to ~|x| = whole useful float range / ~1e15 double. Each
            // Pi has trailing zero mantissa bits so kf*Pi is exact.
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

        // sin(r), |r| <= π/4: odd minimax, sin(r) = r·P(r²). float deg-3 (~0.05 ULP), double deg-6.
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

        // cos(r), |r| <= π/4: even minimax, cos(r) = Q(r²). float deg-4 (~0.001 ULP), double deg-7.
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

        // Full-range sin/cos: reduce, evaluate both polys, then branch-free quadrant select. Sin and
        // Cos each compute the full thing (matching math.sin / math.cos individually).
        public static fProxy Sin(fProxy x)
        {
            ReduceTrig(x, out fProxy r, out int quad);
            fProxy s = SinPoly(r);
            fProxy c = CosPoly(r);
            fProxy sw = (fProxy)(quad & 1);                       // q odd → use cos poly
            fProxy baseS = s + sw * (c - s);
            fProxy sign = (fProxy)1 - (fProxy)2 * (fProxy)((quad >> 1) & 1);
            return TrigGuard(x, sign * baseS);
        }

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

        // Branch-free log domain guard: log(0)=-inf, log(+inf)=+inf, x<0→NaN, NaN→NaN. Computed then
        // masked so the loop still vectorizes.
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

        // log(x), x > 0: split x = m·2^e via the exponent bits, center m to [√2/2, √2), then
        // log(m) = 2s·B(s²) with s = (m-1)/(m+1) (atanh form, no cancellation near m=1). B is minimax
        // (float deg-3 ~0.01 ULP, double deg-7 ~0.01 ULP). log(x) = e·ln2 + log(m), ln2 split hi/lo.
        // Subnormal inputs are scaled into the normal range first (branch-free); LogGuard handles
        // x<=0 / inf / NaN.
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
            m = m - adj * (m * 0.5f);                                // if m>√2: halve → [√2/2,√2)
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

        // atan(x), any real x. Fold |x|>1 via atan(x)=π/2-atan(1/x) → [0,1], then split at tan(π/8)
        // using atan(x)=π/4+atan((x-1)/(x+1)) so the polynomial argument stays in ~[-0.29,0.41]. Odd
        // minimax atan(xr)=xr·P(xr²) (float deg-4 ~0.3 ULP, double deg-10 ~0.33 ULP). Branch-free via
        // math.select (both branches computed, then blended → vectorizes). The one hard primitive: a
        // single wide-range poly is hopeless (deg-12 on [0,1] is ~8e4 ULP), the reduction is essential.
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

        // ---- FUSED sincos: ONE reduction feeds both, ~40% cheaper than Sin()+Cos() separately.
        // (Sin and Cos above each redo the reduction + both polys; this does it once.) The natural
        // primitive for rotation-by-angle / dft twiddles, which always want both.
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

        // ---- Derived functions: FUSED from the core primitives (not naive re-compositions). Each
        // shares one reduction / one exp where a naive version would pay for two.

        // tan = sin/cos: one reduction, both polys, one divide (vs two full reductions).
        public static fProxy Tan(fProxy x) { SinCos(x, out fProxy s, out fProxy c); return s / c; }

        // exp2/log2/log10: scaled exp/log — one core call, one multiply.
        public static fProxy Exp2(fProxy x)  => ExpAcc(x * (fProxy)0.6931471805599453);
        public static fProxy Log2(fProxy x)  => Log(x) * (fProxy)1.4426950408889634;
        public static fProxy Log10(fProxy x) => Log(x) * (fProxy)0.4342944819032518;

        // sinh/cosh share ONE exp + one reciprocal (naive pays two exp calls). tanh from e^{2x}.
        public static void SinhCosh(fProxy x, out fProxy sh, out fProxy ch)
        {
            fProxy e = ExpAcc(x);
            fProxy er = (fProxy)1 / e;
            sh = (e - er) * (fProxy)0.5;
            ch = (e + er) * (fProxy)0.5;
        }
        public static fProxy Sinh(fProxy x) { SinhCosh(x, out fProxy sh, out fProxy ch); return sh; }
        public static fProxy Cosh(fProxy x) { SinhCosh(x, out fProxy sh, out fProxy ch); return ch; }
        public static fProxy Tanh(fProxy x) { fProxy e = ExpAcc((fProxy)2 * x); return (e - (fProxy)1) / (e + (fProxy)1); }

        // pow(x,y) = exp(y·log x), x > 0 (prototype: no sign/zero/integer-exponent edge handling).
        public static fProxy Pow(fProxy x, fProxy y) => ExpAcc(y * Log(x));

        // atan2 from atan + branch-free quadrant. x=0 handled via ±inf into Atan (folds to ±π/2).
        public static fProxy Atan2(fProxy y, fProxy x)
        {
            const double PI_D = 3.141592653589793;
            fProxy PI = (fProxy)PI_D;
            fProxy a = Atan(y / x);
            a = math.select(a, a + PI, (x < (fProxy)0) & (y >= (fProxy)0));
            a = math.select(a, a - PI, (x < (fProxy)0) & (y <  (fProxy)0));
            return a;
        }

        // asin/acos via atan of the half-angle form. |x| <= 1.
        public static fProxy Asin(fProxy x) => Atan(x / math.sqrt((fProxy)1 - x * x));
        public static fProxy Acos(fProxy x) => (fProxy)1.5707963267948966 - Asin(x);
    }

    // ---- fused sincos vs math.sincos (both compute sin AND cos) ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SinCosCompareJobFProxy : IJob
    {
        public fProxyN src, dst;
        public int variant;   // 0 = math.sincos, 1 = det.SinCos
        public int single;

        public void Execute()
        {
            int n = src.N;
            if (single == 0)
            {
                if (variant == 0) for (int i = 0; i < n; i++) { dst[i] = math.sin(src[i]) + math.cos(src[i]); }
                else              for (int i = 0; i < n; i++) { DetMathProtoFProxy.SinCos(src[i], out fProxy s, out fProxy c); dst[i] = s + c; }
            }
            else
            {
                fProxy acc = (fProxy)0, tiny = (fProxy)1e-20;
                if (variant == 0) for (int i = 0; i < n; i++) { fProxy x = src[i] + acc * tiny; acc = math.sin(x) + math.cos(x); }
                else              for (int i = 0; i < n; i++) { DetMathProtoFProxy.SinCos(src[i] + acc * tiny, out fProxy s, out fProxy c); acc = s + c; }
                dst[0] = acc;
            }
        }
    }

    // ---- correctness of the derived layer vs math.* (reference in double) at sample inputs ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct DerivedVerifyJobFProxy : IJob
    {
        public fProxyN src;                 // inputs in (0.2, 0.9): positive, < 1, below the tan pole
        public NativeArray<double> maxUlp;  // [tan,exp2,log2,log10,sinh,cosh,tanh,pow,atan2,asin,acos]
        public void Execute()
        {
            const double eps = /*+choose[1.1920929e-7|2.220446049250313e-16]*/1.1920929e-7/*-choose*/;
            for (int k = 0; k < maxUlp.Length; k++) maxUlp[k] = 0.0;
            int n = src.N;
            for (int i = 0; i < n; i++)
            {
                double x = (double)src[i];
                Acc(maxUlp, 0, (double)DetMathProtoFProxy.Tan(src[i]),          math.tan(x), eps);
                Acc(maxUlp, 1, (double)DetMathProtoFProxy.Exp2(src[i]),         math.exp2(x), eps);
                Acc(maxUlp, 2, (double)DetMathProtoFProxy.Log2(src[i]),         math.log2(x), eps);
                Acc(maxUlp, 3, (double)DetMathProtoFProxy.Log10(src[i]),        math.log10(x), eps);
                Acc(maxUlp, 4, (double)DetMathProtoFProxy.Sinh(src[i]),         math.sinh(x), eps);
                Acc(maxUlp, 5, (double)DetMathProtoFProxy.Cosh(src[i]),         math.cosh(x), eps);
                Acc(maxUlp, 6, (double)DetMathProtoFProxy.Tanh(src[i]),         math.tanh(x), eps);
                Acc(maxUlp, 7, (double)DetMathProtoFProxy.Pow(src[i], (fProxy)2.5), math.pow(x, 2.5), eps);
                Acc(maxUlp, 8, (double)DetMathProtoFProxy.Atan2(src[i], (fProxy)0.7), math.atan2(x, 0.7), eps);
                Acc(maxUlp, 9, (double)DetMathProtoFProxy.Asin(src[i]),         math.asin(x), eps);
                Acc(maxUlp, 10, (double)DetMathProtoFProxy.Acos(src[i]),        math.acos(x), eps);
            }
        }
        static void Acc(NativeArray<double> a, int k, double got, double refv, double eps)
        {
            double denom = math.max(math.abs(refv), 1e-30);
            double ulp = math.abs(got - refv) / denom / eps;
            if (ulp > a[k]) a[k] = ulp;
        }
    }

    public static partial class DetMathBenchmark
    {
        static string SinCosRowFProxy(int variant, bool single, string label, int n)
        {
            var src = new fProxyN(n, Allocator.Persistent);
            var dst = new fProxyN(n, Allocator.Persistent);
            var rng = new Unity.Mathematics.Random(0x51C0u ^ (uint)variant);
            for (int i = 0; i < n; i++) src[i] = rng.NextFProxy(-10f, 10f);
            var job = new SinCosCompareJobFProxy { src = src, dst = dst, variant = variant, single = single ? 1 : 0 };
            var stat = Bench.Time(() => job.Run());
            src.Dispose(); dst.Dispose();
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-20} {1,-10} {2,11:F4} {3,11:F4} {4,11:F4}", label, n, stat.Min, stat.Median, stat.Mean);
        }

        // Returns the 11 max-ULP values (tan,exp2,log2,log10,sinh,cosh,tanh,pow,atan2,asin,acos);
        // the hand-written harness attaches the names.
        static double[] DerivedVerifyFProxy()
        {
            int n = 100000;
            var src = new fProxyN(n, Allocator.Persistent);
            var rng = new Unity.Mathematics.Random(0xDE71u);
            for (int i = 0; i < n; i++) src[i] = rng.NextFProxy(0.2f, 0.9f);
            var maxUlp = new NativeArray<double>(11, Allocator.Persistent);
            new DerivedVerifyJobFProxy { src = src, maxUlp = maxUlp }.Run();
            var result = new double[11];
            for (int k = 0; k < 11; k++) result[k] = maxUlp[k];
            maxUlp.Dispose();
            src.Dispose();
            return result;
        }
    }

    // ---- native math.* throughput (batch) ----
    // func: 0=sin 1=cos 2=exp 3=log 4=atan. Switch is OUTSIDE the loop so each function runs a tight,
    // independently-vectorizable body.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MathFuncThroughputJobFProxy : IJob
    {
        public fProxyN src;
        public fProxyN dst;
        public int func;

        public void Execute()
        {
            int n = src.N;
            switch (func)
            {
                case 0: for (int i = 0; i < n; i++) dst[i] = math.sin(src[i]);  break;
                case 1: for (int i = 0; i < n; i++) dst[i] = math.cos(src[i]);  break;
                case 2: for (int i = 0; i < n; i++) dst[i] = math.exp(src[i]);  break;
                case 3: for (int i = 0; i < n; i++) dst[i] = math.log(src[i]);  break;
                default: for (int i = 0; i < n; i++) dst[i] = math.atan(src[i]); break;
            }
        }
    }

    // ---- exp comparison: variant {0 math, 1 det-acc, 2 det-fast} x mode {batch, single} ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ExpCompareJobFProxy : IJob
    {
        public fProxyN src;
        public fProxyN dst;    // batch output (length N); dst[0] holds the sink in single mode
        public int variant;
        public int single;     // 0 = batch, 1 = single (dependent chain)

        public void Execute()
        {
            int n = src.N;
            if (single == 0)
            {
                switch (variant)
                {
                    case 0: for (int i = 0; i < n; i++) dst[i] = math.exp(src[i]);                       break;
                    case 1: for (int i = 0; i < n; i++) dst[i] = DetMathProtoFProxy.ExpAcc(src[i]);       break;
                    case 2: for (int i = 0; i < n; i++) dst[i] = DetMathProtoFProxy.ExpAccEstrin(src[i]); break;
                    default: for (int i = 0; i < n; i++) dst[i] = DetMathProtoFProxy.ExpFast(src[i]);     break;
                }
            }
            else
            {
                // Dependent chain: acc feeds the next argument (scaled to a negligible perturbation) so
                // the compiler cannot vectorize across iterations — this measures per-call latency.
                fProxy acc = (fProxy)0;
                fProxy tiny = (fProxy)1e-20;
                switch (variant)
                {
                    case 0: for (int i = 0; i < n; i++) acc = math.exp(src[i] + acc * tiny);                       break;
                    case 1: for (int i = 0; i < n; i++) acc = DetMathProtoFProxy.ExpAcc(src[i] + acc * tiny);       break;
                    case 2: for (int i = 0; i < n; i++) acc = DetMathProtoFProxy.ExpAccEstrin(src[i] + acc * tiny); break;
                    default: for (int i = 0; i < n; i++) acc = DetMathProtoFProxy.ExpFast(src[i] + acc * tiny);     break;
                }
                dst[0] = acc;
            }
        }
    }

    // ---- sin/cos comparison: variant {0 math.sin, 1 det.sin, 2 math.cos, 3 det.cos} x {batch,single} ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TrigCompareJobFProxy : IJob
    {
        public fProxyN src;
        public fProxyN dst;
        public int variant;
        public int single;

        public void Execute()
        {
            int n = src.N;
            if (single == 0)
            {
                switch (variant)
                {
                    case 0: for (int i = 0; i < n; i++) dst[i] = math.sin(src[i]);                 break;
                    case 1: for (int i = 0; i < n; i++) dst[i] = DetMathProtoFProxy.Sin(src[i]);     break;
                    case 2: for (int i = 0; i < n; i++) dst[i] = math.cos(src[i]);                 break;
                    default: for (int i = 0; i < n; i++) dst[i] = DetMathProtoFProxy.Cos(src[i]);   break;
                }
            }
            else
            {
                fProxy acc = (fProxy)0;
                fProxy tiny = (fProxy)1e-20;
                switch (variant)
                {
                    case 0: for (int i = 0; i < n; i++) acc = math.sin(src[i] + acc * tiny);                 break;
                    case 1: for (int i = 0; i < n; i++) acc = DetMathProtoFProxy.Sin(src[i] + acc * tiny);     break;
                    case 2: for (int i = 0; i < n; i++) acc = math.cos(src[i] + acc * tiny);                 break;
                    default: for (int i = 0; i < n; i++) acc = DetMathProtoFProxy.Cos(src[i] + acc * tiny);   break;
                }
                dst[0] = acc;
            }
        }
    }

    // ---- log comparison: variant {0 math.log, 1 det.log} x {batch,single} ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LogCompareJobFProxy : IJob
    {
        public fProxyN src;
        public fProxyN dst;
        public int variant;
        public int single;

        public void Execute()
        {
            int n = src.N;
            if (single == 0)
            {
                if (variant == 0) for (int i = 0; i < n; i++) dst[i] = math.log(src[i]);
                else              for (int i = 0; i < n; i++) dst[i] = DetMathProtoFProxy.Log(src[i]);
            }
            else
            {
                fProxy acc = (fProxy)1;
                fProxy tiny = (fProxy)1e-20;
                if (variant == 0) for (int i = 0; i < n; i++) acc = math.log(src[i] + acc * tiny);
                else              for (int i = 0; i < n; i++) acc = DetMathProtoFProxy.Log(src[i] + acc * tiny);
                dst[0] = acc;
            }
        }
    }

    // ---- atan comparison: variant {0 math.atan, 1 det.atan} x {batch,single} ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct AtanCompareJobFProxy : IJob
    {
        public fProxyN src;
        public fProxyN dst;
        public int variant;
        public int single;

        public void Execute()
        {
            int n = src.N;
            if (single == 0)
            {
                if (variant == 0) for (int i = 0; i < n; i++) dst[i] = math.atan(src[i]);
                else              for (int i = 0; i < n; i++) dst[i] = DetMathProtoFProxy.Atan(src[i]);
            }
            else
            {
                fProxy acc = (fProxy)0;
                fProxy tiny = (fProxy)1e-20;
                if (variant == 0) for (int i = 0; i < n; i++) acc = math.atan(src[i] + acc * tiny);
                else              for (int i = 0; i < n; i++) acc = DetMathProtoFProxy.Atan(src[i] + acc * tiny);
                dst[0] = acc;
            }
        }
    }

    public static partial class DetMathBenchmark
    {
        static string AtanRowFProxy(int variant, bool single, string label, int n)
        {
            var src = new fProxyN(n, Allocator.Persistent);
            var dst = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(0xA7A11u ^ (uint)variant);
            for (int i = 0; i < n; i++) src[i] = rng.NextFProxy(-20f, 20f);   // exercises both folds

            var job = new AtanCompareJobFProxy { src = src, dst = dst, variant = variant, single = single ? 1 : 0 };
            var stat = Bench.Time(() => job.Run());

            double maxAbs = 0.0;
            if (!single)
            {
                for (int i = 0; i < n; i++)
                {
                    double refv = System.Math.Atan((double)src[i]);
                    double err  = System.Math.Abs((double)dst[i] - refv);
                    if (err > maxAbs) maxAbs = err;
                }
            }
            src.Dispose(); dst.Dispose();

            double eps = /*+choose[1.1920929e-7|2.220446049250313e-16]*/1.1920929e-7/*-choose*/;
            string errStr = single ? "(chain)" : maxAbs.ToString("E3", CultureInfo.InvariantCulture);
            string ulpStr = single ? "-" : (maxAbs / eps).ToString("F2", CultureInfo.InvariantCulture);
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-20} {1,-10} {2,11:F4} {3,11:F4} {4,11:F4} {5,13} {6,11}",
                label, n, stat.Min, stat.Median, stat.Mean, errStr, ulpStr);
        }

        static string LogRowFProxy(int variant, bool single, string label, int n)
        {
            var src = new fProxyN(n, Allocator.Persistent);
            var dst = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(0x106106u ^ (uint)variant);
            for (int i = 0; i < n; i++) src[i] = rng.NextFProxy(0.1f, 10f);   // x > 0

            var job = new LogCompareJobFProxy { src = src, dst = dst, variant = variant, single = single ? 1 : 0 };
            var stat = Bench.Time(() => job.Run());

            double maxAbs = 0.0;   // absolute error (log crosses 0 at x=1 → relative would blow up there)
            if (!single)
            {
                for (int i = 0; i < n; i++)
                {
                    double refv = System.Math.Log((double)src[i]);
                    double err  = System.Math.Abs((double)dst[i] - refv);
                    if (err > maxAbs) maxAbs = err;
                }
            }
            src.Dispose(); dst.Dispose();

            double eps = /*+choose[1.1920929e-7|2.220446049250313e-16]*/1.1920929e-7/*-choose*/;
            string errStr = single ? "(chain)" : maxAbs.ToString("E3", CultureInfo.InvariantCulture);
            string ulpStr = single ? "-" : (maxAbs / eps).ToString("F2", CultureInfo.InvariantCulture);
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-20} {1,-10} {2,11:F4} {3,11:F4} {4,11:F4} {5,13} {6,11}",
                label, n, stat.Min, stat.Median, stat.Mean, errStr, ulpStr);
        }

        static string TrigRowFProxy(int variant, bool single, string label, int n)
        {
            var src = new fProxyN(n, Allocator.Persistent);
            var dst = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(0x5EED1234u ^ (uint)variant);
            for (int i = 0; i < n; i++) src[i] = rng.NextFProxy(-10f, 10f);

            var job = new TrigCompareJobFProxy { src = src, dst = dst, variant = variant, single = single ? 1 : 0 };
            var stat = Bench.Time(() => job.Run());

            bool isCos = variant >= 2;
            double maxAbs = 0.0;   // sin/cos: absolute error (result in [-1,1]; relative blows up near zeros)
            if (!single)
            {
                for (int i = 0; i < n; i++)
                {
                    double refv = isCos ? System.Math.Cos((double)src[i]) : System.Math.Sin((double)src[i]);
                    double err  = System.Math.Abs((double)dst[i] - refv);
                    if (err > maxAbs) maxAbs = err;
                }
            }
            src.Dispose(); dst.Dispose();

            double eps = /*+choose[1.1920929e-7|2.220446049250313e-16]*/1.1920929e-7/*-choose*/;
            string errStr = single ? "(chain)" : maxAbs.ToString("E3", CultureInfo.InvariantCulture);
            string ulpStr = single ? "-" : (maxAbs / eps).ToString("F2", CultureInfo.InvariantCulture);
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-20} {1,-10} {2,11:F4} {3,11:F4} {4,11:F4} {5,13} {6,11}",
                label, n, stat.Min, stat.Median, stat.Mean, errStr, ulpStr);
        }

        static string MathThroughputFProxy(int func, string label, int n)
        {
            var src = new fProxyN(n, Allocator.Persistent);
            var dst = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)func ^ 0x9E3779B9u);
            for (int i = 0; i < n; i++) src[i] = rng.NextFProxy(0.5f, 3f);

            var job = new MathFuncThroughputJobFProxy { src = src, dst = dst, func = func };
            var stat = Bench.Time(() => job.Run());

            src.Dispose(); dst.Dispose();
            return Bench.RowTime(label, n, stat);
        }

        static string ExpRowFProxy(int variant, bool single, string label, int n)
        {
            var src = new fProxyN(n, Allocator.Persistent);
            var dst = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(0xB16B00B5u ^ (uint)variant);
            for (int i = 0; i < n; i++) src[i] = rng.NextFProxy(-10f, 10f);

            var job = new ExpCompareJobFProxy { src = src, dst = dst, variant = variant, single = single ? 1 : 0 };
            var stat = Bench.Time(() => job.Run());

            // Accuracy (batch only — single overwrites just dst[0]): max relative error vs a double
            // System.Math.Exp reference over the whole input. Computed once, untimed.
            double maxRel = 0.0;
            if (!single)
            {
                for (int i = 0; i < n; i++)
                {
                    double refv = System.Math.Exp((double)src[i]);
                    double got  = (double)dst[i];
                    double rel  = System.Math.Abs(got - refv) / System.Math.Abs(refv);
                    if (rel > maxRel) maxRel = rel;
                }
            }
            src.Dispose(); dst.Dispose();

            double eps = /*+choose[1.1920929e-7|2.220446049250313e-16]*/1.1920929e-7/*-choose*/;
            string relStr = single ? "(chain)" : maxRel.ToString("E3", CultureInfo.InvariantCulture);
            string ulpStr = single ? "-" : (maxRel / eps).ToString("F2", CultureInfo.InvariantCulture);
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-20} {1,-10} {2,11:F4} {3,11:F4} {4,11:F4} {5,13} {6,11}",
                label, n, stat.Min, stat.Median, stat.Mean, relStr, ulpStr);
        }
    }
}
