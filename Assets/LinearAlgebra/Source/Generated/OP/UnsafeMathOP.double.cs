using System.Runtime.CompilerServices;

using Unity.Burst;
using Unity.Mathematics;

namespace LinearAlgebra.Internal
{
    public static unsafe partial class UnsafeMathOP
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void setAll([NoAlias] double* x, int n, double s)
        {
            for (int i = 0; i < n; i++)
                x[i] = s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void setIndexZero([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = i;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void setIndexOne([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = i+1;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void abs([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.abs(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sign([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.sign(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sqrt([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.sqrt(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void acos([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.acos(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void asin([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.asin(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void atan([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.atan(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void ceil([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.ceil(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cos([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.cos(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void cosh([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.cosh(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void exp([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.exp(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void exp2([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.exp2(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void exp10([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.pow(10, x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void floor([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.floor(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void log([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.log(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void log2([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.log2(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void log10([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.log10(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void max([NoAlias] double* x, [NoAlias] double* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.max(x[i], y[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void min([NoAlias] double* x, [NoAlias] double* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.min(x[i], y[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void round([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.round(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sin([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.sin(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sinh([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.sinh(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void tan([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.tan(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void tanh([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.tanh(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void pow([NoAlias] double* x, int n, int pow)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.pow(x[i], pow);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void atan2([NoAlias] double* y, [NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                y[i] = math.atan2(y[i], x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void acosh([NoAlias] double* x, int n)
        {
            // acosh(x) = log(x + sqrt(x^2 - 1)), domain x >= 1.
            // Unity.Mathematics has no math.acosh, so it is computed directly.
            for (int i = 0; i < n; i++)
                x[i] = math.log(x[i] + math.sqrt(x[i] * x[i] - (double)1));
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void clamp([NoAlias] double* x, int n, double min, double max)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.clamp(x[i], min, max);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void degrees([NoAlias] double* radians, int n)
        {
            for (int i = 0; i < n; i++)
                radians[i] = math.degrees(radians[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void radians([NoAlias] double* degrees, int n)
        {
            for (int i = 0; i < n; i++)
                degrees[i] = math.radians(degrees[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void lerp([NoAlias] double* a, [NoAlias] double* b, int n, double t)
        {
            for (int i = 0; i < n; i++)
                a[i] = math.lerp(a[i], b[i], t);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void unlerp([NoAlias] double* a, [NoAlias] double* b, int n, double t)
        {
            for (int i = 0; i < n; i++)
                a[i] = math.unlerp(a[i], b[i], t);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void smoothstep([NoAlias] double* a, [NoAlias] double* b, int n, double t)
        {
            for (int i = 0; i < n; i++)
                a[i] = math.smoothstep(a[i], b[i], t);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void rcp([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.rcp(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void mod([NoAlias] double* x, double y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] % y;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void relu([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < 0? 0 : x[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void remap([NoAlias] double* x, int n, double oldMin, double oldMax, double newMin, double newMax)
        {
            // Unity.Mathematics' math.remap signature is (srcStart, srcEnd, dstStart, dstEnd, value) -
            // the value is LAST, not first.
            for (int i = 0; i < n; i++)
                x[i] = math.remap(oldMin, oldMax, newMin, newMax, x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void mad([NoAlias] double* a, [NoAlias] double* b, [NoAlias] double* c, int n)
        {
            for (int i = 0; i < n; i++)
                a[i] = math.mad(a[i], b[i], c[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void saturate([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.saturate(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void frac([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] -= math.floor(x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void fmod([NoAlias] double* x, double y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.fmod(x[i], y);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void rsqrt([NoAlias] double* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.rsqrt(x[i]);
        }

        // Componentwise, NOT a whole-vector Euclidean distance despite math.distance's name: for two
        // SCALAR double operands math.distance(a,b) is just |a-b| (and math.distancesq is (a-b)^2) -
        // there is no cross-index reduction here. Named absDiff/sqrDiff (not "distance") so the public
        // Comp wrapper isn't mistaken for a whole-vector geometry op.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void absDiff([NoAlias] double* x, [NoAlias] double* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.distance(x[i], y[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sqrDiff([NoAlias] double* x, [NoAlias] double* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.distancesq(x[i], y[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void smoothstep([NoAlias] double* x, int n, double a, double b)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.smoothstep(a, b, x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void step([NoAlias] double* x, int n, double y)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.step(y, x[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double dot([NoAlias] double* x, [NoAlias] double* y, int n)
        {
            double sum = 0f;
            for (int i = 0; i < n; i++)
                sum += x[i] * y[i];

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void reflect([NoAlias] double* incident, [NoAlias] double* normal, int n)
        {
            var d = dot(incident, normal, n);
            for (int i = 0; i < n; i++)
                incident[i] = incident[i] - 2f * normal[i] * d;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void refract([NoAlias] double* incident, [NoAlias] double* normal, int n, double eta)
        {
            var d = dot(incident, normal, n);
            var k = 1f - eta * eta * (1f - d * d);
            if (k < 0f)
                setAll(incident, n, 0f);
            else
                for (int i = 0; i < n; i++)
                    incident[i] = eta * incident[i] - (eta * d + math.sqrt(k)) * normal[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void project([NoAlias] double* a, [NoAlias] double* b, int n)
        {
            var uDot = dot(a, b, n);
            var lDot = dot(b, b, n);
            double div = uDot / lDot;
            for (int i = 0; i < n; i++)
                a[i] = div * b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sincos([NoAlias] double* x, int n, [NoAlias] double* sin, [NoAlias] double* cos)
        {
            // more cache efficient than calling sin&cos at same time and writing to both arrays
            for (int i = 0; i < n; i++)
                sin[i] = math.sin(x[i]);

            for (int i = 0; i < n; i++)
                cos[i] = math.cos(x[i]);
        }
    }
}
