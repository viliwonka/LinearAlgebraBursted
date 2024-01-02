using System.Runtime.CompilerServices;

using Unity.Burst;
using Unity.Mathematics;

namespace LinearAlgebra
{
    
    public static unsafe class mathUnsafefProxy
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setAll([NoAlias] fProxy* x, int n, fProxy s)
        {
            for (int i = 0; i < n; i++)
                x[i] = s;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setIndexZero([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = i;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void setIndexOne([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = i+1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void abs([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.abs(x[i]);   
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void sign([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.sign(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void sqrt([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.sqrt(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void acos([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.acos(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void asin([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.asin(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void atan([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.atan(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ceil([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.ceil(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void cos([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.cos(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void cosh([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.cosh(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void exp([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.exp(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void exp2([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.exp2(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void exp10([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.pow(10, x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void floor([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.floor(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void log([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.log(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void log2([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.log2(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void log10([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.log10(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void max([NoAlias] fProxy* x, [NoAlias] fProxy* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.max(x[i], y[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void min([NoAlias] fProxy* x, [NoAlias] fProxy* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.min(x[i], y[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void round([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.round(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void sin([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.sin(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void sinh([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.sinh(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void tan([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.tan(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void tanh([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.tanh(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void pow([NoAlias] fProxy* x, int n, int pow)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.pow(x[i], pow);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void atan2([NoAlias] fProxy* y, [NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                y[i] = math.atan2(y[i], x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void acosh([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.acos(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void clamp([NoAlias] fProxy* x, int n, fProxy min, fProxy max)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.clamp(x[i], min, max);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void degrees([NoAlias] fProxy* radians, int n)
        {
            for (int i = 0; i < n; i++)
                radians[i] = math.degrees(radians[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void radians([NoAlias] fProxy* degrees, int n)
        {
            for (int i = 0; i < n; i++)
                degrees[i] = math.radians(degrees[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void lerp([NoAlias] fProxy* a, [NoAlias] fProxy* b, int n, fProxy t)
        {
            for (int i = 0; i < n; i++)
                a[i] = math.lerp(a[i], b[i], t);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void unlerp([NoAlias] fProxy* a, [NoAlias] fProxy* b, int n, fProxy t)
        {
            for (int i = 0; i < n; i++)
                a[i] = math.unlerp(a[i], b[i], t);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void smoothstep([NoAlias] fProxy* a, [NoAlias] fProxy* b, int n, fProxy t)
        {
            for (int i = 0; i < n; i++)
                a[i] = math.smoothstep(a[i], b[i], t);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rcp([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.rcp(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mod([NoAlias] fProxy* x, fProxy y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] % y;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void relu([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < 0? 0 : x[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void remap([NoAlias] fProxy* x, int n, fProxy oldMin, fProxy oldMax, fProxy newMin, fProxy newMax)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.remap(x[i], oldMin, oldMax, newMin, newMax);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void mad([NoAlias] fProxy* a, [NoAlias] fProxy* b, [NoAlias] fProxy* c, int n)
        {
            for (int i = 0; i < n; i++)
                a[i] = math.mad(a[i], b[i], c[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void saturate([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.saturate(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void frac([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] -= math.floor(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void fmod([NoAlias] fProxy* x, fProxy y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.fmod(x[i], y);
        }

        /*
        public static void modf([NoAlias] fProxy* x, int n, [NoAlias] fProxy* y)
        {
            for (int i = 0; i < n; i++)
                y[i] = math.modf(x[i], out (float)x[i]);
        }*/

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rsqrt([NoAlias] fProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.rsqrt(x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void distance([NoAlias] fProxy* x, [NoAlias] fProxy* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.distance(x[i], y[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void distancesq([NoAlias] fProxy* x, [NoAlias] fProxy* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.distancesq(x[i], y[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void smoothstep([NoAlias] fProxy* x, int n, fProxy a, fProxy b)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.smoothstep(a, b, x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void step([NoAlias] fProxy* x, int n, fProxy y)
        {
            for (int i = 0; i < n; i++)
                x[i] = math.step(y, x[i]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy dot([NoAlias] fProxy* x, [NoAlias] fProxy* y, int n)
        {
            fProxy sum = 0f;
            for (int i = 0; i < n; i++)
                sum += x[i] * y[i];

            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void reflect([NoAlias] fProxy* incident, [NoAlias] fProxy* normal, int n)
        {
            var d = dot(incident, normal, n);
            for (int i = 0; i < n; i++)
                incident[i] = incident[i] - 2f * normal[i] * d;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void refract([NoAlias] fProxy* incident, [NoAlias] fProxy* normal, int n, fProxy eta)
        {
            var d = dot(incident, normal, n);
            var k = 1f - eta * eta * (1f - d * d);
            if (k < 0f)
                setAll(incident, n, 0f);
            else
                for (int i = 0; i < n; i++)
                    incident[i] = eta * incident[i] - (eta * d + math.sqrt(k)) * normal[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void project([NoAlias] fProxy* a, [NoAlias] fProxy* b, int n)
        {
            var uDot = dot(a, b, n);
            var lDot = dot(b, b, n);
            fProxy div = uDot / lDot;
            for (int i = 0; i < n; i++)
                a[i] = div * b[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void sincos([NoAlias] fProxy* x, int n, [NoAlias] fProxy* sin, [NoAlias] fProxy* cos)
        {
            // more cache efficient than calling sin&cos at same time and writing to both arrays
            for (int i = 0; i < n; i++)
                sin[i] = math.sin(x[i]);

            for (int i = 0; i < n; i++)
                cos[i] = math.cos(x[i]);
        }
    }
}