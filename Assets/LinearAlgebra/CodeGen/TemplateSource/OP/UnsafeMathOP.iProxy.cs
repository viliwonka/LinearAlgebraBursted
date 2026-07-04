using System.Runtime.CompilerServices;

using Unity.Burst;
using Unity.Mathematics;

//alsoExpand[uint]// element-wise kernels. abs/relu are signed-only - there is no unsigned notion of
//a negative value, so no absolute-value/negative-clamp to apply, and neither C#'s Math.Abs nor
//Unity.Mathematics' math.abs defines an overload for uint - both are skipFor-marked below, mirroring
//signFlip's pattern in UnsafeOP.iProxy.cs (do not write that marker's literal token in prose here -
//the codegen parser is content-sensitive, not comment-aware). Everything else (min/max/clamp/mod/
//mad/dot) is unsigned-clean as-is.

namespace LinearAlgebra.Internal
{

    public static unsafe partial class UnsafeMathOP
    {
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void setAll([NoAlias] iProxy* x, int n, iProxy s)
        {
            for (int i = 0; i < n; i++)
                x[i] = (iProxy)s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void setIndexZero([NoAlias] iProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (iProxy)i;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void setIndexOne([NoAlias] iProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (iProxy)(i+1);
        }

        //+skipFor[u]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void abs([NoAlias] iProxy* x, int n)
        {
            for (int i = 0; i < n; i++) {
                iProxy v = x[i];
                x[i] = v < 0? (iProxy)(-v) : v;
            }
        }
        //-skipFor

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void max([NoAlias] iProxy* x, [NoAlias] iProxy* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] > y[i]? x[i]: y[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void min([NoAlias] iProxy* x, [NoAlias] iProxy* y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = x[i] < y[i] ? x[i] : y[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void clamp([NoAlias] iProxy* x, int n, iProxy min, iProxy max)
        {
            for (int i = 0; i < n; i++)
                x[i] = (iProxy)math.max(min, math.min(max, x[i]));
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void mod([NoAlias] iProxy* x, iProxy y, int n)
        {
            for (int i = 0; i < n; i++)
                x[i] = (iProxy)(x[i] % y);
        }

        //+skipFor[u]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void relu([NoAlias] iProxy* x, int n)
        {
            for (int i = 0; i < n; i++) {
                iProxy v = x[i];
                x[i] = v < 0? (iProxy)0 : v;
            }
        }
        //-skipFor

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void mad([NoAlias] iProxy* a, [NoAlias] iProxy* b, [NoAlias] iProxy* c, int n)
        {
            for (int i = 0; i < n; i++)
                a[i] = (iProxy)(a[i] * b[i] + c[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static iProxy dot([NoAlias] iProxy* x, [NoAlias] iProxy* y, int n)
        {
            iProxy sum = 0;
            for (int i = 0; i < n; i++)
                sum += (iProxy)(x[i] * y[i]);

            return sum;
        }

    }
}
