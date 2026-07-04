#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using Unity.Burst;
using LinearAlgebra.Internal;

//alsoExpand[uint]// select is pure data movement (dst[i] = c[i] ? b[i] : a[i]) - no comparison,
//sign flip, or overflow-sensitive arithmetic is involved, so it is fully unsigned-safe with no
//skipFor-marked exclusions needed.

namespace LinearAlgebra
{

    // Class summary lives on the fProxy partial (SelectOP.fProxy.cs). Integer-family select
    // overloads (int/short/long, plus uint via the alsoExpand marker above).
    public static partial class Select
    {
        // ref-dest primitive. No alias guard: select is elementwise (dst[i] = c[i] ? b[i] : a[i]),
        // so the destination may alias a or b safely.
        public static void select(in iProxyN a, in iProxyN b, in boolN c, ref iProxyN dest)
        {
            Assume.SameDim(in a, in b);
            Assume.SameDim(in a, in c);

            if (dest.N != a.N)
                throw new ArgumentException("select: dest.N must equal a.N");

            unsafe
            {
                UnsafeSelectOP.selectiProxy(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, dest.Data.Ptr, a.N);
            }
        }

        public static iProxyN select(in iProxyN a, in iProxyN b, in boolN c)
        {
            iProxyN res = a.iProxyTempVec(a.N, true);
            select(in a, in b, in c, ref res);
            return res;
        }

        // ref-dest primitive. No alias guard: elementwise op.
        public static void select(in iProxyMxN a, in iProxyMxN b, in boolMxN c, ref iProxyMxN dest)
        {
            Assume.SameDim(in a, in b);
            Assume.SameDim(in a, in c);

            if (dest.M_Rows != a.M_Rows || dest.N_Cols != a.N_Cols)
                throw new ArgumentException("select: dest dimensions must match a");

            unsafe
            {
                UnsafeSelectOP.selectiProxy(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, dest.Data.Ptr, a.M_Rows * a.N_Cols);
            }
        }

        public static iProxyMxN select(in iProxyMxN a, in iProxyMxN b, in boolMxN c)
        {
            iProxyMxN res = a.iProxyTempMat(a.M_Rows, a.N_Cols, true);
            select(in a, in b, in c, ref res);
            return res;
        }

        // ref-dest primitive. No alias guard: scalar bool selects the whole source unchanged.
        public static void select(in iProxyN a, in iProxyN b, in bool c, ref iProxyN dest)
        {
            Assume.SameDim(in a, in b);

            if (dest.N != a.N)
                throw new ArgumentException("select: dest.N must equal a.N");

            if (c)
                dest.Data.CopyFrom(b.Data);
            else
                dest.Data.CopyFrom(a.Data);
        }

        public static iProxyN select(in iProxyN a, in iProxyN b, in bool c)
        {
            return c ? b.TempCopy() : a.TempCopy();
        }

        // Matrix analog of the scalar-bool vector overload above (same no-alias reasoning).
        public static void select(in iProxyMxN a, in iProxyMxN b, in bool c, ref iProxyMxN dest)
        {
            Assume.SameDim(in a, in b);

            if (dest.M_Rows != a.M_Rows || dest.N_Cols != a.N_Cols)
                throw new ArgumentException("select: dest dimensions must match a");

            if (c)
                dest.Data.CopyFrom(b.Data);
            else
                dest.Data.CopyFrom(a.Data);
        }

        public static iProxyMxN select(in iProxyMxN a, in iProxyMxN b, in bool c)
        {
            return c ? b.TempCopy() : a.TempCopy();
        }
    }
}

namespace LinearAlgebra.Internal
{
    public static unsafe partial class UnsafeSelectOP
    {
        public static void selectiProxy([NoAlias] iProxy* a, [NoAlias] iProxy* b, [NoAlias] bool* c, iProxy* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = c[i] ? b[i] : a[i];
        }
    }
}
