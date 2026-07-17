using System;
using Unity.Burst;
using Unity.Mathematics;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{

    /// <summary>
    /// Elementwise and scalar-bool select: returns b where c is true, a otherwise. Overloaded over
    /// vector/matrix and per-element/scalar bool <c>c</c>, for fProxy, iProxy, and bool element types.
    /// </summary>
    public static partial class Select
    {
        // ref-dest primitive. No alias guard: select is elementwise (dst[i] = c[i] ? b[i] : a[i]),
        // so the destination may alias a or b safely.
        public static void select(in fProxyN a, in fProxyN b, in boolN c, ref fProxyN dest)
        {
            Assume.SameDim(in a, in b);
            Assume.SameDim(in a, in c);

            if (dest.N != a.N)
                throw new ArgumentException("select: dest.N must equal a.N");

            unsafe
            {
                UnsafeSelectOP.selectfProxy(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, dest.Data.Ptr, a.N);
            }
        }

        public static fProxyN select(in fProxyN a, in fProxyN b, in boolN c)
        {
            fProxyN res = a.fProxyTempVec(a.N, true);
            select(in a, in b, in c, ref res);
            return res;
        }

        // ref-dest primitive. No alias guard: elementwise op.
        public static void select(in fProxyMxN a, in fProxyMxN b, in boolMxN c, ref fProxyMxN dest)
        {
            Assume.SameDim(in a, in b);
            Assume.SameDim(in a, in c);

            if (dest.M_Rows != a.M_Rows || dest.N_Cols != a.N_Cols)
                throw new ArgumentException("select: dest dimensions must match a");

            unsafe
            {
                UnsafeSelectOP.selectfProxy(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, dest.Data.Ptr, a.M_Rows * a.N_Cols);
            }
        }

        public static fProxyMxN select(in fProxyMxN a, in fProxyMxN b, in boolMxN c)
        {
            fProxyMxN res = a.fProxyTempMat(a.M_Rows, a.N_Cols, true);
            select(in a, in b, in c, ref res);
            return res;
        }

        // ref-dest primitive. No alias guard: scalar bool selects the whole source unchanged.
        public static void select(in fProxyN a, in fProxyN b, in bool c, ref fProxyN dest)
        {
            Assume.SameDim(in a, in b);

            if (dest.N != a.N)
                throw new ArgumentException("select: dest.N must equal a.N");

            if (c)
                dest.Data.CopyFrom(b.Data);
            else
                dest.Data.CopyFrom(a.Data);
        }

        public static fProxyN select(in fProxyN a, in fProxyN b, in bool c)
        {
            return c ? b.TempCopy() : a.TempCopy();
        }

        // Matrix analog of the scalar-bool vector overload above (same no-alias reasoning).
        public static void select(in fProxyMxN a, in fProxyMxN b, in bool c, ref fProxyMxN dest)
        {
            Assume.SameDim(in a, in b);

            if (dest.M_Rows != a.M_Rows || dest.N_Cols != a.N_Cols)
                throw new ArgumentException("select: dest dimensions must match a");

            if (c)
                dest.Data.CopyFrom(b.Data);
            else
                dest.Data.CopyFrom(a.Data);
        }

        public static fProxyMxN select(in fProxyMxN a, in fProxyMxN b, in bool c)
        {
            return c ? b.TempCopy() : a.TempCopy();
        }
    }
}

namespace LinearAlgebra.Internal
{
    public static unsafe partial class UnsafeSelectOP
    {
        // a/b carry no [NoAlias]: the public select contract allows target to alias either input
        // (elementwise, each index reads before it writes). c is a different element type and
        // cannot alias the fProxy pointers through the public API.
        public static void selectfProxy(fProxy* a, fProxy* b, [NoAlias] bool* c, fProxy* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = math.select(a[i], b[i], c[i]);
        }
    }
}
