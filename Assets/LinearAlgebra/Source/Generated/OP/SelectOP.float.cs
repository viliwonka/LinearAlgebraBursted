#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using Unity.Burst;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{

    /// <summary>Returns b if c is true, a otherwise.</summary>
    /// <param name="a">Value to use if c is false.</param>
    /// <param name="b">Value to use if c is true.</param>
    /// <param name="c">Bool value to choose between a and b.</param>
    /// <returns>The selection between a and b according to bool c.</returns>
    public static partial class Select_OP
    {
        // ref-dest primitive. No alias guard: select is elementwise (dst[i] = c[i] ? b[i] : a[i]),
        // so the destination may alias a or b safely.
        public static void select(in floatN a, in floatN b, in boolN c, ref floatN dest)
        {
            Assume.SameDim(in a, in b);
            Assume.SameDim(in a, in c);

            if (dest.N != a.N)
                throw new ArgumentException("select: dest.N must equal a.N");

            unsafe
            {
                UnsafeSelect_OP.selectfloat(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, dest.Data.Ptr, a.N);
            }
        }

        public static floatN select(in floatN a, in floatN b, in boolN c)
        {
            floatN res = a.tempfloatVec(a.N, true);
            select(in a, in b, in c, ref res);
            return res;
        }

        // ref-dest primitive. No alias guard: elementwise op.
        public static void select(in floatMxN a, in floatMxN b, in boolMxN c, ref floatMxN dest)
        {
            Assume.SameDim(in a, in b);
            Assume.SameDim(in a, in c);

            if (dest.M_Rows != a.M_Rows || dest.N_Cols != a.N_Cols)
                throw new ArgumentException("select: dest dimensions must match a");

            unsafe
            {
                UnsafeSelect_OP.selectfloat(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, dest.Data.Ptr, a.M_Rows * a.N_Cols);
            }
        }

        public static floatMxN select(in floatMxN a, in floatMxN b, in boolMxN c)
        {
            floatMxN res = a.tempfloatMat(a.M_Rows, a.N_Cols, true);
            select(in a, in b, in c, ref res);
            return res;
        }

        // ref-dest primitive. No alias guard: scalar bool selects the whole source unchanged.
        public static void select(in floatN a, in floatN b, in bool c, ref floatN dest)
        {
            Assume.SameDim(in a, in b);

            if (dest.N != a.N)
                throw new ArgumentException("select: dest.N must equal a.N");

            if (c)
                dest.Data.CopyFrom(b.Data);
            else
                dest.Data.CopyFrom(a.Data);
        }

        public static floatN select(in floatN a, in floatN b, in bool c)
        {
            return c ? b.TempCopy() : a.TempCopy();
        }

        // ref-dest primitive. No alias guard: scalar bool selects the whole source unchanged.
        public static void select(in floatMxN a, in floatMxN b, in bool c, ref floatMxN dest)
        {
            Assume.SameDim(in a, in b);

            if (dest.M_Rows != a.M_Rows || dest.N_Cols != a.N_Cols)
                throw new ArgumentException("select: dest dimensions must match a");

            if (c)
                dest.Data.CopyFrom(b.Data);
            else
                dest.Data.CopyFrom(a.Data);
        }

        public static floatMxN select(in floatMxN a, in floatMxN b, in bool c)
        {
            return c ? b.TempCopy() : a.TempCopy();
        }
    }
}

namespace LinearAlgebra.Internal
{
    public static unsafe partial class UnsafeSelect_OP
    {
        public static void selectfloat([NoAlias] float* a, [NoAlias] float* b, [NoAlias] bool* c, float* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = c[i] ? b[i] : a[i];
        }
    }
}
