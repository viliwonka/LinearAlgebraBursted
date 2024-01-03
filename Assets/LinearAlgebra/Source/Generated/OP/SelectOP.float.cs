#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Burst;

namespace LinearAlgebra
{

    /// <summary>Returns b if c is true, a otherwise.</summary>
    /// <param name="a">Value to use if c is false.</param>
    /// <param name="b">Value to use if c is true.</param>
    /// <param name="c">Bool value to choose between a and b.</param>
    /// <returns>The selection between a and b according to bool c.</returns>
    public static partial class SelectOP
    {
        public static floatN select(in floatN a, in floatN b, in boolN c)
        {
            Assume.SameDim(in a, in b);
            Assume.SameDim(in a, in c);

            floatN res = a.tempfloatVec(a.N, true);

            unsafe
            {
                UnsafeSelectOP.selectfloat(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, res.Data.Ptr, a.N);
            }

            return res;
        }

        public static floatMxN select(in floatMxN a, in floatMxN b, in boolMxN c)
        {
            Assume.SameDim(in a, in b);
            Assume.SameDim(in a, in c);

            floatMxN res = a.tempfloatMat(a.M_Rows, a.N_Cols, true);

            unsafe
            {
                UnsafeSelectOP.selectfloat(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, res.Data.Ptr, a.M_Rows * a.N_Cols);
            }

            return res;
        }

        public static floatN select(in floatN a, in floatN b, in bool c)
        {
            Assume.SameDim(in a, in b);

            return c ? b.TempCopy() : a.TempCopy();
        }

        public static floatMxN select(in floatMxN a, in floatMxN b, in bool c)
        {
            Assume.SameDim(in a, in b);

            return c ? b.TempCopy() : a.TempCopy();
        }
    }

    public static unsafe partial class UnsafeSelectOP
    {
        public static void selectfloat([NoAlias] float* a, [NoAlias] float* b, [NoAlias] bool* c, float* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = c[i] ? b[i] : a[i];
        }
    }
}
