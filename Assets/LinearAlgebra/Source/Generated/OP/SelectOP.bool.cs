#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Burst;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{

    // Class summary lives on the fProxy partial (SelectOP.fProxy.cs). Bool-buffer select overloads.
    public static partial class Select
    {
        public static boolN select(in boolN a, in boolN b, in boolN c)
        {
            Assume.SameDim(in a, in b);
            Assume.SameDim(in a, in c);

            boolN res = a.boolTempVec(a.N, true);

            unsafe
            {
                UnsafeSelectOP.selectBool(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, res.Data.Ptr, a.N);
            }

            return res;
        }

        public static boolMxN select(in boolMxN a, in boolMxN b, in boolMxN c)
        {
            Assume.SameDim(in a, in b);
            Assume.SameDim(in a, in c);

            boolMxN res = a.boolTempMat(a.M_Rows, a.N_Cols, true);

            unsafe
            {
                UnsafeSelectOP.selectBool(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, res.Data.Ptr, a.M_Rows * a.N_Cols);
            }

            return res;
        }

        public static boolN select(in boolN a, in boolN b, in bool c)
        {
            Assume.SameDim(in a, in b);

            return c ? b.TempCopy() : a.TempCopy();
        }

        public static boolMxN select(in boolMxN a, in boolMxN b, in bool c)
        {
            Assume.SameDim(in a, in b);

            return c ? b.TempCopy() : a.TempCopy();
        }
    }
}

namespace LinearAlgebra.Internal
{
    public static unsafe partial class UnsafeSelectOP
    {
        public static void selectBool([NoAlias] bool* a, [NoAlias] bool* b, [NoAlias] bool* c, bool* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = c[i] ? b[i] : a[i];
        }
    }
}
