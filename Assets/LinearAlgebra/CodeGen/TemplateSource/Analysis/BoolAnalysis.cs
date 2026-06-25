#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Burst;
namespace LinearAlgebra
{

    [BurstCompile]
    public static partial class BoolAnalysis {
        
        public static bool IsDiagonal(in boolMxN bm, bool compare = true)
        {
            compare = !compare;
            for (int i = 0; i < bm.M_Rows; i++)
            {
                for (int j = 0; j < bm.N_Cols; j++)
                {
                    if ((bm[i, j] == (i == j)) == compare)
                        return false;
                }
            }

            return true;
        }

        // if all booleans are same with each other
        public static bool IsAllSame<T>(in T x) where T : unmanaged, IUnsafeBoolArray
        {
            for (int i = 1; i < x.Data.Length; i++)
            {
                if (x.Data[i-1] != x.Data[i])
                    return false;
            }
            return true;
        } 

        // if all bools in x are equal to y
        public static bool IsAllEqualTo<T>(in T x, bool y) where T : unmanaged, IUnsafeBoolArray 
        {
            for (int i = 0; i < x.Data.Length; i++)
            {
                if (x.Data[i] != y)
                    return false;
            }
            return true;
        }

        


        public static bool IsAnyEqualTo<T>(in T x, bool y) where T : unmanaged, IUnsafeBoolArray
        {
            for (int i = 0; i < x.Data.Length; i++)
            {
                if (x.Data[i] == y)
                    return true;
            }
            return false;
        }

        // --- QueryOP bridge: bool mask → scalar counts ---------------------------
        // whichTrue (ref intN) lives in Source/OP/QueryOP.Indices.cs (cross-type).

        /// <summary>
        /// Returns the count of true elements in mask (no index buffer needed).
        /// Use whichTrue (in QueryOP.Indices) when you also need the indices.
        /// </summary>
        public static int countTrue(in boolN mask)
        {
            int count = 0;
            for (int i = 0; i < mask.N; i++)
                if (mask.Data[i]) count++;
            return count;
        }

        /// <summary>
        /// Matrix overload: returns the count of true elements in mask.
        /// </summary>
        public static int countTrue(in boolMxN mask)
        {
            int total = mask.M_Rows * mask.N_Cols;
            int count = 0;
            for (int i = 0; i < total; i++)
                if (mask.Data[i]) count++;
            return count;
        }
    }
}
