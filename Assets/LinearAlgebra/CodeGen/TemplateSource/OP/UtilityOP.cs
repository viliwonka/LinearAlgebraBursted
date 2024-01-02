#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Collections.LowLevel.Unsafe;
//singularFile//

namespace LinearAlgebra
{
    public static class UtilityOP {

        // zeroes out a vector inplace
        //+copyReplace
        public static void zeroInpl(in fProxyN vec) {

            unsafe
            {
                var sizeOf = sizeof(fProxy);
                UnsafeUtility.MemClear(vec.Data.Ptr, vec.N * sizeOf);
            }
        }
        //-copyReplace

    }
}
