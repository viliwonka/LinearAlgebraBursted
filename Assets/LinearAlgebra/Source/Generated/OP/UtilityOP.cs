#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Collections.LowLevel.Unsafe;
//singularFile//

namespace LinearAlgebra
{
    public static class UtilityOP {

        // zeroes out a vector inplace
        
        public static void zeroInpl(in floatN vec) {

            unsafe
            {
                var sizeOf = sizeof(float);
                UnsafeUtility.MemClear(vec.Data.Ptr, vec.N * sizeOf);
            }
        }
        
        public static void zeroInpl(in doubleN vec) {

            unsafe
            {
                var sizeOf = sizeof(double);
                UnsafeUtility.MemClear(vec.Data.Ptr, vec.N * sizeOf);
            }
        }
        

    }
}
