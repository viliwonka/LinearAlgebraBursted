#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Collections.LowLevel.Unsafe;
//singularFile//

namespace LinearAlgebra
{
    public static partial class fProxyComp {

        
        public static void zeroInPlace(in floatN vec) {

            unsafe
            {
                var sizeOf = sizeof(float);
                UnsafeUtility.MemClear(vec.Data.Ptr, vec.N * sizeOf);
            }
        }
        
        public static void zeroInPlace(in doubleN vec) {

            unsafe
            {
                var sizeOf = sizeof(double);
                UnsafeUtility.MemClear(vec.Data.Ptr, vec.N * sizeOf);
            }
        }
        

    }
}
