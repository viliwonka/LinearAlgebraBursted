#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using Unity.Collections.LowLevel.Unsafe;
//singularFile//

namespace LinearAlgebra
{
    //+copyReplace
    public static partial class fProxyComp {

        public static void zeroInPlace(in fProxyN vec) {

            unsafe
            {
                var sizeOf = sizeof(fProxy);
                UnsafeUtility.MemClear(vec.Data.Ptr, vec.N * sizeOf);
            }
        }

    }
    //-copyReplace
}
