using System;
using System.Runtime.CompilerServices;

//alsoExpand[uint]// plain element access, no signed-only ops here.

namespace LinearAlgebra
{

    public partial struct iProxyN {

        // Direct array accessor (both int and System.Index, from-end supported).
        public ref iProxy this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Assume.IndexInsideBounds(Data.Length, index);
#endif
                return ref Data.ElementAt(index);
            }
        }

        public ref iProxy this[System.Index index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                var i = index.IsFromEnd ? Data.Length - index.Value : index.Value;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Assume.IndexInsideBounds(Data.Length, i);
#endif
                return ref Data.ElementAt(i);
            }
        }
    }
}