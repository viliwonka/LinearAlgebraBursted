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
            get => ref Data.ElementAt(index);
        }

        public ref iProxy this[System.Index index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data.ElementAt(index.IsFromEnd ? Data.Length - index.Value : index.Value);
        }
    }
}