using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct intN {

        // Direct array accessor (both int and System.Index, from-end supported).
        public ref int this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data.ElementAt(index);
        }

        public ref int this[System.Index index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data.ElementAt(index.IsFromEnd ? Data.Length - index.Value : index.Value);
        }
    }
}