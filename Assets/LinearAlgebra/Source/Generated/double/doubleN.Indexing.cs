using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct doubleN {

        // Direct array accessor (both int and System.Index, from-end supported).
        public ref double this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data.ElementAt(index);
        }

        public ref double this[System.Index index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data.ElementAt(index.IsFromEnd ? Data.Length - index.Value : index.Value);
        }
    }
}