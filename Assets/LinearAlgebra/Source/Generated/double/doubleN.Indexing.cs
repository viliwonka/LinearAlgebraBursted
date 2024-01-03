using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct doubleN {

        /// <summary>
        /// Direct array accessor
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public ref double this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data.ElementAt(index);
        }

        public ref double this[System.Index index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data.ElementAt(index.IsFromEnd? (Data.Length - 1) - index.Value: index.Value);
        }
    }
}