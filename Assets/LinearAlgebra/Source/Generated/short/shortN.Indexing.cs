using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct shortN {

        /// <summary>
        /// Direct array accessor
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public ref short this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data.ElementAt(index);
        }

        public ref short this[System.Index index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data.ElementAt(index.IsFromEnd? (Data.Length - 1) - index.Value: index.Value);
        }
    }
}