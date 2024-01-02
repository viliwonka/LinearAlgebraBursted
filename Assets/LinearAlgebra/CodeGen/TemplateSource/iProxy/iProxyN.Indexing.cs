using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct iProxyN {

        /// <summary>
        /// Direct array accessor
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public ref iProxy this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data.ElementAt(index);
        }

        public ref iProxy this[System.Index index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data.ElementAt(index.IsFromEnd? (Data.Length - 1) - index.Value: index.Value);
        }
    }
}