using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct fProxyN {

        /// <summary>
        /// Direct array accessor
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public ref fProxy this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data.ElementAt(index);
        }

        public ref fProxy this[System.Index index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data.ElementAt(index.IsFromEnd? (Data.Length - 1) - index.Value: index.Value);
        }
    }
}