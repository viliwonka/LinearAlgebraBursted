using Unity.Mathematics;
using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    // A m x n matrix
    // m = rows
    // n = cols
    public partial struct fProxyMxN : IDisposable, IUnsafefProxyArray {
       
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

        /// <summary>
        /// Reverse direct array accessor
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public ref fProxy this[System.Index index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data.ElementAt(index.IsFromEnd ? Data.Length - 1 - index.Value : index.Value);
        }


        /// <summary>
        /// Direct array accessor
        /// </summary>
        /// <param name="r">row, where m = rows</param>
        /// <param name="c">col, where n = cols</param>
        /// <returns></returns>
        public ref fProxy this[int r, int c]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {

#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Assume.IndexInsideBounds(new int2(M_Rows, N_Cols), new int2(r, c));
#endif
                return ref Data.ElementAt(r * N_Cols + c);
            }
        }
        /// <summary>
        /// Direct array accessor
        /// </summary>
        /// <param name="r">row, where m = rows</param>
        /// <param name="indexC">col, where n = cols</param>
        /// <returns></returns>
        public ref fProxy this[int r, System.Index indexC]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {

#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Assume.IndexInsideBounds(new int2(M_Rows, N_Cols), new int2(r, indexC.Value));
#endif
                var c = indexC.IsFromEnd? N_Cols - 1 - indexC.Value : indexC.Value;

                return ref Data.ElementAt(r * N_Cols + c);
            }
        }

        /// <summary>
        /// Direct array accessor
        /// </summary>
        /// <param name="r">row, where m = rows</param>
        /// <param name="indexC">col, where n = cols</param>
        /// <returns></returns>
        public ref fProxy this[System.Index indexR, int c]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {

#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Assume.IndexInsideBounds(new int2(M_Rows, N_Cols), new int2(indexR.Value, c));
#endif
                var r = indexR.IsFromEnd ? M_Rows - 1 - indexR.Value : indexR.Value;

                return ref Data.ElementAt(r * N_Cols + c);
            }
        }

        /// <summary>
        /// Direct array accessor
        /// </summary>
        /// <param name="r">row, where m = rows</param>
        /// <param name="indexC">col, where n = cols</param>
        /// <returns></returns>
        public ref fProxy this[System.Index indexR, System.Index indexC]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {

#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Assume.IndexInsideBounds(new int2(M_Rows, N_Cols), new int2(indexR.Value, indexC.Value));
#endif
                var r = indexR.IsFromEnd ? M_Rows - 1 - indexR.Value : indexR.Value;
                var c = indexC.IsFromEnd ? N_Cols - 1 - indexC.Value : indexC.Value;

                return ref Data.ElementAt(r * N_Cols + c);
            }
        }

    }
}