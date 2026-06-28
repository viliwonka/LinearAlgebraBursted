using Unity.Mathematics;
using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    // A m x n matrix
    // m = rows
    // n = cols
    public partial struct iProxyMxN {
       
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

        /// <summary>
        /// Reverse direct array accessor
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public ref iProxy this[System.Index index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data.ElementAt(index.IsFromEnd ? Data.Length - index.Value : index.Value);
        }


        /// <summary>
        /// Direct array accessor
        /// </summary>
        /// <param name="r">row, where m = rows</param>
        /// <param name="c">col, where n = cols</param>
        /// <returns></returns>
        public ref iProxy this[int r, int c]
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
        public ref iProxy this[int r, System.Index indexC]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {

                var c = indexC.IsFromEnd ? N_Cols - indexC.Value : indexC.Value;
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
        public ref iProxy this[System.Index indexR, int c]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {

                var r = indexR.IsFromEnd ? M_Rows - indexR.Value : indexR.Value;
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
        public ref iProxy this[System.Index indexR, System.Index indexC]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {

                var r = indexR.IsFromEnd ? M_Rows - indexR.Value : indexR.Value;
                var c = indexC.IsFromEnd ? N_Cols - indexC.Value : indexC.Value;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                Assume.IndexInsideBounds(new int2(M_Rows, N_Cols), new int2(r, c));
#endif
                return ref Data.ElementAt(r * N_Cols + c);
            }
        }

    }
}