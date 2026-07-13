using Unity.Mathematics;
using System;
using System.Runtime.CompilerServices;

//alsoExpand[uint]// plain element access, no signed-only ops here.

namespace LinearAlgebra
{

    public partial struct iProxyMxN {

        // Direct array accessors: linear index (int or System.Index, from-end supported)
        // and (row, col) with all int/System.Index combinations.
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