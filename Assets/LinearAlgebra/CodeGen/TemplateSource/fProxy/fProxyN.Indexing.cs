using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    public partial struct fProxyN {

        // Direct array accessor (both int and System.Index, from-end supported).
        public ref fProxy this[int index]
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

        public ref fProxy this[System.Index index]
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

        /// <summary>Handle identity: true iff both views wrap the SAME buffer. NOT elementwise —
        /// use the == operator for an elementwise mask.</summary>
        public override unsafe bool Equals(object obj) =>
            obj is fProxyN other && Data.Ptr == other.Data.Ptr && Data.Length == other.Data.Length;

        public override unsafe int GetHashCode() =>
            unchecked(((int)(long)Data.Ptr * 397) ^ Data.Length);
    }
}