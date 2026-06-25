using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{
    /// <summary>
    /// A zero-alloc index buffer backed by an UnsafeList&lt;int&gt;.
    /// Used as the output buffer for QueryOP index-returning operations
    /// (rowArgMin/Max, kNearestRows, rowsWithinRadius, nonzero, whichTrue, etc.)
    /// so that float-in/int-out operations can live in fProxy templates.
    /// Arena-lifetime: allocate via arena.Indices(n) and the arena owns disposal.
    /// </summary>
    public struct Indices
    {
        private UnsafeList<int> _data;

        public int N => _data.Length;

        public int this[int i]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                if (i < 0 || i >= _data.Length)
                    throw new System.ArgumentOutOfRangeException("i", "Indices index out of range");
                return _data[i];
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                if (i < 0 || i >= _data.Length)
                    throw new System.ArgumentOutOfRangeException("i", "Indices index out of range");
                _data[i] = value;
            }
        }

        public Indices(int n, Allocator allocator = Allocator.Temp)
        {
            _data = new UnsafeList<int>(n, allocator, Unity.Collections.NativeArrayOptions.UninitializedMemory);
            _data.Resize(n, Unity.Collections.NativeArrayOptions.UninitializedMemory);
        }

        public void Dispose()
        {
            _data.Dispose();
        }
    }
}
