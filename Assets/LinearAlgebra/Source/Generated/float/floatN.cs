using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;

namespace LinearAlgebra
{
    [StructLayout(LayoutKind.Sequential)]
    public partial struct floatN : IDisposable, IUnsafefloatArray {

        // Value handle, not a pointer: copying a floatN (including compiler-inserted defensive
        // copies of `in` parameters) copies this 8-byte handle, which still resolves to the SAME
        // heap-allocated ArenaCore. This retired the old "arena identity captures a dangling
        // stack address" failure mode (docs/rfc-memory-model.md FM2) -- see Arena.cs.
        private Arena _arena;

        public int N => Data.Length;

        public UnsafeList<float> Data { get; private set; }

        /// <summary>
        /// Creates a new arena-backed vector of dimension n.
        /// </summary>
        public unsafe floatN(int n, in Arena arena, bool uninit = false) {

            _arena = arena;

            var allocator = arena.Allocator;

            var data = new UnsafeList<float>(n, allocator, uninit? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);
            data.Resize(n, NativeArrayOptions.UninitializedMemory);

            Data = data;
        }

        /// <summary>
        /// Creates a copy of the vector with a new allocation.
        /// </summary>
        public unsafe floatN(in floatN orig, Allocator allocator = Allocator.Invalid) {

            _arena = orig._arena;

            // guard a standalone (null-arena) source — was dereferencing null for the default allocator
            if(allocator == Allocator.Invalid)
                allocator = _arena.HasCore ? _arena.Allocator : Allocator.Temp;

            var data = new UnsafeList<float>(orig.N, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(orig.N, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);

            Data = data;
        }

        /// <summary>
        /// Creates a new standalone (non-arena) vector with its own allocation.
        /// </summary>
        public unsafe floatN(int n, Allocator allocator = Allocator.Invalid, bool uninit = false)
        {
            _arena = default;

            // standalone (non-arena) vector — fall back to Temp instead of dereferencing a null core.
            if (allocator == Allocator.Invalid)
                allocator = Allocator.Temp;

            var data = new UnsafeList<float>(n, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(n, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);

            Data = data;
        }

        public unsafe floatN Copy()
        {
            if (!_arena.HasCore)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return _arena.floatVec(in this);
        }

        public unsafe floatN TempCopy()
        {
            if (!_arena.HasCore)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return _arena.tempfloatVec(in this);   // temp pool (was wrongly the persistent Copy path)
        }

        public void CopyTo(in floatN vec)
        {
            if (this.N != vec.N)
                throw new ArgumentException("CopyTo: dimensions do not match!");

            vec.Data.CopyFrom(Data);
        }

        public void CopyFrom(in floatN vec) {

            if (this.N != vec.N)
                throw new ArgumentException("CopyFrom: dimensions do not match!");

            Data.CopyFrom(vec.Data);
        }

        public void Dispose() {
#if LINALG_DEBUG
            // poison the buffer so a read-after-dispose surfaces as NaN instead of stale data
            for (int i = 0; i < N; i++) this[i] = float.NaN;
#endif
            Data.Dispose();
        }

        public override string ToString()
        {
            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < N; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(this[i]);
            }
            return sb.ToString();
        }
    }
}