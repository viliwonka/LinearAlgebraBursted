using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;

namespace LinearAlgebra
{
    [StructLayout(LayoutKind.Sequential)]
    public partial struct doubleN : IDisposable, IUnsafedoubleArray {

        // Value handle, not a pointer: copying a doubleN (including compiler-inserted defensive
        // copies of `in` parameters) copies this 8-byte handle, which still resolves to the SAME
        // heap-allocated ArenaCore. This retired the old "arena identity captures a dangling
        // stack address" failure mode (docs/rfc-memory-model.md FM2) -- see Arena.cs.
        private Arena _arena;

        public int N => Data.Length;

        public UnsafeList<double> Data { get; private set; }

        /// <summary>
        /// Creates a new vector of dimension N
        /// </summary>
        /// <param name="n"></param>
        /// <param name="allocator"></param>
        public unsafe doubleN(int n, in Arena arena, bool uninit = false) {

            _arena = arena;

            var allocator = arena.Allocator;

            var data = new UnsafeList<double>(n, allocator, uninit? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);
            data.Resize(n, NativeArrayOptions.UninitializedMemory);

            Data = data;
        }

        /// <summary>
        /// Creates a copy of vector with new allocation
        /// </summary>
        /// <param name="orig"></param>
        public unsafe doubleN(in doubleN orig, Allocator allocator = Allocator.Invalid) {

            _arena = orig._arena;

            // guard a standalone (null-arena) source — was dereferencing null for the default allocator
            if(allocator == Allocator.Invalid)
                allocator = _arena.HasCore ? _arena.Allocator : Allocator.Temp;

            var data = new UnsafeList<double>(orig.N, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(orig.N, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);

            Data = data;
        }

        /// <summary>
        /// Creates a copy of vector with new allocation
        /// </summary>
        /// <param name="orig"></param>
        public unsafe doubleN(int n, Allocator allocator = Allocator.Invalid, bool uninit = false)
        {
            _arena = default;

            // standalone (non-arena) vector — fall back to Temp instead of dereferencing a null core.
            if (allocator == Allocator.Invalid)
                allocator = Allocator.Temp;

            var data = new UnsafeList<double>(n, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(n, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);

            Data = data;
        }

        public unsafe doubleN Copy()
        {
            if (!_arena.HasCore)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return _arena.doubleVec(in this);
        }

        public unsafe doubleN TempCopy()
        {
            if (!_arena.HasCore)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return _arena.tempdoubleVec(in this);   // temp pool (was wrongly the persistent Copy path)
        }

        public void CopyTo(in doubleN vec)
        {
            if (this.N != vec.N)
                throw new ArgumentException("CopyTo: dimensions do not match!");

            vec.Data.CopyFrom(Data);
        }

        public void CopyFrom(in doubleN vec) {

            if (this.N != vec.N)
                throw new ArgumentException("CopyFrom: dimensions do not match!");

            Data.CopyFrom(vec.Data);
        }

        public void Dispose() {
#if LINALG_DEBUG
            // poison the buffer so a read-after-dispose surfaces as NaN instead of stale data
            for (int i = 0; i < N; i++) this[i] = double.NaN;
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