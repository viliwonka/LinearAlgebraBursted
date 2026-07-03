using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;

namespace LinearAlgebra
{

    [StructLayout(LayoutKind.Sequential)]
    public partial struct boolN : IDisposable, IUnsafeBoolArray {

        // Value handle, not a pointer: copying a boolN (including compiler-inserted defensive
        // copies of `in` parameters) copies this 8-byte handle, which still resolves to the SAME
        // heap-allocated ArenaCore. This retired the old "arena identity captures a dangling
        // stack address" failure mode (docs/rfc-memory-model.md FM2) -- see Arena.cs.
        private Arena _arena;

        public int N => Data.Length;

        public UnsafeList<bool> Data { get; private set; }

        /// <summary>
        /// Creates a new vector of dimension N
        /// </summary>
        /// <param name="n"></param>
        /// <param name="allocator"></param>
        public unsafe boolN(int n, in Arena arena, bool uninit = false) {

            _arena = arena;

            var allocator = arena.Allocator;

            var data = new UnsafeList<bool>(n, allocator, uninit? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);
            data.Resize(n, NativeArrayOptions.UninitializedMemory);

            Data = data;
        }

        /// <summary>
        /// Creates a copy of vector with new allocation
        /// </summary>
        /// <param name="orig"></param>
        public unsafe boolN(in boolN orig, Allocator allocator = Allocator.Invalid)
        {
            _arena = orig._arena;

            // guard a standalone (null-arena) source — was dereferencing null for the default allocator
            if (allocator == Allocator.Invalid)
                allocator = _arena.HasCore ? _arena.Allocator : Allocator.Temp;

            var data = new UnsafeList<bool>(orig.N, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(orig.N, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);

            Data = data;
        }

        public unsafe boolN Copy()
        {
            if (!_arena.HasCore)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return _arena.boolVec(in this);
        }

        public unsafe boolN TempCopy()
        {
            if (!_arena.HasCore)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return _arena.boolTempVec(in this);
        }

        public void Dispose() {

            Data.Dispose();
        }

    }
}