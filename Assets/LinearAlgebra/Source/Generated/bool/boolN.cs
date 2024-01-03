using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;

namespace LinearAlgebra
{

    [StructLayout(LayoutKind.Sequential)]
    public partial struct boolN : IDisposable, IUnsafeBoolArray {

        [NativeDisableUnsafePtrRestriction]
        private unsafe Arena* _arenaPtr;

        public int N => Data.Length;
        
        public UnsafeList<bool> Data { get; private set; }

        /// <summary>
        /// Creates a new vector of dimension N
        /// </summary>
        /// <param name="n"></param>
        /// <param name="allocator"></param>
        public unsafe boolN(int n, in Arena arena, bool uninit = false) {

            fixed (Arena* arenaPtr = &arena)
                _arenaPtr = arenaPtr;

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
            _arenaPtr = orig._arenaPtr;

            if (allocator == Allocator.Invalid)
                allocator = _arenaPtr->Allocator;

            var data = new UnsafeList<bool>(orig.N, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(orig.N, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);

            Data = data;
        }

        public unsafe boolN Copy() => _arenaPtr->BoolVector(in this);
        
        public unsafe boolN TempCopy() => _arenaPtr->TempBoolVector(in this);

        public void Dispose() {

            Data.Dispose();
        }

    }
}