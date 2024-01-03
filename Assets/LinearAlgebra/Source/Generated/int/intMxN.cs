using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;

namespace LinearAlgebra
{

    // A m x n matrix
    // m = rows
    // n = cols
    public partial struct intMxN : IDisposable, IUnsafeintArray {
        
        public int M_Rows;
        public int N_Cols;

        public UnsafeList<int> Data { get; private set; }

        [NativeDisableUnsafePtrRestriction]
        private unsafe Arena* _arenaPtr;

        public readonly int Length;

        public bool IsSquare => M_Rows == N_Cols;

        public unsafe intMxN(int M_rows, int N_cols, Allocator allocator, bool uninit = false)
        {
            _arenaPtr = null;

            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = M_Rows * N_Cols;
            var data = new UnsafeList<int>(Length, allocator, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            Data = data;
        }
        /// <summary>
        /// Creates a new matrix of dimension N
        /// </summary>
        /// <param name="N_cols"></param>
        /// <param name="allocator"></param>
        public unsafe intMxN(int M_rows, int N_cols, in Arena arena, bool uninit = false)
        {
            fixed (Arena* arenaPtr = &arena)
                _arenaPtr = arenaPtr;

            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = M_Rows * N_Cols;
            var data = new UnsafeList<int>(Length, _arenaPtr->Allocator, uninit? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory );
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            Data = data;
        }

        /// <summary>
        /// Creates a copy of vector with new allocation
        /// </summary>
        /// <param name="orig"></param>
        public unsafe intMxN(in intMxN orig, Allocator allocator = Allocator.Invalid)
        {
            if (allocator == Allocator.Invalid)
                allocator = orig._arenaPtr->Allocator;

            _arenaPtr = orig._arenaPtr;
            M_Rows = orig.M_Rows;
            N_Cols = orig.N_Cols;
            Length = orig.Length;
            var data = new UnsafeList<int>(Length, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);
            Data = data;
        }

        public unsafe intMxN Copy()
        {

            return _arenaPtr->intMat(in this);
        }

        public unsafe intMxN TempCopy()
        {
            return _arenaPtr->tempintMat(in this);
        }

        public void Dispose() {

            Data.Dispose();
        }
    }
}