using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;

namespace LinearAlgebra
{

    // A m x n matrix
    // m = rows
    // n = cols
    public partial struct longMxN : IDisposable, IUnsafelongArray {
        
        public int M_Rows;
        public int N_Cols;

        public UnsafeList<long> Data { get; private set; }

        [NativeDisableUnsafePtrRestriction]
        private unsafe Arena* _arenaPtr;

        public readonly int Length;

        public bool IsSquare => M_Rows == N_Cols;

        public unsafe longMxN(int M_rows, int N_cols, Allocator allocator, bool uninit = false)
        {
            _arenaPtr = null;

            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = M_Rows * N_Cols;
            var data = new UnsafeList<long>(Length, allocator, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            Data = data;
        }
        /// <summary>
        /// Creates a new matrix of dimension N
        /// </summary>
        /// <param name="N_cols"></param>
        /// <param name="allocator"></param>
        public unsafe longMxN(int M_rows, int N_cols, in Arena arena, bool uninit = false)
        {
            fixed (Arena* arenaPtr = &arena)
                _arenaPtr = arenaPtr;

            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = M_Rows * N_Cols;
            var data = new UnsafeList<long>(Length, _arenaPtr->Allocator, uninit? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory );
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            Data = data;
        }

        /// <summary>
        /// Creates a copy of vector with new allocation
        /// </summary>
        /// <param name="orig"></param>
        public unsafe longMxN(in longMxN orig, Allocator allocator = Allocator.Invalid)
        {
            // guard a standalone (null-arena) source — was dereferencing null for the default allocator
            if (allocator == Allocator.Invalid)
                allocator = orig._arenaPtr != null ? orig._arenaPtr->Allocator : Allocator.Temp;

            _arenaPtr = orig._arenaPtr;
            M_Rows = orig.M_Rows;
            N_Cols = orig.N_Cols;
            Length = orig.Length;
            var data = new UnsafeList<long>(Length, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);
            Data = data;
        }

        public unsafe longMxN Copy()
        {

            return _arenaPtr->longMat(in this);
        }

        public unsafe longMxN TempCopy()
        {
            return _arenaPtr->templongMat(in this);
        }

        public void Dispose() {

            Data.Dispose();
        }

        public override string ToString()
        {
            var sb = new System.Text.StringBuilder();
            for (int r = 0; r < M_Rows; r++)
            {
                sb.Append("[ ");
                for (int c = 0; c < N_Cols; c++)
                {
                    if (c > 0) sb.Append("  ");
                    sb.Append(this[r, c]);
                }
                sb.Append(" ]");
                if (r < M_Rows - 1) sb.AppendLine();
            }
            return sb.ToString();
        }
    }
}