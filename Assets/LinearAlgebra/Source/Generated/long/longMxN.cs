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

        // Value handle, not a pointer: copying an longMxN (including compiler-inserted
        // defensive copies of `in` parameters) copies this 8-byte handle, which still resolves
        // to the SAME heap-allocated ArenaCore. This retired the old "arena identity captures a
        // dangling stack address" failure mode (docs/rfc-memory-model.md FM2) -- see Arena.cs.
        private Arena _arena;

        public readonly int Length;

        public bool IsSquare => M_Rows == N_Cols;

        public unsafe longMxN(int M_rows, int N_cols, Allocator allocator, bool uninit = false)
        {
            _arena = default;

            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = M_Rows * N_Cols;
            var data = new UnsafeList<long>(Length, allocator, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            Data = data;
        }
        /// <summary>
        /// Creates a new arena-backed M_Rows x N_Cols matrix.
        /// </summary>
        public unsafe longMxN(int M_rows, int N_cols, in Arena arena, bool uninit = false)
        {
            _arena = arena;

            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = M_Rows * N_Cols;
            var data = new UnsafeList<long>(Length, _arena.Allocator, uninit? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory );
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            Data = data;
        }

        /// <summary>
        /// Creates a copy of the matrix with a new allocation.
        /// </summary>
        public unsafe longMxN(in longMxN orig, Allocator allocator = Allocator.Invalid)
        {
            // guard a standalone (null-arena) source — was dereferencing null for the default allocator
            if (allocator == Allocator.Invalid)
                allocator = orig._arena.HasCore ? orig._arena.Allocator : Allocator.Temp;

            _arena = orig._arena;
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
            if (!_arena.HasCore)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return _arena.longMat(in this);
        }

        public unsafe longMxN TempCopy()
        {
            if (!_arena.HasCore)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return _arena.longTempMat(in this);
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