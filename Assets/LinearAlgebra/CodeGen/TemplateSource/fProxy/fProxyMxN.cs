using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;

namespace LinearAlgebra
{
    // A m x n matrix
    // m = rows
    // n = cols
    public partial struct fProxyMxN : IDisposable, IUnsafefProxyArray, IMatrix<fProxy> {
        
        public int M_Rows;
        public int N_Cols;

        public UnsafeList<fProxy> Data { get; private set; }

        // Value handle, not a pointer: copying a fProxyMxN (including compiler-inserted
        // defensive copies of `in` parameters) copies this 8-byte handle, which still resolves
        // to the SAME heap-allocated ArenaCore. This retired the old "arena identity captures a
        // dangling stack address" failure mode (docs/rfc-memory-model.md FM2) -- see Arena.cs.
        private Arena _arena;

        public readonly int Length;

        public bool IsSquare => M_Rows == N_Cols;

        int IMatrix<fProxy>.M_Rows => M_Rows;

        int IMatrix<fProxy>.N_Cols => N_Cols;

        public unsafe fProxyMxN(int M_rows, int N_cols, Allocator allocator, bool uninit = false)
        {
            _arena = default;
            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = M_Rows * N_Cols;
            var data = new UnsafeList<fProxy>(Length, allocator, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            Data = data;
        }
        /// <summary>
        /// Creates a new matrix of dimension N
        /// </summary>
        /// <param name="N_cols"></param>
        /// <param name="allocator"></param>
        public unsafe fProxyMxN(int M_rows, int N_cols, in Arena arena, bool uninit = false)
        {
            _arena = arena;

            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = M_Rows * N_Cols;
            var data = new UnsafeList<fProxy>(Length, _arena.Allocator, uninit? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory );
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            Data = data;
        }

        /// <summary>
        /// Creates a copy of vector with new allocation
        /// </summary>
        /// <param name="orig"></param>
        public unsafe fProxyMxN(in fProxyMxN orig, Allocator allocator = Allocator.Invalid)
        {
            // guard a standalone (null-arena) source — was dereferencing null for the default allocator
            if (allocator == Allocator.Invalid)
                allocator = orig._arena.HasCore ? orig._arena.Allocator : Allocator.Temp;

            _arena = orig._arena;
            M_Rows = orig.M_Rows;
            N_Cols = orig.N_Cols;
            Length = orig.Length;
            var data = new UnsafeList<fProxy>(Length, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);
            Data = data;
        }

        public unsafe fProxyMxN Copy()
        {
            if (!_arena.HasCore)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return _arena.fProxyMat(in this);
        }

        public unsafe fProxyMxN TempCopy()
        {
            if (!_arena.HasCore)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return _arena.tempfProxyMat(in this);
        }

        public void Dispose() {
#if LINALG_DEBUG
            // poison the buffer so a read-after-dispose surfaces as NaN instead of stale data
            for (int i = 0; i < Length; i++) this[i] = fProxy.NaN;
#endif
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

        void IMatrix<fProxy>.CopyTo(IMatrix<fProxy> destination) {
            throw new NotImplementedException();
        }

        void IMatrix<fProxy>.CopyFrom(IMatrix<fProxy> source) {
            throw new NotImplementedException();
        }
    }
}