using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;

namespace LinearAlgebra
{
    // A m x n matrix of boolean values
    // m = rows
    // n = cols
    [StructLayout(LayoutKind.Sequential)]
    public partial struct boolMxN : IDisposable, IUnsafeBoolArray
    {
        public int M_Rows;
        public int N_Cols;

        private UnsafeList<bool> _data;

        public unsafe UnsafeList<bool> Data
        {
            get => _data;
            private set => _data = value;
        }

        public readonly int Length;

        public bool IsSquare => M_Rows == N_Cols;

        /// <summary>True while this matrix has a live allocation, including views; false for
        /// default(boolMxN) and after Dispose().</summary>
        public unsafe bool IsCreated => _data.IsCreated;

        /// <summary>
        /// Creates a VIEW over <paramref name="viewOf"/>'s memory as a row-major
        /// M_rows x N_cols matrix (viewOf.Length must equal M_rows*N_cols) -- no copy, no
        /// ownership. Element reads/writes go straight to the array. Valid only while the source
        /// array is alive; Dispose() releases nothing (the array keeps ownership). The view is
        /// outside the job-safety system: it does not carry the array's safety handle, so the
        /// caller owns the aliasing/race discipline.
        /// </summary>
        public unsafe boolMxN(int M_rows, int N_cols, NativeArray<bool> viewOf)
        {
            if (M_rows * N_cols != viewOf.Length)
                throw new ArgumentException("boolMxN view: viewOf.Length must equal M_rows * N_cols");

            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = viewOf.Length;
            _data = new UnsafeList<bool>((bool*)viewOf.GetUnsafePtr(), viewOf.Length);
        }

        public unsafe boolMxN(int M_rows, int N_cols, Allocator allocator = Allocator.Temp, bool uninit = false)
        {
            _data = default;
            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = M_Rows * N_Cols;
            var data = new UnsafeList<bool>(Length, allocator, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            Data = data;
        }

        public unsafe boolMxN(in boolMxN orig, Allocator allocator = Allocator.Temp)
        {
            _data = default;

            M_Rows = orig.M_Rows;
            N_Cols = orig.N_Cols;
            Length = orig.Length;
            var data = new UnsafeList<bool>(Length, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);
            Data = data;
        }

        /// <summary>Returns an Allocator.Temp copy.</summary>
        public unsafe boolMxN Copy()
        {
            return new boolMxN(in this, Allocator.Temp);
        }

        /// <summary>Returns an Allocator.Temp copy.</summary>
        public unsafe boolMxN TempCopy()
        {
            return new boolMxN(in this, Allocator.Temp);
        }

        /// <summary>Copies every element into <paramref name="mat"/> (dimensions must match). Fixed-size: never resizes <paramref name="mat"/>.</summary>
        public unsafe void CopyTo(in boolMxN mat)
        {
            if (M_Rows != mat.M_Rows || N_Cols != mat.N_Cols)
                throw new ArgumentException("CopyTo: dimensions do not match!");

            UnsafeUtility.MemCpy(mat.Data.Ptr, Data.Ptr, (long)M_Rows * N_Cols * sizeof(bool));
        }

        /// <summary>Copies every element from <paramref name="mat"/> (dimensions must match). Fixed-size: never resizes this matrix.</summary>
        public unsafe void CopyFrom(in boolMxN mat)
        {
            if (M_Rows != mat.M_Rows || N_Cols != mat.N_Cols)
                throw new ArgumentException("CopyFrom: dimensions do not match!");

            UnsafeUtility.MemCpy(Data.Ptr, mat.Data.Ptr, (long)M_Rows * N_Cols * sizeof(bool));
        }

        /// <summary>Copies every element (row-major) into <paramref name="dst"/> (dst.Length must equal M_Rows*N_Cols).</summary>
        public unsafe void CopyTo(NativeArray<bool> dst)
        {
            if (Length != dst.Length)
                throw new ArgumentException("CopyTo: dst.Length must equal M_Rows * N_Cols");

            UnsafeUtility.MemCpy(dst.GetUnsafePtr(), Data.Ptr, (long)Length * sizeof(bool));
        }

        /// <summary>Copies every element (row-major) from <paramref name="src"/> (src.Length must equal M_Rows*N_Cols).</summary>
        public unsafe void CopyFrom(NativeArray<bool> src)
        {
            if (Length != src.Length)
                throw new ArgumentException("CopyFrom: src.Length must equal M_Rows * N_Cols");

            UnsafeUtility.MemCpy(Data.Ptr, src.GetUnsafeReadOnlyPtr(), (long)Length * sizeof(bool));
        }

        public unsafe void Dispose()
        {
            _data.Dispose();
        }
    }
}
