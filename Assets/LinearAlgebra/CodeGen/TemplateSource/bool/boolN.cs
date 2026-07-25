using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;

namespace BULA
{

    [StructLayout(LayoutKind.Sequential)]
    public partial struct boolN : IDisposable, IUnsafeBoolArray {

        private UnsafeList<bool> _data;

        public int N => Data.Length;

        public unsafe UnsafeList<bool> Data
        {
            get => _data;
            private set => _data = value;
        }

        /// <summary>True while this vector has a live allocation, including views; false for
        /// default(boolN) and after Dispose().</summary>
        public unsafe bool IsCreated => _data.IsCreated;

        /// <summary>
        /// Creates a VIEW over <paramref name="viewOf"/>'s memory -- no copy, no
        /// ownership. Element reads/writes go straight to the array. Valid only while the source
        /// array is alive; Dispose() releases nothing (the array keeps ownership). The view is
        /// outside the job-safety system: it does not carry the array's safety handle, so the
        /// caller owns the aliasing/race discipline.
        /// </summary>
        public unsafe boolN(NativeArray<bool> viewOf)
        {
            _data = new UnsafeList<bool>((bool*)viewOf.GetUnsafePtr(), viewOf.Length);
        }

        /// <summary>
        /// Creates a new vector with its own allocation.
        /// </summary>
        public unsafe boolN(int n, Allocator allocator = Allocator.Temp, bool uninit = false)
        {
            _data = default;

            var data = new UnsafeList<bool>(n, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(n, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);

            Data = data;
        }

        /// <summary>
        /// Creates a copy of vector with new allocation
        /// </summary>
        /// <param name="orig"></param>
        public unsafe boolN(in boolN orig, Allocator allocator = Allocator.Temp)
        {
            _data = default;

            var data = new UnsafeList<bool>(orig.N, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(orig.N, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);

            Data = data;
        }

        /// <summary>Returns an Allocator.Temp copy.</summary>
        public unsafe boolN Copy()
        {
            return new boolN(in this, Allocator.Temp);
        }

        /// <summary>Returns an Allocator.Temp copy.</summary>
        public unsafe boolN TempCopy()
        {
            return new boolN(in this, Allocator.Temp);
        }

        /// <summary>Copies every component into <paramref name="dst"/> (lengths must match).</summary>
        public unsafe void CopyTo(NativeArray<bool> dst)
        {
            if (this.N != dst.Length)
                throw new ArgumentException("CopyTo: dst.Length must equal N");

            UnsafeUtility.MemCpy(dst.GetUnsafePtr(), Data.Ptr, (long)N * sizeof(bool));
        }

        /// <summary>Copies every component from <paramref name="src"/> (lengths must match).</summary>
        public unsafe void CopyFrom(NativeArray<bool> src)
        {
            if (this.N != src.Length)
                throw new ArgumentException("CopyFrom: src.Length must equal N");

            UnsafeUtility.MemCpy(Data.Ptr, src.GetUnsafeReadOnlyPtr(), (long)N * sizeof(bool));
        }

        public unsafe void Dispose() {
            _data.Dispose();
        }

    }
}
