using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;

namespace BULA
{
    [StructLayout(LayoutKind.Sequential)]
    public partial struct fProxyN : IDisposable, IUnsafefProxyArray {

        private UnsafeList<fProxy> _data;

        public int N => Data.Length;

        public unsafe UnsafeList<fProxy> Data
        {
            get => _data;
            private set => _data = value;
        }

        /// <summary>True while this vector has a live allocation, including views; false for
        /// default(fProxyN) and after Dispose().</summary>
        public unsafe bool IsCreated => _data.IsCreated;

        /// <summary>
        /// Creates a new vector with its own allocation.
        /// </summary>
        public unsafe fProxyN(int n, Allocator allocator = Allocator.Temp, bool uninit = false)
        {
            _data = default;

            var data = new UnsafeList<fProxy>(n, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(n, uninit? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);

            Data = data;
        }

        /// <summary>
        /// Creates a VIEW over <paramref name="viewOf"/>'s memory -- no copy, no
        /// ownership. Element reads/writes go straight to the array. Valid only while the source
        /// array is alive; Dispose() releases nothing (the array keeps ownership). The view is
        /// outside the job-safety system: it does not carry the array's safety handle, so the
        /// caller owns the aliasing/race discipline.
        /// </summary>
        public unsafe fProxyN(NativeArray<fProxy> viewOf)
        {
            _data = new UnsafeList<fProxy>((fProxy*)viewOf.GetUnsafePtr(), viewOf.Length);
        }

        /// <summary>
        /// Creates a copy of the vector with a new allocation.
        /// </summary>
        public unsafe fProxyN(in fProxyN orig, Allocator allocator = Allocator.Temp) {

            _data = default;

            var data = new UnsafeList<fProxy>(orig.N, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(orig.N, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);

            Data = data;
        }

        /// <summary>Returns an Allocator.Temp copy.</summary>
        public unsafe fProxyN Copy()
        {
            return new fProxyN(in this, Allocator.Temp);
        }

        /// <summary>Returns an Allocator.Temp copy.</summary>
        public unsafe fProxyN TempCopy()
        {
            return new fProxyN(in this, Allocator.Temp);
        }

        /// <summary>Copies every component into <paramref name="vec"/> (lengths must match). Fixed-size: never resizes <paramref name="vec"/>.</summary>
        public unsafe void CopyTo(in fProxyN vec)
        {
            if (this.N != vec.N)
                throw new ArgumentException("CopyTo: dimensions do not match!");

            UnsafeUtility.MemCpy(vec.Data.Ptr, Data.Ptr, (long)N * sizeof(fProxy));
        }

        /// <summary>Copies every component from <paramref name="vec"/> (lengths must match). Fixed-size: never resizes this vector.</summary>
        public unsafe void CopyFrom(in fProxyN vec) {

            if (this.N != vec.N)
                throw new ArgumentException("CopyFrom: dimensions do not match!");

            UnsafeUtility.MemCpy(Data.Ptr, vec.Data.Ptr, (long)N * sizeof(fProxy));
        }

        /// <summary>Copies every component into <paramref name="dst"/> (lengths must match).</summary>
        public unsafe void CopyTo(NativeArray<fProxy> dst)
        {
            if (this.N != dst.Length)
                throw new ArgumentException("CopyTo: dst.Length must equal N");

            UnsafeUtility.MemCpy(dst.GetUnsafePtr(), Data.Ptr, (long)N * sizeof(fProxy));
        }

        /// <summary>Copies every component from <paramref name="src"/> (lengths must match).</summary>
        public unsafe void CopyFrom(NativeArray<fProxy> src)
        {
            if (this.N != src.Length)
                throw new ArgumentException("CopyFrom: src.Length must equal N");

            UnsafeUtility.MemCpy(Data.Ptr, src.GetUnsafeReadOnlyPtr(), (long)N * sizeof(fProxy));
        }

        public unsafe void Dispose() {
            _data.Dispose();
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
