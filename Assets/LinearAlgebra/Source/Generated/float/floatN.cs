using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;

namespace LinearAlgebra
{
    [StructLayout(LayoutKind.Sequential)]
    public partial struct floatN : IDisposable, IUnsafefloatArray {

        [NativeDisableUnsafePtrRestriction]
        private unsafe Arena* _arenaPtr;

        public int N => Data.Length;
        
        public UnsafeList<float> Data { get; private set; }

        /// <summary>
        /// Creates a new vector of dimension N
        /// </summary>
        /// <param name="n"></param>
        /// <param name="allocator"></param>
        public unsafe floatN(int n, in Arena arena, bool uninit = false) { 

            fixed (Arena* arenaPtr = &arena)
                _arenaPtr = arenaPtr;

            var allocator = arena.Allocator;
            //var allocator1 = _arenaPtr->Allocator;
            //UnityEngine.Debug.Log($"Vector: {allocator}");
            //UnityEngine.Debug.Log($"Vector: {allocator1}");

            var data = new UnsafeList<float>(n, allocator, uninit? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);
            data.Resize(n, NativeArrayOptions.UninitializedMemory);

            Data = data;
        }

        /// <summary>
        /// Creates a copy of vector with new allocation
        /// </summary>
        /// <param name="orig"></param>
        public unsafe floatN(in floatN orig, Allocator allocator = Allocator.Invalid) {

            _arenaPtr = orig._arenaPtr;

            // guard a standalone (null-arena) source — was dereferencing null for the default allocator
            if(allocator == Allocator.Invalid)
                allocator = _arenaPtr != null ? _arenaPtr->Allocator : Allocator.Temp;

            //var allocator1 = _arenaPtr->Allocator;
            //UnityEngine.Debug.Log($"Vector: {allocator}");
            //UnityEngine.Debug.Log($"Vector: {allocator1}");

            var data = new UnsafeList<float>(orig.N, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(orig.N, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);

            Data = data;
        }

        /// <summary>
        /// Creates a copy of vector with new allocation
        /// </summary>
        /// <param name="orig"></param>
        public unsafe floatN(int n, Allocator allocator = Allocator.Invalid, bool uninit = false)
        {
            _arenaPtr = null;

            // standalone (non-arena) vector — fall back to Temp instead of dereferencing the null
            // _arenaPtr (which crashed for the default Allocator.Invalid).
            if (allocator == Allocator.Invalid)
                allocator = Allocator.Temp;

            var data = new UnsafeList<float>(n, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(n, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);
            
            Data = data;
        }

        public unsafe floatN Copy()
        {
            return _arenaPtr->floatVec(in this);
        }

        public unsafe floatN TempCopy()
        {
            return _arenaPtr->tempfloatVec(in this);   // temp pool (was wrongly the persistent Copy path)
        }

        public void CopyTo(in floatN vec)
        {
            if (this.N != vec.N)
                throw new ArgumentException("CopyTo: dimensions do not match!");

            vec.Data.CopyFrom(Data);
        }

        public void CopyFrom(in floatN vec) {

            if (this.N != vec.N)
                throw new ArgumentException("CopyFrom: dimensions do not match!");

            Data.CopyFrom(vec.Data);
        }

        public void Dispose() {
#if LINALG_DEBUG
            // poison the buffer so a read-after-dispose surfaces as NaN instead of stale data
            for (int i = 0; i < N; i++) this[i] = float.NaN;
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