using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;

//alsoExpand[uint]// data type + construction/copy: no signed-only ops here.

namespace LinearAlgebra
{

    [StructLayout(LayoutKind.Sequential)]
    public partial struct iProxyN : IDisposable, IUnsafeiProxyArray {

        // Arena-tracked path: a stable pointer into the arena's ChunkedRecordTable<iProxyVecRecord>.
        // null for a standalone (non-arena) vector, in which case Data resolves to the inline
        // _inlineData field below instead -- see the Data property.
        [NativeDisableUnsafePtrRestriction] private unsafe iProxyVecRecord* _rec;

        // Standalone-path backing store -- the ONLY thing that changes for a non-arena vector.
        // Stays default(UnsafeList<iProxy>) whenever _rec != null (arena-tracked).
        private UnsafeList<iProxy> _inlineData;

        public int N => Data.Length;

        public unsafe UnsafeList<iProxy> Data
        {
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                AssertRecordAlive();
#endif
                return _rec != null ? _rec->Data : _inlineData;
            }
            private set { if (_rec != null) _rec->Data = value; else _inlineData = value; }
        }

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        // Debug-only liveness check (compiles out of player builds). Throws on a read after
        // Dispose()/Clear()/ClearTemp() on this record. Does not catch a stale handle into a
        // since-recycled slot -- this struct has no generation stamp to check that (see the MxN
        // family's AssertRecordValid, which does). Uses the direct pointer-cast IsAliveFast to
        // avoid a per-element chunk-scan cost, since this getter runs on every read.
        private unsafe void AssertRecordAlive()
        {
            if (_rec != null && !ChunkedRecordTable<iProxyVecRecord>.IsAliveFast(_rec))
                throw new InvalidOperationException("iProxyN.Data: use of disposed/cleared arena allocation");
        }
#endif

        // Reconstructs a live Arena handle from this record's owner core; used by Copy()/
        // TempCopy() and the cross-type allocation shortcuts. Only meaningful when _rec != null --
        // callers must only call this on an arena-backed instance.
        private unsafe Arena OwnerArena => new Arena(_rec->Owner);

        /// <summary>True while this vector has a live allocation (arena-tracked or standalone,
        /// including views); false for default(iProxyN) and after Dispose().</summary>
        public unsafe bool IsCreated
        {
            get
            {
                if (_rec == null) return _inlineData.IsCreated;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                if (!ChunkedRecordTable<iProxyVecRecord>.IsAliveFast(_rec)) return false;
#endif
                return _rec->Data.IsCreated;
            }
        }

        /// <summary>
        /// Creates a standalone VIEW over <paramref name="viewOf"/>'s memory -- no copy, no
        /// ownership. Element reads/writes go straight to the array. Valid only while the source
        /// array is alive; Dispose() releases nothing (the array keeps ownership). The view is
        /// outside the job-safety system: it does not carry the array's safety handle, so the
        /// caller owns the aliasing/race discipline.
        /// </summary>
        public unsafe iProxyN(NativeArray<iProxy> viewOf)
        {
            _rec = null;
            _inlineData = new UnsafeList<iProxy>((iProxy*)viewOf.GetUnsafePtr(), viewOf.Length);
        }

        /// <summary>
        /// Creates a new standalone (non-arena) vector with its own allocation.
        /// </summary>
        public unsafe iProxyN(int n, Allocator allocator = Allocator.Invalid, bool uninit = false)
        {
            _rec = null;
            _inlineData = default;

            // standalone (non-arena) vector — fall back to Temp instead of dereferencing a null core.
            if (allocator == Allocator.Invalid)
                allocator = Allocator.Temp;

            var data = new UnsafeList<iProxy>(n, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(n, uninit? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);

            Data = data;
        }

        /// <summary>
        /// Creates a standalone copy of the vector with a new allocation. If <paramref name="orig"/>
        /// is arena-backed and no allocator is given, falls back to that arena's allocator (matching
        /// the historical behavior) -- the COPY itself is always standalone (untracked).
        /// </summary>
        public unsafe iProxyN(in iProxyN orig, Allocator allocator = Allocator.Invalid) {

            _rec = null;
            _inlineData = default;

            // guard a standalone (null-record) source: fall back to Temp if no allocator is given
            if(allocator == Allocator.Invalid)
                allocator = orig._rec != null ? orig._rec->Owner->Allocator : Allocator.Temp;

            var data = new UnsafeList<iProxy>(orig.N, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(orig.N, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);

            Data = data;
        }

        /// <summary>
        /// Arena-tracked constructor. <paramref name="rec"/> is a slot already carved from the
        /// arena's record table by the caller (Arena.iProxyVec/iProxyTempVec) -- this ctor only
        /// fills in the record's Data, it does not allocate or own the slot itself.
        /// </summary>
        internal unsafe iProxyN(int n, iProxyVecRecord* rec, Allocator allocator, bool uninit = false) {

            _rec = rec;
            _inlineData = default;

            var data = new UnsafeList<iProxy>(n, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(n, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);

            Data = data;
        }

        /// <summary>Arena-tracked copy constructor -- same pre-allocated-record contract as above.</summary>
        internal unsafe iProxyN(in iProxyN orig, iProxyVecRecord* rec, Allocator allocator) {

            _rec = rec;
            _inlineData = default;

            var data = new UnsafeList<iProxy>(orig.N, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(orig.N, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);

            Data = data;
        }

        public unsafe iProxyN Copy()
        {
            if (_rec == null)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return OwnerArena.iProxyVec(in this);
        }

        public unsafe iProxyN TempCopy()
        {
            if (_rec == null)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return OwnerArena.iProxyTempVec(in this);   // temp pool
        }

        public void CopyTo(in iProxyN vec)
        {
            if (this.N != vec.N)
                throw new ArgumentException("CopyTo: dimensions do not match!");

            vec.Data.CopyFrom(Data);
        }

        public void CopyFrom(in iProxyN vec) {

            if (this.N != vec.N)
                throw new ArgumentException("CopyFrom: dimensions do not match!");

            Data.CopyFrom(vec.Data);
        }

        /// <summary>Copies every component into <paramref name="dst"/> (lengths must match).</summary>
        public unsafe void CopyTo(NativeArray<iProxy> dst)
        {
            if (this.N != dst.Length)
                throw new ArgumentException("CopyTo: dst.Length must equal N");

            UnsafeUtility.MemCpy(dst.GetUnsafePtr(), Data.Ptr, (long)N * sizeof(iProxy));
        }

        /// <summary>Copies every component from <paramref name="src"/> (lengths must match).</summary>
        public unsafe void CopyFrom(NativeArray<iProxy> src)
        {
            if (this.N != src.Length)
                throw new ArgumentException("CopyFrom: src.Length must equal N");

            UnsafeUtility.MemCpy(Data.Ptr, src.GetUnsafeReadOnlyPtr(), (long)N * sizeof(iProxy));
        }

        public unsafe void Dispose() {

            if (_rec != null)
            {
                // Cache Data before Free(): Free() marks the slot dead without clearing the
                // payload. Free() runs before the native Dispose() call, so an aliased
                // double-dispose (a different struct copy sharing this record) throws here, via
                // the table's double-Free guard, before any native memory is touched a second
                // time. Disposing the SAME variable twice is a safe no-op: this call nulls _rec,
                // so a second call takes the standalone branch instead.
                var data = _rec->Data;
                _rec->Table->Free(_rec->SelfIndex);
                data.Dispose();
                _rec = null;
            }
            else
            {
                _inlineData.Dispose();
            }
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
