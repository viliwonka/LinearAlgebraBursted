using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;


namespace LinearAlgebra
{

    [StructLayout(LayoutKind.Sequential)]
    public partial struct shortN : IDisposable, IUnsafeshortArray {

        // Arena-tracked path: a stable pointer into the arena's ChunkedRecordTable<shortVecRecord>.
        // null for a standalone (non-arena) vector, in which case Data resolves to the inline
        // _inlineData field below instead -- see the Data property.
        [NativeDisableUnsafePtrRestriction] private unsafe shortVecRecord* _rec;

        // Standalone-path backing store -- the ONLY thing that changes for a non-arena vector.
        // Stays default(UnsafeList<short>) whenever _rec != null (arena-tracked).
        private UnsafeList<short> _inlineData;

        public int N => Data.Length;

        public unsafe UnsafeList<short> Data
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
            if (_rec != null && !ChunkedRecordTable<shortVecRecord>.IsAliveFast(_rec))
                throw new InvalidOperationException("shortN.Data: use of disposed/cleared arena allocation");
        }
#endif

        // Reconstructs a live Arena handle from this record's owner core; used by Copy()/
        // TempCopy() and the cross-type allocation shortcuts. Only meaningful when _rec != null --
        // callers must only call this on an arena-backed instance.
        private unsafe Arena OwnerArena => new Arena(_rec->Owner);

        /// <summary>
        /// Creates a new standalone (non-arena) vector with its own allocation.
        /// </summary>
        public unsafe shortN(int n, Allocator allocator = Allocator.Invalid, bool uninit = false)
        {
            _rec = null;
            _inlineData = default;

            // standalone (non-arena) vector — fall back to Temp instead of dereferencing a null core.
            if (allocator == Allocator.Invalid)
                allocator = Allocator.Temp;

            var data = new UnsafeList<short>(n, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(n, uninit? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);

            Data = data;
        }

        /// <summary>
        /// Creates a standalone copy of the vector with a new allocation. If <paramref name="orig"/>
        /// is arena-backed and no allocator is given, falls back to that arena's allocator (matching
        /// the historical behavior) -- the COPY itself is always standalone (untracked).
        /// </summary>
        public unsafe shortN(in shortN orig, Allocator allocator = Allocator.Invalid) {

            _rec = null;
            _inlineData = default;

            // guard a standalone (null-record) source: fall back to Temp if no allocator is given
            if(allocator == Allocator.Invalid)
                allocator = orig._rec != null ? orig._rec->Owner->Allocator : Allocator.Temp;

            var data = new UnsafeList<short>(orig.N, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(orig.N, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);

            Data = data;
        }

        /// <summary>
        /// Arena-tracked constructor. <paramref name="rec"/> is a slot already carved from the
        /// arena's record table by the caller (Arena.shortVec/shortTempVec) -- this ctor only
        /// fills in the record's Data, it does not allocate or own the slot itself.
        /// </summary>
        internal unsafe shortN(int n, shortVecRecord* rec, Allocator allocator, bool uninit = false) {

            _rec = rec;
            _inlineData = default;

            var data = new UnsafeList<short>(n, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(n, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);

            Data = data;
        }

        /// <summary>Arena-tracked copy constructor -- same pre-allocated-record contract as above.</summary>
        internal unsafe shortN(in shortN orig, shortVecRecord* rec, Allocator allocator) {

            _rec = rec;
            _inlineData = default;

            var data = new UnsafeList<short>(orig.N, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(orig.N, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);

            Data = data;
        }

        public unsafe shortN Copy()
        {
            if (_rec == null)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return OwnerArena.shortVec(in this);
        }

        public unsafe shortN TempCopy()
        {
            if (_rec == null)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return OwnerArena.shortTempVec(in this);   // temp pool
        }

        public void CopyTo(in shortN vec)
        {
            if (this.N != vec.N)
                throw new ArgumentException("CopyTo: dimensions do not match!");

            vec.Data.CopyFrom(Data);
        }

        public void CopyFrom(in shortN vec) {

            if (this.N != vec.N)
                throw new ArgumentException("CopyFrom: dimensions do not match!");

            Data.CopyFrom(vec.Data);
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
