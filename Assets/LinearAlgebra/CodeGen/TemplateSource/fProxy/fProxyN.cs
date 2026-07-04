using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;

namespace LinearAlgebra
{
    [StructLayout(LayoutKind.Sequential)]
    public partial struct fProxyN : IDisposable, IUnsafefProxyArray {

        // Arena-tracked path: a stable pointer into the arena's ChunkedRecordTable<fProxyVecRecord>
        // (docs/rfc-memory-model.md §4 Option A). null for a standalone (non-arena) vector, in which
        // case Data resolves to the inline _inlineData field below instead -- see the Data property.
        // Replaces the old `Arena _arena` handle field: retiring it keeps this struct's size
        // unchanged (both are a single pointer-width field), and the record's own `Owner`
        // back-pointer is what Copy()/TempCopy()/the cross-type shortcuts resolve through instead.
        [NativeDisableUnsafePtrRestriction] private unsafe fProxyVecRecord* _rec;

        // Standalone-path backing store -- the ONLY thing that changes for a non-arena vector.
        // Stays default(UnsafeList<fProxy>) whenever _rec != null (arena-tracked).
        private UnsafeList<fProxy> _inlineData;

        public int N => Data.Length;

        public unsafe UnsafeList<fProxy> Data
        {
            get => _rec != null ? _rec->Data : _inlineData;
            private set { if (_rec != null) _rec->Data = value; else _inlineData = value; }
        }

        // Reconstructs a live Arena handle from this record's owner core -- used by Copy()/
        // TempCopy() and the cross-type allocation shortcuts (fProxyN.Shortcuts.cs) that used to
        // read a private `_arena` field directly. Only meaningful when _rec != null; callers guard
        // (Copy()/TempCopy()) or otherwise only call this on an arena-backed instance, exactly as
        // the old `_arena` field required.
        private unsafe Arena OwnerArena => new Arena(_rec->Owner);

        /// <summary>
        /// Creates a new standalone (non-arena) vector with its own allocation.
        /// </summary>
        public unsafe fProxyN(int n, Allocator allocator = Allocator.Invalid, bool uninit = false)
        {
            _rec = null;
            _inlineData = default;

            // standalone (non-arena) vector — fall back to Temp instead of dereferencing a null core.
            if (allocator == Allocator.Invalid)
                allocator = Allocator.Temp;

            var data = new UnsafeList<fProxy>(n, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(n, uninit? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);

            Data = data;
        }

        /// <summary>
        /// Creates a standalone copy of the vector with a new allocation. If <paramref name="orig"/>
        /// is arena-backed and no allocator is given, falls back to that arena's allocator (matching
        /// the historical behavior) -- the COPY itself is always standalone (untracked).
        /// </summary>
        public unsafe fProxyN(in fProxyN orig, Allocator allocator = Allocator.Invalid) {

            _rec = null;
            _inlineData = default;

            // guard a standalone (null-record) source — was dereferencing null for the default allocator
            if(allocator == Allocator.Invalid)
                allocator = orig._rec != null ? orig._rec->Owner->Allocator : Allocator.Temp;

            var data = new UnsafeList<fProxy>(orig.N, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(orig.N, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);

            Data = data;
        }

        /// <summary>
        /// Arena-tracked constructor. <paramref name="rec"/> is a slot already carved from the
        /// arena's record table by the caller (Arena.fProxyVec/fProxyTempVec) -- this ctor only
        /// fills in the record's Data, it does not allocate or own the slot itself.
        /// </summary>
        internal unsafe fProxyN(int n, fProxyVecRecord* rec, Allocator allocator, bool uninit = false) {

            _rec = rec;
            _inlineData = default;

            var data = new UnsafeList<fProxy>(n, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(n, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);

            Data = data;
        }

        /// <summary>Arena-tracked copy constructor -- same pre-allocated-record contract as above.</summary>
        internal unsafe fProxyN(in fProxyN orig, fProxyVecRecord* rec, Allocator allocator) {

            _rec = rec;
            _inlineData = default;

            var data = new UnsafeList<fProxy>(orig.N, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(orig.N, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);

            Data = data;
        }

        public unsafe fProxyN Copy()
        {
            if (_rec == null)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return OwnerArena.fProxyVec(in this);
        }

        public unsafe fProxyN TempCopy()
        {
            if (_rec == null)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return OwnerArena.fProxyTempVec(in this);   // temp pool (was wrongly the persistent Copy path)
        }

        public void CopyTo(in fProxyN vec)
        {
            if (this.N != vec.N)
                throw new ArgumentException("CopyTo: dimensions do not match!");

            vec.Data.CopyFrom(Data);
        }

        public void CopyFrom(in fProxyN vec) {

            if (this.N != vec.N)
                throw new ArgumentException("CopyFrom: dimensions do not match!");

            Data.CopyFrom(vec.Data);
        }

        public unsafe void Dispose() {
#if LINALG_DEBUG
            // poison the buffer so a read-after-dispose surfaces as NaN instead of stale data
            for (int i = 0; i < N; i++) this[i] = fProxy.NaN;
#endif
            if (_rec != null)
            {
                // Cache Data BEFORE Free(): Free() marks the slot dead and (per
                // ChunkedRecordTable's own documented contract) does NOT poison/clear the record's
                // payload today -- but Dispose() must not rely on that as an implicit invariant, so
                // read Data into a local first rather than reading _rec->Data again after the slot
                // is already dead. Free() still runs BEFORE the native Dispose() call: an ALIASED
                // double-dispose (a different struct copy sharing this SAME record) throws HERE,
                // from the table's own double-Free guard, before any native memory would be touched
                // a second time -- instead of the old silent double-free through a stale value-copy
                // in the arena's tracking list. (Disposing the SAME variable twice is a separate,
                // safe no-op: this call nulls _rec below, so a second call on that variable takes
                // the standalone branch instead of reaching here at all.) See also Arena.cs's
                // Clear()/ClearTemp(), which use the opposite order (dispose-then-Free) safely for a
                // different reason -- see the comment there.
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