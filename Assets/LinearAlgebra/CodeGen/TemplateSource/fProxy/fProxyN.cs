using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;

namespace LinearAlgebra
{
    [StructLayout(LayoutKind.Sequential)]
    public partial struct fProxyN : IDisposable, IUnsafefProxyArray {

        // Arena-tracked path: a stable pointer into the arena's ChunkedRecordTable<fProxyVecRecord>
        // (docs/dev/rfc-memory-model.md §4 Option A). null for a standalone (non-arena) vector, in which
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
        // Editor/test-only guard (ENABLE_UNITY_COLLECTIONS_CHECKS is defined automatically by the
        // Unity Editor, including every test run, and compiles out of player builds entirely --
        // struct size is identical in both configs either way, since this adds no field). fProxyN
        // has no spare bits to pack a generation
        // stamp into (32B = 8 _rec + 24 UnsafeList<fProxy>, exactly -- see docs/dev/rfc-memory-model.md
        // §6.2 and ArenaLayoutTests.VectorStructsAreExpectedSize), so this only checks Alive: it
        // catches a read after Dispose()/Clear()/ClearTemp() on THIS record, but not a stale handle
        // into a slot that has since been recycled by a fresh Allocate() (that needs a generation
        // stamp, which fProxyMxN/fProxyBSR carry in their own padding hole -- see those types).
        //
        // Uses ChunkedRecordTable's IsAliveFast(TRecord*) -- a direct pointer cast, no index, no
        // chunk-scan lookup -- rather than the index-based IsAlive(int) (i.e. NOT _rec->Table->
        // IsAlive(_rec->SelfIndex)): this getter runs on EVERY read (i.e. per element, since an
        // indexer routes through Data), so the index-based path's chunk scan would be a real
        // per-element cost. See IsAliveFast's own doc comment (ChunkedRecordTable.cs) for the
        // container-of rationale.
        private unsafe void AssertRecordAlive()
        {
            if (_rec != null && !ChunkedRecordTable<fProxyVecRecord>.IsAliveFast(_rec))
                throw new InvalidOperationException("fProxyN.Data: use of disposed/cleared arena allocation");
        }
#endif

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
            // LINALG_DEBUG NaN-poison-on-dispose removed (2026-07-05): the symbol was defined
            // nowhere in the project, so that block was dead code that had never executed.
            // Superseded by the record table's own unconditional guards below -- a double-dispose
            // (aliased or not) throws deterministically via Free()'s double-Free check, in every
            // build config, not just a debug one -- plus the ENABLE_UNITY_COLLECTIONS_CHECKS
            // generational overlay on the Data getter, which catches a stale read (use-after-
            // dispose/Clear, or a handle into a since-recycled slot) instead of returning garbage.
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