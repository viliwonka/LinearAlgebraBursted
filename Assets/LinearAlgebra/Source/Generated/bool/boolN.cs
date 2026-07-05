using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;

namespace LinearAlgebra
{

    [StructLayout(LayoutKind.Sequential)]
    public partial struct boolN : IDisposable, IUnsafeBoolArray {

        // Arena-tracked path: a stable pointer into the arena's ChunkedRecordTable<boolVecRecord>
        // (docs/rfc-memory-model.md §4 Option A). null for a standalone (non-arena) vector, in which
        // case Data resolves to the inline _inlineData field below instead -- see the Data property.
        // Replaces the old `Arena _arena` handle field: retiring it keeps this struct's size
        // unchanged (both are a single pointer-width field), and the record's own `Owner`
        // back-pointer is what Copy()/TempCopy()/the cross-type shortcuts resolve through instead.
        [NativeDisableUnsafePtrRestriction] private unsafe boolVecRecord* _rec;

        // Standalone-path backing store -- the ONLY thing that changes for a non-arena vector.
        // Stays default(UnsafeList<bool>) whenever _rec != null (arena-tracked).
        private UnsafeList<bool> _inlineData;

        public int N => Data.Length;

        public unsafe UnsafeList<bool> Data
        {
            get => _rec != null ? _rec->Data : _inlineData;
            private set { if (_rec != null) _rec->Data = value; else _inlineData = value; }
        }

        // Reconstructs a live Arena handle from this record's owner core -- used by Copy()/
        // TempCopy() and the cross-type allocation shortcuts (boolN.Shortcuts.cs) that used to
        // read a private `_arena` field directly. Only meaningful when _rec != null; callers guard
        // (Copy()/TempCopy()) or otherwise only call this on an arena-backed instance, exactly as
        // the old `_arena` field required.
        private unsafe Arena OwnerArena => new Arena(_rec->Owner);

        /// <summary>
        /// Creates a copy of vector with new allocation
        /// </summary>
        /// <param name="orig"></param>
        public unsafe boolN(in boolN orig, Allocator allocator = Allocator.Invalid)
        {
            _rec = null;
            _inlineData = default;

            // guard a standalone (null-record) source — was dereferencing null for the default allocator
            if (allocator == Allocator.Invalid)
                allocator = orig._rec != null ? orig._rec->Owner->Allocator : Allocator.Temp;

            var data = new UnsafeList<bool>(orig.N, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(orig.N, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);

            Data = data;
        }

        /// <summary>
        /// Arena-tracked constructor. <paramref name="rec"/> is a slot already carved from the
        /// arena's record table by the caller (Arena.boolVec/boolTempVec) -- this ctor only fills
        /// in the record's Data, it does not allocate or own the slot itself.
        /// </summary>
        internal unsafe boolN(int n, boolVecRecord* rec, Allocator allocator, bool uninit = false) {

            _rec = rec;
            _inlineData = default;

            var data = new UnsafeList<bool>(n, allocator, uninit? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);
            data.Resize(n, NativeArrayOptions.UninitializedMemory);

            Data = data;
        }

        /// <summary>Arena-tracked copy constructor -- same pre-allocated-record contract as above.</summary>
        internal unsafe boolN(in boolN orig, boolVecRecord* rec, Allocator allocator) {

            _rec = rec;
            _inlineData = default;

            var data = new UnsafeList<bool>(orig.N, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(orig.N, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);

            Data = data;
        }

        public unsafe boolN Copy()
        {
            if (_rec == null)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return OwnerArena.boolVec(in this);
        }

        public unsafe boolN TempCopy()
        {
            if (_rec == null)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return OwnerArena.boolTempVec(in this);
        }

        public unsafe void Dispose() {

            if (_rec != null)
            {
                // Cache Data BEFORE Free() -- same ordering rationale as every other migrated
                // family's N.Dispose() (e.g. floatN/intN): Free() runs BEFORE the native Dispose()
                // call so an ALIASED double-dispose (a
                // different struct copy sharing this SAME record) throws from the table's own
                // double-Free guard, before any native memory would be touched a second time.
                // (Disposing the SAME variable twice is a separate, safe no-op: this call nulls
                // _rec below, so a second call on that variable takes the standalone branch
                // instead of reaching here at all.) See also Arena.cs's Clear()/ClearTemp(), which
                // use the opposite order (dispose-then-Free) safely for a different reason.
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

    }
}
