using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;

namespace LinearAlgebra
{
    // A m x n matrix of boolean values
    // m = rows
    // n = cols
    //
    // [StructLayout(Sequential)]: pins field order/packing explicitly (matches boolN's existing
    // attribute) instead of leaving it to the compiler's default Auto layout. This is what makes
    // the trailing padding hole below -- and therefore _gen's placement in it -- a guarantee rather
    // than an implementation detail: Auto layout is free to reorder/repack fields, so relying on
    // "the compiler currently happens to leave 4 bytes at the end" without Sequential would be
    // fragile. See _gen's own doc comment for the padding-hole analysis.
    [StructLayout(LayoutKind.Sequential)]
    public partial struct boolMxN : IDisposable, IUnsafeBoolArray
    {
        public int M_Rows;
        public int N_Cols;

        // Arena-tracked path -- see boolN.cs's `_rec` doc comment for the full rationale (same
        // Option A record-pointer design, mirrored here for the matrix family). null for a
        // standalone (non-arena) matrix, in which case Data resolves to _inlineData instead.
        [NativeDisableUnsafePtrRestriction] private unsafe boolMatRecord* _rec;

        // Standalone-path backing store. Stays default(UnsafeList<bool>) whenever _rec != null.
        private UnsafeList<bool> _inlineData;

        public unsafe UnsafeList<bool> Data
        {
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                AssertRecordValid();
#endif
                return _rec != null ? _rec->Data : _inlineData;
            }
            private set { if (_rec != null) _rec->Data = value; else _inlineData = value; }
        }

        // Reconstructs a live Arena handle from this record's owner core -- used by Copy()/
        // TempCopy() and the cross-type allocation shortcuts (boolMxN.Shortcuts.cs) that used to
        // read a private `_arena` field directly. Only meaningful when _rec != null.
        private unsafe Arena OwnerArena => new Arena(_rec->Owner);

        public readonly int Length;

        // Generation stamp captured from the record's slot at construction time (0/unused on the
        // standalone path). Free-riding on real spare bytes, not a size-growing addition: with
        // [StructLayout(Sequential)] and natural alignment, M_Rows(4)+N_Cols(4)+_rec(8)+
        // _inlineData(24)+Length(4) = 44 bytes, and this struct's 8-byte alignment (forced by the
        // pointer/UnsafeList fields) rounds that up to 48 regardless -- there were already 4 unused
        // trailing bytes here (docs/rfc-memory-model.md §6.2's "padding analysis", confirmed by
        // ArenaLayoutTests.MatrixStructsAreExpectedSize staying at 48 with this field present). Only
        // meaningful when _rec != null: AssertRecordValid() compares it against the table's CURRENT
        // GetGeneration(SelfIndex) to detect a stale handle into a since-recycled slot (Alive alone,
        // boolN's option, cannot tell "still the same allocation" from "a new one that reused this
        // slot number").
        private readonly int _gen;

        public bool IsSquare => M_Rows == N_Cols;

        public unsafe boolMxN(int M_rows, int N_cols, Allocator allocator, bool uninit = false)
        {
            _rec = null;
            _inlineData = default;
            _gen = 0; // standalone (non-arena): never read (AssertRecordValid short-circuits on _rec == null)
            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = M_Rows * N_Cols;
            var data = new UnsafeList<bool>(Length, allocator, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            Data = data;
        }

        /// <summary>
        /// Arena-tracked constructor. <paramref name="rec"/> is a slot already carved from the
        /// arena's record table by the caller (Arena.boolMat/boolTempMat) -- this ctor only fills
        /// in the record's Data, it does not allocate or own the slot itself.
        /// </summary>
        internal unsafe boolMxN(int M_rows, int N_cols, boolMatRecord* rec, Allocator allocator, bool uninit = false)
        {
            _rec = rec;
            _inlineData = default;
            _gen = rec->Table->GetGeneration(rec->SelfIndex); // stamp this fresh allocation's generation

            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = M_Rows * N_Cols;
            var data = new UnsafeList<bool>(Length, allocator, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            Data = data;
        }

        public unsafe boolMxN(in boolMxN orig, Allocator allocator = Allocator.Invalid)
        {
            _rec = null;
            _inlineData = default;
            _gen = 0; // standalone (non-arena): never read (AssertRecordValid short-circuits on _rec == null)

            // guard a standalone (null-record) source — was dereferencing null for the default allocator
            if (allocator == Allocator.Invalid)
                allocator = orig._rec != null ? orig._rec->Owner->Allocator : Allocator.Temp;

            M_Rows = orig.M_Rows;
            N_Cols = orig.N_Cols;
            Length = orig.Length;
            var data = new UnsafeList<bool>(Length, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);
            Data = data;
        }

        /// <summary>Arena-tracked copy constructor -- same pre-allocated-record contract as above.</summary>
        internal unsafe boolMxN(in boolMxN orig, boolMatRecord* rec, Allocator allocator)
        {
            _rec = rec;
            _inlineData = default;
            _gen = rec->Table->GetGeneration(rec->SelfIndex); // stamp this fresh allocation's generation (NOT orig's)

            M_Rows = orig.M_Rows;
            N_Cols = orig.N_Cols;
            Length = orig.Length;
            var data = new UnsafeList<bool>(Length, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);
            Data = data;
        }

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        // Editor/test-only guard (ENABLE_UNITY_COLLECTIONS_CHECKS is defined automatically by the
        // Unity Editor, including every test run, and compiles out of player builds entirely --
        // struct size is identical in both configs either way, since _gen occupies a byte range
        // that was already-wasted alignment padding, not a size-growing addition -- see _gen's own
        // doc comment). Unlike boolN (Alive-only), boolMxN also has
        // a free generation stamp (_gen, see its doc comment) to check, so this catches BOTH bug
        // classes: a read after Dispose()/Clear()/ClearTemp() on THIS record (Alive fails), and a
        // stale handle into a slot that was freed and later recycled by a fresh Allocate() for a
        // DIFFERENT allocation (Alive is true again, but the generation moved on).
        //
        // Uses ChunkedRecordTable's IsAliveFast/GenerationFast(TRecord*) -- direct pointer casts,
        // no index, no chunk-scan lookup -- rather than the index-based IsAlive(int)/
        // GetGeneration(int) (i.e. NOT _rec->Table->IsAlive(_rec->SelfIndex) etc.): this getter
        // runs on EVERY read (i.e. per element, since an indexer routes through Data), so the
        // index-based path's chunk scan would be a real per-element cost. See IsAliveFast's own
        // doc comment (ChunkedRecordTable.cs) for the container-of rationale.
        private unsafe void AssertRecordValid()
        {
            if (_rec != null && (!ChunkedRecordTable<boolMatRecord>.IsAliveFast(_rec) || ChunkedRecordTable<boolMatRecord>.GenerationFast(_rec) != _gen))
                throw new InvalidOperationException("boolMxN.Data: use of disposed/cleared arena allocation");
        }
#endif

        public unsafe boolMxN Copy()
        {
            if (_rec == null)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return OwnerArena.boolMat(in this);
        }

        public unsafe boolMxN TempCopy()
        {
            if (_rec == null)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return OwnerArena.boolTempMat(in this);
        }

        public unsafe void Dispose()
        {
            if (_rec != null)
            {
                // Cache Data BEFORE Free() -- same ordering rationale as boolN.Dispose() and every
                // other migrated family's MxN.Dispose() (e.g. floatMxN/intMxN): guards against an
                // ALIASED double-dispose throwing before any native memory is touched a second
                // time. See Arena.cs's
                // Clear()/ClearTemp(), which use the opposite order safely for a different reason.
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
