using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;


namespace LinearAlgebra
{

    // A m x n matrix
    // m = rows
    // n = cols
    //
    // [StructLayout(Sequential)]: pins field order/packing explicitly (matches uintN's existing
    // attribute) instead of leaving it to the compiler's default Auto layout. This is what makes
    // the trailing padding hole below -- and therefore _gen's placement in it -- a guarantee rather
    // than an implementation detail: Auto layout is free to reorder/repack fields, so relying on
    // "the compiler currently happens to leave 4 bytes at the end" without Sequential would be
    // fragile. See _gen's own doc comment for the padding-hole analysis.
    [StructLayout(LayoutKind.Sequential)]
    public partial struct uintMxN : IDisposable, IUnsafeuintArray {

        public int M_Rows;
        public int N_Cols;

        // Arena-tracked path -- see uintN.cs's `_rec` doc comment for the full rationale (same
        // Option A record-pointer design, mirrored here for the matrix family). null for a
        // standalone (non-arena) matrix, in which case Data resolves to _inlineData instead.
        [NativeDisableUnsafePtrRestriction] private unsafe uintMatRecord* _rec;

        // Standalone-path backing store. Stays default(UnsafeList<uint>) whenever _rec != null.
        private UnsafeList<uint> _inlineData;

        public unsafe UnsafeList<uint> Data
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
        // TempCopy() and the cross-type allocation shortcuts (uintMxN.Shortcuts.cs) that used to
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
        // uintN's option, cannot tell "still the same allocation" from "a new one that reused this
        // slot number").
        private readonly int _gen;

        public bool IsSquare => M_Rows == N_Cols;

        public unsafe uintMxN(int M_rows, int N_cols, Allocator allocator, bool uninit = false)
        {
            _rec = null;
            _inlineData = default;
            _gen = 0; // standalone (non-arena): never read (AssertRecordValid short-circuits on _rec == null)
            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = M_Rows * N_Cols;
            var data = new UnsafeList<uint>(Length, allocator, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            Data = data;
        }

        /// <summary>
        /// Arena-tracked constructor. <paramref name="rec"/> is a slot already carved from the
        /// arena's record table by the caller (Arena.uintMat/uintTempMat) -- this ctor only
        /// fills in the record's Data, it does not allocate or own the slot itself.
        /// </summary>
        internal unsafe uintMxN(int M_rows, int N_cols, uintMatRecord* rec, Allocator allocator, bool uninit = false)
        {
            _rec = rec;
            _inlineData = default;
            _gen = rec->Table->GetGeneration(rec->SelfIndex); // stamp this fresh allocation's generation

            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = M_Rows * N_Cols;
            var data = new UnsafeList<uint>(Length, allocator, uninit? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory );
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            Data = data;
        }

        /// <summary>
        /// Creates a standalone copy of the matrix with a new allocation.
        /// </summary>
        public unsafe uintMxN(in uintMxN orig, Allocator allocator = Allocator.Invalid)
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
            var data = new UnsafeList<uint>(Length, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);
            Data = data;
        }

        /// <summary>Arena-tracked copy constructor -- same pre-allocated-record contract as above.</summary>
        internal unsafe uintMxN(in uintMxN orig, uintMatRecord* rec, Allocator allocator)
        {
            _rec = rec;
            _inlineData = default;
            _gen = rec->Table->GetGeneration(rec->SelfIndex); // stamp this fresh allocation's generation (NOT orig's)

            M_Rows = orig.M_Rows;
            N_Cols = orig.N_Cols;
            Length = orig.Length;
            var data = new UnsafeList<uint>(Length, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);
            Data = data;
        }

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        // Editor/test-only guard (ENABLE_UNITY_COLLECTIONS_CHECKS is defined automatically by the
        // Unity Editor, including every test run, and compiles out of player builds entirely --
        // struct size is identical in both configs either way, since _gen occupies a byte range
        // that was already-wasted alignment padding, not a size-growing addition -- see _gen's own
        // doc comment). Unlike uintN (Alive-only), uintMxN also has
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
            if (_rec != null && (!ChunkedRecordTable<uintMatRecord>.IsAliveFast(_rec) || ChunkedRecordTable<uintMatRecord>.GenerationFast(_rec) != _gen))
                throw new InvalidOperationException("uintMxN.Data: use of disposed/cleared arena allocation");
        }
#endif

        public unsafe uintMxN Copy()
        {
            if (_rec == null)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return OwnerArena.uintMat(in this);
        }

        public unsafe uintMxN TempCopy()
        {
            if (_rec == null)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return OwnerArena.uintTempMat(in this);
        }

        public unsafe void Dispose() {

            if (_rec != null)
            {
                // Cache Data BEFORE Free(): Free() marks the slot dead and (per
                // ChunkedRecordTable's own documented contract) does NOT poison/clear the record's
                // payload today -- but Dispose() must not rely on that as an implicit invariant, so
                // read Data into a local first rather than reading _rec->Data again after the slot
                // is already dead. Free() still runs BEFORE the native Dispose() call: an ALIASED
                // double-dispose (a different struct copy sharing this SAME record) throws HERE,
                // from the table's own double-Free guard, before any native memory would be touched
                // a second time. (Disposing the SAME variable twice is a separate, safe no-op: this
                // call nulls _rec below, so a second call on that variable takes the standalone
                // branch instead of reaching here at all.) See also Arena.cs's Clear()/ClearTemp(),
                // which use the opposite order (dispose-then-Free) safely for a different reason --
                // see the comment there.
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
            for (int r = 0; r < M_Rows; r++)
            {
                sb.Append("[ ");
                for (int c = 0; c < N_Cols; c++)
                {
                    if (c > 0) sb.Append("  ");
                    sb.Append(this[r, c]);
                }
                sb.Append(" ]");
                if (r < M_Rows - 1) sb.AppendLine();
            }
            return sb.ToString();
        }
    }
}
