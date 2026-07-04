using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;

namespace LinearAlgebra
{
    // A m x n matrix
    // m = rows
    // n = cols
    public partial struct doubleMxN : IDisposable, IUnsafedoubleArray {

        public int M_Rows;
        public int N_Cols;

        // Arena-tracked path -- see doubleN.cs's `_rec` doc comment for the full rationale (same
        // Option A record-pointer design, mirrored here for the matrix family). null for a
        // standalone (non-arena) matrix, in which case Data resolves to _inlineData instead.
        [NativeDisableUnsafePtrRestriction] private unsafe doubleMatRecord* _rec;

        // Standalone-path backing store. Stays default(UnsafeList<double>) whenever _rec != null.
        private UnsafeList<double> _inlineData;

        public unsafe UnsafeList<double> Data
        {
            get => _rec != null ? _rec->Data : _inlineData;
            private set { if (_rec != null) _rec->Data = value; else _inlineData = value; }
        }

        // Reconstructs a live Arena handle from this record's owner core -- used by Copy()/
        // TempCopy() and the cross-type allocation shortcuts (doubleMxN.Shortcuts.cs) that used to
        // read a private `_arena` field directly. Only meaningful when _rec != null.
        private unsafe Arena OwnerArena => new Arena(_rec->Owner);

        public readonly int Length;

        public bool IsSquare => M_Rows == N_Cols;

        public unsafe doubleMxN(int M_rows, int N_cols, Allocator allocator, bool uninit = false)
        {
            _rec = null;
            _inlineData = default;
            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = M_Rows * N_Cols;
            var data = new UnsafeList<double>(Length, allocator, uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            Data = data;
        }

        /// <summary>
        /// Arena-tracked constructor. <paramref name="rec"/> is a slot already carved from the
        /// arena's record table by the caller (Arena.doubleMat/doubleTempMat) -- this ctor only
        /// fills in the record's Data, it does not allocate or own the slot itself.
        /// </summary>
        internal unsafe doubleMxN(int M_rows, int N_cols, doubleMatRecord* rec, Allocator allocator, bool uninit = false)
        {
            _rec = rec;
            _inlineData = default;

            M_Rows = M_rows;
            N_Cols = N_cols;
            Length = M_Rows * N_Cols;
            var data = new UnsafeList<double>(Length, allocator, uninit? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory );
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            Data = data;
        }

        /// <summary>
        /// Creates a standalone copy of the matrix with a new allocation.
        /// </summary>
        public unsafe doubleMxN(in doubleMxN orig, Allocator allocator = Allocator.Invalid)
        {
            _rec = null;
            _inlineData = default;

            // guard a standalone (null-record) source — was dereferencing null for the default allocator
            if (allocator == Allocator.Invalid)
                allocator = orig._rec != null ? orig._rec->Owner->Allocator : Allocator.Temp;

            M_Rows = orig.M_Rows;
            N_Cols = orig.N_Cols;
            Length = orig.Length;
            var data = new UnsafeList<double>(Length, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);
            Data = data;
        }

        /// <summary>Arena-tracked copy constructor -- same pre-allocated-record contract as above.</summary>
        internal unsafe doubleMxN(in doubleMxN orig, doubleMatRecord* rec, Allocator allocator)
        {
            _rec = rec;
            _inlineData = default;

            M_Rows = orig.M_Rows;
            N_Cols = orig.N_Cols;
            Length = orig.Length;
            var data = new UnsafeList<double>(Length, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);
            Data = data;
        }

        public unsafe doubleMxN Copy()
        {
            if (_rec == null)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return OwnerArena.doubleMat(in this);
        }

        public unsafe doubleMxN TempCopy()
        {
            if (_rec == null)
                throw new System.InvalidOperationException("Copy()/TempCopy() require an arena-backed matrix/vector; use new <T>(in this, allocator) for a standalone copy.");

            return OwnerArena.doubleTempMat(in this);
        }

        public unsafe void Dispose() {
#if LINALG_DEBUG
            // poison the buffer so a read-after-dispose surfaces as NaN instead of stale data
            for (int i = 0; i < Length; i++) this[i] = double.NaN;
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
