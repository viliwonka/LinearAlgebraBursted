using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;

namespace LinearAlgebra
{
    // A m x n matrix of boolean values
    // m = rows
    // n = cols
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
            get => _rec != null ? _rec->Data : _inlineData;
            private set { if (_rec != null) _rec->Data = value; else _inlineData = value; }
        }

        // Reconstructs a live Arena handle from this record's owner core -- used by Copy()/
        // TempCopy() and the cross-type allocation shortcuts (boolMxN.Shortcuts.cs) that used to
        // read a private `_arena` field directly. Only meaningful when _rec != null.
        private unsafe Arena OwnerArena => new Arena(_rec->Owner);

        public readonly int Length;

        public bool IsSquare => M_Rows == N_Cols;

        public unsafe boolMxN(int M_rows, int N_cols, Allocator allocator, bool uninit = false)
        {
            _rec = null;
            _inlineData = default;
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

            M_Rows = orig.M_Rows;
            N_Cols = orig.N_Cols;
            Length = orig.Length;
            var data = new UnsafeList<bool>(Length, allocator, NativeArrayOptions.UninitializedMemory);
            data.Resize(Length, NativeArrayOptions.UninitializedMemory);
            data.CopyFrom(orig.Data);
            Data = data;
        }

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
