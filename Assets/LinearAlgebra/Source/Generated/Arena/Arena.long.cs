using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Internal;


namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        // Pointer-stable allocation-record tables (docs/rfc-memory-model.md §4 Option A) -- replace
        // the old value-copy-tracking UnsafeList<longN>/UnsafeList<longMxN> lists. longN/
        // longMxN now hold a stable longVecRecord*/longMatRecord* pointing INTO one of these
        // tables instead of storing their Data inline + being tracked by a separate value copy.
        internal ChunkedRecordTable<longVecRecord> longVecRecords;
        internal ChunkedRecordTable<longMatRecord> longMatRecords;
        internal ChunkedRecordTable<longVecRecord> longTempVecRecords;
        internal ChunkedRecordTable<longMatRecord> longTempMatRecords;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public longN longVec(int N, bool uninit = false) {

            longVecRecord* rec = _core->longVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->longVecRecords;
            rec->SelfIndex = slot;
            return new longN(N, rec, Allocator, uninit);
        }

        public longN longVec(int N, long s)
        {
            longVecRecord* rec = _core->longVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->longVecRecords;
            rec->SelfIndex = slot;
            var vec = new longN(N, rec, Allocator, true);
            unsafe {
                UnsafeMathOP.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal longN longVec(in longN orig)
        {
            longVecRecord* rec = _core->longVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->longVecRecords;
            rec->SelfIndex = slot;
            return new longN(in orig, rec, Allocator);   // persistent (backs Copy()); was wrongly the temp list
        }

        internal longN longTempVec(int N, bool uninit = false)
        {
            longVecRecord* rec = _core->longTempVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->longTempVecRecords;
            rec->SelfIndex = slot;
            return new longN(N, rec, Allocator, uninit);
        }

        internal longN longTempVec(in longN orig)
        {
            longVecRecord* rec = _core->longTempVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->longTempVecRecords;
            rec->SelfIndex = slot;
            return new longN(in orig, rec, Allocator);
        }
        #endregion

        #region MATRIX
        public longMxN longMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED in longMatRecords —
            // the direct `new longMxN(...)` here was untracked and leaked on Dispose.
            return longMat(dim, dim, uninit);
        }

        public longMxN longMat(int M_rows, int N_cols, bool uninit = false)
        {
            longMatRecord* rec = _core->longMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->longMatRecords;
            rec->SelfIndex = slot;
            return new longMxN(M_rows, N_cols, rec, Allocator, uninit);
        }

        public longMxN longMat(int M_rows, int N_cols, long s)
        {
            longMatRecord* rec = _core->longMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->longMatRecords;
            rec->SelfIndex = slot;
            var matrix = new longMxN(M_rows, N_cols, rec, Allocator, false);
            unsafe
            {
                UnsafeMathOP.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public longMxN longMat(in longMxN orig)
        {
            longMatRecord* rec = _core->longMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->longMatRecords;
            rec->SelfIndex = slot;
            return new longMxN(in orig, rec, Allocator);
        }

        internal longMxN longTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            longMatRecord* rec = _core->longTempMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->longTempMatRecords;
            rec->SelfIndex = slot;
            return new longMxN(M_rows, M_cols, rec, Allocator, uninit);
        }

        internal longMxN longTempMat(in longMxN orig)
        {
            longMatRecord* rec = _core->longTempMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->longTempMatRecords;
            rec->SelfIndex = slot;
            return new longMxN(in orig, rec, Allocator);
        }
        #endregion

        // --- debug pool checks (see Arena.fProxy.cs for the full rationale) ---
        public bool isPersistent(in longN v) {
            for (int i = 0; i < _core->longVecRecords.Count; i++)
                if (_core->longVecRecords.IsAlive(i) && _core->longVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in longN v) {
            for (int i = 0; i < _core->longTempVecRecords.Count; i++)
                if (_core->longTempVecRecords.IsAlive(i) && _core->longTempVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isPersistent(in longMxN m) {
            for (int i = 0; i < _core->longMatRecords.Count; i++)
                if (_core->longMatRecords.IsAlive(i) && _core->longMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in longMxN m) {
            for (int i = 0; i < _core->longTempMatRecords.Count; i++)
                if (_core->longTempMatRecords.IsAlive(i) && _core->longTempMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }

    }

}
