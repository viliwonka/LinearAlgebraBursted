using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Internal;


namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        // Pointer-stable allocation-record tables (docs/rfc-memory-model.md §4 Option A) -- replace
        // the old value-copy-tracking UnsafeList<shortN>/UnsafeList<shortMxN> lists. shortN/
        // shortMxN now hold a stable shortVecRecord*/shortMatRecord* pointing INTO one of these
        // tables instead of storing their Data inline + being tracked by a separate value copy.
        internal ChunkedRecordTable<shortVecRecord> shortVecRecords;
        internal ChunkedRecordTable<shortMatRecord> shortMatRecords;
        internal ChunkedRecordTable<shortVecRecord> shortTempVecRecords;
        internal ChunkedRecordTable<shortMatRecord> shortTempMatRecords;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public shortN shortVec(int N, bool uninit = false) {

            shortVecRecord* rec = _core->shortVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->shortVecRecords;
            rec->SelfIndex = slot;
            return new shortN(N, rec, Allocator, uninit);
        }

        public shortN shortVec(int N, short s)
        {
            shortVecRecord* rec = _core->shortVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->shortVecRecords;
            rec->SelfIndex = slot;
            var vec = new shortN(N, rec, Allocator, true);
            unsafe {
                UnsafeMathOP.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal shortN shortVec(in shortN orig)
        {
            shortVecRecord* rec = _core->shortVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->shortVecRecords;
            rec->SelfIndex = slot;
            return new shortN(in orig, rec, Allocator);   // persistent (backs Copy()); was wrongly the temp list
        }

        internal shortN shortTempVec(int N, bool uninit = false)
        {
            shortVecRecord* rec = _core->shortTempVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->shortTempVecRecords;
            rec->SelfIndex = slot;
            return new shortN(N, rec, Allocator, uninit);
        }

        internal shortN shortTempVec(in shortN orig)
        {
            shortVecRecord* rec = _core->shortTempVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->shortTempVecRecords;
            rec->SelfIndex = slot;
            return new shortN(in orig, rec, Allocator);
        }
        #endregion

        #region MATRIX
        public shortMxN shortMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED in shortMatRecords —
            // the direct `new shortMxN(...)` here was untracked and leaked on Dispose.
            return shortMat(dim, dim, uninit);
        }

        public shortMxN shortMat(int M_rows, int N_cols, bool uninit = false)
        {
            shortMatRecord* rec = _core->shortMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->shortMatRecords;
            rec->SelfIndex = slot;
            return new shortMxN(M_rows, N_cols, rec, Allocator, uninit);
        }

        public shortMxN shortMat(int M_rows, int N_cols, short s)
        {
            shortMatRecord* rec = _core->shortMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->shortMatRecords;
            rec->SelfIndex = slot;
            var matrix = new shortMxN(M_rows, N_cols, rec, Allocator, false);
            unsafe
            {
                UnsafeMathOP.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public shortMxN shortMat(in shortMxN orig)
        {
            shortMatRecord* rec = _core->shortMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->shortMatRecords;
            rec->SelfIndex = slot;
            return new shortMxN(in orig, rec, Allocator);
        }

        internal shortMxN shortTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            shortMatRecord* rec = _core->shortTempMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->shortTempMatRecords;
            rec->SelfIndex = slot;
            return new shortMxN(M_rows, M_cols, rec, Allocator, uninit);
        }

        internal shortMxN shortTempMat(in shortMxN orig)
        {
            shortMatRecord* rec = _core->shortTempMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->shortTempMatRecords;
            rec->SelfIndex = slot;
            return new shortMxN(in orig, rec, Allocator);
        }
        #endregion

        // --- debug pool checks (see Arena.fProxy.cs for the full rationale) ---
        public bool isPersistent(in shortN v) {
            for (int i = 0; i < _core->shortVecRecords.Count; i++)
                if (_core->shortVecRecords.IsAlive(i) && _core->shortVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in shortN v) {
            for (int i = 0; i < _core->shortTempVecRecords.Count; i++)
                if (_core->shortTempVecRecords.IsAlive(i) && _core->shortTempVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isPersistent(in shortMxN m) {
            for (int i = 0; i < _core->shortMatRecords.Count; i++)
                if (_core->shortMatRecords.IsAlive(i) && _core->shortMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in shortMxN m) {
            for (int i = 0; i < _core->shortTempMatRecords.Count; i++)
                if (_core->shortTempMatRecords.IsAlive(i) && _core->shortTempMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }

    }

}
