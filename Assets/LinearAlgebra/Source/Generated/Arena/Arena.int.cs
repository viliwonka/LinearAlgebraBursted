using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Internal;


namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        // Pointer-stable allocation-record tables (docs/rfc-memory-model.md §4 Option A) -- replace
        // the old value-copy-tracking UnsafeList<intN>/UnsafeList<intMxN> lists. intN/
        // intMxN now hold a stable intVecRecord*/intMatRecord* pointing INTO one of these
        // tables instead of storing their Data inline + being tracked by a separate value copy.
        internal ChunkedRecordTable<intVecRecord> intVecRecords;
        internal ChunkedRecordTable<intMatRecord> intMatRecords;
        internal ChunkedRecordTable<intVecRecord> intTempVecRecords;
        internal ChunkedRecordTable<intMatRecord> intTempMatRecords;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public intN intVec(int N, bool uninit = false) {

            intVecRecord* rec = _core->intVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->intVecRecords;
            rec->SelfIndex = slot;
            return new intN(N, rec, Allocator, uninit);
        }

        public intN intVec(int N, int s)
        {
            intVecRecord* rec = _core->intVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->intVecRecords;
            rec->SelfIndex = slot;
            var vec = new intN(N, rec, Allocator, true);
            unsafe {
                UnsafeMathOP.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal intN intVec(in intN orig)
        {
            intVecRecord* rec = _core->intVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->intVecRecords;
            rec->SelfIndex = slot;
            return new intN(in orig, rec, Allocator);   // persistent (backs Copy()); was wrongly the temp list
        }

        internal intN intTempVec(int N, bool uninit = false)
        {
            intVecRecord* rec = _core->intTempVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->intTempVecRecords;
            rec->SelfIndex = slot;
            return new intN(N, rec, Allocator, uninit);
        }

        internal intN intTempVec(in intN orig)
        {
            intVecRecord* rec = _core->intTempVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->intTempVecRecords;
            rec->SelfIndex = slot;
            return new intN(in orig, rec, Allocator);
        }
        #endregion

        #region MATRIX
        public intMxN intMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED in intMatRecords —
            // the direct `new intMxN(...)` here was untracked and leaked on Dispose.
            return intMat(dim, dim, uninit);
        }

        public intMxN intMat(int M_rows, int N_cols, bool uninit = false)
        {
            intMatRecord* rec = _core->intMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->intMatRecords;
            rec->SelfIndex = slot;
            return new intMxN(M_rows, N_cols, rec, Allocator, uninit);
        }

        public intMxN intMat(int M_rows, int N_cols, int s)
        {
            intMatRecord* rec = _core->intMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->intMatRecords;
            rec->SelfIndex = slot;
            var matrix = new intMxN(M_rows, N_cols, rec, Allocator, false);
            unsafe
            {
                UnsafeMathOP.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public intMxN intMat(in intMxN orig)
        {
            intMatRecord* rec = _core->intMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->intMatRecords;
            rec->SelfIndex = slot;
            return new intMxN(in orig, rec, Allocator);
        }

        internal intMxN intTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            intMatRecord* rec = _core->intTempMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->intTempMatRecords;
            rec->SelfIndex = slot;
            return new intMxN(M_rows, M_cols, rec, Allocator, uninit);
        }

        internal intMxN intTempMat(in intMxN orig)
        {
            intMatRecord* rec = _core->intTempMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->intTempMatRecords;
            rec->SelfIndex = slot;
            return new intMxN(in orig, rec, Allocator);
        }
        #endregion

        // --- debug pool checks (see Arena.fProxy.cs for the full rationale) ---
        public bool isPersistent(in intN v) {
            for (int i = 0; i < _core->intVecRecords.Count; i++)
                if (_core->intVecRecords.IsAlive(i) && _core->intVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in intN v) {
            for (int i = 0; i < _core->intTempVecRecords.Count; i++)
                if (_core->intTempVecRecords.IsAlive(i) && _core->intTempVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isPersistent(in intMxN m) {
            for (int i = 0; i < _core->intMatRecords.Count; i++)
                if (_core->intMatRecords.IsAlive(i) && _core->intMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in intMxN m) {
            for (int i = 0; i < _core->intTempMatRecords.Count; i++)
                if (_core->intTempMatRecords.IsAlive(i) && _core->intTempMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }

    }

}
