using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        // Pointer-stable allocation-record tables (docs/rfc-memory-model.md §4 Option A) -- replace
        // the old value-copy-tracking UnsafeList<fProxyN>/UnsafeList<fProxyMxN> lists. fProxyN/
        // fProxyMxN now hold a stable fProxyVecRecord*/fProxyMatRecord* pointing INTO one of these
        // tables instead of storing their Data inline + being tracked by a separate value copy.
        internal ChunkedRecordTable<fProxyVecRecord> fProxyVecRecords;
        internal ChunkedRecordTable<fProxyMatRecord> fProxyMatRecords;
        internal ChunkedRecordTable<fProxyVecRecord> fProxyTempVecRecords;
        internal ChunkedRecordTable<fProxyMatRecord> fProxyTempMatRecords;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public fProxyN fProxyVec(int N, bool uninit = false) {

            fProxyVecRecord* rec = _core->fProxyVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->fProxyVecRecords;
            rec->SelfIndex = slot;
            return new fProxyN(N, rec, Allocator, uninit);
        }

        public fProxyN fProxyVec(int N, fProxy s)
        {
            fProxyVecRecord* rec = _core->fProxyVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->fProxyVecRecords;
            rec->SelfIndex = slot;
            var vec = new fProxyN(N, rec, Allocator, true);
            unsafe {
                UnsafeMathOP.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal fProxyN fProxyVec(in fProxyN orig)
        {
            fProxyVecRecord* rec = _core->fProxyVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->fProxyVecRecords;
            rec->SelfIndex = slot;
            return new fProxyN(in orig, rec, Allocator);   // persistent (backs Copy()); was wrongly the temp list
        }

        internal fProxyN fProxyTempVec(int N, bool uninit = false)
        {
            fProxyVecRecord* rec = _core->fProxyTempVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->fProxyTempVecRecords;
            rec->SelfIndex = slot;
            return new fProxyN(N, rec, Allocator, uninit);
        }

        internal fProxyN fProxyTempVec(in fProxyN orig)
        {
            fProxyVecRecord* rec = _core->fProxyTempVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->fProxyTempVecRecords;
            rec->SelfIndex = slot;
            return new fProxyN(in orig, rec, Allocator);
        }
        #endregion

        #region MATRIX
        public fProxyMxN fProxyMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED in fProxyMatRecords —
            // the direct `new fProxyMxN(...)` here was untracked and leaked on Dispose.
            return fProxyMat(dim, dim, uninit);
        }

        public fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false)
        {
            fProxyMatRecord* rec = _core->fProxyMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->fProxyMatRecords;
            rec->SelfIndex = slot;
            return new fProxyMxN(M_rows, N_cols, rec, Allocator, uninit);
        }

        public fProxyMxN fProxyMat(int M_rows, int N_cols, fProxy s)
        {
            fProxyMatRecord* rec = _core->fProxyMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->fProxyMatRecords;
            rec->SelfIndex = slot;
            var matrix = new fProxyMxN(M_rows, N_cols, rec, Allocator, false);
            unsafe
            {
                UnsafeMathOP.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public fProxyMxN fProxyMat(in fProxyMxN orig)
        {
            fProxyMatRecord* rec = _core->fProxyMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->fProxyMatRecords;
            rec->SelfIndex = slot;
            return new fProxyMxN(in orig, rec, Allocator);
        }

        internal fProxyMxN fProxyTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            fProxyMatRecord* rec = _core->fProxyTempMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->fProxyTempMatRecords;
            rec->SelfIndex = slot;
            return new fProxyMxN(M_rows, M_cols, rec, Allocator, uninit);
        }

        internal fProxyMxN fProxyTempMat(in fProxyMxN orig)
        {
            fProxyMatRecord* rec = _core->fProxyTempMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->fProxyTempMatRecords;
            rec->SelfIndex = slot;
            return new fProxyMxN(in orig, rec, Allocator);
        }
        #endregion

        // --- debug pool checks: confirm a buffer lives in the expected (persistent vs temp) table,
        //     e.g. to assert an op didn't silently move a persistent input into the temp pool ---
        //     Walk the table via its Count/IsAlive/Resolve iteration surface (ChunkedRecordTable has
        //     no ForEachAlive callback -- Burst has no managed delegates to hang one off).
        public bool isPersistent(in fProxyN v) {
            for (int i = 0; i < _core->fProxyVecRecords.Count; i++)
                if (_core->fProxyVecRecords.IsAlive(i) && _core->fProxyVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in fProxyN v) {
            for (int i = 0; i < _core->fProxyTempVecRecords.Count; i++)
                if (_core->fProxyTempVecRecords.IsAlive(i) && _core->fProxyTempVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isPersistent(in fProxyMxN m) {
            for (int i = 0; i < _core->fProxyMatRecords.Count; i++)
                if (_core->fProxyMatRecords.IsAlive(i) && _core->fProxyMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in fProxyMxN m) {
            for (int i = 0; i < _core->fProxyTempMatRecords.Count; i++)
                if (_core->fProxyTempMatRecords.IsAlive(i) && _core->fProxyTempMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }

    }

}
