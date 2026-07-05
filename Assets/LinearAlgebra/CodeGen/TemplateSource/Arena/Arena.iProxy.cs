using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Internal;

//alsoExpand[uint]// core bump-allocator factories (arena.uintVec/uintMat); no signed-only ops here.
//This flag ALSO widens the ArenaCore partial's record-table field declarations right below (each
//generated Arena.<type>.cs gets its own uintVecRecords/uintMatRecords/uintTempVecRecords/
//uintTempMatRecords fields, mirroring intVecRecords/shortVecRecords/longVecRecords) - see
//iProxyRecords.iProxy.cs's identical alsoExpand note for the record TYPES themselves.

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        // Pointer-stable allocation-record tables (docs/rfc-memory-model.md §4 Option A) -- replace
        // the old value-copy-tracking UnsafeList<iProxyN>/UnsafeList<iProxyMxN> lists. iProxyN/
        // iProxyMxN now hold a stable iProxyVecRecord*/iProxyMatRecord* pointing INTO one of these
        // tables instead of storing their Data inline + being tracked by a separate value copy.
        internal ChunkedRecordTable<iProxyVecRecord> iProxyVecRecords;
        internal ChunkedRecordTable<iProxyMatRecord> iProxyMatRecords;
        internal ChunkedRecordTable<iProxyVecRecord> iProxyTempVecRecords;
        internal ChunkedRecordTable<iProxyMatRecord> iProxyTempMatRecords;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public iProxyN iProxyVec(int N, bool uninit = false) {

            iProxyVecRecord* rec = _core->iProxyVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->iProxyVecRecords;
            rec->SelfIndex = slot;
            return new iProxyN(N, rec, Allocator, uninit);
        }

        public iProxyN iProxyVec(int N, iProxy s)
        {
            iProxyVecRecord* rec = _core->iProxyVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->iProxyVecRecords;
            rec->SelfIndex = slot;
            var vec = new iProxyN(N, rec, Allocator, true);
            unsafe {
                UnsafeMathOP.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal iProxyN iProxyVec(in iProxyN orig)
        {
            iProxyVecRecord* rec = _core->iProxyVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->iProxyVecRecords;
            rec->SelfIndex = slot;
            return new iProxyN(in orig, rec, Allocator);   // persistent (backs Copy()); was wrongly the temp list
        }

        internal iProxyN iProxyTempVec(int N, bool uninit = false)
        {
            iProxyVecRecord* rec = _core->iProxyTempVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->iProxyTempVecRecords;
            rec->SelfIndex = slot;
            return new iProxyN(N, rec, Allocator, uninit);
        }

        internal iProxyN iProxyTempVec(in iProxyN orig)
        {
            iProxyVecRecord* rec = _core->iProxyTempVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->iProxyTempVecRecords;
            rec->SelfIndex = slot;
            return new iProxyN(in orig, rec, Allocator);
        }
        #endregion

        #region MATRIX
        public iProxyMxN iProxyMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED in iProxyMatRecords —
            // the direct `new iProxyMxN(...)` here was untracked and leaked on Dispose.
            return iProxyMat(dim, dim, uninit);
        }

        public iProxyMxN iProxyMat(int M_rows, int N_cols, bool uninit = false)
        {
            iProxyMatRecord* rec = _core->iProxyMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->iProxyMatRecords;
            rec->SelfIndex = slot;
            return new iProxyMxN(M_rows, N_cols, rec, Allocator, uninit);
        }

        public iProxyMxN iProxyMat(int M_rows, int N_cols, iProxy s)
        {
            iProxyMatRecord* rec = _core->iProxyMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->iProxyMatRecords;
            rec->SelfIndex = slot;
            var matrix = new iProxyMxN(M_rows, N_cols, rec, Allocator, false);
            unsafe
            {
                UnsafeMathOP.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public iProxyMxN iProxyMat(in iProxyMxN orig)
        {
            iProxyMatRecord* rec = _core->iProxyMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->iProxyMatRecords;
            rec->SelfIndex = slot;
            return new iProxyMxN(in orig, rec, Allocator);
        }

        internal iProxyMxN iProxyTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            iProxyMatRecord* rec = _core->iProxyTempMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->iProxyTempMatRecords;
            rec->SelfIndex = slot;
            return new iProxyMxN(M_rows, M_cols, rec, Allocator, uninit);
        }

        internal iProxyMxN iProxyTempMat(in iProxyMxN orig)
        {
            iProxyMatRecord* rec = _core->iProxyTempMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->iProxyTempMatRecords;
            rec->SelfIndex = slot;
            return new iProxyMxN(in orig, rec, Allocator);
        }
        #endregion

        // --- debug pool checks (see Arena.fProxy.cs for the full rationale) ---
        public bool isPersistent(in iProxyN v) {
            for (int i = 0; i < _core->iProxyVecRecords.Count; i++)
                if (_core->iProxyVecRecords.IsAlive(i) && _core->iProxyVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in iProxyN v) {
            for (int i = 0; i < _core->iProxyTempVecRecords.Count; i++)
                if (_core->iProxyTempVecRecords.IsAlive(i) && _core->iProxyTempVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isPersistent(in iProxyMxN m) {
            for (int i = 0; i < _core->iProxyMatRecords.Count; i++)
                if (_core->iProxyMatRecords.IsAlive(i) && _core->iProxyMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in iProxyMxN m) {
            for (int i = 0; i < _core->iProxyTempMatRecords.Count; i++)
                if (_core->iProxyTempMatRecords.IsAlive(i) && _core->iProxyTempMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }

    }

}
