using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Internal;


namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        // Pointer-stable allocation-record tables (docs/rfc-memory-model.md §4 Option A) -- replace
        // the old value-copy-tracking UnsafeList<uintN>/UnsafeList<uintMxN> lists. uintN/
        // uintMxN now hold a stable uintVecRecord*/uintMatRecord* pointing INTO one of these
        // tables instead of storing their Data inline + being tracked by a separate value copy.
        internal ChunkedRecordTable<uintVecRecord> uintVecRecords;
        internal ChunkedRecordTable<uintMatRecord> uintMatRecords;
        internal ChunkedRecordTable<uintVecRecord> uintTempVecRecords;
        internal ChunkedRecordTable<uintMatRecord> uintTempMatRecords;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public uintN uintVec(int N, bool uninit = false) {

            uintVecRecord* rec = _core->uintVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->uintVecRecords;
            rec->SelfIndex = slot;
            return new uintN(N, rec, Allocator, uninit);
        }

        public uintN uintVec(int N, uint s)
        {
            uintVecRecord* rec = _core->uintVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->uintVecRecords;
            rec->SelfIndex = slot;
            var vec = new uintN(N, rec, Allocator, true);
            unsafe {
                UnsafeMathOP.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal uintN uintVec(in uintN orig)
        {
            uintVecRecord* rec = _core->uintVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->uintVecRecords;
            rec->SelfIndex = slot;
            return new uintN(in orig, rec, Allocator);   // persistent (backs Copy()); was wrongly the temp list
        }

        internal uintN uintTempVec(int N, bool uninit = false)
        {
            uintVecRecord* rec = _core->uintTempVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->uintTempVecRecords;
            rec->SelfIndex = slot;
            return new uintN(N, rec, Allocator, uninit);
        }

        internal uintN uintTempVec(in uintN orig)
        {
            uintVecRecord* rec = _core->uintTempVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->uintTempVecRecords;
            rec->SelfIndex = slot;
            return new uintN(in orig, rec, Allocator);
        }
        #endregion

        #region MATRIX
        public uintMxN uintMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED in uintMatRecords —
            // the direct `new uintMxN(...)` here was untracked and leaked on Dispose.
            return uintMat(dim, dim, uninit);
        }

        public uintMxN uintMat(int M_rows, int N_cols, bool uninit = false)
        {
            uintMatRecord* rec = _core->uintMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->uintMatRecords;
            rec->SelfIndex = slot;
            return new uintMxN(M_rows, N_cols, rec, Allocator, uninit);
        }

        public uintMxN uintMat(int M_rows, int N_cols, uint s)
        {
            uintMatRecord* rec = _core->uintMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->uintMatRecords;
            rec->SelfIndex = slot;
            var matrix = new uintMxN(M_rows, N_cols, rec, Allocator, false);
            unsafe
            {
                UnsafeMathOP.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public uintMxN uintMat(in uintMxN orig)
        {
            uintMatRecord* rec = _core->uintMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->uintMatRecords;
            rec->SelfIndex = slot;
            return new uintMxN(in orig, rec, Allocator);
        }

        internal uintMxN uintTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            uintMatRecord* rec = _core->uintTempMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->uintTempMatRecords;
            rec->SelfIndex = slot;
            return new uintMxN(M_rows, M_cols, rec, Allocator, uninit);
        }

        internal uintMxN uintTempMat(in uintMxN orig)
        {
            uintMatRecord* rec = _core->uintTempMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->uintTempMatRecords;
            rec->SelfIndex = slot;
            return new uintMxN(in orig, rec, Allocator);
        }
        #endregion

        // --- debug pool checks (see Arena.fProxy.cs for the full rationale) ---
        public bool isPersistent(in uintN v) {
            for (int i = 0; i < _core->uintVecRecords.Count; i++)
                if (_core->uintVecRecords.IsAlive(i) && _core->uintVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in uintN v) {
            for (int i = 0; i < _core->uintTempVecRecords.Count; i++)
                if (_core->uintTempVecRecords.IsAlive(i) && _core->uintTempVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isPersistent(in uintMxN m) {
            for (int i = 0; i < _core->uintMatRecords.Count; i++)
                if (_core->uintMatRecords.IsAlive(i) && _core->uintMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in uintMxN m) {
            for (int i = 0; i < _core->uintTempMatRecords.Count; i++)
                if (_core->uintTempMatRecords.IsAlive(i) && _core->uintTempMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }

    }

}
