using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        // Pointer-stable allocation-record tables (docs/rfc-memory-model.md §4 Option A) -- replace
        // the old value-copy-tracking UnsafeList<floatN>/UnsafeList<floatMxN> lists. floatN/
        // floatMxN now hold a stable floatVecRecord*/floatMatRecord* pointing INTO one of these
        // tables instead of storing their Data inline + being tracked by a separate value copy.
        internal ChunkedRecordTable<floatVecRecord> floatVecRecords;
        internal ChunkedRecordTable<floatMatRecord> floatMatRecords;
        internal ChunkedRecordTable<floatVecRecord> floatTempVecRecords;
        internal ChunkedRecordTable<floatMatRecord> floatTempMatRecords;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public floatN floatVec(int N, bool uninit = false) {

            floatVecRecord* rec = _core->floatVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->floatVecRecords;
            rec->SelfIndex = slot;
            return new floatN(N, rec, Allocator, uninit);
        }

        public floatN floatVec(int N, float s)
        {
            floatVecRecord* rec = _core->floatVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->floatVecRecords;
            rec->SelfIndex = slot;
            var vec = new floatN(N, rec, Allocator, true);
            unsafe {
                UnsafeMathOP.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal floatN floatVec(in floatN orig)
        {
            floatVecRecord* rec = _core->floatVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->floatVecRecords;
            rec->SelfIndex = slot;
            return new floatN(in orig, rec, Allocator);   // persistent (backs Copy()); was wrongly the temp list
        }

        internal floatN floatTempVec(int N, bool uninit = false)
        {
            floatVecRecord* rec = _core->floatTempVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->floatTempVecRecords;
            rec->SelfIndex = slot;
            return new floatN(N, rec, Allocator, uninit);
        }

        internal floatN floatTempVec(in floatN orig)
        {
            floatVecRecord* rec = _core->floatTempVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->floatTempVecRecords;
            rec->SelfIndex = slot;
            return new floatN(in orig, rec, Allocator);
        }
        #endregion

        #region MATRIX
        public floatMxN floatMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED in floatMatRecords —
            // the direct `new floatMxN(...)` here was untracked and leaked on Dispose.
            return floatMat(dim, dim, uninit);
        }

        public floatMxN floatMat(int M_rows, int N_cols, bool uninit = false)
        {
            floatMatRecord* rec = _core->floatMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->floatMatRecords;
            rec->SelfIndex = slot;
            return new floatMxN(M_rows, N_cols, rec, Allocator, uninit);
        }

        public floatMxN floatMat(int M_rows, int N_cols, float s)
        {
            floatMatRecord* rec = _core->floatMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->floatMatRecords;
            rec->SelfIndex = slot;
            var matrix = new floatMxN(M_rows, N_cols, rec, Allocator, false);
            unsafe
            {
                UnsafeMathOP.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public floatMxN floatMat(in floatMxN orig)
        {
            floatMatRecord* rec = _core->floatMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->floatMatRecords;
            rec->SelfIndex = slot;
            return new floatMxN(in orig, rec, Allocator);
        }

        internal floatMxN floatTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            floatMatRecord* rec = _core->floatTempMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->floatTempMatRecords;
            rec->SelfIndex = slot;
            return new floatMxN(M_rows, M_cols, rec, Allocator, uninit);
        }

        internal floatMxN floatTempMat(in floatMxN orig)
        {
            floatMatRecord* rec = _core->floatTempMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->floatTempMatRecords;
            rec->SelfIndex = slot;
            return new floatMxN(in orig, rec, Allocator);
        }
        #endregion

        // --- debug pool checks: confirm a buffer lives in the expected (persistent vs temp) table,
        //     e.g. to assert an op didn't silently move a persistent input into the temp pool ---
        //     Walk the table via its Count/IsAlive/Resolve iteration surface (ChunkedRecordTable has
        //     no ForEachAlive callback -- Burst has no managed delegates to hang one off).
        public bool isPersistent(in floatN v) {
            for (int i = 0; i < _core->floatVecRecords.Count; i++)
                if (_core->floatVecRecords.IsAlive(i) && _core->floatVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in floatN v) {
            for (int i = 0; i < _core->floatTempVecRecords.Count; i++)
                if (_core->floatTempVecRecords.IsAlive(i) && _core->floatTempVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isPersistent(in floatMxN m) {
            for (int i = 0; i < _core->floatMatRecords.Count; i++)
                if (_core->floatMatRecords.IsAlive(i) && _core->floatMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in floatMxN m) {
            for (int i = 0; i < _core->floatTempMatRecords.Count; i++)
                if (_core->floatTempMatRecords.IsAlive(i) && _core->floatTempMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }

    }

}
