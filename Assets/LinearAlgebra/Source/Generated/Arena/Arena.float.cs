using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        // Pointer-stable allocation-record tables (docs/dev/rfc-memory-model.md §4 Option A) -- replace
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

        // Guarded (docs/features/dense-types.md's threading contract): _core->EnterMutation()/
        // ExitMutation() bracket every TERMINAL factory body below (the ones that actually touch
        // a record table's Allocate) under ENABLE_UNITY_COLLECTIONS_CHECKS -- see ArenaCore's
        // _busy field doc (Arena.cs) for why this is safe against reentrancy without a counter.
        // Each also starts with an UNCONDITIONAL `_core == null` guard (matching Pivot/Indices
        // above) -- without it, calling a factory on a disposed/default handle dereferences a null
        // _core before EnterMutation() (or the Allocate call, without checks) ever runs.
        public floatN floatVec(int N, bool uninit = false) {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.floatVec/floatMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                floatVecRecord* rec = _core->floatVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->floatVecRecords;
                rec->SelfIndex = slot;
                return new floatN(N, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        public floatN floatVec(int N, float s)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.floatVec/floatMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                floatVecRecord* rec = _core->floatVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->floatVecRecords;
                rec->SelfIndex = slot;
                var vec = new floatN(N, rec, Allocator, true);
                unsafe {
                    UnsafeMathOP.setAll(vec.Data.Ptr, N, s);
                }
                return vec;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal floatN floatVec(in floatN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.floatVec/floatMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                floatVecRecord* rec = _core->floatVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->floatVecRecords;
                rec->SelfIndex = slot;
                return new floatN(in orig, rec, Allocator);   // persistent (backs Copy()); was wrongly the temp list
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal floatN floatTempVec(int N, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.floatVec/floatMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                floatVecRecord* rec = _core->floatTempVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->floatTempVecRecords;
                rec->SelfIndex = slot;
                return new floatN(N, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal floatN floatTempVec(in floatN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.floatVec/floatMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                floatVecRecord* rec = _core->floatTempVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->floatTempVecRecords;
                rec->SelfIndex = slot;
                return new floatN(in orig, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }
        #endregion

        #region MATRIX
        public floatMxN floatMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED in floatMatRecords —
            // the direct `new floatMxN(...)` here was untracked and leaked on Dispose. NOT
            // guarded here (pure forwarding wrapper) -- the (rows, cols) overload below is the
            // terminal call that actually touches a record table, so IT holds the guard (and the
            // null check); guarding both would nest EnterMutation() on the same thread and trip
            // the tripwire on ourselves (see ArenaCore's _busy field doc).
            return floatMat(dim, dim, uninit);
        }

        public floatMxN floatMat(int M_rows, int N_cols, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.floatVec/floatMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                floatMatRecord* rec = _core->floatMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->floatMatRecords;
                rec->SelfIndex = slot;
                return new floatMxN(M_rows, N_cols, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        public floatMxN floatMat(int M_rows, int N_cols, float s)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.floatVec/floatMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
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
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        public floatMxN floatMat(in floatMxN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.floatVec/floatMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                floatMatRecord* rec = _core->floatMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->floatMatRecords;
                rec->SelfIndex = slot;
                return new floatMxN(in orig, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal floatMxN floatTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.floatVec/floatMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                floatMatRecord* rec = _core->floatTempMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->floatTempMatRecords;
                rec->SelfIndex = slot;
                return new floatMxN(M_rows, M_cols, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal floatMxN floatTempMat(in floatMxN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.floatVec/floatMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                floatMatRecord* rec = _core->floatTempMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->floatTempMatRecords;
                rec->SelfIndex = slot;
                return new floatMxN(in orig, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }
        #endregion

        // --- debug pool checks: confirm a buffer lives in the expected (persistent vs temp) table,
        //     e.g. to assert an op didn't silently move a persistent input into the temp pool ---
        //     Walk the table via its Count/IsAlive/Resolve iteration surface (ChunkedRecordTable has
        //     no ForEachAlive callback -- Burst has no managed delegates to hang one off).
        //     READ-ONLY: not guarded (element reads are out of scope -- see the threading-contract
        //     doc on Arena's class comment).
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
