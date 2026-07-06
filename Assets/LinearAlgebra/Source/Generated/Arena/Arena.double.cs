using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        // Pointer-stable allocation-record tables (docs/dev/rfc-memory-model.md §4 Option A) -- replace
        // the old value-copy-tracking UnsafeList<doubleN>/UnsafeList<doubleMxN> lists. doubleN/
        // doubleMxN now hold a stable doubleVecRecord*/doubleMatRecord* pointing INTO one of these
        // tables instead of storing their Data inline + being tracked by a separate value copy.
        internal ChunkedRecordTable<doubleVecRecord> doubleVecRecords;
        internal ChunkedRecordTable<doubleMatRecord> doubleMatRecords;
        internal ChunkedRecordTable<doubleVecRecord> doubleTempVecRecords;
        internal ChunkedRecordTable<doubleMatRecord> doubleTempMatRecords;
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
        public doubleN doubleVec(int N, bool uninit = false) {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.doubleVec/doubleMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                doubleVecRecord* rec = _core->doubleVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->doubleVecRecords;
                rec->SelfIndex = slot;
                return new doubleN(N, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        public doubleN doubleVec(int N, double s)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.doubleVec/doubleMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                doubleVecRecord* rec = _core->doubleVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->doubleVecRecords;
                rec->SelfIndex = slot;
                var vec = new doubleN(N, rec, Allocator, true);
                unsafe {
                    UnsafeMathOP.setAll(vec.Data.Ptr, N, s);
                }
                return vec;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal doubleN doubleVec(in doubleN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.doubleVec/doubleMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                doubleVecRecord* rec = _core->doubleVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->doubleVecRecords;
                rec->SelfIndex = slot;
                return new doubleN(in orig, rec, Allocator);   // persistent (backs Copy()); was wrongly the temp list
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal doubleN doubleTempVec(int N, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.doubleVec/doubleMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                doubleVecRecord* rec = _core->doubleTempVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->doubleTempVecRecords;
                rec->SelfIndex = slot;
                return new doubleN(N, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal doubleN doubleTempVec(in doubleN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.doubleVec/doubleMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                doubleVecRecord* rec = _core->doubleTempVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->doubleTempVecRecords;
                rec->SelfIndex = slot;
                return new doubleN(in orig, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }
        #endregion

        #region MATRIX
        public doubleMxN doubleMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED in doubleMatRecords —
            // the direct `new doubleMxN(...)` here was untracked and leaked on Dispose. NOT
            // guarded here (pure forwarding wrapper) -- the (rows, cols) overload below is the
            // terminal call that actually touches a record table, so IT holds the guard (and the
            // null check); guarding both would nest EnterMutation() on the same thread and trip
            // the tripwire on ourselves (see ArenaCore's _busy field doc).
            return doubleMat(dim, dim, uninit);
        }

        public doubleMxN doubleMat(int M_rows, int N_cols, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.doubleVec/doubleMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                doubleMatRecord* rec = _core->doubleMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->doubleMatRecords;
                rec->SelfIndex = slot;
                return new doubleMxN(M_rows, N_cols, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        public doubleMxN doubleMat(int M_rows, int N_cols, double s)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.doubleVec/doubleMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                doubleMatRecord* rec = _core->doubleMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->doubleMatRecords;
                rec->SelfIndex = slot;
                var matrix = new doubleMxN(M_rows, N_cols, rec, Allocator, false);
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

        public doubleMxN doubleMat(in doubleMxN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.doubleVec/doubleMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                doubleMatRecord* rec = _core->doubleMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->doubleMatRecords;
                rec->SelfIndex = slot;
                return new doubleMxN(in orig, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal doubleMxN doubleTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.doubleVec/doubleMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                doubleMatRecord* rec = _core->doubleTempMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->doubleTempMatRecords;
                rec->SelfIndex = slot;
                return new doubleMxN(M_rows, M_cols, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal doubleMxN doubleTempMat(in doubleMxN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.doubleVec/doubleMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                doubleMatRecord* rec = _core->doubleTempMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->doubleTempMatRecords;
                rec->SelfIndex = slot;
                return new doubleMxN(in orig, rec, Allocator);
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
        public bool isPersistent(in doubleN v) {
            for (int i = 0; i < _core->doubleVecRecords.Count; i++)
                if (_core->doubleVecRecords.IsAlive(i) && _core->doubleVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in doubleN v) {
            for (int i = 0; i < _core->doubleTempVecRecords.Count; i++)
                if (_core->doubleTempVecRecords.IsAlive(i) && _core->doubleTempVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isPersistent(in doubleMxN m) {
            for (int i = 0; i < _core->doubleMatRecords.Count; i++)
                if (_core->doubleMatRecords.IsAlive(i) && _core->doubleMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in doubleMxN m) {
            for (int i = 0; i < _core->doubleTempMatRecords.Count; i++)
                if (_core->doubleTempMatRecords.IsAlive(i) && _core->doubleTempMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }

    }

}
