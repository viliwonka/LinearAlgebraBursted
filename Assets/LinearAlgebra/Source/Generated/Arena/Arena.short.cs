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

        // Guarded (docs/features/dense-types.md's threading contract): _core->EnterMutation()/
        // ExitMutation() bracket every TERMINAL factory body below (the ones that actually touch
        // a record table's Allocate) under ENABLE_UNITY_COLLECTIONS_CHECKS -- see ArenaCore's
        // _busy field doc (Arena.cs) for why this is safe against reentrancy without a counter.
        // Each also starts with an UNCONDITIONAL `_core == null` guard (matching Pivot/Indices
        // above) -- without it, calling a factory on a disposed/default handle dereferences a null
        // _core before EnterMutation() (or the Allocate call, without checks) ever runs.
        public shortN shortVec(int N, bool uninit = false) {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.shortVec/shortMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                shortVecRecord* rec = _core->shortVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->shortVecRecords;
                rec->SelfIndex = slot;
                return new shortN(N, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        public shortN shortVec(int N, short s)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.shortVec/shortMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                shortVecRecord* rec = _core->shortVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->shortVecRecords;
                rec->SelfIndex = slot;
                var vec = new shortN(N, rec, Allocator, true);
                unsafe {
                    UnsafeMathOP.setAll(vec.Data.Ptr, N, s);
                }
                return vec;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal shortN shortVec(in shortN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.shortVec/shortMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                shortVecRecord* rec = _core->shortVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->shortVecRecords;
                rec->SelfIndex = slot;
                return new shortN(in orig, rec, Allocator);   // persistent (backs Copy()); was wrongly the temp list
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal shortN shortTempVec(int N, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.shortVec/shortMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                shortVecRecord* rec = _core->shortTempVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->shortTempVecRecords;
                rec->SelfIndex = slot;
                return new shortN(N, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal shortN shortTempVec(in shortN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.shortVec/shortMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                shortVecRecord* rec = _core->shortTempVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->shortTempVecRecords;
                rec->SelfIndex = slot;
                return new shortN(in orig, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }
        #endregion

        #region MATRIX
        public shortMxN shortMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED in shortMatRecords —
            // the direct `new shortMxN(...)` here was untracked and leaked on Dispose. NOT
            // guarded here (pure forwarding wrapper) -- the (rows, cols) overload below is the
            // terminal call that actually touches a record table, so IT holds the guard (and the
            // null check); guarding both would nest EnterMutation() on the same thread and trip
            // the tripwire on ourselves (see ArenaCore's _busy field doc).
            return shortMat(dim, dim, uninit);
        }

        public shortMxN shortMat(int M_rows, int N_cols, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.shortVec/shortMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                shortMatRecord* rec = _core->shortMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->shortMatRecords;
                rec->SelfIndex = slot;
                return new shortMxN(M_rows, N_cols, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        public shortMxN shortMat(int M_rows, int N_cols, short s)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.shortVec/shortMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
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
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        public shortMxN shortMat(in shortMxN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.shortVec/shortMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                shortMatRecord* rec = _core->shortMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->shortMatRecords;
                rec->SelfIndex = slot;
                return new shortMxN(in orig, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal shortMxN shortTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.shortVec/shortMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                shortMatRecord* rec = _core->shortTempMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->shortTempMatRecords;
                rec->SelfIndex = slot;
                return new shortMxN(M_rows, M_cols, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal shortMxN shortTempMat(in shortMxN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.shortVec/shortMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                shortMatRecord* rec = _core->shortTempMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->shortTempMatRecords;
                rec->SelfIndex = slot;
                return new shortMxN(in orig, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }
        #endregion

        // --- debug pool checks (see Arena.fProxy.cs for the full rationale) ---
        // READ-ONLY: not guarded (element reads are out of scope -- see the threading-contract doc
        // on Arena's class comment).
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
