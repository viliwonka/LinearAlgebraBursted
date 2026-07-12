using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        // Pointer-stable allocation-record tables: fProxyN/fProxyMxN hold a stable
        // fProxyVecRecord*/fProxyMatRecord* pointing INTO one of these tables (records never move
        // when a table grows), rather than storing their Data inline.
        internal ChunkedRecordTable<fProxyVecRecord> fProxyVecRecords;
        internal ChunkedRecordTable<fProxyMatRecord> fProxyMatRecords;
        internal ChunkedRecordTable<fProxyVecRecord> fProxyTempVecRecords;
        internal ChunkedRecordTable<fProxyMatRecord> fProxyTempMatRecords;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        // Guarded under ENABLE_UNITY_COLLECTIONS_CHECKS (_core->EnterMutation()/ExitMutation()
        // bracket every terminal factory body below that touches a record table's Allocate);
        // throws on a default/disposed arena.
        public fProxyN fProxyVec(int N, bool uninit = false) {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.fProxyVec/fProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                fProxyVecRecord* rec = _core->fProxyVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->fProxyVecRecords;
                rec->SelfIndex = slot;
                return new fProxyN(N, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        public fProxyN fProxyVec(int N, fProxy s)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.fProxyVec/fProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                fProxyVecRecord* rec = _core->fProxyVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->fProxyVecRecords;
                rec->SelfIndex = slot;
                var vec = new fProxyN(N, rec, Allocator, true);
                unsafe {
                    UnsafeMathOP.setAll(vec.Data.Ptr, N, s);
                }
                return vec;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal fProxyN fProxyVec(in fProxyN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.fProxyVec/fProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                fProxyVecRecord* rec = _core->fProxyVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->fProxyVecRecords;
                rec->SelfIndex = slot;
                return new fProxyN(in orig, rec, Allocator);   // persistent (backs Copy())
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal fProxyN fProxyTempVec(int N, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.fProxyVec/fProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                fProxyVecRecord* rec = _core->fProxyTempVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->fProxyTempVecRecords;
                rec->SelfIndex = slot;
                return new fProxyN(N, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal fProxyN fProxyTempVec(in fProxyN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.fProxyVec/fProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                fProxyVecRecord* rec = _core->fProxyTempVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->fProxyTempVecRecords;
                rec->SelfIndex = slot;
                return new fProxyN(in orig, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }
        #endregion

        #region MATRIX
        public fProxyMxN fProxyMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED in fProxyMatRecords. NOT
            // guarded here (pure forwarding wrapper) -- the (rows, cols) overload below is the
            // terminal call that actually touches a record table, so IT holds the guard (and the
            // null check); guarding both would nest EnterMutation() on the same thread and trip
            // the tripwire on ourselves (see ArenaCore's _busy field doc).
            return fProxyMat(dim, dim, uninit);
        }

        public fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.fProxyVec/fProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                fProxyMatRecord* rec = _core->fProxyMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->fProxyMatRecords;
                rec->SelfIndex = slot;
                return new fProxyMxN(M_rows, N_cols, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        public fProxyMxN fProxyMat(int M_rows, int N_cols, fProxy s)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.fProxyVec/fProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
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
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        public fProxyMxN fProxyMat(in fProxyMxN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.fProxyVec/fProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                fProxyMatRecord* rec = _core->fProxyMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->fProxyMatRecords;
                rec->SelfIndex = slot;
                return new fProxyMxN(in orig, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal fProxyMxN fProxyTempMat(int M_rows, int N_cols, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.fProxyVec/fProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                fProxyMatRecord* rec = _core->fProxyTempMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->fProxyTempMatRecords;
                rec->SelfIndex = slot;
                return new fProxyMxN(M_rows, N_cols, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal fProxyMxN fProxyTempMat(in fProxyMxN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.fProxyVec/fProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                fProxyMatRecord* rec = _core->fProxyTempMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->fProxyTempMatRecords;
                rec->SelfIndex = slot;
                return new fProxyMxN(in orig, rec, Allocator);
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
        public bool isPersistent(in fProxyN v) {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.isPersistent: arena is not initialized (default or disposed).");
            for (int i = 0; i < _core->fProxyVecRecords.Count; i++)
                if (_core->fProxyVecRecords.IsAlive(i) && _core->fProxyVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in fProxyN v) {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.isTemp: arena is not initialized (default or disposed).");
            for (int i = 0; i < _core->fProxyTempVecRecords.Count; i++)
                if (_core->fProxyTempVecRecords.IsAlive(i) && _core->fProxyTempVecRecords.Resolve(i)->Data.Ptr == v.Data.Ptr)
                    return true;
            return false;
        }
        public bool isPersistent(in fProxyMxN m) {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.isPersistent: arena is not initialized (default or disposed).");
            for (int i = 0; i < _core->fProxyMatRecords.Count; i++)
                if (_core->fProxyMatRecords.IsAlive(i) && _core->fProxyMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }
        public bool isTemp(in fProxyMxN m) {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.isTemp: arena is not initialized (default or disposed).");
            for (int i = 0; i < _core->fProxyTempMatRecords.Count; i++)
                if (_core->fProxyTempMatRecords.IsAlive(i) && _core->fProxyTempMatRecords.Resolve(i)->Data.Ptr == m.Data.Ptr)
                    return true;
            return false;
        }

    }

}
