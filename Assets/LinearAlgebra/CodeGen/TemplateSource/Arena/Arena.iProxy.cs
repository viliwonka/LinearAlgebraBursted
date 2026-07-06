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
        // Pointer-stable allocation-record tables (docs/dev/rfc-memory-model.md §4 Option A) -- replace
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

        // Guarded (docs/features/dense-types.md's threading contract): _core->EnterMutation()/
        // ExitMutation() bracket every TERMINAL factory body below (the ones that actually touch
        // a record table's Allocate) under ENABLE_UNITY_COLLECTIONS_CHECKS -- see ArenaCore's
        // _busy field doc (Arena.cs) for why this is safe against reentrancy without a counter.
        // Each also starts with an UNCONDITIONAL `_core == null` guard (matching Pivot/Indices
        // above) -- without it, calling a factory on a disposed/default handle dereferences a null
        // _core before EnterMutation() (or the Allocate call, without checks) ever runs.
        public iProxyN iProxyVec(int N, bool uninit = false) {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.iProxyVec/iProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                iProxyVecRecord* rec = _core->iProxyVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->iProxyVecRecords;
                rec->SelfIndex = slot;
                return new iProxyN(N, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        public iProxyN iProxyVec(int N, iProxy s)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.iProxyVec/iProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                iProxyVecRecord* rec = _core->iProxyVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->iProxyVecRecords;
                rec->SelfIndex = slot;
                var vec = new iProxyN(N, rec, Allocator, true);
                unsafe {
                    UnsafeMathOP.setAll(vec.Data.Ptr, N, s);
                }
                return vec;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal iProxyN iProxyVec(in iProxyN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.iProxyVec/iProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                iProxyVecRecord* rec = _core->iProxyVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->iProxyVecRecords;
                rec->SelfIndex = slot;
                return new iProxyN(in orig, rec, Allocator);   // persistent (backs Copy()); was wrongly the temp list
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal iProxyN iProxyTempVec(int N, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.iProxyVec/iProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                iProxyVecRecord* rec = _core->iProxyTempVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->iProxyTempVecRecords;
                rec->SelfIndex = slot;
                return new iProxyN(N, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal iProxyN iProxyTempVec(in iProxyN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.iProxyVec/iProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                iProxyVecRecord* rec = _core->iProxyTempVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->iProxyTempVecRecords;
                rec->SelfIndex = slot;
                return new iProxyN(in orig, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }
        #endregion

        #region MATRIX
        public iProxyMxN iProxyMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED in iProxyMatRecords —
            // the direct `new iProxyMxN(...)` here was untracked and leaked on Dispose. NOT
            // guarded here (pure forwarding wrapper) -- the (rows, cols) overload below is the
            // terminal call that actually touches a record table, so IT holds the guard (and the
            // null check); guarding both would nest EnterMutation() on the same thread and trip
            // the tripwire on ourselves (see ArenaCore's _busy field doc).
            return iProxyMat(dim, dim, uninit);
        }

        public iProxyMxN iProxyMat(int M_rows, int N_cols, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.iProxyVec/iProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                iProxyMatRecord* rec = _core->iProxyMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->iProxyMatRecords;
                rec->SelfIndex = slot;
                return new iProxyMxN(M_rows, N_cols, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        public iProxyMxN iProxyMat(int M_rows, int N_cols, iProxy s)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.iProxyVec/iProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
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
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        public iProxyMxN iProxyMat(in iProxyMxN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.iProxyVec/iProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                iProxyMatRecord* rec = _core->iProxyMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->iProxyMatRecords;
                rec->SelfIndex = slot;
                return new iProxyMxN(in orig, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal iProxyMxN iProxyTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.iProxyVec/iProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                iProxyMatRecord* rec = _core->iProxyTempMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->iProxyTempMatRecords;
                rec->SelfIndex = slot;
                return new iProxyMxN(M_rows, M_cols, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal iProxyMxN iProxyTempMat(in iProxyMxN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.iProxyVec/iProxyMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                iProxyMatRecord* rec = _core->iProxyTempMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->iProxyTempMatRecords;
                rec->SelfIndex = slot;
                return new iProxyMxN(in orig, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }
        #endregion

        // --- debug pool checks (see Arena.fProxy.cs for the full rationale) ---
        // READ-ONLY: not guarded (element reads are out of scope -- see the threading-contract doc
        // on Arena's class comment).
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
