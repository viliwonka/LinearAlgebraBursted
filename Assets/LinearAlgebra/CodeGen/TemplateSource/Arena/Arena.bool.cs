//using System;
namespace LinearAlgebra
{

    // Allocation helper
    public unsafe partial struct Arena {

        // Guarded under ENABLE_UNITY_COLLECTIONS_CHECKS (_core->EnterMutation()/ExitMutation()
        // bracket every factory body below); throws on a default/disposed arena.
        #region BOOLVECTOR
        public boolN boolVec(int N, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.boolVec/boolMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                boolVecRecord* rec = _core->BoolVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->BoolVecRecords;
                rec->SelfIndex = slot;
                return new boolN(N, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal boolN boolTempVec(int N, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.boolVec/boolMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                boolVecRecord* rec = _core->TempBoolVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->TempBoolVecRecords;
                rec->SelfIndex = slot;
                return new boolN(N, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal boolN boolVec(in boolN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.boolVec/boolMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                boolVecRecord* rec = _core->BoolVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->BoolVecRecords;
                rec->SelfIndex = slot;
                return new boolN(in orig, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal boolN boolTempVec(in boolN orig)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.boolVec/boolMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                boolVecRecord* rec = _core->TempBoolVecRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->TempBoolVecRecords;
                rec->SelfIndex = slot;
                return new boolN(in orig, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        #endregion

        #region BOOLMATRIX

        public boolMxN boolMat(int M_rows, int N_cols, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.boolVec/boolMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                boolMatRecord* rec = _core->BoolMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->BoolMatRecords;
                rec->SelfIndex = slot;
                return new boolMxN(M_rows, N_cols, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        public boolMxN boolMat(in boolMxN mat)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.boolVec/boolMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                boolMatRecord* rec = _core->BoolMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->BoolMatRecords;
                rec->SelfIndex = slot;
                return new boolMxN(in mat, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal boolMxN boolTempMat(int M_rows, int N_cols, bool uninit = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.boolVec/boolMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                boolMatRecord* rec = _core->TempBoolMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->TempBoolMatRecords;
                rec->SelfIndex = slot;
                return new boolMxN(M_rows, N_cols, rec, Allocator, uninit);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        internal boolMxN boolTempMat(in boolMxN mat)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.boolVec/boolMat: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                boolMatRecord* rec = _core->TempBoolMatRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->TempBoolMatRecords;
                rec->SelfIndex = slot;
                return new boolMxN(in mat, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        #endregion

    }

}
