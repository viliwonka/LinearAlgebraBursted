//using System;
namespace LinearAlgebra
{

    // Allocation helper
    public unsafe partial struct Arena {

        #region BOOLVECTOR
        public boolN boolVec(int N, bool uninit = false)
        {
            boolVecRecord* rec = _core->BoolVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->BoolVecRecords;
            rec->SelfIndex = slot;
            return new boolN(N, rec, Allocator, uninit);
        }

        public boolN boolTempVec(int N, bool uninit = false)
        {
            boolVecRecord* rec = _core->TempBoolVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->TempBoolVecRecords;
            rec->SelfIndex = slot;
            return new boolN(N, rec, Allocator, uninit);
        }

        internal boolN boolVec(in boolN orig)
        {
            boolVecRecord* rec = _core->BoolVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->BoolVecRecords;
            rec->SelfIndex = slot;
            return new boolN(in orig, rec, Allocator);
        }

        internal boolN boolTempVec(in boolN orig)
        {
            boolVecRecord* rec = _core->TempBoolVecRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->TempBoolVecRecords;
            rec->SelfIndex = slot;
            return new boolN(in orig, rec, Allocator);
        }

        #endregion

        #region BOOLMATRIX

        public boolMxN boolMat(int M_rows, int N_cols, bool uninit = false)
        {
            boolMatRecord* rec = _core->BoolMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->BoolMatRecords;
            rec->SelfIndex = slot;
            return new boolMxN(M_rows, N_cols, rec, Allocator, uninit);
        }

        public boolMxN boolMat(in boolMxN mat)
        {
            boolMatRecord* rec = _core->BoolMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->BoolMatRecords;
            rec->SelfIndex = slot;
            return new boolMxN(in mat, rec, Allocator);
        }

        public boolMxN boolTempMat(int M_rows, int N_cols, bool uninit = false)
        {
            boolMatRecord* rec = _core->TempBoolMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->TempBoolMatRecords;
            rec->SelfIndex = slot;
            return new boolMxN(M_rows, N_cols, rec, Allocator, uninit);
        }

        public boolMxN boolTempMat(in boolMxN mat)
        {
            boolMatRecord* rec = _core->TempBoolMatRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->TempBoolMatRecords;
            rec->SelfIndex = slot;
            return new boolMxN(in mat, rec, Allocator);
        }

        #endregion

    }

}
