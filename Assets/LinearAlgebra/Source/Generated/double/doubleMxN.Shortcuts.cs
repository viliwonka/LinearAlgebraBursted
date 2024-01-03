// Generated
// Shortcuts for creating new vectors and matrices test
using System.Runtime.CompilerServices;

using LinearAlgebra;

namespace LinearAlgebra {

    public partial struct doubleMxN : IArenaShortcuts
    {
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatN floatVec(int N, bool uninit = false) => _arenaPtr->floatVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatN tempfloatVec(int N, bool uninit = false) => _arenaPtr->tempfloatVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatMxN floatMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->floatMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatMxN tempfloatMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->tempfloatMat(M_rows, N_cols, uninit);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleN doubleVec(int N, bool uninit = false) => _arenaPtr->doubleVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleN tempdoubleVec(int N, bool uninit = false) => _arenaPtr->tempdoubleVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleMxN doubleMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->doubleMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleMxN tempdoubleMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->tempdoubleMat(M_rows, N_cols, uninit);
        

        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intN intVec(int N, bool uninit = false) => _arenaPtr->intVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intN tempintVec(int N, bool uninit = false) => _arenaPtr->tempintVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intMxN intMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->intMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intMxN tempintMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->tempintMat(M_rows, N_cols, uninit);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortN shortVec(int N, bool uninit = false) => _arenaPtr->shortVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortN tempshortVec(int N, bool uninit = false) => _arenaPtr->tempshortVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortMxN shortMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->shortMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortMxN tempshortMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->tempshortMat(M_rows, N_cols, uninit);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longN longVec(int N, bool uninit = false) => _arenaPtr->longVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longN templongVec(int N, bool uninit = false) => _arenaPtr->templongVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longMxN longMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->longMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longMxN templongMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->templongMat(M_rows, N_cols, uninit);
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolN boolVec(int n, bool uninit = false) => _arenaPtr->BoolVector(n, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolN tempBoolVec(int n, bool uninit = false) => _arenaPtr->TempBoolVector(n, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->BoolMatrix(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolMxN tempBoolMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->TempBoolMatrix(M_rows, N_cols, uninit);
    }
}