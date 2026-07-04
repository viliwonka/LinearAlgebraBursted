// Generated
// Shortcuts for creating new vectors and matrices
using System.Runtime.CompilerServices;

using LinearAlgebra;


namespace LinearAlgebra {

    public partial struct uintMxN : IArenaShortcuts
    {
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatN floatVec(int N, bool uninit = false) => _arena.floatVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatN floatTempVec(int N, bool uninit = false) => _arena.floatTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatMxN floatMat(int M_rows, int N_cols, bool uninit = false) => _arena.floatMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatMxN floatTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.floatTempMat(M_rows, N_cols, uninit);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleN doubleVec(int N, bool uninit = false) => _arena.doubleVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleN doubleTempVec(int N, bool uninit = false) => _arena.doubleTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleMxN doubleMat(int M_rows, int N_cols, bool uninit = false) => _arena.doubleMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleMxN doubleTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.doubleTempMat(M_rows, N_cols, uninit);
        

        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intN intVec(int N, bool uninit = false) => _arena.intVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intN intTempVec(int N, bool uninit = false) => _arena.intTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intMxN intMat(int M_rows, int N_cols, bool uninit = false) => _arena.intMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intMxN intTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.intTempMat(M_rows, N_cols, uninit);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortN shortVec(int N, bool uninit = false) => _arena.shortVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortN shortTempVec(int N, bool uninit = false) => _arena.shortTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortMxN shortMat(int M_rows, int N_cols, bool uninit = false) => _arena.shortMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortMxN shortTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.shortTempMat(M_rows, N_cols, uninit);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longN longVec(int N, bool uninit = false) => _arena.longVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longN longTempVec(int N, bool uninit = false) => _arena.longTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longMxN longMat(int M_rows, int N_cols, bool uninit = false) => _arena.longMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longMxN longTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.longTempMat(M_rows, N_cols, uninit);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe uintN uintVec(int N, bool uninit = false) => _arena.uintVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe uintN uintTempVec(int N, bool uninit = false) => _arena.uintTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe uintMxN uintMat(int M_rows, int N_cols, bool uninit = false) => _arena.uintMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe uintMxN uintTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.uintTempMat(M_rows, N_cols, uninit);
        


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolN boolVec(int n, bool uninit = false) => _arena.boolVec(n, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolN boolTempVec(int n, bool uninit = false) => _arena.boolTempVec(n, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => _arena.boolMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolMxN boolTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.boolTempMat(M_rows, N_cols, uninit);
    }
}