// Generated
// Shortcuts for creating new vectors and matrices
using System.Runtime.CompilerServices;

using LinearAlgebra;


namespace LinearAlgebra {

    public partial struct longN : IArenaShortcuts
    {
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatN floatVec(int N, bool uninit = false) => OwnerArena.floatVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatN floatTempVec(int N, bool uninit = false) => OwnerArena.floatTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatMxN floatMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.floatMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatMxN floatTempMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.floatTempMat(M_rows, N_cols, uninit);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleN doubleVec(int N, bool uninit = false) => OwnerArena.doubleVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleN doubleTempVec(int N, bool uninit = false) => OwnerArena.doubleTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleMxN doubleMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.doubleMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleMxN doubleTempMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.doubleTempMat(M_rows, N_cols, uninit);
        

        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intN intVec(int N, bool uninit = false) => OwnerArena.intVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intN intTempVec(int N, bool uninit = false) => OwnerArena.intTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intMxN intMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.intMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intMxN intTempMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.intTempMat(M_rows, N_cols, uninit);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortN shortVec(int N, bool uninit = false) => OwnerArena.shortVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortN shortTempVec(int N, bool uninit = false) => OwnerArena.shortTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortMxN shortMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.shortMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortMxN shortTempMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.shortTempMat(M_rows, N_cols, uninit);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longN longVec(int N, bool uninit = false) => OwnerArena.longVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longN longTempVec(int N, bool uninit = false) => OwnerArena.longTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longMxN longMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.longMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longMxN longTempMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.longTempMat(M_rows, N_cols, uninit);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe uintN uintVec(int N, bool uninit = false) => OwnerArena.uintVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe uintN uintTempVec(int N, bool uninit = false) => OwnerArena.uintTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe uintMxN uintMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.uintMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe uintMxN uintTempMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.uintTempMat(M_rows, N_cols, uninit);
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolN boolVec(int n, bool uninit = false) => OwnerArena.boolVec(n, uninit);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolN boolTempVec(int n, bool uninit = false) => OwnerArena.boolTempVec(n, uninit);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.boolMat(M_rows, N_cols, uninit);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolMxN boolTempMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.boolTempMat(M_rows, N_cols, uninit);
    }
}