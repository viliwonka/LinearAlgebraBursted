// Generated
// Shortcuts for creating new vectors and matrices test
using System.Runtime.CompilerServices;

using LinearAlgebra;

namespace LinearAlgebra {

    public partial struct floatMxN : IArenaShortcuts
    {
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatN floatVec(int N, bool uninit = false) => _arena.floatVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatN tempfloatVec(int N, bool uninit = false) => _arena.tempfloatVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatMxN floatMat(int M_rows, int N_cols, bool uninit = false) => _arena.floatMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe floatMxN tempfloatMat(int M_rows, int N_cols, bool uninit = false) => _arena.tempfloatMat(M_rows, N_cols, uninit);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleN doubleVec(int N, bool uninit = false) => _arena.doubleVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleN tempdoubleVec(int N, bool uninit = false) => _arena.tempdoubleVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleMxN doubleMat(int M_rows, int N_cols, bool uninit = false) => _arena.doubleMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe doubleMxN tempdoubleMat(int M_rows, int N_cols, bool uninit = false) => _arena.tempdoubleMat(M_rows, N_cols, uninit);
        

        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intN intVec(int N, bool uninit = false) => _arena.intVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intN tempintVec(int N, bool uninit = false) => _arena.tempintVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intMxN intMat(int M_rows, int N_cols, bool uninit = false) => _arena.intMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe intMxN tempintMat(int M_rows, int N_cols, bool uninit = false) => _arena.tempintMat(M_rows, N_cols, uninit);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortN shortVec(int N, bool uninit = false) => _arena.shortVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortN tempshortVec(int N, bool uninit = false) => _arena.tempshortVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortMxN shortMat(int M_rows, int N_cols, bool uninit = false) => _arena.shortMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe shortMxN tempshortMat(int M_rows, int N_cols, bool uninit = false) => _arena.tempshortMat(M_rows, N_cols, uninit);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longN longVec(int N, bool uninit = false) => _arena.longVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longN templongVec(int N, bool uninit = false) => _arena.templongVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longMxN longMat(int M_rows, int N_cols, bool uninit = false) => _arena.longMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe longMxN templongMat(int M_rows, int N_cols, bool uninit = false) => _arena.templongMat(M_rows, N_cols, uninit);
        

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolN boolVec(int n, bool uninit = false) => _arena.boolVec(n, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolN tempBoolVec(int n, bool uninit = false) => _arena.tempBoolVec(n, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => _arena.boolMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolMxN tempBoolMat(int M_rows, int N_cols, bool uninit = false) => _arena.tempBoolMat(M_rows, N_cols, uninit);
    }
}
