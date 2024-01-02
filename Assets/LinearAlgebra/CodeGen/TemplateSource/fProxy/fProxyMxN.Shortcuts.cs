// Generated
// Shortcuts for creating new vectors and matrices test
using System.Runtime.CompilerServices;

using LinearAlgebra;

namespace LinearAlgebra {

    public partial struct fProxyMxN : IArenaShortcuts
    {
        //+copyReplace
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyN fProxyVec(int N, bool uninit = false) => _arenaPtr->fProxyVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyN tempfProxyVec(int N, bool uninit = false) => _arenaPtr->tempfProxyVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->fProxyMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyMxN tempfProxyMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->tempfProxyMat(M_rows, N_cols, uninit);
        //-copyReplace

        //+copyReplace
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyN iProxyVec(int N, bool uninit = false) => _arenaPtr->iProxyVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyN tempiProxyVec(int N, bool uninit = false) => _arenaPtr->tempiProxyVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyMxN iProxyMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->iProxyMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyMxN tempiProxyMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->tempiProxyMat(M_rows, N_cols, uninit);
        //-copyReplace

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