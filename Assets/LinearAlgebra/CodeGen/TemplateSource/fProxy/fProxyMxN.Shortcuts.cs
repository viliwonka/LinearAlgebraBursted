// Generated
// Shortcuts for creating new vectors and matrices
using System.Runtime.CompilerServices;

using LinearAlgebra;

//alsoExpand[uint]// widens the iProxy-family copy-replace block below to a 4th (uint) copy, giving
//fProxyMxN a uintVec/uintTempVec/uintMat/uintTempMat cross-shortcut alongside its existing int/
//short/long ones - mirrors the identical iProxyMxN.Shortcuts.cs/iProxyN.Shortcuts.cs treatment;
//Hash.fProxy.cs's rowHashes/colHashes allocating wrappers need it to allocate their uintN result
//from A's own arena without direct access to fProxyMxN's private _arena field.

namespace LinearAlgebra {

    public partial struct fProxyMxN : IArenaShortcuts
    {
        //+copyReplace
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyN fProxyVec(int N, bool uninit = false) => _arena.fProxyVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyN fProxyTempVec(int N, bool uninit = false) => _arena.fProxyTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false) => _arena.fProxyMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyMxN fProxyTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.fProxyTempMat(M_rows, N_cols, uninit);
        //-copyReplace

        //+copyReplace
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyN iProxyVec(int N, bool uninit = false) => _arena.iProxyVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyN iProxyTempVec(int N, bool uninit = false) => _arena.iProxyTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyMxN iProxyMat(int M_rows, int N_cols, bool uninit = false) => _arena.iProxyMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyMxN iProxyTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.iProxyTempMat(M_rows, N_cols, uninit);
        //-copyReplace

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
