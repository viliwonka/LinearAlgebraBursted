// Generated
// Shortcuts for creating new vectors and matrices
using Unity.Collections;
using LinearAlgebra;

//alsoExpand[uint]// widens the iProxy-family copy-replace block below to a 4th (uint) copy, giving
//boolMxN an int/short/long/uint cross-shortcut set (intVec/shortVec/longVec/uintVec, and the
//matching Mat/TempVec/TempMat forms) alongside its existing float/double one - mirrors the
//identical fProxyMxN.Shortcuts.cs/iProxyMxN.Shortcuts.cs treatment; Hash.iProxy.cs's bool-sourced
//rowHashes/colHashes allocating wrappers need boolMxN.uintVec() specifically, to allocate their
//uintN result from A's own arena without direct access to boolMxN's private _rec field.

namespace LinearAlgebra {

    public partial struct boolMxN : IArenaShortcuts
    {
        //+copyReplace
        public unsafe fProxyN fProxyVec(int N, bool uninit = false) => _rec != null ? OwnerArena.fProxyVec(N, uninit) : new fProxyN(N, Allocator.Temp, uninit);

        public unsafe fProxyN fProxyTempVec(int N, bool uninit = false) => _rec != null ? OwnerArena.fProxyTempVec(N, uninit) : new fProxyN(N, Allocator.Temp, uninit);

        public unsafe fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false) => _rec != null ? OwnerArena.fProxyMat(M_rows, N_cols, uninit) : new fProxyMxN(M_rows, N_cols, Allocator.Temp, uninit);

        public unsafe fProxyMxN fProxyTempMat(int M_rows, int N_cols, bool uninit = false) => _rec != null ? OwnerArena.fProxyTempMat(M_rows, N_cols, uninit) : new fProxyMxN(M_rows, N_cols, Allocator.Temp, uninit);
        //-copyReplace

        //+copyReplace
        public unsafe iProxyN iProxyVec(int N, bool uninit = false) => _rec != null ? OwnerArena.iProxyVec(N, uninit) : new iProxyN(N, Allocator.Temp, uninit);

        public unsafe iProxyN iProxyTempVec(int N, bool uninit = false) => _rec != null ? OwnerArena.iProxyTempVec(N, uninit) : new iProxyN(N, Allocator.Temp, uninit);

        public unsafe iProxyMxN iProxyMat(int M_rows, int N_cols, bool uninit = false) => _rec != null ? OwnerArena.iProxyMat(M_rows, N_cols, uninit) : new iProxyMxN(M_rows, N_cols, Allocator.Temp, uninit);

        public unsafe iProxyMxN iProxyTempMat(int M_rows, int N_cols, bool uninit = false) => _rec != null ? OwnerArena.iProxyTempMat(M_rows, N_cols, uninit) : new iProxyMxN(M_rows, N_cols, Allocator.Temp, uninit);
        //-copyReplace


        public unsafe boolN boolVec(int n, bool uninit = false) => _rec != null ? OwnerArena.boolVec(n, uninit) : new boolN(n, Allocator.Temp, uninit);

        public unsafe boolN boolTempVec(int n, bool uninit = false) => _rec != null ? OwnerArena.boolTempVec(n, uninit) : new boolN(n, Allocator.Temp, uninit);

        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => _rec != null ? OwnerArena.boolMat(M_rows, N_cols, uninit) : new boolMxN(M_rows, N_cols, Allocator.Temp, uninit);

        public unsafe boolMxN boolTempMat(int M_rows, int N_cols, bool uninit = false) => _rec != null ? OwnerArena.boolTempMat(M_rows, N_cols, uninit) : new boolMxN(M_rows, N_cols, Allocator.Temp, uninit);
    }
}