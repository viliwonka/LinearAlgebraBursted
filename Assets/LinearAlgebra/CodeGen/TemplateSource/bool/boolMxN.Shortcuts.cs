// Generated
// Shortcuts for creating new vectors and matrices
using LinearAlgebra;

//alsoExpand[uint]// widens the iProxy-family copy-replace block below to a 4th (uint) copy, giving
//boolMxN an int/short/long/uint cross-shortcut set (intVec/shortVec/longVec/uintVec, and the
//matching Mat/TempVec/TempMat forms) alongside its existing float/double one - mirrors the
//identical fProxyMxN.Shortcuts.cs/iProxyMxN.Shortcuts.cs treatment; Hash.iProxy.cs's bool-sourced
//rowHashes/colHashes allocating wrappers need boolMxN.uintVec() specifically, to allocate their
//uintN result from A's own arena without direct access to boolMxN's private _arena field.

namespace LinearAlgebra {

    public partial struct boolMxN : IArenaShortcuts
    {
        //+copyReplace
        public unsafe fProxyN fProxyVec(int N, bool uninit = false) => _arena.fProxyVec(N, uninit);

        public unsafe fProxyN fProxyTempVec(int N, bool uninit = false) => _arena.fProxyTempVec(N, uninit);

        public unsafe fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false) => _arena.fProxyMat(M_rows, N_cols, uninit);

        public unsafe fProxyMxN fProxyTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.fProxyTempMat(M_rows, N_cols, uninit);
        //-copyReplace

        //+copyReplace
        public unsafe iProxyN iProxyVec(int N, bool uninit = false) => _arena.iProxyVec(N, uninit);

        public unsafe iProxyN iProxyTempVec(int N, bool uninit = false) => _arena.iProxyTempVec(N, uninit);

        public unsafe iProxyMxN iProxyMat(int M_rows, int N_cols, bool uninit = false) => _arena.iProxyMat(M_rows, N_cols, uninit);

        public unsafe iProxyMxN iProxyTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.iProxyTempMat(M_rows, N_cols, uninit);
        //-copyReplace


        public unsafe boolN boolVec(int n, bool uninit = false) => _arena.boolVec(n, uninit);

        public unsafe boolN boolTempVec(int n, bool uninit = false) => _arena.boolTempVec(n, uninit);

        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => _arena.boolMat(M_rows, N_cols, uninit);

        public unsafe boolMxN boolTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.boolTempMat(M_rows, N_cols, uninit);
    }
}