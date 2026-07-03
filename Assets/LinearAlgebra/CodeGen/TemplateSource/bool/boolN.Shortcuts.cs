// Generated
// Shortcuts for creating new vectors and matrices
using LinearAlgebra;

namespace LinearAlgebra {

    public partial struct boolN : IArenaShortcuts
    {
        //+copyReplace
        public unsafe fProxyN fProxyVec(int N, bool uninit = false) => _arena.fProxyVec(N, uninit);

        public unsafe fProxyN fProxyTempVec(int N, bool uninit = false) => _arena.fProxyTempVec(N, uninit);

        public unsafe fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false) => _arena.fProxyMat(M_rows, N_cols, uninit);

        public unsafe fProxyMxN fProxyTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.fProxyTempMat(M_rows, N_cols, uninit);
        //-copyReplace


        public unsafe boolN boolVec(int n, bool uninit = false) => _arena.boolVec(n, uninit);

        public unsafe boolN boolTempVec(int n, bool uninit = false) => _arena.boolTempVec(n, uninit);

        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => _arena.boolMat(M_rows, N_cols, uninit);

        public unsafe boolMxN boolTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.boolTempMat(M_rows, N_cols, uninit);
    }
}