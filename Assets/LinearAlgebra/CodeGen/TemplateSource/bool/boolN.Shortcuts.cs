// Generated
// Shortcuts for creating new vectors and matrices test
using LinearAlgebra;

namespace LinearAlgebra {

    public partial struct boolN : IArenaShortcuts
    {
        //+copyReplace
        public unsafe fProxyN fProxyVec(int N, bool uninit = false) => _arena.fProxyVec(N, uninit);

        public unsafe fProxyN tempfProxyVec(int N, bool uninit = false) => _arena.tempfProxyVec(N, uninit);

        public unsafe fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false) => _arena.fProxyMat(M_rows, N_cols, uninit);

        public unsafe fProxyMxN tempfProxyMat(int M_rows, int N_cols, bool uninit = false) => _arena.tempfProxyMat(M_rows, N_cols, uninit);
        //-copyReplace


        public unsafe boolN boolVec(int n, bool uninit = false) => _arena.boolVec(n, uninit);

        public unsafe boolN tempBoolVec(int n, bool uninit = false) => _arena.tempBoolVec(n, uninit);

        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => _arena.boolMat(M_rows, N_cols, uninit);

        public unsafe boolMxN tempBoolMat(int M_rows, int N_cols, bool uninit = false) => _arena.tempBoolMat(M_rows, N_cols, uninit);
    }
}