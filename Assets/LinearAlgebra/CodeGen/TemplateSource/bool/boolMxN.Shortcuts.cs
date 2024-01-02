// Generated
// Shortcuts for creating new vectors and matrices test
using LinearAlgebra;

namespace LinearAlgebra {

    public partial struct boolMxN : IArenaShortcuts
    {
        //+copyReplace
        public unsafe fProxyN fProxyVec(int N, bool uninit = false) => _arenaPtr->fProxyVec(N, uninit);

        public unsafe fProxyN tempfProxyVec(int N, bool uninit = false) => _arenaPtr->tempfProxyVec(N, uninit);

        public unsafe fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->fProxyMat(M_rows, N_cols, uninit);

        public unsafe fProxyMxN tempfProxyMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->tempfProxyMat(M_rows, N_cols, uninit);
        //-copyReplace


        public unsafe boolN boolVec(int n, bool uninit = false) => _arenaPtr->BoolVector(n, uninit);

        public unsafe boolN tempBoolVec(int n, bool uninit = false) => _arenaPtr->TempBoolVector(n, uninit);

        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->BoolMatrix(M_rows, N_cols, uninit);

        public unsafe boolMxN tempBoolMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->TempBoolMatrix(M_rows, N_cols, uninit);
    }
}