// Generated
// Shortcuts for creating new vectors and matrices test
using LinearAlgebra;

namespace LinearAlgebra {

    public partial struct boolMxN : IArenaShortcuts
    {
        
        public unsafe floatN floatVec(int N, bool uninit = false) => _arenaPtr->floatVec(N, uninit);

        public unsafe floatN tempfloatVec(int N, bool uninit = false) => _arenaPtr->tempfloatVec(N, uninit);

        public unsafe floatMxN floatMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->floatMat(M_rows, N_cols, uninit);

        public unsafe floatMxN tempfloatMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->tempfloatMat(M_rows, N_cols, uninit);
        
        public unsafe doubleN doubleVec(int N, bool uninit = false) => _arenaPtr->doubleVec(N, uninit);

        public unsafe doubleN tempdoubleVec(int N, bool uninit = false) => _arenaPtr->tempdoubleVec(N, uninit);

        public unsafe doubleMxN doubleMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->doubleMat(M_rows, N_cols, uninit);

        public unsafe doubleMxN tempdoubleMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->tempdoubleMat(M_rows, N_cols, uninit);
        


        public unsafe boolN boolVec(int n, bool uninit = false) => _arenaPtr->BoolVector(n, uninit);

        public unsafe boolN tempBoolVec(int n, bool uninit = false) => _arenaPtr->TempBoolVector(n, uninit);

        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->BoolMatrix(M_rows, N_cols, uninit);

        public unsafe boolMxN tempBoolMat(int M_rows, int N_cols, bool uninit = false) => _arenaPtr->TempBoolMatrix(M_rows, N_cols, uninit);
    }
}