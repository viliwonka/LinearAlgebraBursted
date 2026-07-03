// Generated
// Shortcuts for creating new vectors and matrices
using LinearAlgebra;

namespace LinearAlgebra {

    public partial struct boolN : IArenaShortcuts
    {
        
        public unsafe floatN floatVec(int N, bool uninit = false) => _arena.floatVec(N, uninit);

        public unsafe floatN floatTempVec(int N, bool uninit = false) => _arena.floatTempVec(N, uninit);

        public unsafe floatMxN floatMat(int M_rows, int N_cols, bool uninit = false) => _arena.floatMat(M_rows, N_cols, uninit);

        public unsafe floatMxN floatTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.floatTempMat(M_rows, N_cols, uninit);
        
        public unsafe doubleN doubleVec(int N, bool uninit = false) => _arena.doubleVec(N, uninit);

        public unsafe doubleN doubleTempVec(int N, bool uninit = false) => _arena.doubleTempVec(N, uninit);

        public unsafe doubleMxN doubleMat(int M_rows, int N_cols, bool uninit = false) => _arena.doubleMat(M_rows, N_cols, uninit);

        public unsafe doubleMxN doubleTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.doubleTempMat(M_rows, N_cols, uninit);
        


        public unsafe boolN boolVec(int n, bool uninit = false) => _arena.boolVec(n, uninit);

        public unsafe boolN boolTempVec(int n, bool uninit = false) => _arena.boolTempVec(n, uninit);

        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => _arena.boolMat(M_rows, N_cols, uninit);

        public unsafe boolMxN boolTempMat(int M_rows, int N_cols, bool uninit = false) => _arena.boolTempMat(M_rows, N_cols, uninit);
    }
}