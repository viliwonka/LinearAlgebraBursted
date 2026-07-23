// Generated
// Shortcuts for creating new vectors and matrices
using Unity.Collections;
using LinearAlgebra;

namespace LinearAlgebra {

    public partial struct boolN : IArenaShortcuts
    {
        //+copyReplace
        public unsafe fProxyN fProxyVec(int N, bool uninit = false) => new fProxyN(N, Allocator.Temp, uninit);

        public unsafe fProxyN fProxyTempVec(int N, bool uninit = false) => new fProxyN(N, Allocator.Temp, uninit);

        public unsafe fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false) => new fProxyMxN(M_rows, N_cols, Allocator.Temp, uninit);

        public unsafe fProxyMxN fProxyTempMat(int M_rows, int N_cols, bool uninit = false) => new fProxyMxN(M_rows, N_cols, Allocator.Temp, uninit);
        //-copyReplace


        public unsafe boolN boolVec(int n, bool uninit = false) => new boolN(n, Allocator.Temp, uninit);

        public unsafe boolN boolTempVec(int n, bool uninit = false) => new boolN(n, Allocator.Temp, uninit);

        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => new boolMxN(M_rows, N_cols, Allocator.Temp, uninit);

        public unsafe boolMxN boolTempMat(int M_rows, int N_cols, bool uninit = false) => new boolMxN(M_rows, N_cols, Allocator.Temp, uninit);
    }
}
