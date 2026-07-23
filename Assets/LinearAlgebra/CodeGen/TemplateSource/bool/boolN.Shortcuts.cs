// Generated
// Shortcuts for creating new vectors and matrices
using Unity.Collections;
using LinearAlgebra;

namespace LinearAlgebra {

    public partial struct boolN : IArenaShortcuts
    {
        //+copyReplace
        public unsafe fProxyN fProxyVec(int N, bool uninit = false) => _rec != null ? OwnerArena.fProxyVec(N, uninit) : new fProxyN(N, Allocator.Temp, uninit);

        public unsafe fProxyN fProxyTempVec(int N, bool uninit = false) => _rec != null ? OwnerArena.fProxyTempVec(N, uninit) : new fProxyN(N, Allocator.Temp, uninit);

        public unsafe fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false) => _rec != null ? OwnerArena.fProxyMat(M_rows, N_cols, uninit) : new fProxyMxN(M_rows, N_cols, Allocator.Temp, uninit);

        public unsafe fProxyMxN fProxyTempMat(int M_rows, int N_cols, bool uninit = false) => _rec != null ? OwnerArena.fProxyTempMat(M_rows, N_cols, uninit) : new fProxyMxN(M_rows, N_cols, Allocator.Temp, uninit);
        //-copyReplace


        public unsafe boolN boolVec(int n, bool uninit = false) => _rec != null ? OwnerArena.boolVec(n, uninit) : new boolN(n, Allocator.Temp, uninit);

        public unsafe boolN boolTempVec(int n, bool uninit = false) => _rec != null ? OwnerArena.boolTempVec(n, uninit) : new boolN(n, Allocator.Temp, uninit);

        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => _rec != null ? OwnerArena.boolMat(M_rows, N_cols, uninit) : new boolMxN(M_rows, N_cols, Allocator.Temp, uninit);

        public unsafe boolMxN boolTempMat(int M_rows, int N_cols, bool uninit = false) => _rec != null ? OwnerArena.boolTempMat(M_rows, N_cols, uninit) : new boolMxN(M_rows, N_cols, Allocator.Temp, uninit);
    }
}