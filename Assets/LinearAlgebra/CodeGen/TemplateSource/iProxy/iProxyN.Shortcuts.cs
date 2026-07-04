// Generated
// Shortcuts for creating new vectors and matrices
using System.Runtime.CompilerServices;

using LinearAlgebra;

//alsoExpand[uint]// gives uintN an IArenaShortcuts implementation (float/double/bool cross-type
//shortcuts below, unaffected by this file's own outer-type opt-in). The inner iProxy-family
//copy-replace block below ALSO widens to a 4th (uint) copy from this same flag - TemplateConverter
//resolves it once per file and threads it into that block too - giving every int-family type
//(including uintN itself) a uintVec/uintTempVec/uintMat/uintTempMat cross-shortcut, which
//Blas/OP.Dot.iProxy.cs's a.iProxyTempMat(...)/x.iProxyTempVec(...) calls need to resolve for uint.

namespace LinearAlgebra {

    public partial struct iProxyN : IArenaShortcuts
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