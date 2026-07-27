// Generated
// Shortcuts for creating new vectors and matrices
using System.Runtime.CompilerServices;

using Unity.Collections;
using BULA;

//alsoExpand[uint]// gives uintN an IAllocShortcuts implementation (float/double/bool cross-type
//shortcuts below, unaffected by this file's own outer-type opt-in). The inner iProxy-family
//copy-replace block below ALSO widens to a 4th (uint) copy from this same flag - TemplateConverter
//resolves it once per file and threads it into that block too - giving every int-family type
//(including uintN itself) a uintVec/uintTempVec/uintMat/uintTempMat cross-shortcut, which
//Blas/OP.Dot.iProxy.cs's a.iProxyTempMat(...)/x.iProxyTempVec(...) calls need to resolve for uint.

namespace BULA {

    public partial struct iProxyN : IAllocShortcuts
    {
        //+copyReplace
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyN fProxyVec(int N, bool uninit = false) => new fProxyN(N, Allocator.Temp, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyN fProxyTempVec(int N, bool uninit = false) => new fProxyN(N, Allocator.Temp, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false) => new fProxyMxN(M_rows, N_cols, Allocator.Temp, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyMxN fProxyTempMat(int M_rows, int N_cols, bool uninit = false) => new fProxyMxN(M_rows, N_cols, Allocator.Temp, uninit);
        //-copyReplace

        //+copyReplace
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyN iProxyVec(int N, bool uninit = false) => new iProxyN(N, Allocator.Temp, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyN iProxyTempVec(int N, bool uninit = false) => new iProxyN(N, Allocator.Temp, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyMxN iProxyMat(int M_rows, int N_cols, bool uninit = false) => new iProxyMxN(M_rows, N_cols, Allocator.Temp, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyMxN iProxyTempMat(int M_rows, int N_cols, bool uninit = false) => new iProxyMxN(M_rows, N_cols, Allocator.Temp, uninit);
        //-copyReplace

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolN boolVec(int n, bool uninit = false) => new boolN(n, Allocator.Temp, uninit);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolN boolTempVec(int n, bool uninit = false) => new boolN(n, Allocator.Temp, uninit);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => new boolMxN(M_rows, N_cols, Allocator.Temp, uninit);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolMxN boolTempMat(int M_rows, int N_cols, bool uninit = false) => new boolMxN(M_rows, N_cols, Allocator.Temp, uninit);
    }
}
