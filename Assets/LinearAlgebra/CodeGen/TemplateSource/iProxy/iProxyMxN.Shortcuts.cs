// Generated
// Shortcuts for creating new vectors and matrices
using System.Runtime.CompilerServices;

using Unity.Collections;
using BULA;

//alsoExpand[uint]// gives uintMxN an IArenaShortcuts implementation; the inner iProxy-family
//copy-replace block below widens to a 4th (uint) copy from this same flag, giving every int-family
//type a uintVec/uintTempVec/uintMat/uintTempMat cross-shortcut - see the identical note in
//iProxyN.Shortcuts.cs.

namespace BULA {

    public partial struct iProxyMxN : IArenaShortcuts
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
