// Generated
// Shortcuts for creating new vectors and matrices
using System.Runtime.CompilerServices;

using LinearAlgebra;
using LinearAlgebra.Sparse;

namespace LinearAlgebra {

    public partial struct fProxyN : IArenaShortcuts
    {
        //+copyReplace
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyN fProxyVec(int N, bool uninit = false) => OwnerArena.fProxyVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyN fProxyTempVec(int N, bool uninit = false) => OwnerArena.fProxyTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.fProxyMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyMxN fProxyTempMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.fProxyTempMat(M_rows, N_cols, uninit);
        //-copyReplace

        // NOT wrapped in copyReplace: there is no iProxy BSR equivalent, so this only needs to
        // exist for the fProxy float/double types this file already generates. Forwards to the
        // arena `b` carries so Solvers.fProxy.cs can materialize A^T once per solve via
        // `b.fProxyBSRTranspose(in A)` without direct access to fProxyN's private _rec field.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyBSR fProxyBSRTranspose(in fProxyBSR A) => OwnerArena.fProxyBSRTranspose(in A);

        //+copyReplace
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyN iProxyVec(int N, bool uninit = false) => OwnerArena.iProxyVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyN iProxyTempVec(int N, bool uninit = false) => OwnerArena.iProxyTempVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyMxN iProxyMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.iProxyMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyMxN iProxyTempMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.iProxyTempMat(M_rows, N_cols, uninit);
        //-copyReplace

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolN boolVec(int n, bool uninit = false) => OwnerArena.boolVec(n, uninit);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolN boolTempVec(int n, bool uninit = false) => OwnerArena.boolTempVec(n, uninit);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.boolMat(M_rows, N_cols, uninit);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolMxN boolTempMat(int M_rows, int N_cols, bool uninit = false) => OwnerArena.boolTempMat(M_rows, N_cols, uninit);
    }
}