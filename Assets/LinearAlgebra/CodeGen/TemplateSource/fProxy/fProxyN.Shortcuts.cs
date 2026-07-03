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
        public unsafe fProxyN fProxyVec(int N, bool uninit = false) => _arena.fProxyVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyN tempfProxyVec(int N, bool uninit = false) => _arena.tempfProxyVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false) => _arena.fProxyMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyMxN tempfProxyMat(int M_rows, int N_cols, bool uninit = false) => _arena.tempfProxyMat(M_rows, N_cols, uninit);
        //-copyReplace

        // NOT wrapped in copyReplace: there is no iProxy BSM equivalent, so this only needs to
        // exist for the fProxy float/double types this file already generates. Forwards to the
        // arena `b` carries so Solvers.fProxy.cs can materialize A^T once per solve via
        // `b.fProxyBSMTranspose(in A)` without direct access to fProxyN's private _arena field.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe fProxyBSM fProxyBSMTranspose(in fProxyBSM A) => _arena.fProxyBSMTranspose(in A);

        //+copyReplace
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyN iProxyVec(int N, bool uninit = false) => _arena.iProxyVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyN tempiProxyVec(int N, bool uninit = false) => _arena.tempiProxyVec(N, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyMxN iProxyMat(int M_rows, int N_cols, bool uninit = false) => _arena.iProxyMat(M_rows, N_cols, uninit);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe iProxyMxN tempiProxyMat(int M_rows, int N_cols, bool uninit = false) => _arena.tempiProxyMat(M_rows, N_cols, uninit);
        //-copyReplace

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolN boolVec(int n, bool uninit = false) => _arena.boolVec(n, uninit);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolN tempBoolVec(int n, bool uninit = false) => _arena.tempBoolVec(n, uninit);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolMxN boolMat(int M_rows, int N_cols, bool uninit = false) => _arena.boolMat(M_rows, N_cols, uninit);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe boolMxN tempBoolMat(int M_rows, int N_cols, bool uninit = false) => _arena.tempBoolMat(M_rows, N_cols, uninit);
    }
}