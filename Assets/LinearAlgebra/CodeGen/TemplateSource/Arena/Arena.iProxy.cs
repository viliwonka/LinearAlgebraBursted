using Unity.Collections.LowLevel.Unsafe;

//alsoExpand[uint]// core bump-allocator factories (arena.uintVec/uintMat); no signed-only ops here.

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        internal UnsafeList<iProxyN> iProxyVectors;
        internal UnsafeList<iProxyMxN> iProxyMatrices;
        internal UnsafeList<iProxyN> iProxyTempVectors;
        internal UnsafeList<iProxyMxN> iProxyTempMatrices;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public iProxyN iProxyVec(int N, bool uninit = false) {

            var vec = new iProxyN(N, in this, uninit);
            _core->iProxyVectors.Add(in vec);
            return vec;
        }

        public iProxyN iProxyVec(int N, iProxy s)
        {
            var vec = new iProxyN(N, in this, true);
            _core->iProxyVectors.Add(in vec);
            unsafe {
                mathUnsafeiProxy.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal iProxyN iProxyVec(in iProxyN orig)
        {
            var vec = new iProxyN(in orig);
            _core->iProxyVectors.Add(in vec);   // persistent (backs Copy()); was wrongly the temp list
            return vec;
        }

        internal iProxyN iProxyTempVec(int N, bool uninit = false)
        {
            var vec = new iProxyN(N, in this, uninit);
            _core->iProxyTempVectors.Add(in vec);
            return vec;
        }

        internal iProxyN iProxyTempVec(in iProxyN orig)
        {
            var vec = new iProxyN(in orig);
            _core->iProxyTempVectors.Add(in vec);
            return vec;
        }
        #endregion

        #region MATRIX
        public iProxyMxN iProxyMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED (was leaking on Dispose).
            return iProxyMat(dim, dim, uninit);
        }

        public iProxyMxN iProxyMat(int M_rows, int N_cols, bool uninit = false)
        {
            var matrix = new iProxyMxN(M_rows, N_cols, in this, uninit);
            _core->iProxyMatrices.Add(in matrix);
            return matrix;
        }

        public iProxyMxN iProxyMat(int M_rows, int N_cols, iProxy s)
        {
            var matrix = new iProxyMxN(M_rows, N_cols, in this, false);
            _core->iProxyMatrices.Add(in matrix);
            unsafe
            {
                mathUnsafeiProxy.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public iProxyMxN iProxyMat(in iProxyMxN orig)
        {
            var matrix = new iProxyMxN(in orig);
            _core->iProxyMatrices.Add(in matrix);
            return matrix;
        }

        internal iProxyMxN iProxyTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            var matrix = new iProxyMxN(M_rows, M_cols, in this, uninit);
            _core->iProxyTempMatrices.Add(in matrix);
            return matrix;
        }

        internal iProxyMxN iProxyTempMat(in iProxyMxN orig)
        {
            var matrix = new iProxyMxN(orig);
            _core->iProxyTempMatrices.Add(in matrix);
            return matrix;
        }
        #endregion

        // --- debug pool checks (see Arena.fProxy) ---
        public bool isPersistent(in iProxyN v) {
            for (int i = 0; i < _core->iProxyVectors.Length; i++) if (_core->iProxyVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in iProxyN v) {
            for (int i = 0; i < _core->iProxyTempVectors.Length; i++) if (_core->iProxyTempVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isPersistent(in iProxyMxN m) {
            for (int i = 0; i < _core->iProxyMatrices.Length; i++) if (_core->iProxyMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in iProxyMxN m) {
            for (int i = 0; i < _core->iProxyTempMatrices.Length; i++) if (_core->iProxyTempMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }

    }

}
