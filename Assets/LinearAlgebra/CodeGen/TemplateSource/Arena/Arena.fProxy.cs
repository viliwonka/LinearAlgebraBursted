using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        internal UnsafeList<fProxyN> fProxyVectors;
        internal UnsafeList<fProxyMxN> fProxyMatrices;
        internal UnsafeList<fProxyN> fProxyTempVectors;
        internal UnsafeList<fProxyMxN> fProxyTempMatrices;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public fProxyN fProxyVec(int N, bool uninit = false) {

            var vec = new fProxyN(N, in this, uninit);
            _core->fProxyVectors.Add(in vec);
            return vec;
        }

        public fProxyN fProxyVec(int N, fProxy s)
        {
            var vec = new fProxyN(N, in this, true);
            _core->fProxyVectors.Add(in vec);
            unsafe {
                mathUnsafefProxy.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal fProxyN fProxyVec(in fProxyN orig)
        {
            var vec = new fProxyN(in orig);
            _core->fProxyVectors.Add(in vec);   // persistent (backs Copy()); was wrongly the temp list
            return vec;
        }

        internal fProxyN fProxyTempVec(int N, bool uninit = false)
        {
            var vec = new fProxyN(N, in this, uninit);
            _core->fProxyTempVectors.Add(in vec);
            return vec;
        }

        internal fProxyN fProxyTempVec(in fProxyN orig)
        {
            var vec = new fProxyN(in orig);
            _core->fProxyTempVectors.Add(in vec);
            return vec;
        }
        #endregion

        #region MATRIX
        public fProxyMxN fProxyMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED in fProxyMatrices —
            // the direct `new fProxyMxN(...)` here was untracked and leaked on Dispose.
            return fProxyMat(dim, dim, uninit);
        }

        public fProxyMxN fProxyMat(int M_rows, int N_cols, bool uninit = false)
        {
            var matrix = new fProxyMxN(M_rows, N_cols, in this, uninit);
            _core->fProxyMatrices.Add(in matrix);
            return matrix;
        }

        public fProxyMxN fProxyMat(int M_rows, int N_cols, fProxy s)
        {
            var matrix = new fProxyMxN(M_rows, N_cols, in this, false);
            _core->fProxyMatrices.Add(in matrix);
            unsafe
            {
                mathUnsafefProxy.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public fProxyMxN fProxyMat(in fProxyMxN orig)
        {
            var matrix = new fProxyMxN(in orig);
            _core->fProxyMatrices.Add(in matrix);
            return matrix;
        }

        internal fProxyMxN fProxyTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            var matrix = new fProxyMxN(M_rows, M_cols, in this, uninit);
            _core->fProxyTempMatrices.Add(in matrix);
            return matrix;
        }

        internal fProxyMxN fProxyTempMat(in fProxyMxN orig)
        {
            var matrix = new fProxyMxN(orig);
            _core->fProxyTempMatrices.Add(in matrix);
            return matrix;
        }
        #endregion

        // --- debug pool checks: confirm a buffer lives in the expected (persistent vs temp) list,
        //     e.g. to assert an op didn't silently move a persistent input into the temp pool ---
        public bool isPersistent(in fProxyN v) {
            for (int i = 0; i < _core->fProxyVectors.Length; i++) if (_core->fProxyVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in fProxyN v) {
            for (int i = 0; i < _core->fProxyTempVectors.Length; i++) if (_core->fProxyTempVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isPersistent(in fProxyMxN m) {
            for (int i = 0; i < _core->fProxyMatrices.Length; i++) if (_core->fProxyMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in fProxyMxN m) {
            for (int i = 0; i < _core->fProxyTempMatrices.Length; i++) if (_core->fProxyTempMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }

    }

}
