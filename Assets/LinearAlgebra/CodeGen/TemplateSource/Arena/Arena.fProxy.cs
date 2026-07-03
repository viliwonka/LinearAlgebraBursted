using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        internal UnsafeList<fProxyN> fProxyVectors;
        internal UnsafeList<fProxyMxN> fProxyMatrices;
        internal UnsafeList<fProxyN> tempfProxyVectors;
        internal UnsafeList<fProxyMxN> tempfProxyMatrices;
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

        internal fProxyN tempfProxyVec(int N, bool uninit = false)
        {
            var vec = new fProxyN(N, in this, uninit);
            _core->tempfProxyVectors.Add(in vec);
            return vec;
        }

        internal fProxyN tempfProxyVec(in fProxyN orig)
        {
            var vec = new fProxyN(in orig);
            _core->tempfProxyVectors.Add(in vec);
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

        internal fProxyMxN tempfProxyMat(int M_rows, int M_cols, bool uninit = false)
        {
            var matrix = new fProxyMxN(M_rows, M_cols, in this, uninit);
            _core->tempfProxyMatrices.Add(in matrix);
            return matrix;
        }

        internal fProxyMxN tempfProxyMat(in fProxyMxN orig)
        {
            var matrix = new fProxyMxN(orig);
            _core->tempfProxyMatrices.Add(in matrix);
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
            for (int i = 0; i < _core->tempfProxyVectors.Length; i++) if (_core->tempfProxyVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isPersistent(in fProxyMxN m) {
            for (int i = 0; i < _core->fProxyMatrices.Length; i++) if (_core->fProxyMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in fProxyMxN m) {
            for (int i = 0; i < _core->tempfProxyMatrices.Length; i++) if (_core->tempfProxyMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }

    }

}
