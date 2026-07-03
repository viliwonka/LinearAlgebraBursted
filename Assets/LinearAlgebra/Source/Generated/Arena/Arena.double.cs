using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        internal UnsafeList<doubleN> doubleVectors;
        internal UnsafeList<doubleMxN> doubleMatrices;
        internal UnsafeList<doubleN> doubleTempVectors;
        internal UnsafeList<doubleMxN> doubleTempMatrices;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public doubleN doubleVec(int N, bool uninit = false) {

            var vec = new doubleN(N, in this, uninit);
            _core->doubleVectors.Add(in vec);
            return vec;
        }

        public doubleN doubleVec(int N, double s)
        {
            var vec = new doubleN(N, in this, true);
            _core->doubleVectors.Add(in vec);
            unsafe {
                mathUnsafedouble.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal doubleN doubleVec(in doubleN orig)
        {
            var vec = new doubleN(in orig);
            _core->doubleVectors.Add(in vec);   // persistent (backs Copy()); was wrongly the temp list
            return vec;
        }

        internal doubleN doubleTempVec(int N, bool uninit = false)
        {
            var vec = new doubleN(N, in this, uninit);
            _core->doubleTempVectors.Add(in vec);
            return vec;
        }

        internal doubleN doubleTempVec(in doubleN orig)
        {
            var vec = new doubleN(in orig);
            _core->doubleTempVectors.Add(in vec);
            return vec;
        }
        #endregion

        #region MATRIX
        public doubleMxN doubleMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED in doubleMatrices —
            // the direct `new doubleMxN(...)` here was untracked and leaked on Dispose.
            return doubleMat(dim, dim, uninit);
        }

        public doubleMxN doubleMat(int M_rows, int N_cols, bool uninit = false)
        {
            var matrix = new doubleMxN(M_rows, N_cols, in this, uninit);
            _core->doubleMatrices.Add(in matrix);
            return matrix;
        }

        public doubleMxN doubleMat(int M_rows, int N_cols, double s)
        {
            var matrix = new doubleMxN(M_rows, N_cols, in this, false);
            _core->doubleMatrices.Add(in matrix);
            unsafe
            {
                mathUnsafedouble.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public doubleMxN doubleMat(in doubleMxN orig)
        {
            var matrix = new doubleMxN(in orig);
            _core->doubleMatrices.Add(in matrix);
            return matrix;
        }

        internal doubleMxN doubleTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            var matrix = new doubleMxN(M_rows, M_cols, in this, uninit);
            _core->doubleTempMatrices.Add(in matrix);
            return matrix;
        }

        internal doubleMxN doubleTempMat(in doubleMxN orig)
        {
            var matrix = new doubleMxN(orig);
            _core->doubleTempMatrices.Add(in matrix);
            return matrix;
        }
        #endregion

        // --- debug pool checks: confirm a buffer lives in the expected (persistent vs temp) list,
        //     e.g. to assert an op didn't silently move a persistent input into the temp pool ---
        public bool isPersistent(in doubleN v) {
            for (int i = 0; i < _core->doubleVectors.Length; i++) if (_core->doubleVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in doubleN v) {
            for (int i = 0; i < _core->doubleTempVectors.Length; i++) if (_core->doubleTempVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isPersistent(in doubleMxN m) {
            for (int i = 0; i < _core->doubleMatrices.Length; i++) if (_core->doubleMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in doubleMxN m) {
            for (int i = 0; i < _core->doubleTempMatrices.Length; i++) if (_core->doubleTempMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }

    }

}
