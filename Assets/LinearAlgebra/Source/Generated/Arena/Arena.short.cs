using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        internal UnsafeList<shortN> shortVectors;
        internal UnsafeList<shortMxN> shortMatrices;
        internal UnsafeList<shortN> tempshortVectors;
        internal UnsafeList<shortMxN> tempshortMatrices;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public shortN shortVec(int N, bool uninit = false) {

            var vec = new shortN(N, in this, uninit);
            _core->shortVectors.Add(in vec);
            return vec;
        }

        // creates vector with s values
        public shortN shortVec(int N, short s)
        {
            var vec = new shortN(N, in this, true);
            _core->shortVectors.Add(in vec);
            unsafe {
                mathUnsafeshort.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal shortN shortVec(in shortN orig)
        {
            var vec = new shortN(in orig);
            _core->shortVectors.Add(in vec);   // persistent (backs Copy()); was wrongly the temp list
            return vec;
        }

        internal shortN tempshortVec(int N, bool uninit = false)
        {
            var vec = new shortN(N, in this, uninit);
            _core->tempshortVectors.Add(in vec);
            return vec;
        }

        internal shortN tempshortVec(in shortN orig)
        {
            var vec = new shortN(in orig);
            _core->tempshortVectors.Add(in vec);
            return vec;
        }
        #endregion

        #region MATRIX
        public shortMxN shortMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED (was leaking on Dispose).
            return shortMat(dim, dim, uninit);
        }

        public shortMxN shortMat(int M_rows, int N_cols, bool uninit = false)
        {
            var matrix = new shortMxN(M_rows, N_cols, in this, uninit);
            _core->shortMatrices.Add(in matrix);
            return matrix;
        }

        // creates vector with s values
        public shortMxN shortMat(int M_rows, int N_cols, short s)
        {
            var matrix = new shortMxN(M_rows, N_cols, in this, false);
            _core->shortMatrices.Add(in matrix);
            unsafe
            {
                mathUnsafeshort.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public shortMxN shortMat(in shortMxN orig)
        {
            var matrix = new shortMxN(in orig);
            _core->shortMatrices.Add(in matrix);
            return matrix;
        }

        internal shortMxN tempshortMat(int M_rows, int M_cols, bool uninit = false)
        {
            var matrix = new shortMxN(M_rows, M_cols, in this, uninit);
            _core->tempshortMatrices.Add(in matrix);
            return matrix;
        }

        internal shortMxN tempshortMat(in shortMxN orig)
        {
            var matrix = new shortMxN(orig);
            _core->tempshortMatrices.Add(in matrix);
            return matrix;
        }
        #endregion

        // --- debug pool checks (see Arena.fProxy) ---
        public bool isPersistent(in shortN v) {
            for (int i = 0; i < _core->shortVectors.Length; i++) if (_core->shortVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in shortN v) {
            for (int i = 0; i < _core->tempshortVectors.Length; i++) if (_core->tempshortVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isPersistent(in shortMxN m) {
            for (int i = 0; i < _core->shortMatrices.Length; i++) if (_core->shortMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in shortMxN m) {
            for (int i = 0; i < _core->tempshortMatrices.Length; i++) if (_core->tempshortMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }

    }

}
