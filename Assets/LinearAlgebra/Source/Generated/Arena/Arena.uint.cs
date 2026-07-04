using Unity.Collections.LowLevel.Unsafe;


namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        internal UnsafeList<uintN> uintVectors;
        internal UnsafeList<uintMxN> uintMatrices;
        internal UnsafeList<uintN> uintTempVectors;
        internal UnsafeList<uintMxN> uintTempMatrices;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public uintN uintVec(int N, bool uninit = false) {

            var vec = new uintN(N, in this, uninit);
            _core->uintVectors.Add(in vec);
            return vec;
        }

        public uintN uintVec(int N, uint s)
        {
            var vec = new uintN(N, in this, true);
            _core->uintVectors.Add(in vec);
            unsafe {
                mathUnsafeuint.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal uintN uintVec(in uintN orig)
        {
            var vec = new uintN(in orig);
            _core->uintVectors.Add(in vec);   // persistent (backs Copy()); was wrongly the temp list
            return vec;
        }

        internal uintN uintTempVec(int N, bool uninit = false)
        {
            var vec = new uintN(N, in this, uninit);
            _core->uintTempVectors.Add(in vec);
            return vec;
        }

        internal uintN uintTempVec(in uintN orig)
        {
            var vec = new uintN(in orig);
            _core->uintTempVectors.Add(in vec);
            return vec;
        }
        #endregion

        #region MATRIX
        public uintMxN uintMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED (was leaking on Dispose).
            return uintMat(dim, dim, uninit);
        }

        public uintMxN uintMat(int M_rows, int N_cols, bool uninit = false)
        {
            var matrix = new uintMxN(M_rows, N_cols, in this, uninit);
            _core->uintMatrices.Add(in matrix);
            return matrix;
        }

        public uintMxN uintMat(int M_rows, int N_cols, uint s)
        {
            var matrix = new uintMxN(M_rows, N_cols, in this, false);
            _core->uintMatrices.Add(in matrix);
            unsafe
            {
                mathUnsafeuint.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public uintMxN uintMat(in uintMxN orig)
        {
            var matrix = new uintMxN(in orig);
            _core->uintMatrices.Add(in matrix);
            return matrix;
        }

        internal uintMxN uintTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            var matrix = new uintMxN(M_rows, M_cols, in this, uninit);
            _core->uintTempMatrices.Add(in matrix);
            return matrix;
        }

        internal uintMxN uintTempMat(in uintMxN orig)
        {
            var matrix = new uintMxN(orig);
            _core->uintTempMatrices.Add(in matrix);
            return matrix;
        }
        #endregion

        // --- debug pool checks (see Arena.fProxy) ---
        public bool isPersistent(in uintN v) {
            for (int i = 0; i < _core->uintVectors.Length; i++) if (_core->uintVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in uintN v) {
            for (int i = 0; i < _core->uintTempVectors.Length; i++) if (_core->uintTempVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isPersistent(in uintMxN m) {
            for (int i = 0; i < _core->uintMatrices.Length; i++) if (_core->uintMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in uintMxN m) {
            for (int i = 0; i < _core->uintTempMatrices.Length; i++) if (_core->uintTempMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }

    }

}
