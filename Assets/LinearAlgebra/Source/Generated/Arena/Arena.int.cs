using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        internal UnsafeList<intN> intVectors;
        internal UnsafeList<intMxN> intMatrices;
        internal UnsafeList<intN> tempintVectors;
        internal UnsafeList<intMxN> tempintMatrices;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public intN intVec(int N, bool uninit = false) {

            var vec = new intN(N, in this, uninit);
            _core->intVectors.Add(in vec);
            return vec;
        }

        // creates vector with s values
        public intN intVec(int N, int s)
        {
            var vec = new intN(N, in this, true);
            _core->intVectors.Add(in vec);
            unsafe {
                mathUnsafeint.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal intN intVec(in intN orig)
        {
            var vec = new intN(in orig);
            _core->intVectors.Add(in vec);   // persistent (backs Copy()); was wrongly the temp list
            return vec;
        }

        internal intN tempintVec(int N, bool uninit = false)
        {
            var vec = new intN(N, in this, uninit);
            _core->tempintVectors.Add(in vec);
            return vec;
        }

        internal intN tempintVec(in intN orig)
        {
            var vec = new intN(in orig);
            _core->tempintVectors.Add(in vec);
            return vec;
        }
        #endregion

        #region MATRIX
        public intMxN intMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED (was leaking on Dispose).
            return intMat(dim, dim, uninit);
        }

        public intMxN intMat(int M_rows, int N_cols, bool uninit = false)
        {
            var matrix = new intMxN(M_rows, N_cols, in this, uninit);
            _core->intMatrices.Add(in matrix);
            return matrix;
        }

        // creates vector with s values
        public intMxN intMat(int M_rows, int N_cols, int s)
        {
            var matrix = new intMxN(M_rows, N_cols, in this, false);
            _core->intMatrices.Add(in matrix);
            unsafe
            {
                mathUnsafeint.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public intMxN intMat(in intMxN orig)
        {
            var matrix = new intMxN(in orig);
            _core->intMatrices.Add(in matrix);
            return matrix;
        }

        internal intMxN tempintMat(int M_rows, int M_cols, bool uninit = false)
        {
            var matrix = new intMxN(M_rows, M_cols, in this, uninit);
            _core->tempintMatrices.Add(in matrix);
            return matrix;
        }

        internal intMxN tempintMat(in intMxN orig)
        {
            var matrix = new intMxN(orig);
            _core->tempintMatrices.Add(in matrix);
            return matrix;
        }
        #endregion

        // --- debug pool checks (see Arena.fProxy) ---
        public bool isPersistent(in intN v) {
            for (int i = 0; i < _core->intVectors.Length; i++) if (_core->intVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in intN v) {
            for (int i = 0; i < _core->tempintVectors.Length; i++) if (_core->tempintVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isPersistent(in intMxN m) {
            for (int i = 0; i < _core->intMatrices.Length; i++) if (_core->intMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in intMxN m) {
            for (int i = 0; i < _core->tempintMatrices.Length; i++) if (_core->tempintMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }

    }

}
