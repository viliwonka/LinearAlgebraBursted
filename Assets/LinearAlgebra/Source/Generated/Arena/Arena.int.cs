using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Internal;


namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        internal UnsafeList<intN> intVectors;
        internal UnsafeList<intMxN> intMatrices;
        internal UnsafeList<intN> intTempVectors;
        internal UnsafeList<intMxN> intTempMatrices;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public intN intVec(int N, bool uninit = false) {

            var vec = new intN(N, in this, uninit);
            _core->intVectors.Add(in vec);
            return vec;
        }

        public intN intVec(int N, int s)
        {
            var vec = new intN(N, in this, true);
            _core->intVectors.Add(in vec);
            unsafe {
                UnsafeMathOP.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal intN intVec(in intN orig)
        {
            var vec = new intN(in orig);
            _core->intVectors.Add(in vec);   // persistent (backs Copy()); was wrongly the temp list
            return vec;
        }

        internal intN intTempVec(int N, bool uninit = false)
        {
            var vec = new intN(N, in this, uninit);
            _core->intTempVectors.Add(in vec);
            return vec;
        }

        internal intN intTempVec(in intN orig)
        {
            var vec = new intN(in orig);
            _core->intTempVectors.Add(in vec);
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

        public intMxN intMat(int M_rows, int N_cols, int s)
        {
            var matrix = new intMxN(M_rows, N_cols, in this, false);
            _core->intMatrices.Add(in matrix);
            unsafe
            {
                UnsafeMathOP.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public intMxN intMat(in intMxN orig)
        {
            var matrix = new intMxN(in orig);
            _core->intMatrices.Add(in matrix);
            return matrix;
        }

        internal intMxN intTempMat(int M_rows, int M_cols, bool uninit = false)
        {
            var matrix = new intMxN(M_rows, M_cols, in this, uninit);
            _core->intTempMatrices.Add(in matrix);
            return matrix;
        }

        internal intMxN intTempMat(in intMxN orig)
        {
            var matrix = new intMxN(orig);
            _core->intTempMatrices.Add(in matrix);
            return matrix;
        }
        #endregion

        // --- debug pool checks (see Arena.fProxy) ---
        public bool isPersistent(in intN v) {
            for (int i = 0; i < _core->intVectors.Length; i++) if (_core->intVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in intN v) {
            for (int i = 0; i < _core->intTempVectors.Length; i++) if (_core->intTempVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isPersistent(in intMxN m) {
            for (int i = 0; i < _core->intMatrices.Length; i++) if (_core->intMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in intMxN m) {
            for (int i = 0; i < _core->intTempMatrices.Length; i++) if (_core->intTempMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }

    }

}
