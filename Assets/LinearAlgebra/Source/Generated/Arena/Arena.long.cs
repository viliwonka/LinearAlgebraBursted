using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        internal UnsafeList<longN> longVectors;
        internal UnsafeList<longMxN> longMatrices;
        internal UnsafeList<longN> templongVectors;
        internal UnsafeList<longMxN> templongMatrices;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public longN longVec(int N, bool uninit = false) {

            var vec = new longN(N, in this, uninit);
            _core->longVectors.Add(in vec);
            return vec;
        }

        // creates vector with s values
        public longN longVec(int N, long s)
        {
            var vec = new longN(N, in this, true);
            _core->longVectors.Add(in vec);
            unsafe {
                mathUnsafelong.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal longN longVec(in longN orig)
        {
            var vec = new longN(in orig);
            _core->longVectors.Add(in vec);   // persistent (backs Copy()); was wrongly the temp list
            return vec;
        }

        internal longN templongVec(int N, bool uninit = false)
        {
            var vec = new longN(N, in this, uninit);
            _core->templongVectors.Add(in vec);
            return vec;
        }

        internal longN templongVec(in longN orig)
        {
            var vec = new longN(in orig);
            _core->templongVectors.Add(in vec);
            return vec;
        }
        #endregion

        #region MATRIX
        public longMxN longMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED (was leaking on Dispose).
            return longMat(dim, dim, uninit);
        }

        public longMxN longMat(int M_rows, int N_cols, bool uninit = false)
        {
            var matrix = new longMxN(M_rows, N_cols, in this, uninit);
            _core->longMatrices.Add(in matrix);
            return matrix;
        }

        // creates vector with s values
        public longMxN longMat(int M_rows, int N_cols, long s)
        {
            var matrix = new longMxN(M_rows, N_cols, in this, false);
            _core->longMatrices.Add(in matrix);
            unsafe
            {
                mathUnsafelong.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public longMxN longMat(in longMxN orig)
        {
            var matrix = new longMxN(in orig);
            _core->longMatrices.Add(in matrix);
            return matrix;
        }

        internal longMxN templongMat(int M_rows, int M_cols, bool uninit = false)
        {
            var matrix = new longMxN(M_rows, M_cols, in this, uninit);
            _core->templongMatrices.Add(in matrix);
            return matrix;
        }

        internal longMxN templongMat(in longMxN orig)
        {
            var matrix = new longMxN(orig);
            _core->templongMatrices.Add(in matrix);
            return matrix;
        }
        #endregion

        // --- debug pool checks (see Arena.fProxy) ---
        public bool isPersistent(in longN v) {
            for (int i = 0; i < _core->longVectors.Length; i++) if (_core->longVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in longN v) {
            for (int i = 0; i < _core->templongVectors.Length; i++) if (_core->templongVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isPersistent(in longMxN m) {
            for (int i = 0; i < _core->longMatrices.Length; i++) if (_core->longMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in longMxN m) {
            for (int i = 0; i < _core->templongMatrices.Length; i++) if (_core->templongMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }

    }

}
