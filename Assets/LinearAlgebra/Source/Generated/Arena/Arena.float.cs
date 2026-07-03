using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        internal UnsafeList<floatN> floatVectors;
        internal UnsafeList<floatMxN> floatMatrices;
        internal UnsafeList<floatN> tempfloatVectors;
        internal UnsafeList<floatMxN> tempfloatMatrices;
    }

    public unsafe partial struct Arena {

        #region VECTOR

        public floatN floatVec(int N, bool uninit = false) {

            var vec = new floatN(N, in this, uninit);
            _core->floatVectors.Add(in vec);
            return vec;
        }

        public floatN floatVec(int N, float s)
        {
            var vec = new floatN(N, in this, true);
            _core->floatVectors.Add(in vec);
            unsafe {
                mathUnsafefloat.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal floatN floatVec(in floatN orig)
        {
            var vec = new floatN(in orig);
            _core->floatVectors.Add(in vec);   // persistent (backs Copy()); was wrongly the temp list
            return vec;
        }

        internal floatN tempfloatVec(int N, bool uninit = false)
        {
            var vec = new floatN(N, in this, uninit);
            _core->tempfloatVectors.Add(in vec);
            return vec;
        }

        internal floatN tempfloatVec(in floatN orig)
        {
            var vec = new floatN(in orig);
            _core->tempfloatVectors.Add(in vec);
            return vec;
        }
        #endregion

        #region MATRIX
        public floatMxN floatMat(int dim, bool uninit = false)
        {
            // forward to the (rows, cols) overload so the matrix is TRACKED in floatMatrices —
            // the direct `new floatMxN(...)` here was untracked and leaked on Dispose.
            return floatMat(dim, dim, uninit);
        }

        public floatMxN floatMat(int M_rows, int N_cols, bool uninit = false)
        {
            var matrix = new floatMxN(M_rows, N_cols, in this, uninit);
            _core->floatMatrices.Add(in matrix);
            return matrix;
        }

        public floatMxN floatMat(int M_rows, int N_cols, float s)
        {
            var matrix = new floatMxN(M_rows, N_cols, in this, false);
            _core->floatMatrices.Add(in matrix);
            unsafe
            {
                mathUnsafefloat.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public floatMxN floatMat(in floatMxN orig)
        {
            var matrix = new floatMxN(in orig);
            _core->floatMatrices.Add(in matrix);
            return matrix;
        }

        internal floatMxN tempfloatMat(int M_rows, int M_cols, bool uninit = false)
        {
            var matrix = new floatMxN(M_rows, M_cols, in this, uninit);
            _core->tempfloatMatrices.Add(in matrix);
            return matrix;
        }

        internal floatMxN tempfloatMat(in floatMxN orig)
        {
            var matrix = new floatMxN(orig);
            _core->tempfloatMatrices.Add(in matrix);
            return matrix;
        }
        #endregion

        // --- debug pool checks: confirm a buffer lives in the expected (persistent vs temp) list,
        //     e.g. to assert an op didn't silently move a persistent input into the temp pool ---
        public bool isPersistent(in floatN v) {
            for (int i = 0; i < _core->floatVectors.Length; i++) if (_core->floatVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in floatN v) {
            for (int i = 0; i < _core->tempfloatVectors.Length; i++) if (_core->tempfloatVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public bool isPersistent(in floatMxN m) {
            for (int i = 0; i < _core->floatMatrices.Length; i++) if (_core->floatMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }
        public bool isTemp(in floatMxN m) {
            for (int i = 0; i < _core->tempfloatMatrices.Length; i++) if (_core->tempfloatMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }

    }

}
