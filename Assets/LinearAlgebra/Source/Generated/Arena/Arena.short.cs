using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    public partial struct Arena {

        private UnsafeList<shortN> shortVectors;
        private UnsafeList<shortMxN> shortMatrices;
        private UnsafeList<shortN> tempshortVectors;
        private UnsafeList<shortMxN> tempshortMatrices;


        #region VECTOR
        
        public shortN shortVec(int N, bool uninit = false) {

            var vec = new shortN(N, in this, uninit);
            shortVectors.Add(in vec);
            return vec;
        }

        // creates vector with s values
        public shortN shortVec(int N, short s)
        {
            var vec = new shortN(N, in this, true);
            shortVectors.Add(in vec);
            unsafe {
                mathUnsafeshort.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal shortN shortVec(in shortN orig)
        {
            var vec = new shortN(in orig);
            shortVectors.Add(in vec);   // persistent (backs Copy()); was wrongly the temp list
            return vec;
        }

        internal shortN tempshortVec(int N, bool uninit = false)
        {
            var vec = new shortN(N, in this, uninit);
            tempshortVectors.Add(in vec);
            return vec;
        }

        internal shortN tempshortVec(in shortN orig)
        {
            var vec = new shortN(in orig);
            tempshortVectors.Add(in vec);
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
            shortMatrices.Add(in matrix);
            return matrix;
        }

        // creates vector with s values
        public shortMxN shortMat(int M_rows, int N_cols, short s)
        {
            var matrix = new shortMxN(M_rows, N_cols, in this, false);
            shortMatrices.Add(in matrix);
            unsafe
            {
                mathUnsafeshort.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public shortMxN shortMat(in shortMxN orig)
        {
            var matrix = new shortMxN(in orig);
            shortMatrices.Add(in matrix);
            return matrix;
        }   

        internal shortMxN tempshortMat(int M_rows, int M_cols, bool uninit = false)
        {
            var matrix = new shortMxN(M_rows, M_cols, in this, uninit);
            tempshortMatrices.Add(in matrix);
            return matrix;
        }

        internal shortMxN tempshortMat(in shortMxN orig)
        {
            var matrix = new shortMxN(orig);
            tempshortMatrices.Add(in matrix);
            return matrix;
        }
        #endregion

        // --- debug pool checks (see Arena.fProxy) ---
        public unsafe bool DB_isPersistent(in shortN v) {
            for (int i = 0; i < shortVectors.Length; i++) if (shortVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public unsafe bool DB_isTemp(in shortN v) {
            for (int i = 0; i < tempshortVectors.Length; i++) if (tempshortVectors[i].Data.Ptr == v.Data.Ptr) return true;
            return false;
        }
        public unsafe bool DB_isPersistent(in shortMxN m) {
            for (int i = 0; i < shortMatrices.Length; i++) if (shortMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }
        public unsafe bool DB_isTemp(in shortMxN m) {
            for (int i = 0; i < tempshortMatrices.Length; i++) if (tempshortMatrices[i].Data.Ptr == m.Data.Ptr) return true;
            return false;
        }

    }

}