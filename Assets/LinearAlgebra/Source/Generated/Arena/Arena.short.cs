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
            tempshortVectors.Add(in vec);
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
            return new shortMxN(dim, dim, in this, uninit);
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

    }

}