using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    public partial struct Arena {

        private UnsafeList<longN> longVectors;
        private UnsafeList<longMxN> longMatrices;
        private UnsafeList<longN> templongVectors;
        private UnsafeList<longMxN> templongMatrices;


        #region VECTOR
        
        public longN longVec(int N, bool uninit = false) {

            var vec = new longN(N, in this, uninit);
            longVectors.Add(in vec);
            return vec;
        }

        // creates vector with s values
        public longN longVec(int N, long s)
        {
            var vec = new longN(N, in this, true);
            longVectors.Add(in vec);
            unsafe {
                mathUnsafelong.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal longN longVec(in longN orig)
        {
            var vec = new longN(in orig);
            templongVectors.Add(in vec);
            return vec;
        }

        internal longN templongVec(int N, bool uninit = false)
        {
            var vec = new longN(N, in this, uninit);
            templongVectors.Add(in vec);
            return vec;
        }

        internal longN templongVec(in longN orig)
        {
            var vec = new longN(in orig);
            templongVectors.Add(in vec);
            return vec;
        }
        #endregion

        #region MATRIX
        public longMxN longMat(int dim, bool uninit = false)
        {
            return new longMxN(dim, dim, in this, uninit);
        }

        public longMxN longMat(int M_rows, int N_cols, bool uninit = false)
        {
            var matrix = new longMxN(M_rows, N_cols, in this, uninit);
            longMatrices.Add(in matrix);
            return matrix;
        }

        // creates vector with s values
        public longMxN longMat(int M_rows, int N_cols, long s)
        {
            var matrix = new longMxN(M_rows, N_cols, in this, false);
            longMatrices.Add(in matrix);
            unsafe
            {
                mathUnsafelong.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public longMxN longMat(in longMxN orig)
        {
            var matrix = new longMxN(in orig);
            longMatrices.Add(in matrix);
            return matrix;
        }   

        internal longMxN templongMat(int M_rows, int M_cols, bool uninit = false)
        {
            var matrix = new longMxN(M_rows, M_cols, in this, uninit);
            templongMatrices.Add(in matrix);
            return matrix;
        }

        internal longMxN templongMat(in longMxN orig)
        {
            var matrix = new longMxN(orig);
            templongMatrices.Add(in matrix);
            return matrix;
        }
        #endregion

    }

}