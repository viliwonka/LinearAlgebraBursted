using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    public partial struct Arena {

        private UnsafeList<intN> intVectors;
        private UnsafeList<intMxN> intMatrices;
        private UnsafeList<intN> tempintVectors;
        private UnsafeList<intMxN> tempintMatrices;


        #region VECTOR
        
        public intN intVec(int N, bool uninit = false) {

            var vec = new intN(N, in this, uninit);
            intVectors.Add(in vec);
            return vec;
        }

        // creates vector with s values
        public intN intVec(int N, int s)
        {
            var vec = new intN(N, in this, true);
            intVectors.Add(in vec);
            unsafe {
                mathUnsafeint.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal intN intVec(in intN orig)
        {
            var vec = new intN(in orig);
            tempintVectors.Add(in vec);
            return vec;
        }

        internal intN tempintVec(int N, bool uninit = false)
        {
            var vec = new intN(N, in this, uninit);
            tempintVectors.Add(in vec);
            return vec;
        }

        internal intN tempintVec(in intN orig)
        {
            var vec = new intN(in orig);
            tempintVectors.Add(in vec);
            return vec;
        }
        #endregion

        #region MATRIX
        public intMxN intMat(int dim, bool uninit = false)
        {
            return new intMxN(dim, dim, in this, uninit);
        }

        public intMxN intMat(int M_rows, int N_cols, bool uninit = false)
        {
            var matrix = new intMxN(M_rows, N_cols, in this, uninit);
            intMatrices.Add(in matrix);
            return matrix;
        }

        // creates vector with s values
        public intMxN intMat(int M_rows, int N_cols, int s)
        {
            var matrix = new intMxN(M_rows, N_cols, in this, false);
            intMatrices.Add(in matrix);
            unsafe
            {
                mathUnsafeint.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public intMxN intMat(in intMxN orig)
        {
            var matrix = new intMxN(in orig);
            intMatrices.Add(in matrix);
            return matrix;
        }   

        internal intMxN tempintMat(int M_rows, int M_cols, bool uninit = false)
        {
            var matrix = new intMxN(M_rows, M_cols, in this, uninit);
            tempintMatrices.Add(in matrix);
            return matrix;
        }

        internal intMxN tempintMat(in intMxN orig)
        {
            var matrix = new intMxN(orig);
            tempintMatrices.Add(in matrix);
            return matrix;
        }
        #endregion

    }

}