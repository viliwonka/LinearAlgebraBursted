using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    public partial struct Arena {

        private UnsafeList<iProxyN> iProxyVectors;
        private UnsafeList<iProxyMxN> iProxyMatrices;
        private UnsafeList<iProxyN> tempiProxyVectors;
        private UnsafeList<iProxyMxN> tempiProxyMatrices;


        #region VECTOR
        
        public iProxyN iProxyVec(int N, bool uninit = false) {

            var vec = new iProxyN(N, in this, uninit);
            iProxyVectors.Add(in vec);
            return vec;
        }

        // creates vector with s values
        public iProxyN iProxyVec(int N, iProxy s)
        {
            var vec = new iProxyN(N, in this, true);
            iProxyVectors.Add(in vec);
            unsafe {
                mathUnsafeiProxy.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal iProxyN iProxyVec(in iProxyN orig)
        {
            var vec = new iProxyN(in orig);
            tempiProxyVectors.Add(in vec);
            return vec;
        }

        internal iProxyN tempiProxyVec(int N, bool uninit = false)
        {
            var vec = new iProxyN(N, in this, uninit);
            tempiProxyVectors.Add(in vec);
            return vec;
        }

        internal iProxyN tempiProxyVec(in iProxyN orig)
        {
            var vec = new iProxyN(in orig);
            tempiProxyVectors.Add(in vec);
            return vec;
        }
        #endregion

        #region MATRIX
        public iProxyMxN iProxyMat(int dim, bool uninit = false)
        {
            return new iProxyMxN(dim, dim, in this, uninit);
        }

        public iProxyMxN iProxyMat(int M_rows, int N_cols, bool uninit = false)
        {
            var matrix = new iProxyMxN(M_rows, N_cols, in this, uninit);
            iProxyMatrices.Add(in matrix);
            return matrix;
        }

        // creates vector with s values
        public iProxyMxN iProxyMat(int M_rows, int N_cols, iProxy s)
        {
            var matrix = new iProxyMxN(M_rows, N_cols, in this, false);
            iProxyMatrices.Add(in matrix);
            unsafe
            {
                mathUnsafeiProxy.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public iProxyMxN iProxyMat(in iProxyMxN orig)
        {
            var matrix = new iProxyMxN(in orig);
            iProxyMatrices.Add(in matrix);
            return matrix;
        }   

        internal iProxyMxN tempiProxyMat(int M_rows, int M_cols, bool uninit = false)
        {
            var matrix = new iProxyMxN(M_rows, M_cols, in this, uninit);
            tempiProxyMatrices.Add(in matrix);
            return matrix;
        }

        internal iProxyMxN tempiProxyMat(in iProxyMxN orig)
        {
            var matrix = new iProxyMxN(orig);
            tempiProxyMatrices.Add(in matrix);
            return matrix;
        }
        #endregion

    }

}