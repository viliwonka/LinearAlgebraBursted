using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    public partial struct Arena {

        private UnsafeList<doubleN> doubleVectors;
        private UnsafeList<doubleMxN> doubleMatrices;
        private UnsafeList<doubleN> tempdoubleVectors;
        private UnsafeList<doubleMxN> tempdoubleMatrices;


        #region VECTOR
        
        public doubleN doubleVec(int N, bool uninit = false) {

            var vec = new doubleN(N, in this, uninit);
            doubleVectors.Add(in vec);
            return vec;
        }

        // creates vector with s values
        public doubleN doubleVec(int N, float s)
        {
            var vec = new doubleN(N, in this, true);
            doubleVectors.Add(in vec);
            unsafe {
                mathUnsafedouble.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        internal doubleN doubleVec(in doubleN orig)
        {
            var vec = new doubleN(in orig);
            tempdoubleVectors.Add(in vec);
            return vec;
        }

        internal doubleN tempdoubleVec(int N, bool uninit = false)
        {
            var vec = new doubleN(N, in this, uninit);
            tempdoubleVectors.Add(in vec);
            return vec;
        }

        internal doubleN tempdoubleVec(in doubleN orig)
        {
            var vec = new doubleN(in orig);
            tempdoubleVectors.Add(in vec);
            return vec;
        }
        #endregion

        #region MATRIX
        public doubleMxN doubleMat(int dim, bool uninit = false)
        {
            return new doubleMxN(dim, dim, in this, uninit);
        }

        public doubleMxN doubleMat(int M_rows, int N_cols, bool uninit = false)
        {
            var matrix = new doubleMxN(M_rows, N_cols, in this, uninit);
            doubleMatrices.Add(in matrix);
            return matrix;
        }

        // creates vector with s values
        public doubleMxN doubleMat(int M_rows, int N_cols, float s)
        {
            var matrix = new doubleMxN(M_rows, N_cols, in this, false);
            doubleMatrices.Add(in matrix);
            unsafe
            {
                mathUnsafedouble.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        public doubleMxN doubleMat(in doubleMxN orig)
        {
            var matrix = new doubleMxN(in orig);
            doubleMatrices.Add(in matrix);
            return matrix;
        }   

        internal doubleMxN tempdoubleMat(int M_rows, int M_cols, bool uninit = false)
        {
            var matrix = new doubleMxN(M_rows, M_cols, in this, uninit);
            tempdoubleMatrices.Add(in matrix);
            return matrix;
        }

        internal doubleMxN tempdoubleMat(in doubleMxN orig)
        {
            var matrix = new doubleMxN(orig);
            tempdoubleMatrices.Add(in matrix);
            return matrix;
        }
        #endregion

    }

}