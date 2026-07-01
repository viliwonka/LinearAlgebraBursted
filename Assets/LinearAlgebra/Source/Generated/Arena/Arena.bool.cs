//using System;
namespace LinearAlgebra
{

    // Allocation helper
    public partial struct Arena {

        #region BOOLVECTOR
        public boolN boolVec(int N, bool uninit = false)
        {
            var vec = new boolN(N, in this, uninit);
            BoolVectors.Add(in vec);
            return vec;
        }

        public boolN tempBoolVec(int N, bool uninit = false)
        {
            var vec = new boolN(N, in this, uninit);
            TempBoolVectors.Add(in vec);
            return vec;
        }

        internal boolN boolVec(in boolN orig)
        {
            var vec = new boolN(in orig);
            BoolVectors.Add(in vec);
            return vec;
        }

        internal boolN tempBoolVec(in boolN orig)
        {
            var vec = new boolN(in orig);
            TempBoolVectors.Add(in vec);
            return vec;
        }

        #endregion

        #region BOOLMATRIX
        
        public boolMxN boolMat(int M_rows, int N_cols, bool uninit = false)
        {
            var matrix = new boolMxN(M_rows, N_cols, in this, uninit);
            BoolMatrices.Add(in matrix);
            return matrix;
        }

        public boolMxN boolMat(in boolMxN mat)
        {
            var matrix = new boolMxN(mat);
            BoolMatrices.Add(in matrix);
            return matrix;
        }

        public boolMxN tempBoolMat(int M_rows, int N_cols, bool uninit = false)
        {
            var matrix = new boolMxN(M_rows, N_cols, in this, uninit);
            TempBoolMatrices.Add(in matrix);
            return matrix;
        }

        public boolMxN tempBoolMat(in boolMxN mat)
        {
            var matrix = new boolMxN(mat);
            TempBoolMatrices.Add(in matrix);
            return matrix;
        }

        #endregion

    }

}